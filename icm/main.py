"""
CLI for Internal Coherence Maximization (ICM).
"""

import argparse
import json
from pathlib import Path

# Local imports
from icm.dataset_utils import load_truthfulqa_from_json
from icm.icm import ICMSearcher


def save_labeled_examples_jsonl(labeled_examples, filepath):
    """Save labeled examples to JSONL format."""
    with open(filepath, 'w') as f:
        for example in labeled_examples:
            f.write(json.dumps(example) + '\n')
    print(f"Saved labeled examples to {filepath}")


def save_statistics(result, filepath):
    """Save detailed statistics to JSON."""
    label_counts = {}
    for ex in result.labeled_examples:
        label = ex["label"]
        label_counts[label] = label_counts.get(label, 0) + 1

    stats = {
        "num_examples": len(result.labeled_examples),
        "final_score": result.score,
        "iterations": result.iterations,
        "convergence_info": result.convergence_info,
        "metadata": result.metadata,
        "label_distribution": label_counts,
        "label_percentages": {
            label: (count / len(result.labeled_examples)) * 100
            for label, count in label_counts.items()
        }
    }

    with open(filepath, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Saved statistics to {filepath}")


def parse_args():
    """Command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Internal Coherence Maximization (ICM) - Unsupervised Elicitation"
    )

    # Data parameters
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/train_truthfulqa.json",
        help="Path to TruthfulQA JSON file"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Number of examples to use (None for all)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="icm_results",
        help="Output directory"
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Output file name prefix (auto-generated if not provided)"
    )

    # Model parameters
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Meta-Llama-3.1-405B",
        help="Hyperbolic model name"
    )

    # ICM algorithm parameters
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Weight for mutual predictability"
    )
    parser.add_argument(
        "--initial-temperature",
        type=float,
        default=3.0,
        help="Initial temperature for simulated annealing"
    )
    parser.add_argument(
        "--final-temperature",
        type=float,
        default=0.001,
        help="Final temperature for simulated annealing"
    )
    parser.add_argument(
        "--cooling-rate",
        type=float,
        default=0.98,
        help="Temperature cooling rate"
    )
    parser.add_argument(
        "--initial-examples",
        type=int,
        default=20,
        help="Number of initial randomly labeled examples (K)"
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=1000,
        help="Maximum iterations"
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Maximum number of examples to process"
    )

    # Generation parameters
    parser.add_argument(
        "--generation-temperature",
        type=float,
        default=0.2,
        help="Temperature for text generation"
    )
    parser.add_argument(
        "--generation-top-p",
        type=float,
        default=0.9,
        help="Top-p for generation"
    )

    # System parameters
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    return parser.parse_args()


def main():
    """Run the ICM algorithm on TruthfulQA dataset."""
    args = parse_args()

    print("="*60)
    print("Internal Coherence Maximization (ICM)")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Data: {args.data_path}")
    print(f"Parameters: alpha={args.alpha}, temp={args.initial_temperature}->{args.final_temperature}")
    print("="*60)

    # Load TruthfulQA dataset
    print("\nLoading TruthfulQA dataset...")
    dataset = load_truthfulqa_from_json(
        json_path=args.data_path,
        sample_size=args.sample_size,
        seed=args.seed
    )

    # Print dataset info
    print(f"Loaded {len(dataset)} examples")
    stats = dataset.get_stats()
    print(f"Average input length: {stats['avg_input_length']:.1f} characters")

    # Create ICM searcher
    print("\nInitializing ICM searcher...")
    searcher = ICMSearcher(
        model_name=args.model,
        alpha=args.alpha,
        initial_temperature=args.initial_temperature,
        final_temperature=args.final_temperature,
        cooling_rate=args.cooling_rate,
        initial_examples=args.initial_examples,
        max_iterations=args.max_iterations,
        generation_temperature=args.generation_temperature,
        generation_top_p=args.generation_top_p,
        seed=args.seed
    )

    # Run ICM search
    print("\nStarting ICM search...")
    result = searcher.search(
        dataset=dataset,
        max_examples=args.max_examples
    )

    print("\nICM search completed!")
    print(f"Final score: {result.score:.4f}")
    print(f"Generated {len(result.labeled_examples)} labeled examples")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate output name
    if args.output_name:
        output_name = args.output_name
    else:
        model_name = args.model.split('/')[-1].replace('-', '_').lower()
        output_name = f"truthfulqa_{model_name}_icm"

    # Save results
    print(f"\nSaving results to {output_dir}...")

    # 1. Save full result as JSON
    result_path = output_dir / f"{output_name}_result.json"
    result.save_to_json(str(result_path))

    # 2. Save labeled examples as JSONL
    examples_path = output_dir / f"{output_name}_labeled.jsonl"
    save_labeled_examples_jsonl(result.labeled_examples, str(examples_path))

    # 3. Save statistics
    stats_path = output_dir / f"{output_name}_stats.json"
    save_statistics(result, str(stats_path))

    # Print summary statistics
    print("\n" + "="*60)
    print("LABEL DISTRIBUTION:")
    print("="*60)

    label_counts = {}
    for ex in result.labeled_examples:
        label = ex["label"]
        label_counts[label] = label_counts.get(label, 0) + 1

    for label in sorted(label_counts.keys()):
        count = label_counts[label]
        percentage = (count / len(result.labeled_examples)) * 100
        print(f"{label:10s}: {count:4d} ({percentage:5.1f}%)")

    print("="*60)
    print(f"\nResults saved to: {output_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
