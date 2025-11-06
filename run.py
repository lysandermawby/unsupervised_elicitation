"""
End-to-end script to run the complete ICM reproduction.
Runs ICM search on training data and evaluates the test data on all baselines (zero-shot, zero-shot (chat), and golden labels)
"""

import argparse
import subprocess
import sys
from pathlib import Path
import json


def run_command(cmd, description):
    """Run a command and handle errors."""
    print("\n" + "="*60)
    print(f"STEP: {description}")
    print("="*60)
    print(f"Running: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        print(f"Error: {description} failed with code {result.returncode}")
        sys.exit(1)

    print(f"{description} completed successfully")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run complete ICM reproduction pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test run (fast, ~30 min)
  python run_reproduction.py --quick

  # Full reproduction (slow, several hours)
  python run_reproduction.py --full

  # Custom settings
  python run_reproduction.py --sample-size 100 --max-iterations 200
        """
    )

    # Presets
    preset = parser.add_mutually_exclusive_group()
    preset.add_argument(
        "--quick",
        action="store_true",
        help="Quick test run (50 examples, 100 iterations, skip zero-shot)"
    )
    preset.add_argument(
        "--full",
        action="store_true",
        help="Full reproduction (256 examples, 500 iterations, with zero-shot)"
    )

    # Data parameters
    parser.add_argument(
        "--train-data",
        type=str,
        default="data/train_truthfulqa.json",
        help="Path to training data"
    )
    parser.add_argument(
        "--test-data",
        type=str,
        default="data/test_truthfulqa.json",
        help="Path to test data"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Number of training examples (None for all)"
    )

    # ICM parameters
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=500,
        help="ICM search iterations"
    )
    parser.add_argument(
        "--initial-examples",
        type=int,
        default=20,
        help="Initial random labels"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Mutual predictability weight"
    )

    # Evaluation parameters
    parser.add_argument(
        "--skip-zero-shot-base",
        action="store_true",
        help="Skip zero-shot base model evaluation (saves API calls)"
    )
    parser.add_argument(
        "--skip-zero-shot-chat",
        action="store_true",
        help="Skip zero-shot chat model evaluation (saves API calls)"
    )
    parser.add_argument(
        "--max-test-examples",
        type=int,
        default=None,
        help="Limit test examples (for faster evaluation)"
    )

    # Output parameters
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory"
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Output name prefix"
    )

    # System parameters
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--skip-icm",
        action="store_true",
        help="Skip ICM training (use existing results)"
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip evaluation (only run ICM)"
    )
    parser.add_argument(
        "--skip-plot",
        action="store_true",
        help="Skip plotting (only run ICM and eval)"
    )

    return parser.parse_args()


def main():
    """Run the complete pipeline."""
    args = parse_args()

    # Apply presets
    if args.quick:
        print("Using QUICK preset (fast test run)")
        args.sample_size = 50
        args.max_iterations = 100
        args.skip_zero_shot_base = True
        args.skip_zero_shot_chat = True
        args.initial_examples = 10
    elif args.full:
        print("Using FULL preset (complete reproduction)")
        args.sample_size = 256
        args.max_iterations = 500
        args.skip_zero_shot_base = False
        args.skip_zero_shot_chat = False

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("ICM REPRODUCTION PIPELINE")
    print("="*60)
    print(f"Training data: {args.train_data}")
    print(f"Test data: {args.test_data}")
    print(f"Sample size: {args.sample_size or 'all'}")
    print(f"Iterations: {args.max_iterations}")
    print(f"Output: {output_dir}")
    print("="*60)

    # Step 1: Run ICM search
    if not args.skip_icm:
        icm_cmd = [
            sys.executable, "-m", "icm.main",
            "--data-path", args.train_data,
            "--output-dir", args.output_dir,
            "--max-iterations", str(args.max_iterations),
            "--initial-examples", str(args.initial_examples),
            "--alpha", str(args.alpha),
            "--seed", str(args.seed)
        ]

        if args.sample_size:
            icm_cmd.extend(["--sample-size", str(args.sample_size)])

        if args.output_name:
            icm_cmd.extend(["--output-name", args.output_name])

        run_command(icm_cmd, "ICM Search (Training)")

        # Find the ICM results file
        if args.output_name:
            icm_results_pattern = f"{args.output_name}_result.json"
        else:
            icm_results_pattern = "*_result.json"

        icm_results_files = list(output_dir.glob(icm_results_pattern))

        if not icm_results_files:
            print(f"\n❌ Error: Could not find ICM results file matching {icm_results_pattern}")
            sys.exit(1)

        icm_results = str(icm_results_files[0])
    else:
        print("\n⏭️  Skipping ICM search (using existing results)")

        # Find existing results
        icm_results_files = list(output_dir.glob("*_result.json"))

        if not icm_results_files:
            print(f"\n❌ Error: No existing ICM results found in {output_dir}")
            sys.exit(1)

        icm_results = str(icm_results_files[0])
        print(f"Using existing results: {icm_results}")

    # Step 2: Evaluate
    if not args.skip_eval:
        eval_output = output_dir / "evaluation_results.json"

        eval_cmd = [
            sys.executable, "evaluate.py",
            "--train-data", args.train_data,
            "--test-data", args.test_data,
            "--icm-results", icm_results,
            "--output", str(eval_output),
            "--seed", str(args.seed)
        ]

        if args.skip_zero_shot_base:
            eval_cmd.append("--skip-zero-shot-base")

        if args.skip_zero_shot_chat:
            eval_cmd.append("--skip-zero-shot-chat")

        if args.max_test_examples:
            eval_cmd.extend(["--max-examples", str(args.max_test_examples)])

        run_command(eval_cmd, "Evaluation (Test Set)")
    else:
        print("\n⏭️  Skipping evaluation")
        eval_output = output_dir / "evaluation_results.json"

    # Step 3: Plot results
    if not args.skip_plot:
        plot_output = output_dir / "results_plot.png"

        plot_cmd = [
            sys.executable, "plot_results.py",
            "--results", str(eval_output),
            "--output", str(plot_output),
            "--title", "TruthfulQA Performance (ICM Reproduction)"
        ]

        run_command(plot_cmd, "Visualization")

        print("\n" + "="*60)
        print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nGenerated files:")
        print(f"  ICM results:  {icm_results}")
        print(f"  Evaluation:   {eval_output}")
        print(f"  Plot:         {plot_output}")

        # Print evaluation summary
        if eval_output.exists():
            with open(eval_output, 'r') as f:
                eval_data = json.load(f)

            print("\n" + "="*60)
            print("RESULTS SUMMARY")
            print("="*60)
            for result in eval_data["results"]:
                method = result["method"].replace("_", " ").title()
                accuracy = result["accuracy"] * 100
                print(f"  {method:20s}: {accuracy:5.1f}%")
            print("="*60)

        print(f"\nView the plot: open {plot_output}")
        print(f"Or open: {str(output_dir)}")

    else:
        print("Skipping plotting")


if __name__ == "__main__":
    main()
