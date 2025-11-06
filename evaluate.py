"""
Evaluation script for ICM and baselines on TruthfulQA.

This script evaluates:
1. Zero-shot (Base) - Base model with HHH system prompt
2. Zero-shot (Chat) - Instruct model without demonstrations
3. Golden Labels - Many-shot prompting with ground truth
4. ICM - Many-shot prompting with discovered labels
"""

import json
import random
import argparse
from typing import Dict, List, Any
from tqdm import tqdm
from dotenv import load_dotenv

from icm.dataset_utils import load_truthfulqa_from_json, ICMExample
from icm.hyperbolic_client import HyperbolicClient

# Load HHH system prompt
HHH_PROMPT = open("HHH.txt", "r").read().strip()


def evaluate_labels(examples: List[ICMExample], predicted_labels: List[str]) -> Dict[str, float]:
    """
    Evaluate predicted labels against ground truth.

    Args:
        examples: List of ICMExample with ground truth in metadata
        predicted_labels: List of predicted labels

    Returns:
        Dictionary with accuracy metrics
    """
    assert len(examples) == len(predicted_labels), "Mismatch in number of examples and labels"

    correct = 0
    total = len(examples)

    for example, pred_label in zip(examples, predicted_labels):
        ground_truth = example.metadata.get("ground_truth_label", "True")
        if pred_label == ground_truth:
            correct += 1

    accuracy = correct / total if total > 0 else 0.0

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total
    }


def predict_with_demonstrations(
    test_examples: List[ICMExample],
    demo_examples: List[Dict[str, str]],
    model_name: str = "meta-llama/Meta-Llama-3.1-405B",
    max_demos: int = 20
) -> List[str]:
    """
    Predict labels for test examples using in-context learning with demonstrations.

    Args:
        test_examples: Test examples to predict
        demo_examples: Demonstration examples with {"input": ..., "label": ...}
        model_name: Model to use
        max_demos: Maximum number of demonstrations to use

    Returns:
        List of predicted labels
    """
    load_dotenv()
    client = HyperbolicClient()

    # Limit demonstrations to avoid context length issues
    if len(demo_examples) > max_demos:
        demo_examples = random.sample(demo_examples, max_demos)

    predicted_labels = []

    for example in tqdm(test_examples, desc="Predicting labels"):
        # Build prompt with demonstrations (matching reference implementation)
        prompt_parts = []

        # Add demonstrations
        # Format: Question: X\nClaim: Y\nI think this Claim is True/False\n\n
        for demo in demo_examples:
            prompt_parts.append(demo["input"])
            prompt_parts.append(f" {demo['label']}")
            prompt_parts.append("")

        # Add test example
        # Format: Question: X\nClaim: Y\nI think this Claim is
        prompt_parts.append(example.input_text)

        prompt = "\n".join(prompt_parts)

        try:
            response = client.complete(
                model=model_name,
                prompt=prompt,
                temperature=0.0,  # Deterministic
                max_tokens=5,
                top_p=0.9
            )

            # Extract label
            text = response.strip().upper()
            if "TRUE" in text:
                label = "True"
            elif "FALSE" in text:
                label = "False"
            else:
                label = random.choice(["True", "False"])

            predicted_labels.append(label)

        except Exception as e:
            print(f"Error: {e}")
            predicted_labels.append(random.choice(["True", "False"]))

    return predicted_labels


def evaluate_golden_labels(train_dataset, test_dataset, model_name: str) -> Dict[str, Any]:
    """
    Evaluate using golden/ground truth labels as demonstrations.

    This tests: what if we had perfect human labels as training data?
    """
    print("\n" + "="*60)
    print("Evaluating: Golden Labels (Supervised)")
    print("="*60)

    # Create demonstrations from training data with ground truth labels
    demo_examples = [
        {
            "input": example.input_text,
            "label": example.metadata.get("ground_truth_label", "True")
        }
        for example in train_dataset.examples
    ]

    # Predict on test set using these demonstrations
    predicted_labels = predict_with_demonstrations(
        test_dataset.examples,
        demo_examples,
        model_name=model_name
    )

    results = evaluate_labels(test_dataset.examples, predicted_labels)
    print(f"Accuracy: {results['accuracy']:.4f} ({results['correct']}/{results['total']})")

    return {
        "method": "golden_labels",
        "accuracy": results["accuracy"],
        "correct": results["correct"],
        "total": results["total"]
    }


def evaluate_random_baseline(dataset, seed: int = 42) -> Dict[str, Any]:
    """Evaluate random baseline (50/50 True/False)."""
    print("\n" + "="*60)
    print("Evaluating: Random Baseline")
    print("="*60)

    random.seed(seed)
    predicted_labels = [
        random.choice(["True", "False"])
        for _ in dataset.examples
    ]

    results = evaluate_labels(dataset.examples, predicted_labels)
    print(f"Accuracy: {results['accuracy']:.4f} ({results['correct']}/{results['total']})")

    return {
        "method": "random_baseline",
        "accuracy": results["accuracy"],
        "correct": results["correct"],
        "total": results["total"]
    }


def evaluate_zero_shot_base(
    dataset,
    model_name: str = "meta-llama/Meta-Llama-3.1-405B",
    max_examples: int = None
) -> Dict[str, Any]:
    """
    Evaluate zero-shot baseline on BASE model with HHH system prompt.

    This tests: how well does the base model do with an optimized assistant prompt?

    Args:
        dataset: ICMDataset
        model_name: Hyperbolic model name (should be base model)
        max_examples: Maximum examples to evaluate (for speed)

    Returns:
        Evaluation results dictionary
    """
    print("\n" + "="*60)
    print("Evaluating: Zero-Shot (Base Model + HHH Prompt)")
    print("="*60)

    load_dotenv()
    client = HyperbolicClient()

    examples = dataset.examples
    if max_examples:
        examples = examples[:max_examples]

    predicted_labels = []

    for example in tqdm(examples, desc="Zero-shot base predictions"):
        # Build prompt with HHH system prompt
        # The HHH prompt sets up helpful/honest behavior, then we use the standard format
        prompt = f"{HHH_PROMPT}\n\n-----\n\nHuman: {example.input_text}\n\nAssistant:"

        try:
            response = client.complete(
                model=model_name,
                prompt=prompt,
                temperature=0.0,  # Deterministic
                max_tokens=10,
                top_p=0.9
            )

            # Extract label
            text = response.strip().upper()
            if "TRUE" in text:
                label = "True"
            elif "FALSE" in text:
                label = "False"
            else:
                label = random.choice(["True", "False"])

            predicted_labels.append(label)

        except Exception as e:
            print(f"Error: {e}")
            predicted_labels.append(random.choice(["True", "False"]))

    results = evaluate_labels(examples, predicted_labels)
    print(f"Accuracy: {results['accuracy']:.4f} ({results['correct']}/{results['total']})")

    return {
        "method": "zero_shot_base",
        "accuracy": results["accuracy"],
        "correct": results["correct"],
        "total": results["total"]
    }


def evaluate_zero_shot_chat(
    dataset,
    model_name: str = "meta-llama/Meta-Llama-3.1-405B-Instruct",
    max_examples: int = None
) -> Dict[str, Any]:
    """
    Evaluate zero-shot baseline on CHAT/INSTRUCT model.

    This tests: how well does the post-trained chat model do without demonstrations?

    Args:
        dataset: ICMDataset
        model_name: Hyperbolic model name (should be instruct model)
        max_examples: Maximum examples to evaluate (for speed)

    Returns:
        Evaluation results dictionary
    """
    print("\n" + "="*60)
    print("Evaluating: Zero-Shot (Chat/Instruct Model)")
    print("="*60)

    load_dotenv()
    client = HyperbolicClient()

    examples = dataset.examples
    if max_examples:
        examples = examples[:max_examples]

    predicted_labels = []

    for example in tqdm(examples, desc="Zero-shot chat predictions"):
        # Build simple prompt without demonstrations
        # Format: Question: X\nClaim: Y\nI think this Claim is
        # Use chat format for instruct model
        messages = [
            {"role": "user", "content": example.input_text}
        ]

        try:
            response = client.chat_completion(
                model=model_name,
                messages=messages,
                temperature=0.0,  # Deterministic
                max_tokens=10,
                top_p=0.9
            )

            # Extract label
            text = response.strip().upper()
            if "TRUE" in text:
                label = "True"
            elif "FALSE" in text:
                label = "False"
            else:
                label = random.choice(["True", "False"])

            predicted_labels.append(label)

        except Exception as e:
            print(f"Error: {e}")
            predicted_labels.append(random.choice(["True", "False"]))

    results = evaluate_labels(examples, predicted_labels)
    print(f"Accuracy: {results['accuracy']:.4f} ({results['correct']}/{results['total']})")

    return {
        "method": "zero_shot_chat",
        "accuracy": results["accuracy"],
        "correct": results["correct"],
        "total": results["total"]
    }


def evaluate_icm_labels(train_dataset, test_dataset, icm_results_path: str, model_name: str) -> Dict[str, Any]:
    """
    Evaluate ICM-generated labels using in-context learning.

    This tests: can ICM discover useful labels without supervision?

    Args:
        train_dataset: Training dataset (to get input texts)
        test_dataset: Test dataset
        icm_results_path: Path to ICM results JSON file
        model_name: Model to use for predictions

    Returns:
        Evaluation results dictionary
    """
    print("\n" + "="*60)
    print("Evaluating: ICM Labels (Unsupervised)")
    print("="*60)

    # Load ICM results
    with open(icm_results_path, 'r') as f:
        icm_data = json.load(f)

    labeled_examples = icm_data["labeled_examples"]

    # Create demonstrations from ICM-discovered labels
    demo_examples = [
        {
            "input": ex["input"],
            "label": ex["label"]
        }
        for ex in labeled_examples
    ]

    print(f"Using {len(demo_examples)} ICM-labeled demonstrations")

    # Predict on test set using ICM demonstrations
    predicted_labels = predict_with_demonstrations(
        test_dataset.examples,
        demo_examples,
        model_name=model_name
    )

    results = evaluate_labels(test_dataset.examples, predicted_labels)
    print(f"Accuracy: {results['accuracy']:.4f} ({results['correct']}/{results['total']})")

    return {
        "method": "icm",
        "accuracy": results["accuracy"],
        "correct": results["correct"],
        "total": results["total"]
    }


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate ICM and baselines on TruthfulQA"
    )

    parser.add_argument(
        "--train-data",
        type=str,
        default="data/train_truthfulqa.json",
        help="Path to training data JSON"
    )
    parser.add_argument(
        "--test-data",
        type=str,
        default="data/test_truthfulqa.json",
        help="Path to test data JSON"
    )
    parser.add_argument(
        "--icm-results",
        type=str,
        default=None,
        help="Path to ICM results JSON (if available)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Meta-Llama-3.1-405B",
        help="Model for predictions"
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Max test examples (None for all)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results.json",
        help="Output JSON file"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
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
        "--skip-golden",
        action="store_true",
        help="Skip golden labels evaluation (saves API calls)"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="meta-llama/Meta-Llama-3.1-405B",
        help="Base model name"
    )
    parser.add_argument(
        "--chat-model",
        type=str,
        default="meta-llama/Meta-Llama-3.1-405B-Instruct",
        help="Chat/instruct model name"
    )

    return parser.parse_args()


def main():
    """Run all evaluations."""
    args = parse_args()

    print("="*60)
    print("TruthfulQA Evaluation")
    print("="*60)

    # Load training dataset
    print(f"\nLoading training data from {args.train_data}...")
    train_dataset = load_truthfulqa_from_json(
        json_path=args.train_data,
        seed=args.seed
    )
    print(f"Loaded {len(train_dataset)} training examples")

    # Load test dataset
    print(f"\nLoading test data from {args.test_data}...")
    test_dataset = load_truthfulqa_from_json(
        json_path=args.test_data,
        seed=args.seed
    )

    if args.max_examples:
        print(f"Limiting to {args.max_examples} test examples")
        test_dataset.examples = test_dataset.examples[:args.max_examples]

    print(f"Loaded {len(test_dataset)} test examples")

    # Run evaluations
    all_results = []

    # 1. Zero-shot (Base model with HHH prompt)
    if not args.skip_zero_shot_base:
        zero_shot_base_results = evaluate_zero_shot_base(
            test_dataset,
            model_name=args.base_model,
            max_examples=None
        )
        all_results.append(zero_shot_base_results)
    else:
        print("\n⏭️  Skipping zero-shot base evaluation")

    # 2. Zero-shot (Chat/Instruct model)
    if not args.skip_zero_shot_chat:
        zero_shot_chat_results = evaluate_zero_shot_chat(
            test_dataset,
            model_name=args.chat_model,
            max_examples=None
        )
        all_results.append(zero_shot_chat_results)
    else:
        print("\n⏭️  Skipping zero-shot chat evaluation")

    # 3. Golden labels (many-shot with ground truth)
    if not args.skip_golden:
        golden_results = evaluate_golden_labels(
            train_dataset,
            test_dataset,
            model_name=args.base_model  # Use base model for consistency
        )
        all_results.append(golden_results)
    else:
        print("\n⏭️  Skipping golden labels evaluation")

    # 4. ICM labels (many-shot with discovered labels)
    if args.icm_results:
        icm_results = evaluate_icm_labels(
            train_dataset,
            test_dataset,
            args.icm_results,
            model_name=args.base_model  # Use base model for consistency
        )
        all_results.append(icm_results)
    else:
        print("\n⏭️  No ICM results provided, skipping ICM evaluation")

    # Save results
    output = {
        "train_dataset": args.train_data,
        "test_dataset": args.test_data,
        "num_train_examples": len(train_dataset),
        "num_test_examples": len(test_dataset),
        "results": all_results
    }

    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for result in all_results:
        print(f"{result['method']:20s}: {result['accuracy']:.4f}")
    print("="*60)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
