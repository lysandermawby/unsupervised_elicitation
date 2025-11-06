"""
Visualization script for ICM evaluation results.

Creates a bar graph similar to Figure 1 in the paper.
"""

import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_results(results_path: str, output_path: str = "results_plot.png", title: str = "TruthfulQA Performance"):
    """
    Create bar plot from evaluation results.

    Args:
        results_path: Path to evaluation_results.json
        output_path: Path to save plot
        title: Plot title
    """
    # Load results
    with open(results_path, 'r') as f:
        data = json.load(f)

    results = data["results"]

    # Extract method names and accuracies
    methods = []
    accuracies = []
    method_order = {
        "zero_shot_base": 0,
        "zero_shot_chat": 1,
        "golden_labels": 2,
        "icm": 3
    }

    # Sort results by desired order
    sorted_results = sorted(results, key=lambda x: method_order.get(x["method"], 999))

    for result in sorted_results:
        method = result["method"]
        accuracy = result["accuracy"] * 100  # Convert to percentage

        # Format method names for display
        display_names = {
            "zero_shot_base": "Zero-Shot\n(Base)",
            "zero_shot_chat": "Zero-Shot\n(Chat)",
            "golden_labels": "Golden\nLabels",
            "icm": "ICM\n(Ours)"
        }

        methods.append(display_names.get(method, method))
        accuracies.append(accuracy)

    # Create bar plot
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(methods))
    width = 0.6

    # Color scheme: different colors for each bar
    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']  # red, orange, green, blue

    bars = ax.bar(x, accuracies, width, color=colors[:len(methods)], alpha=0.8, edgecolor='black', linewidth=1.5)

    # Customize plot
    ax.set_ylabel('Test Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=12)
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height + 1,
            f'{height:.1f}%',
            ha='center',
            va='bottom',
            fontsize=11,
            fontweight='bold'
        )

    # Add horizontal line at 50% for reference
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.text(len(methods) - 0.5, 51, 'Random Chance', fontsize=9, color='gray', style='italic')

    plt.tight_layout()

    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")

    # Also save as PDF for publication quality
    pdf_path = Path(output_path).with_suffix('.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"PDF saved to: {pdf_path}")

    plt.close()


def plot_with_error_bars(
    results_paths: list,
    output_path: str = "results_plot_with_error.png",
    title: str = "TruthfulQA Performance"
):
    """
    Create bar plot with error bars from multiple runs.

    Args:
        results_paths: List of paths to evaluation_results.json from different runs
        output_path: Path to save plot
        title: Plot title
    """
    # Collect results from all runs
    all_results = {}

    for path in results_paths:
        with open(path, 'r') as f:
            data = json.load(f)

        for result in data["results"]:
            method = result["method"]
            accuracy = result["accuracy"] * 100

            if method not in all_results:
                all_results[method] = []
            all_results[method].append(accuracy)

    # Calculate mean and std
    methods = []
    means = []
    stds = []

    method_order = {
        "random_baseline": 0,
        "zero_shot": 1,
        "icm": 2,
        "golden_labels": 3
    }

    method_order_error = {
        "zero_shot_base": 0,
        "zero_shot_chat": 1,
        "golden_labels": 2,
        "icm": 3
    }

    for method in sorted(all_results.keys(), key=lambda x: method_order_error.get(x, 999)):
        accuracies = all_results[method]

        display_names = {
            "zero_shot_base": "Zero-Shot\n(Base)",
            "zero_shot_chat": "Zero-Shot\n(Chat)",
            "golden_labels": "Golden\nLabels",
            "icm": "ICM\n(Ours)"
        }

        methods.append(display_names.get(method, method))
        means.append(np.mean(accuracies))
        stds.append(np.std(accuracies))

    # Create bar plot
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(methods))
    width = 0.6

    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']

    bars = ax.bar(x, means, width, yerr=stds, color=colors[:len(methods)],
                   alpha=0.8, edgecolor='black', linewidth=1.5,
                   capsize=5, error_kw={'linewidth': 2})

    # Customize plot
    ax.set_ylabel('Test Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=12)
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()
        label = f'{mean:.1f}%\n±{std:.1f}' if std > 0 else f'{mean:.1f}%'
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height + std + 2,
            label,
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold'
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot with error bars saved to: {output_path}")

    plt.close()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot evaluation results"
    )

    parser.add_argument(
        "--results",
        type=str,
        default="evaluation_results.json",
        help="Path to evaluation results JSON"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results_plot.png",
        help="Output plot path"
    )
    parser.add_argument(
        "--title",
        type=str,
        default="TruthfulQA Performance",
        help="Plot title"
    )
    parser.add_argument(
        "--multiple-runs",
        nargs='+',
        default=None,
        help="Paths to multiple result files for error bars"
    )

    return parser.parse_args()


def main():
    """Generate visualization."""
    args = parse_args()

    print("="*60)
    print("Generating Results Visualization")
    print("="*60)

    if args.multiple_runs:
        print(f"Creating plot with error bars from {len(args.multiple_runs)} runs...")
        plot_with_error_bars(
            results_paths=args.multiple_runs,
            output_path=args.output,
            title=args.title
        )
    else:
        print(f"Creating plot from single run: {args.results}")
        plot_results(
            results_path=args.results,
            output_path=args.output,
            title=args.title
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
