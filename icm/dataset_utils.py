"""
Dataset handling for ICM - TruthfulQA from local JSON files.
"""

import json
import random
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ICMExample:
    """Single example for ICM processing."""
    input_text: str
    metadata: Dict[str, Any]

    def __post_init__(self):
        """Validate the example after initialization."""
        if not isinstance(self.input_text, str):
            raise ValueError("input_text must be a string")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a dictionary")


class ICMDataset:
    """Dataset container for ICM examples."""

    def __init__(self, examples: List[ICMExample], metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize ICM dataset.

        Args:
            examples: List of ICM examples
            metadata: Dataset-level metadata
        """
        self.examples = examples
        self.metadata = metadata or {}

    def __len__(self) -> int:
        """Return number of examples."""
        return len(self.examples)

    def __getitem__(self, idx: int) -> ICMExample:
        """Get example by index."""
        return self.examples[idx]

    def shuffle(self, seed: Optional[int] = None) -> 'ICMDataset':
        """Shuffle the dataset."""
        if seed is not None:
            random.seed(seed)
        shuffled_examples = self.examples.copy()
        random.shuffle(shuffled_examples)
        return ICMDataset(shuffled_examples, self.metadata)

    def sample(self, n: int, seed: Optional[int] = None) -> 'ICMDataset':
        """Sample n examples from the dataset."""
        if seed is not None:
            random.seed(seed)
        sampled_examples = random.sample(self.examples, min(n, len(self.examples)))
        return ICMDataset(sampled_examples, self.metadata)

    def get_stats(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        if not self.examples:
            return {
                "num_examples": 0,
                "avg_input_length": 0,
                "metadata_keys": []
            }

        stats = {
            "num_examples": len(self.examples),
            "avg_input_length": sum(len(ex.input_text) for ex in self.examples) / len(self.examples),
            "metadata_keys": set()
        }

        for ex in self.examples:
            stats["metadata_keys"].update(ex.metadata.keys())

        stats["metadata_keys"] = list(stats["metadata_keys"])
        return stats


def load_truthfulqa_from_json(
    json_path: str,
    sample_size: Optional[int] = None,
    seed: int = 42
) -> ICMDataset:
    """
    Load TruthfulQA dataset from local JSON file.

    Expected JSON format:
    [
        {
            "question": str,
            "choice": str,
            "label": int (0 or 1),
            "consistency_id": int
        },
        ...
    ]

    Args:
        json_path: Path to JSON file
        sample_size: Number of examples to sample (None for all)
        seed: Random seed for sampling

    Returns:
        ICMDataset ready for processing
    """
    print(f"Loading TruthfulQA dataset from {json_path}")

    # Load JSON file
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    with open(path, 'r') as f:
        raw_data = json.load(f)

    print(f"Loaded {len(raw_data)} raw examples")

    # Sample if requested
    if sample_size is not None and sample_size < len(raw_data):
        print(f"Sampling {sample_size} examples from {len(raw_data)} available")
        random.seed(seed)
        raw_data = random.sample(raw_data, sample_size)

    # Convert to ICM examples
    examples = []
    for item in raw_data:
        question = item["question"]
        choice = item["choice"]
        label = item["label"]  # 0 or 1
        consistency_id = item.get("consistency_id", -1)

        # Format input text for ICM (matching reference implementation)
        input_text = (
            f"Question: {question}\n"
            f"Claim: {choice}\n"
            f"I think this Claim is"
        )

        metadata = {
            "question": question,
            "choice": choice,
            "ground_truth_label": "True" if label == 1 else "False",
            "label_int": label,
            "consistency_id": consistency_id,
            "task": "truthfulness"
        }

        examples.append(ICMExample(input_text, metadata))

    # Create dataset
    dataset = ICMDataset(
        examples,
        {
            "task_type": "truthfulqa",
            "source": "local_json",
            "json_path": str(path),
            "sample_size": sample_size,
            "seed": seed
        }
    )

    print(f"Created {len(dataset)} ICM examples")
    return dataset


def save_dataset_to_jsonl(dataset: ICMDataset, filepath: str) -> None:
    """
    Save ICM dataset to JSONL file.

    Args:
        dataset: ICMDataset to save
        filepath: Output file path
    """
    with open(filepath, 'w') as f:
        for example in dataset.examples:
            record = {
                "input_text": example.input_text,
                "metadata": example.metadata
            }
            f.write(json.dumps(record) + '\n')

    print(f"Saved {len(dataset)} examples to {filepath}")


def load_dataset_from_jsonl(filepath: str) -> ICMDataset:
    """
    Load ICM dataset from JSONL file.

    Args:
        filepath: Input file path

    Returns:
        ICMDataset
    """
    examples = []

    with open(filepath, 'r') as f:
        for line in f:
            record = json.loads(line)
            examples.append(ICMExample(
                input_text=record["input_text"],
                metadata=record["metadata"]
            ))

    print(f"Loaded {len(examples)} examples from {filepath}")
    return ICMDataset(examples)


# Example usage
if __name__ == "__main__":
    # Load TruthfulQA dataset from local JSON
    dataset = load_truthfulqa_from_json(
        "../data/train_truthfulqa.json",
        sample_size=10,
        seed=42
    )

    # Print statistics
    print("\nDataset Statistics:")
    stats = dataset.get_stats()
    print(json.dumps(stats, indent=2))

    # Show first example
    print("\nFirst Example:")
    print(f"Input: {dataset[0].input_text}")
    print(f"Metadata: {dataset[0].metadata}")
