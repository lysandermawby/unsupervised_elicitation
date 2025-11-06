"""
ICM Algorithm Implementation using Hyperbolic API.

This is a simplified, prompt-based version that doesn't fine-tune models.
"""

import json
import random
import math
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv

from icm.dataset_utils import ICMDataset, ICMExample
from icm.hyperbolic_client import HyperbolicClient


@dataclass
class ICMResult:
    """Result from ICM search containing labeled examples and metadata."""
    labeled_examples: List[Dict[str, Any]]
    score: float
    iterations: int
    convergence_info: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def save_to_json(self, filepath: str) -> None:
        """Save result to JSON file."""
        output = {
            "labeled_examples": self.labeled_examples,
            "score": self.score,
            "iterations": self.iterations,
            "convergence_info": self.convergence_info,
            "metadata": self.metadata
        }
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"Saved ICM result to {filepath}")


class ICMSearcher:
    """
    Internal Coherence Maximization searcher using Hyperbolic API.

    Implements the ICM algorithm using mutual predictability to generate
    labels for unlabeled datasets without external supervision.
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Meta-Llama-3.1-405B",
        alpha: float = 1.0,
        initial_temperature: float = 3.0,
        final_temperature: float = 0.001,
        cooling_rate: float = 0.98,
        initial_examples: int = 20,
        max_iterations: int = 1000,
        generation_temperature: float = 0.2,
        generation_top_p: float = 0.9,
        seed: int = 42
    ):
        """
        Initialize ICM searcher.

        Args:
            model_name: Hyperbolic model name (base model for logprobs)
            alpha: Weight for mutual predictability
            initial_temperature: Initial temperature for simulated annealing
            final_temperature: Final temperature
            cooling_rate: Temperature cooling rate
            initial_examples: Number of initial randomly labeled examples (K)
            max_iterations: Maximum iterations
            generation_temperature: Temperature for label generation
            generation_top_p: Top-p for generation
            seed: Random seed
        """
        load_dotenv()

        self.model_name = model_name
        self.alpha = alpha
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.cooling_rate = cooling_rate
        self.initial_examples = initial_examples
        self.max_iterations = max_iterations
        self.generation_temperature = generation_temperature
        self.generation_top_p = generation_top_p
        self.seed = seed

        # Set random seeds
        random.seed(seed)
        np.random.seed(seed)

        # Initialize Hyperbolic client
        self.client = HyperbolicClient()

        print(f"ICM Searcher initialized with model: {model_name}")

    def search(
        self,
        dataset: ICMDataset,
        max_examples: Optional[int] = None
    ) -> ICMResult:
        """
        Run ICM search on a dataset.

        Args:
            dataset: ICM dataset to search on
            max_examples: Maximum number of examples to process

        Returns:
            ICMResult with labeled examples and metadata
        """
        print(f"Starting ICM search on {len(dataset)} examples")

        examples = dataset.examples

        if max_examples and max_examples < len(examples):
            print(f"Limiting to {max_examples} examples")
            examples = examples[:max_examples]

        # Initialize with K randomly labeled examples
        labeled_data = self._initialize_labeled_data(examples)

        # Main search loop
        best_score = self._calculate_score(labeled_data)
        temperature = self.initial_temperature

        print(f"Initial score: {best_score:.4f}")

        for iteration in tqdm(range(self.max_iterations), desc="ICM Search"):
            # Update temperature (logarithmic cooling)
            temperature = max(
                self.final_temperature,
                self.initial_temperature / (1 + self.cooling_rate * math.log(iteration + 1))
            )

            # Sample example to label
            example_idx = self._sample_example_to_label(examples, labeled_data)
            example = examples[example_idx]

            # Generate label
            new_label = self._generate_label(example, labeled_data)

            # Create new labeled data
            new_labeled_data = labeled_data.copy()
            new_labeled_data[example_idx] = {
                "example": example,
                "label": new_label,
                "index": example_idx
            }

            # Calculate new score
            new_score = self._calculate_score(new_labeled_data)

            # Accept or reject (simulated annealing)
            delta = new_score - best_score

            if delta > 0 or random.random() < math.exp(delta / temperature):
                labeled_data = new_labeled_data
                best_score = new_score

            # Progress logging
            if iteration > 0 and iteration % 100 == 0:
                print(f"Iteration {iteration}: score={best_score:.4f}, temp={temperature:.6f}, labeled={len(labeled_data)}/{len(examples)}")

            # Early stopping if all labeled
            if len(labeled_data) >= len(examples):
                print(f"All {len(examples)} examples labeled. Stopping early.")
                break

        # Convert to final format
        labeled_examples = []
        for idx, data in labeled_data.items():
            labeled_examples.append({
                "input": data["example"].input_text,
                "label": data["label"],
                "metadata": data["example"].metadata
            })

        result = ICMResult(
            labeled_examples=labeled_examples,
            score=best_score,
            iterations=iteration + 1,
            convergence_info={
                "final_temperature": temperature,
                "labeled_count": len(labeled_data)
            },
            metadata={
                "model_name": self.model_name,
                "alpha": self.alpha,
                "dataset_size": len(examples)
            }
        )

        print(f"ICM search completed. Final score: {best_score:.4f}")
        return result

    def _initialize_labeled_data(
        self,
        examples: List[ICMExample]
    ) -> Dict[int, Dict[str, Any]]:
        """Initialize with K randomly labeled examples."""
        labeled_data = {}

        selected_indices = random.sample(
            range(len(examples)),
            min(self.initial_examples, len(examples))
        )

        for idx in selected_indices:
            example = examples[idx]
            label = random.choice(["True", "False"])

            labeled_data[idx] = {
                "example": example,
                "label": label,
                "index": idx
            }

        print(f"Initialized with {len(labeled_data)} randomly labeled examples")
        return labeled_data

    def _sample_example_to_label(
        self,
        examples: List[ICMExample],
        labeled_data: Dict[int, Dict[str, Any]]
    ) -> int:
        """Sample an example to label (uniform random)."""
        unlabeled_indices = [i for i in range(len(examples)) if i not in labeled_data]

        # if all entries have been labelled
        if not unlabeled_indices:
            # Re-label a random already-labeled example
            return random.choice(list(labeled_data.keys()))

        return random.choice(unlabeled_indices)

    def _generate_label(
        self,
        example: ICMExample,
        labeled_data: Dict[int, Dict[str, Any]]
    ) -> str:
        """Generate label for an example based on mutual predictability."""
        # Build prompt with in-context examples
        prompt = self._build_prediction_prompt(example, labeled_data)

        try:
            # Use base model for completion
            # logprobs are required, and the hyperbolic API doesn't support logprobs for instruct models (horrendously)
            response = self.client.complete(
                model=self.model_name,
                prompt=prompt,
                temperature=self.generation_temperature,
                max_tokens=5,
                top_p=self.generation_top_p
            )

            # Extract label from response
            return self._extract_label(response)

        except Exception as e:
            print(f"Error generating label: {e}")
            return random.choice(["True", "False"])

    def _build_prediction_prompt(
        self,
        example: ICMExample,
        labeled_data: Dict[int, Dict[str, Any]]
    ) -> str:
        """Build prompt for label prediction with in-context examples."""
        prompt_parts = []

        # Add labeled examples as demonstrations
        # Format: Question: X\nClaim: Y\nI think this Claim is True/False\n\n
        for data in labeled_data.values():
            prompt_parts.append(data["example"].input_text)
            prompt_parts.append(f" {data['label']}")
            prompt_parts.append("")

        # Add target example
        # Format: Question: X\nClaim: Y\nI think this Claim is
        prompt_parts.append(example.input_text)

        return "\n".join(prompt_parts)

    def _extract_label(self, generated_text: str) -> str:
        """Extract label from generated text."""
        text = generated_text.strip().upper()

        if "TRUE" in text:
            return "True"
        elif "FALSE" in text:
            return "False"
        else:
            # Default to True if unclear
            return random.choice(["True", "False"])

    def _calculate_score(self, labeled_data: Dict[int, Dict[str, Any]]) -> float:
        """
        Calculate ICM scoring function: U(D) = α * P_θ(D).

        Note: Consistency checking is ignored per instructions.
        """
        if not labeled_data:
            return 0.0

        mutual_predictability = self._calculate_mutual_predictability(labeled_data)
        score = self.alpha * mutual_predictability 

        return score

    def _calculate_mutual_predictability(
        self,
        labeled_data: Dict[int, Dict[str, Any]]
    ) -> float:
        """
        Calculate mutual predictability P_θ(D).

        For each example, calculate log P(label | example, other_labeled_examples)
        and average across all examples.
        """
        if len(labeled_data) < 2:
            return 0.0

        total_log_prob = 0.0
        count = 0

        # Sample a subset to reduce API calls
        # Hyperbolic API was slow / often not responding during development
        sample_size = min(10, len(labeled_data))
        sampled_indices = random.sample(list(labeled_data.keys()), sample_size)

        for target_idx in sampled_indices:
            target_data = labeled_data[target_idx]

            # Build context with all other examples
            context_data = {
                idx: data for idx, data in labeled_data.items()
                if idx != target_idx
            }

            if not context_data:
                continue

            log_prob = self._calculate_conditional_probability(
                target_data["example"],
                target_data["label"],
                context_data
            )

            total_log_prob += log_prob
            count += 1

        return total_log_prob / count if count > 0 else 0.0

    def _calculate_conditional_probability(
        self,
        target_example: ICMExample,
        target_label: str,
        context_data: Dict[int, Dict[str, Any]]
    ) -> float:
        """
        Calculate log P(target_label | target_example, context_examples).

        Uses logprobs from Hyperbolic API.
        """
        try:
            # Build prompt
            prompt = self._build_prediction_prompt(target_example, context_data)

            # Get logprobs from API
            response = self.client.get_log_probs(
                model=self.model_name,
                prompt=prompt,
                max_tokens=1,
                logprobs=20
            )

            # Extract logprobs for True/False
            logprobs_dict = self._extract_true_false_logprobs(response)

            # Get probability for target label
            if target_label == "True":
                return logprobs_dict.get("True", -10.0)
            else:
                return logprobs_dict.get("False", -10.0)

        except Exception as e:
            print(f"Error calculating conditional probability: {e}")
            return -5.0  # Neutral score

    def _extract_true_false_logprobs(self, response: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract log probabilities for True and False from API response.

        Args:
            response: API response with logprobs

        Returns:
            Dictionary with "True" and "False" logprobs
        """
        try:
            choices = response.get("choices", [])
            if not choices:
                return {"True": -10.0, "False": -10.0}

            logprobs_data = choices[0].get("logprobs", {})
            top_logprobs = logprobs_data.get("top_logprobs", [])

            if not top_logprobs:
                return {"True": -10.0, "False": -10.0}

            # Get first token's top logprobs
            first_token_logprobs = top_logprobs[0] if isinstance(top_logprobs, list) else top_logprobs

            # Extract True/False logprobs
            true_logprob = -float('inf')
            false_logprob = -float('inf')

            for token, logprob in first_token_logprobs.items():
                token_upper = token.strip().upper()
                if "TRUE" in token_upper:
                    true_logprob = max(true_logprob, logprob)
                elif "FALSE" in token_upper:
                    false_logprob = max(false_logprob, logprob)

            # Normalize using log-sum-exp
            if true_logprob == -float('inf') and false_logprob == -float('inf'):
                return {"True": math.log(0.5), "False": math.log(0.5)}

            # Convert to probabilities
            log_sum = math.log(math.exp(true_logprob) + math.exp(false_logprob))

            return {
                "True": true_logprob - log_sum,
                "False": false_logprob - log_sum
            }

        except Exception as e:
            print(f"Error extracting logprobs: {e}")
            return {"True": math.log(0.5), "False": math.log(0.5)}


# Example usage
if __name__ == "__main__":
    from icm.dataset_utils import load_truthfulqa_from_json

    # Load dataset
    dataset = load_truthfulqa_from_json(
        "../data/train_truthfulqa.json",
        sample_size=50,
        seed=42
    )

    # Initialize searcher
    searcher = ICMSearcher(
        model_name="meta-llama/Meta-Llama-3.1-405B",
        max_iterations=200,
        alpha=1.0
    )

    # Run search
    result = searcher.search(dataset, max_examples=50)

    # Save results
    result.save_to_json("icm_results.json")

    print(f"\nLabeled {len(result.labeled_examples)} examples")
    print(f"Final score: {result.score:.4f}")
