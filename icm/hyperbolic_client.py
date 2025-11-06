"""Client for Hyperbolic API."""

import os
import requests
from typing import List, Dict, Any, Optional
import time


class HyperbolicClient:
    """Client for interacting with Hyperbolic API."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize client with API key."""
        self.api_key = api_key or os.getenv("HYPERBOLIC_API_KEY")
        if not self.api_key:
            raise ValueError("HYPERBOLIC_API_KEY not found in environment")

        self.base_url = "https://api.hyperbolic.xyz/v1"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def complete(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
        top_p: float = 0.9,
    ) -> str:
        """Get completion from model using completions endpoint.

        This is for base models like meta-llama/Meta-Llama-3.1-405B.

        Args:
            model: Model name (e.g., "meta-llama/Meta-Llama-3.1-405B")
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Top-p sampling parameter

        Returns:
            Generated text response
        """
        url = f"{self.base_url}/completions"
        payload = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
        }

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(url, json=payload, headers=self.headers, timeout=60)
                response.raise_for_status()
                result = response.json()
                return result["choices"][0]["text"]
            except Exception as e:
                if attempt == max_retries - 1:
                    raise Exception(f"API call failed after {max_retries} attempts: {e}")
                time.sleep(2 ** attempt)  # Exponential backoff

    def chat_completion(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 512,
        top_p: float = 0.9,
    ) -> str:
        """Get chat completion from model.

        This is for instruct models like meta-llama/Meta-Llama-3.1-405B-Instruct.

        Args:
            model: Model name (e.g., "meta-llama/Meta-Llama-3.1-405B-Instruct")
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Top-p sampling parameter

        Returns:
            Generated text response
        """
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
        }

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(url, json=payload, headers=self.headers, timeout=60)
                response.raise_for_status()
                result = response.json()
                return result["choices"][0]["message"]["content"]
            except Exception as e:
                if attempt == max_retries - 1:
                    raise Exception(f"API call failed after {max_retries} attempts: {e}")
                time.sleep(2 ** attempt)  # Exponential backoff

    def get_log_probs(
        self,
        model: str,
        prompt: str,
        max_tokens: int = 1,
        logprobs: int = 20,
    ) -> Dict[str, Any]:
        """Get log probabilities for completion.

        Args:
            model: Model name (must be base model, not instruct)
            prompt: Input prompt
            max_tokens: Max tokens to generate
            logprobs: Number of top logprobs to return

        Returns:
            Dictionary with logprobs information
        """
        url = f"{self.base_url}/completions"
        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,  # Greedy for logprobs
            "logprobs": logprobs,
            "echo": False,
        }

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(url, json=payload, headers=self.headers, timeout=60)
                response.raise_for_status()
                return response.json()
            except Exception as e:
                if attempt == max_retries - 1:
                    raise Exception(f"API call failed after {max_retries} attempts: {e}")
                time.sleep(2 ** attempt)
