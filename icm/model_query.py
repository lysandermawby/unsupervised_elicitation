#!/usr/bin/env python
"""query models through hyperbolic"""

# import openai
from dotenv import load_dotenv
import os
import click
import math
import requests

ALLOWED_MODELS = ["Llama-3.1-405B", "Llama-3.1-405B-instruct", "base", "instruct"]

@click.command(context_settings=dict(help_option_names=['-h', '--help'], show_default=True))
@click.argument('prompt')
@click.option('--system-prompt', help='system prompt', default=None)
@click.option('-m', '--model', type=click.Choice(ALLOWED_MODELS), default='base')
def query_llama(prompt, model='base', system_prompt=None):
    """query llama model through the hyperbolic API"""
    
    load_dotenv()

    if model.lower() == 'base':
        model = "Llama-3.1-405B"
    elif model.lower() == 'instruct':
        model = "Llama-3.1-405B-instruct"

    if model not in ALLOWED_MODELS:
        click.echo(click.style(f"Error: Model {model} not found in list of allowed models: {', '.join(ALLOWED_MODELS)}", fg='red'))
        return None

    HYPERBOLIC_API_KEY = os.getenv("HYPERBOLIC_API_KEY")

    # base model must be accessed using the completions API
    # instruct model can be accessed conventionally using the chat API
    if model == "Llama-3.1-405B":
        url = "https://api.hyperbolic.xyz/v1/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {HYPERBOLIC_API_KEY}"
        }

        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        # sensible inference parameters
        data = {
            "prompt": full_prompt,
            "model": "meta-llama/Meta-Llama-3.1-405B",
            "max_tokens": 1024, # follows OpenAI standard, refers to the number of tokens being generated
            "temperature": 0.7,
            "top_p": 0.9
        }

        try:
            response = requests.post(url, headers=headers, json=data, timeout=60)
            print(response)
            response.raise_for_status()
            response_json = response.json()
        except requests.exceptions.Timeout:
            raise TimeoutError("API request timed out after 60 seconds")
        except requests.exceptions.HTTPError as e:
            print(f"HTTP Error: {e}")
            print(f"Response: {response.text}")
            raise
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            raise

        return response_json['choices'][0]['text']
    else:
        url = "https://api.hyperbolic.xyz/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {HYPERBOLIC_API_KEY}"
        }

        # create chat message
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        data = {
            "messages": messages,
            "model": "meta-llama/Meta-Llama-3.1-405B-Instruct",
            "max_tokens": 1024,
            "temperature": 0.7,
            "top_p": 0.9
        }

        try:
            response = requests.post(url, headers=headers, json=data, timeout=60)
            response.raise_for_status()
            response_json = response.json()
        except requests.exceptions.Timeout:
            raise TimeoutError("API request timed out after 60 seconds")
        except requests.exceptions.HTTPError as e:
            print(f"HTTP Error: {e}")
            print(f"Response: {response.text}")
            raise
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            raise

        return response_json['choices'][0]['message']['content']


def query_llama_with_logprobs(prompt, model='base'):
    """Query model and get log probabilities for True/False tokens.
    Returns dict with 'logprob_true' and 'logprob_false'
    
    Note: The instruct model via chat completions endpoint doesn't seem to support logprobs.
    Consider using the base model for logprob extraction, or using completions endpoint for instruct model.
    """

    load_dotenv()

    if model == 'base':
        model_name = "meta-llama/Meta-Llama-3.1-405B"
    else:
        model_name = "meta-llama/Meta-Llama-3.1-405B-Instruct"

    HYPERBOLIC_API_KEY = os.getenv("HYPERBOLIC_API_KEY")
    
    if model_name == "meta-llama/Meta-Llama-3.1-405B":
        url = "https://api.hyperbolic.xyz/v1/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {HYPERBOLIC_API_KEY}"
        }

        data = {
            "prompt": prompt,
            "model": "meta-llama/Meta-Llama-3.1-405B",
            "max_tokens": 2,  # Only need 1 token for True/False
            "temperature": 0,
            "top_p": 0.9,
            "logprobs": 20,  # Request logprobs for completions API
            "echo": False
        }

        try:
            # print(f"Sending request to {url} for model {data['model']}...")
            response = requests.post(url, headers=headers, json=data, timeout=30)
            # print(f"Response status: {response.status_code}")
            response.raise_for_status()  # Raise error for bad status codes
            response_json = response.json()
        except requests.exceptions.Timeout:
            raise TimeoutError("API request timed out after 30 seconds")
        except requests.exceptions.HTTPError as e:
            print(f"HTTP Error: {e}")
            print(f"Response: {response.text}")
            raise
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            raise

        # debug print
        # print(f"API request completed with status: {response.status_code}")
        # print(f"Response preview: {str(response.text)[:200]}")
        
        # Debug: print the response to see what we're getting
        if 'choices' not in response_json:
            print(f"Error response: {response_json}")
            raise ValueError(f"Unexpected API response: {response_json}")
        
        # Extract logprobs from completions API response
        choice = response_json['choices'][0]
        
        # The structure for completions API logprobs is different
        if 'logprobs' in choice and choice['logprobs'] is not None:
            logprobs_data = choice['logprobs']
            
            # For completions API, logprobs structure is typically:
            # {'tokens': [...], 'token_logprobs': [...], 'top_logprobs': [{token: logprob, ...}, ...]}
            top_logprobs = logprobs_data.get('top_logprobs', [{}])[0]
            
            logprob_true = -float('inf')
            logprob_false = -float('inf')
            
            for token, logprob in top_logprobs.items():
                token_upper = token.strip().upper()
                if 'TRUE' in token_upper:
                    logprob_true = logprob
                elif 'FALSE' in token_upper:
                    logprob_false = logprob
            
            return {
                'logprob_true': logprob_true,
                'logprob_false': logprob_false,
                'prob_true': math.exp(logprob_true) if logprob_true > -float('inf') else 0,
                'prob_false': math.exp(logprob_false) if logprob_false > -float('inf') else 0
            }
        else:
            raise ValueError("No logprobs returned in response")
    
    else:
        # The Hyperbolic API does not support logprobs for the instruct model
        # The chat/completions endpoint returns 'logprobs': None
        # The completions endpoint returns 500 error for instruct model
        raise NotImplementedError(
            "The Hyperbolic API does not support logprobs for the instruct model "
            "(meta-llama/Meta-Llama-3.1-405B-Instruct). "
            "Please use model='base' instead, or use the instruct model without logprobs."
        )

# def get_prediction_simple(prompt, model, demonstrations=None):
#     """simple prediction without logprobs (for evaluation)"""
#     response = query_llama(prompt, model)

#     return 'TRUE' in response

def get_prediction_simple(prompt, model, demonstrations=None):
    """Simple prediction without logprobs (for evaluation)

    Args:
        prompt: The formatted prompt to send to the model
        model: 'base' or 'instruct'
        demonstrations: Unused parameter for API compatibility (demonstrations should be in prompt)
    """
    load_dotenv()
    
    # Map model names
    if model.lower() == 'base':
        model_name = "meta-llama/Meta-Llama-3.1-405B"
        use_chat = False
    elif model.lower() == 'instruct':
        model_name = "meta-llama/Meta-Llama-3.1-405B-Instruct"
        use_chat = True
    else:
        raise ValueError(f"Unknown model: {model}")
    
    HYPERBOLIC_API_KEY = os.getenv("HYPERBOLIC_API_KEY")
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {HYPERBOLIC_API_KEY}"
    }
    
    if use_chat:
        # Instruct model uses chat API
        url = "https://api.hyperbolic.xyz/v1/chat/completions"
        data = {
            "messages": [{"role": "user", "content": prompt}],
            "model": model_name,
            "max_tokens": 5,  # Just need "True" or "False"
            "temperature": 0,  # Deterministic for evaluation
            "top_p": 0.9
        }
    else:
        # Base model uses completions API
        url = "https://api.hyperbolic.xyz/v1/completions"
        data = {
            "prompt": prompt,
            "model": model_name,
            "max_tokens": 5,  # Just need "True" or "False"
            "temperature": 0,  # Deterministic for evaluation
            "top_p": 0.9
        }
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=60)
        response.raise_for_status()
        response_json = response.json()
        
        # Extract text
        if use_chat:
            text = response_json['choices'][0]['message']['content']
        else:
            text = response_json['choices'][0]['text']
        
        # Check if "True" appears in response
        prediction = 'TRUE' in text.upper()
        
        return prediction
        
    except requests.exceptions.RequestException as e:
        print(f"API Error: {e}")
        raise

if __name__ == '__main__':
    query_llama()
