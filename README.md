# ICM: Internal Coherence Maximisation

Reproduction of the ICM algorithm from [Unsupervised Elicitation of Language Models](https://arxiv.org/abs/2506.10139) on the TruthfulQA dataset.

This is a simplified, **prompt-based only** reproduction (no fine-tuning). Designed to be powered by Hyperbolic for inference of Llama-3-405B base and instruct models.

## Quick Setup

### Install Dependencies

Using `uv` (recommended):
```bash
chmod +x setup.sh
./setup.sh
```

Or using `pip`:
```bash
pip install -r requirements.txt
```

### Configure API Key

Create a `.env` file with your Hyperbolic API key:
```bash
cp .env.example .env
```

Then edit `.env` and add your key:
```
HYPERBOLIC_API_KEY=your_key_here
```

Get your API key from [hyperbolic.xyz](https://www.hyperbolic.ai/).

## Usage

### One-Command Reproduction (Recommended)

The easiest way to run the complete reproduction:

```bash
# Quick test run (~30 minutes)
python run.py --quick

# Full reproduction (~2 hours)
python run.py --full
```

This single command will:
- Run ICM search on the training data
- Evaluate all baselines (zero-shot, zero-shot chat, ICM, and Golden labels)
- Visualise results

### Manual Step-by-step Reproduction

Run ICM on the training data to discover labels:

```bash
python -m icm.main \
  --data-path data/train_truthfulqa.json \
  --output-dir results \
  --max-iterations 500 \
  --sample-size 256 \
  --alpha 1.0 \
  --seed 0
```

Evaluate on the test set:

```bash
python evaluate.py \
  --test-data data/test_truthfulqa.json \
  --icm-results results/truthfulqa_meta_llama_3.1_405b_icm_result.json \
  --output evaluation_results.json \
  --skip-zero-shot
```

Generate a bar graph visualisation of the results:

```bash
python plot_results.py \
  --results evaluation_results.json \
  --output results_plot.png \
  --title "TruthfulQA Performance"
```


