# Autoresearch

An autonomous AI research agent that improves LLM training code overnight.

## How It Works

```
┌─────────────────────────────────────────────────────────────┐
│  1. AI agent reads program.md (instructions)                │
│  2. AI modifies train.py (hyperparameters/architecture)     │
│  3. AI runs train.py (5-minute experiment)                  │
│  4. AI checks val_bpb (lower = better)                      │
│  5. Keep if improved, discard if worse                      │
│  6. Repeat overnight (~100 experiments)                     │
│  7. Wake up to better model + experiment log                │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Data (One-Time Setup)

```bash
python prepare.py
```

This downloads TinyStories dataset (~200MB) and trains a tokenizer.

**Options:**
```bash
python prepare.py --shards 2   # Download 2 shards (default)
python prepare.py --shards 5   # Download more data
```

### 3. Run Baseline Training

```bash
python train.py
```

This runs a 5-minute training experiment and reports `val_bpb`.

### 4. Start Autonomous Research

Open Claude Code (or any AI coding assistant) in this directory and prompt:

```
Read program.md and let's start experimenting!
First, run the baseline, then try to improve val_bpb.
```

The AI will:
- Read the instructions
- Run experiments
- Modify hyperparameters
- Track what works

Let it run overnight for ~100 experiments.

## Project Structure

```
autoresearch/
├── prepare.py          # One-time setup (data + tokenizer)
├── train.py            # Model + training (AI modifies this)
├── program.md          # Instructions for AI agent
├── generate.py         # Generate stories from saved models
├── view_experiments.py # View experiment history
├── requirements.txt
├── README.md
├── checkpoints/        # Saved models (created after training)
│   ├── baseline.pt     # First model (for comparison)
│   └── best.pt         # Best model (lowest val_bpb)
└── experiments.jsonl   # Experiment log (created after training)
```

## What the AI Can Modify

| Category | Parameters |
|----------|------------|
| **Architecture** | N_LAYERS, N_HEADS, N_EMBED, DROPOUT |
| **Training** | BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY |
| **Advanced** | Optimizers, schedulers, activations, attention |

## Metric

**val_bpb** (validation bits per byte)
- Lower is better
- Vocab-size independent (fair comparison across experiments)

## Requirements

- Python 3.10+
- Mac (Apple Silicon) or NVIDIA GPU
- ~10GB disk space
- Claude Code / Codex / Cursor (for AI agent)

## Comparing Models (Before vs After)

After running experiments, compare the baseline vs best model:

### View Experiment Stats

```bash
# See all experiments
python view_experiments.py

# See best experiment only
python view_experiments.py --best
```

### Generate Stories & Compare Quality

```bash
# Compare baseline vs best model side-by-side
python generate.py --compare --prompt "Once upon a time"

# Generate from specific model
python generate.py --model baseline --prompt "The brave princess"
python generate.py --model best --prompt "The brave princess"

# Adjust generation settings
python generate.py --model best --prompt "Once" --tokens 150 --temperature 0.9
```

**Example output:**
```
📊 MODEL COMPARISON
Prompt: "Once upon a time"
======================================================================

🔵 BASELINE MODEL
   val_bpb: 1.4532
   layers: 6, heads: 6, embed: 384
----------------------------------------------------------------------
Once upon a time the the was a little the and the was...

🏆 BEST MODEL
   val_bpb: 1.2891
   layers: 8, heads: 8, embed: 512
----------------------------------------------------------------------
Once upon a time, there was a little rabbit named Lily.
She lived in a cozy burrow under a big oak tree...

✅ Lower val_bpb = better model. Compare the story quality above!
```

### What Gets Saved

| File | When Saved | Purpose |
|------|------------|---------|
| `checkpoints/baseline.pt` | First run only | Compare "before" |
| `checkpoints/best.pt` | When val_bpb improves | Compare "after" |
| `experiments.jsonl` | Every run | Full history |

## Tips

1. **Start small** — 2 shards is enough for experimentation
2. **Track experiments** — AI logs what it tries
3. **Be patient** — Some experiments will fail, that's learning
4. **Check in the morning** — Review the experiment log
5. **Compare stories** — Use `generate.py --compare` to see real improvement

## Based On

Inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) — this is a simplified, Mac-compatible version.

---

Happy researching! 🚀
