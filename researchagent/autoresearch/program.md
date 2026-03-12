# Autoresearch Program

You are an autonomous ML research agent. Your goal is to **improve val_bpb** (validation bits per byte) by modifying `train.py`.

## Your Task

1. **Run an experiment** by modifying `train.py`
2. **Train for 5 minutes** and observe val_bpb
3. **Keep changes that improve val_bpb**, discard those that don't
4. **Repeat** until told to stop

## Rules

- **Only modify `train.py`** — never touch `prepare.py`
- **Lower val_bpb is better** — this is your optimization target
- **Each experiment runs exactly 5 minutes** — don't change TIME_BUDGET
- **Track your experiments** — note what you tried and the result

## What You Can Modify in train.py

### Hyperparameters
```python
N_LAYERS = 6          # Try: 4, 8, 12
N_HEADS = 6           # Try: 4, 8 (must divide N_EMBED)
N_EMBED = 384         # Try: 256, 512 (must be divisible by N_HEADS)
DROPOUT = 0.1         # Try: 0.0, 0.05, 0.2
BATCH_SIZE = 32       # Try: 16, 64 (limited by GPU memory)
LEARNING_RATE = 3e-4  # Try: 1e-4, 6e-4, 1e-3
WEIGHT_DECAY = 0.1    # Try: 0.01, 0.05
```

### Architecture Ideas
- Add learning rate scheduler (warmup + decay)
- Try different activation functions (ReLU, SiLU instead of GELU)
- Add gradient clipping
- Experiment with different attention patterns
- Try rotary position embeddings (RoPE)
- Modify feedforward expansion ratio (currently 4x)

### Training Ideas
- Add gradient accumulation for larger effective batch size
- Try different optimizers (Adam, SGD with momentum)
- Implement mixed precision training (fp16/bf16)

## Experiment Log Format

After each experiment, report:
```
## Experiment N
- Change: [what you modified]
- val_bpb: [result]
- Comparison: [better/worse than baseline]
- Decision: [keep/discard]
```

## Baseline

Run `train.py` without modifications first to establish baseline val_bpb.

## Getting Started

1. Read this file (program.md)
2. Read train.py to understand the code
3. Run baseline: `python train.py`
4. Note baseline val_bpb
5. Make a hypothesis ("lower learning rate might help")
6. Modify train.py
7. Run again: `python train.py`
8. Compare val_bpb
9. Keep or discard change
10. Repeat from step 5

Good luck! 🚀
