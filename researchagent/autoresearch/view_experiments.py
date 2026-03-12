"""
View experiment history from experiments.jsonl

Usage:
    python view_experiments.py          # Show all experiments
    python view_experiments.py --best   # Show best experiment only
"""

import json
import argparse
from pathlib import Path


def load_experiments():
    """Load all experiments from log file."""
    log_file = Path("experiments.jsonl")

    if not log_file.exists():
        print("❌ No experiments found. Run train.py first.")
        return []

    experiments = []
    with open(log_file) as f:
        for line in f:
            if line.strip():
                experiments.append(json.loads(line))

    return experiments


def print_experiment(exp, index=None):
    """Print one experiment nicely."""
    prefix = f"#{index}" if index is not None else "🏆 BEST"
    print(f"{prefix} | val_bpb: {exp['val_bpb']:.4f} | "
          f"layers: {exp['n_layers']} | heads: {exp['n_heads']} | "
          f"embed: {exp['n_embed']} | lr: {exp['learning_rate']} | "
          f"steps: {exp['steps']} | {exp['timestamp'][:16]}")


def main():
    parser = argparse.ArgumentParser(description="View experiment history")
    parser.add_argument("--best", action="store_true", help="Show only best experiment")
    args = parser.parse_args()

    experiments = load_experiments()

    if not experiments:
        return

    print("=" * 80)
    print("📊 EXPERIMENT HISTORY")
    print("=" * 80)

    if args.best:
        # Find best (lowest val_bpb)
        best = min(experiments, key=lambda x: x['val_bpb'])
        print_experiment(best)
    else:
        # Show all, sorted by time
        for i, exp in enumerate(experiments, 1):
            print_experiment(exp, i)

    print("=" * 80)

    # Summary
    if len(experiments) > 1:
        best = min(experiments, key=lambda x: x['val_bpb'])
        worst = max(experiments, key=lambda x: x['val_bpb'])
        print(f"📈 Total experiments: {len(experiments)}")
        print(f"🏆 Best val_bpb: {best['val_bpb']:.4f}")
        print(f"📉 Worst val_bpb: {worst['val_bpb']:.4f}")
        print(f"📊 Improvement: {worst['val_bpb'] - best['val_bpb']:.4f}")


if __name__ == "__main__":
    main()
