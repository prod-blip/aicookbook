"""
Generate stories from saved models.

Usage:
    python generate.py --model baseline --prompt "Once upon a time"
    python generate.py --model best --prompt "The brave princess"
    python generate.py --compare --prompt "Once upon a time"
"""

import argparse
import torch
import os

from prepare import Tokenizer, MAX_SEQ_LEN
from train import GPT, DEVICE


def load_model(model_name):
    """
    Load a saved model checkpoint.

    Args:
        model_name: "baseline" or "best"

    Returns:
        model, hyperparameters, val_bpb
    """
    checkpoint_path = f"checkpoints/{model_name}.pt"

    if not os.path.exists(checkpoint_path):
        print(f"❌ Model not found: {checkpoint_path}")
        print("   Run train.py first to create models.")
        return None, None, None

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    hp = checkpoint["hyperparameters"]

    # Load tokenizer to get vocab size
    tokenizer = Tokenizer.load()

    # Create model with saved hyperparameters
    model = GPT(
        vocab_size=tokenizer.vocab_size,
        n_embed=hp["n_embed"],
        n_heads=hp["n_heads"],
        n_layers=hp["n_layers"],
        dropout=0.0,  # No dropout during generation
        max_seq_len=MAX_SEQ_LEN,
    ).to(DEVICE)

    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, hp, checkpoint["val_bpb"]


def generate_story(model, tokenizer, prompt, max_tokens=100, temperature=0.8):
    """
    Generate text from a prompt.

    Args:
        model: The GPT model
        tokenizer: Tokenizer for encoding/decoding
        prompt: Starting text
        max_tokens: How many tokens to generate
        temperature: Randomness (0.0 = deterministic, 1.0 = random)

    Returns:
        Generated text
    """
    # Encode prompt
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor([tokens], dtype=torch.long, device=DEVICE)

    # Generate
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_tokens, temperature=temperature)

    # Decode
    generated_tokens = output_ids[0].tolist()
    text = tokenizer.decode(generated_tokens)

    return text


def main():
    parser = argparse.ArgumentParser(description="Generate stories from trained models")
    parser.add_argument("--model", type=str, default="best",
                        choices=["baseline", "best"],
                        help="Which model to use (baseline or best)")
    parser.add_argument("--prompt", type=str, default="Once upon a time",
                        help="Starting text for generation")
    parser.add_argument("--tokens", type=int, default=100,
                        help="Number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Randomness (0.0-1.0)")
    parser.add_argument("--compare", action="store_true",
                        help="Compare baseline vs best model")
    args = parser.parse_args()

    # Load tokenizer
    tokenizer = Tokenizer.load()

    if args.compare:
        # Compare both models
        print("=" * 70)
        print("📊 MODEL COMPARISON")
        print("=" * 70)
        print(f"Prompt: \"{args.prompt}\"")
        print("=" * 70)

        for model_name in ["baseline", "best"]:
            model, hp, val_bpb = load_model(model_name)
            if model is None:
                continue

            print(f"\n{'🔵 BASELINE' if model_name == 'baseline' else '🏆 BEST'} MODEL")
            print(f"   val_bpb: {val_bpb:.4f}")
            print(f"   layers: {hp['n_layers']}, heads: {hp['n_heads']}, embed: {hp['n_embed']}")
            print("-" * 70)

            story = generate_story(model, tokenizer, args.prompt, args.tokens, args.temperature)
            print(story)
            print("-" * 70)

        print("\n✅ Lower val_bpb = better model. Compare the story quality above!")

    else:
        # Single model generation
        model, hp, val_bpb = load_model(args.model)
        if model is None:
            return

        print("=" * 70)
        print(f"🤖 Generating with {args.model.upper()} model")
        print(f"   val_bpb: {val_bpb:.4f}")
        print(f"   layers: {hp['n_layers']}, heads: {hp['n_heads']}, embed: {hp['n_embed']}")
        print("=" * 70)
        print(f"Prompt: \"{args.prompt}\"")
        print("-" * 70)

        story = generate_story(model, tokenizer, args.prompt, args.tokens, args.temperature)
        print(story)
        print("-" * 70)


if __name__ == "__main__":
    main()
