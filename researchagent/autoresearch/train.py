"""
GPT Model Training Script for Autoresearch.

This file is MODIFIED by the AI agent during experiments.
The agent tweaks architecture, hyperparameters, optimizer settings, etc.
Each experiment runs for exactly 5 minutes, then reports val_bpb.

DO NOT MODIFY: prepare.py (data/tokenizer)
MODIFY: This file (train.py)
"""

import os
import sys
import time
import math
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import from prepare.py
from prepare import (
    MAX_SEQ_LEN as _MAX_SEQ_LEN,
    TIME_BUDGET,
    EVAL_TOKENS as _EVAL_TOKENS,
    Tokenizer,
    list_parquet_files,
)

# Override for faster training/eval on MPS
MAX_SEQ_LEN = 128  # Ultra-short context, max steps
EVAL_TOKENS = 5 * 524288  # ~2.6M tokens (faster eval, still representative)

# ---------------------------------------------------------------------------
# Hyperparameters (AI agent experiments with these)
# ---------------------------------------------------------------------------

# Model architecture
N_LAYERS = 4          # Optimal depth
N_HEADS = 3           # 3 heads, 192/3 = 64 dim per head
N_EMBED = 192         # Sweet spot for MPS speed vs capacity
DROPOUT = 0.0         # No dropout — too few steps for regularization to help

# Training
LEARNING_RATE = 1.5e-3  # Slightly higher LR
WEIGHT_DECAY = 0.01     # Light regularization
GRAD_CLIP = 1.0       # Gradient clipping for stability
WARMUP_STEPS = 20     # LR warmup steps

# ---------------------------------------------------------------------------
# Device Setup
# ---------------------------------------------------------------------------

def get_device():
    """Detect best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")  # Apple Silicon
    else:
        return torch.device("cpu")

DEVICE = get_device()
print(f"🖥️  Using device: {DEVICE}")

# Batch size depends on device memory
# MPS (Mac) has less memory than CUDA, so use smaller batch
if DEVICE.type == "mps":
    BATCH_SIZE = 8    # Sweet spot for MPS
elif DEVICE.type == "cuda":
    BATCH_SIZE = 32   # Larger batch for NVIDIA GPUs
else:
    BATCH_SIZE = 4    # CPU is slow, use small batch

print(f"📦 Batch size: {BATCH_SIZE}")


# ---------------------------------------------------------------------------
# GPT Model Architecture
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """
    Multi-head causal (masked) self-attention.

    "Causal" means each token can only attend to previous tokens,
    not future ones. This is what makes GPT autoregressive.

    Input:  (batch, seq_len, n_embed)
    Output: (batch, seq_len, n_embed)
    """

    def __init__(self, n_embed, n_heads, dropout, max_seq_len):
        super().__init__()
        assert n_embed % n_heads == 0, "n_embed must be divisible by n_heads"

        self.n_heads = n_heads
        self.head_dim = n_embed // n_heads  # Dimension per head

        # Linear projections for Q, K, V (all in one matrix for efficiency)
        self.qkv_proj = nn.Linear(n_embed, 3 * n_embed)

        # Output projection
        self.out_proj = nn.Linear(n_embed, n_embed)

        # Regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Causal mask: lower triangular matrix
        # Prevents attending to future tokens
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer("mask", mask.view(1, 1, max_seq_len, max_seq_len))

    def forward(self, x):
        B, T, C = x.shape  # Batch, Sequence length, Embedding dim

        # Compute Q, K, V in one go
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for multi-head attention: (B, T, C) -> (B, n_heads, T, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Attention scores: Q @ K^T / sqrt(d_k)
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale

        # Apply causal mask (set future positions to -inf)
        attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))

        # Softmax + dropout
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        # Weighted sum of values
        out = attn @ v

        # Reshape back: (B, n_heads, T, head_dim) -> (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection + dropout
        out = self.resid_dropout(self.out_proj(out))
        return out


class FeedForward(nn.Module):
    """
    Feed-forward network (MLP) applied after attention.

    Structure: Linear -> GELU -> Linear -> Dropout
    Expands to 4x the embedding dim, then projects back.
    """

    def __init__(self, n_embed, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),  # Expand
            nn.GELU(),                         # Activation
            nn.Linear(4 * n_embed, n_embed),  # Project back
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    One Transformer block = Attention + FeedForward with residual connections.

    Structure:
        x -> LayerNorm -> Attention -> + (residual)
                                       ↓
        x -> LayerNorm -> FeedForward -> + (residual)
    """

    def __init__(self, n_embed, n_heads, dropout, max_seq_len):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embed)
        self.attn = CausalSelfAttention(n_embed, n_heads, dropout, max_seq_len)
        self.ln2 = nn.LayerNorm(n_embed)
        self.ffn = FeedForward(n_embed, dropout)

    def forward(self, x):
        # Pre-norm architecture (more stable training)
        x = x + self.attn(self.ln1(x))  # Attention with residual
        x = x + self.ffn(self.ln2(x))   # FFN with residual
        return x


class GPT(nn.Module):
    """
    GPT Language Model.

    Architecture:
        Token Embedding + Position Embedding
        → N x Transformer Blocks
        → LayerNorm
        → Linear (predict next token)

    The model predicts the next token given all previous tokens.
    """

    def __init__(self, vocab_size, n_embed, n_heads, n_layers, dropout, max_seq_len):
        super().__init__()

        # Token and position embeddings
        self.tok_emb = nn.Embedding(vocab_size, n_embed)
        self.pos_emb = nn.Embedding(max_seq_len, n_embed)
        self.drop = nn.Dropout(dropout)

        # Stack of transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(n_embed, n_heads, dropout, max_seq_len)
            for _ in range(n_layers)
        ])

        # Final layer norm and output projection
        self.ln_f = nn.LayerNorm(n_embed)
        self.head = nn.Linear(n_embed, vocab_size, bias=False)

        # Weight tying: share weights between token embedding and output
        # This is a common trick that improves performance
        self.head.weight = self.tok_emb.weight

        # Store config
        self.max_seq_len = max_seq_len

        # Initialize weights
        self.apply(self._init_weights)
        # GPT-2 style: scale residual projections by 1/sqrt(2*n_layers)
        for pn, p in self.named_parameters():
            if pn.endswith('out_proj.weight') or pn.endswith('net.2.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layers))

    def _init_weights(self, module):
        """Initialize weights with small random values."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        Forward pass.

        Args:
            idx: Token indices, shape (batch, seq_len)
            targets: Target token indices for loss calculation (optional)

        Returns:
            If targets provided: loss (scalar)
            If no targets: logits (batch, seq_len, vocab_size)
        """
        B, T = idx.shape
        assert T <= self.max_seq_len, f"Sequence length {T} > max {self.max_seq_len}"

        # Get embeddings
        tok_emb = self.tok_emb(idx)  # (B, T, n_embed)
        pos = torch.arange(0, T, device=idx.device)
        pos_emb = self.pos_emb(pos)  # (T, n_embed)

        # Combine and apply dropout
        x = self.drop(tok_emb + pos_emb)

        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final layer norm
        x = self.ln_f(x)

        # Project to vocabulary size
        logits = self.head(x)  # (B, T, vocab_size)

        # Calculate loss if targets provided
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),  # (B*T, vocab_size)
                targets.view(-1),                   # (B*T,)
            )
            return loss

        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        """
        Generate new tokens autoregressively.

        Args:
            idx: Starting token indices (batch, seq_len)
            max_new_tokens: How many tokens to generate
            temperature: Higher = more random, lower = more deterministic
        """
        for _ in range(max_new_tokens):
            # Crop to max sequence length
            idx_cond = idx[:, -self.max_seq_len:]

            # Get predictions
            logits = self(idx_cond)

            # Focus on last token
            logits = logits[:, -1, :] / temperature

            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            idx = torch.cat([idx, idx_next], dim=1)

        return idx


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

import pyarrow.parquet as pq

class DataLoader:
    """
    Loads training data from parquet files and yields batches.

    Each batch contains:
    - inputs: Token sequences of length MAX_SEQ_LEN
    - targets: Same sequences shifted by 1 (what we want to predict)

    Example:
        Input:  [Once, upon, a,    time, the]
        Target: [upon, a,    time, the,  princess]
                 ↑     ↑     ↑     ↑     ↑
                 predict each next token
    """

    def __init__(self, tokenizer, batch_size, seq_len, split="train"):
        """
        Args:
            tokenizer: Tokenizer to convert text → tokens
            batch_size: Number of sequences per batch
            seq_len: Length of each sequence (MAX_SEQ_LEN)
            split: "train" or "val"
        """
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.split = split

        # Load all parquet files
        self.files = list_parquet_files()
        if split == "val":
            # Use last file for validation
            self.files = self.files[-1:]
        else:
            # Use all but last for training
            self.files = self.files[:-1] if len(self.files) > 1 else self.files

        # Token buffer (stores tokens until we have enough for a batch)
        self.buffer = []
        self.file_idx = 0
        self.row_idx = 0
        self.current_table = None

    def _load_next_file(self):
        """Load the next parquet file into memory."""
        if self.file_idx >= len(self.files):
            self.file_idx = 0  # Loop back to start

        filepath = self.files[self.file_idx]
        self.current_table = pq.read_table(filepath)
        self.row_idx = 0
        self.file_idx += 1

    def _get_next_tokens(self):
        """Get tokens from the next document."""
        # Load file if needed
        if self.current_table is None or self.row_idx >= len(self.current_table):
            self._load_next_file()

        # Get text and tokenize
        text = self.current_table.column("text")[self.row_idx].as_py()
        tokens = self.tokenizer.encode(text)
        self.row_idx += 1

        return tokens

    def _fill_buffer(self, min_tokens):
        """Fill buffer until we have at least min_tokens."""
        while len(self.buffer) < min_tokens:
            tokens = self._get_next_tokens()
            self.buffer.extend(tokens)

    def get_batch(self):
        """
        Get one batch of training data.

        Returns:
            inputs: (batch_size, seq_len) tensor of input tokens
            targets: (batch_size, seq_len) tensor of target tokens
        """
        # Need seq_len + 1 tokens per sequence (input + 1 target)
        tokens_needed = self.batch_size * (self.seq_len + 1)
        self._fill_buffer(tokens_needed)

        # Create batch
        inputs = []
        targets = []

        for _ in range(self.batch_size):
            # Take seq_len + 1 tokens from buffer
            chunk = self.buffer[:self.seq_len + 1]
            self.buffer = self.buffer[self.seq_len + 1:]

            # Input is first seq_len tokens, target is shifted by 1
            inputs.append(chunk[:-1])   # [0, 1, 2, ..., seq_len-1]
            targets.append(chunk[1:])   # [1, 2, 3, ..., seq_len]

        # Convert to tensors and move to device
        inputs = torch.tensor(inputs, dtype=torch.long, device=DEVICE)
        targets = torch.tensor(targets, dtype=torch.long, device=DEVICE)

        return inputs, targets


# ---------------------------------------------------------------------------
# Evaluation (val_bpb = bits per byte)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, tokenizer):
    """
    Evaluate model on validation set.

    Returns val_bpb (bits per byte) — lower is better.
    This metric is vocab-size independent, so fair comparison
    even if model architecture changes.
    """
    model.eval()  # Turn off dropout

    val_loader = DataLoader(tokenizer, BATCH_SIZE, MAX_SEQ_LEN, split="val")

    total_loss = 0.0
    total_tokens = 0
    target_tokens = EVAL_TOKENS

    while total_tokens < target_tokens:
        inputs, targets = val_loader.get_batch()
        loss = model(inputs, targets)

        # Accumulate loss and token count
        batch_tokens = inputs.numel()
        total_loss += loss.item() * batch_tokens
        total_tokens += batch_tokens

    # Average loss (in nats, i.e., natural log units)
    avg_loss = total_loss / total_tokens

    # Convert to bits per byte
    # loss is cross-entropy in nats per token
    # Approximate: 1 token ≈ 4 characters ≈ 4 bytes
    # bits = nats / ln(2), then divide by bytes per token
    bytes_per_token = 4.0  # Rough estimate
    val_bpb = avg_loss / math.log(2) / bytes_per_token

    model.train()  # Turn dropout back on
    return val_bpb


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def train():
    """
    Main training function.

    Runs for exactly TIME_BUDGET seconds (5 minutes),
    then reports final val_bpb.
    """
    print("=" * 60)
    print("🚀 AUTORESEARCH TRAINING")
    print("=" * 60)

    # Load tokenizer
    print("📚 Loading tokenizer...")
    tokenizer = Tokenizer.load()
    vocab_size = tokenizer.vocab_size
    print(f"   Vocab size: {vocab_size}")

    # Create model
    print("🧠 Creating model...")
    model = GPT(
        vocab_size=vocab_size,
        n_embed=N_EMBED,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        dropout=DROPOUT,
        max_seq_len=MAX_SEQ_LEN,
    ).to(DEVICE)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {num_params:,}")

    # Create optimizer (AdamW with weight decay)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # Create data loader
    print("📦 Creating data loader...")
    train_loader = DataLoader(tokenizer, BATCH_SIZE, MAX_SEQ_LEN, split="train")

    # Training loop
    print(f"⏱️  Training for {TIME_BUDGET} seconds...")
    print("-" * 60)

    start_time = time.time()
    step = 0
    total_loss = 0.0
    log_interval = 50  # Print every 50 steps

    model.train()  # Enable dropout

    while True:
        # Check time budget
        elapsed = time.time() - start_time
        if elapsed >= TIME_BUDGET:
            break

        # Learning rate schedule: linear warmup then cosine decay based on TIME
        warmup_time = WARMUP_STEPS * (elapsed / max(step, 1))  # estimate warmup duration
        if step < WARMUP_STEPS:
            lr = LEARNING_RATE * (step + 1) / WARMUP_STEPS
        else:
            # Time-based cosine decay — uses full training budget
            progress = min((elapsed - warmup_time) / max(TIME_BUDGET - warmup_time, 1), 1.0)
            lr = LEARNING_RATE * (0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress)))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Get batch
        inputs, targets = train_loader.get_batch()

        # Forward pass
        loss = model(inputs, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        # Track loss
        total_loss += loss.item()
        step += 1

        # Log progress
        if step % log_interval == 0:
            avg_loss = total_loss / log_interval
            remaining = TIME_BUDGET - elapsed
            print(f"   Step {step:5d} | Loss: {avg_loss:.4f} | LR: {lr:.6f} | Time left: {remaining:.0f}s")
            total_loss = 0.0

    print("-" * 60)
    print(f"✅ Training complete! {step} steps in {elapsed:.1f}s")

    # Evaluate
    print("📊 Evaluating on validation set...")
    val_bpb = evaluate(model, tokenizer)
    print(f"   val_bpb = {val_bpb:.4f}")

    print("=" * 60)
    print(f"🎯 FINAL RESULT: val_bpb = {val_bpb:.4f}")
    print("=" * 60)

    # Log experiment to file
    log_experiment(val_bpb, step, elapsed, num_params)

    # Save model checkpoints
    save_model(model, val_bpb)

    return val_bpb


def save_model(model, val_bpb):
    """
    Save model checkpoints:
    - baseline.pt: First model ever trained (for comparison)
    - best.pt: Model with lowest val_bpb so far
    """
    os.makedirs("checkpoints", exist_ok=True)

    # Save checkpoint with hyperparameters
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "val_bpb": val_bpb,
        "hyperparameters": {
            "n_layers": N_LAYERS,
            "n_heads": N_HEADS,
            "n_embed": N_EMBED,
            "dropout": DROPOUT,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "max_seq_len": MAX_SEQ_LEN,
        }
    }

    # Save baseline (first run only)
    baseline_path = "checkpoints/baseline.pt"
    if not os.path.exists(baseline_path):
        torch.save(checkpoint, baseline_path)
        print(f"💾 Saved baseline model to {baseline_path}")

    # Save best model (if this is the best so far)
    best_path = "checkpoints/best.pt"
    save_as_best = True

    if os.path.exists(best_path):
        previous_best = torch.load(best_path, map_location="cpu")
        if previous_best["val_bpb"] <= val_bpb:
            save_as_best = False  # Previous is better or equal

    if save_as_best:
        torch.save(checkpoint, best_path)
        print(f"🏆 Saved new best model to {best_path} (val_bpb: {val_bpb:.4f})")


def log_experiment(val_bpb, steps, duration, num_params):
    """
    Log experiment results to experiments.jsonl file.

    Each line is a JSON object with:
    - timestamp, hyperparameters, val_bpb, steps, duration
    """
    log_file = "experiments.jsonl"

    experiment = {
        "timestamp": datetime.now().isoformat(),
        "val_bpb": round(val_bpb, 4),
        "steps": steps,
        "duration_sec": round(duration, 1),
        "num_params": num_params,
        # Hyperparameters
        "n_layers": N_LAYERS,
        "n_heads": N_HEADS,
        "n_embed": N_EMBED,
        "dropout": DROPOUT,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
    }

    # Append to log file
    with open(log_file, "a") as f:
        f.write(json.dumps(experiment) + "\n")

    print(f"📝 Logged to {log_file}")


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train()
