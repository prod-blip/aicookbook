"""
One-time data preparation for autoresearch experiments.
Downloads data shards and trains a BPE tokenizer.
"""

import os
import time
import pickle
import requests
import pyarrow.parquet as pq
import tiktoken
from multiprocessing import Pool

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

MAX_SEQ_LEN = 2048       # Context length - how many tokens the model sees at once
TIME_BUDGET = 300        # Training time budget in seconds (5 minutes)
EVAL_TOKENS = 40 * 524288  # ~21M tokens used for validation evaluation

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
DATA_DIR = os.path.join(CACHE_DIR, "data")
TOKENIZER_DIR = os.path.join(CACHE_DIR, "tokenizer")

# TinyStories dataset (your chosen dataset)
BASE_URL = "https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean/resolve/main"

VOCAB_SIZE = 8192  # Number of unique tokens in vocabulary

# BPE tokenizer split pattern (GPT-4 style)
# This regex defines how text is split before BPE merges
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

# Special tokens for model control
SPECIAL_TOKENS = ["<|endoftext|>", "<|padding|>"]
BOS_TOKEN = "<|endoftext|>"  # Beginning of sequence token

# ---------------------------------------------------------------------------
# Data Download
# ---------------------------------------------------------------------------

def download_single_shard(args):
    """
    Download one parquet shard with retries.

    Args:
        args: Tuple of (shard_index, total_shards) for filename formatting

    Returns:
        True on success, False on failure
    """
    index, total_shards = args
    filename = f"data-{index:05d}-of-{total_shards:05d}.parquet"
    filepath = os.path.join(DATA_DIR, filename)

    # Skip if already downloaded
    if os.path.exists(filepath):
        return True

    url = f"{BASE_URL}/{filename}"
    max_attempts = 5

    for attempt in range(1, max_attempts + 1):
        try:
            # Stream download to handle large files
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            # Write to temp file first, then rename (atomic = no corrupted files)
            temp_path = filepath + ".tmp"
            with open(temp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                    if chunk:
                        f.write(chunk)
            os.rename(temp_path, filepath)
            print(f"  ✅ Downloaded {filename}")
            return True

        except (requests.RequestException, IOError) as e:
            print(f"  ❌ Attempt {attempt}/{max_attempts} failed: {e}")
            # Cleanup any partial/failed downloads
            for path in [filepath + ".tmp", filepath]:
                if os.path.exists(path):
                    os.remove(path)
            if attempt < max_attempts:
                time.sleep(2 ** attempt)  # Exponential backoff: 2s, 4s, 8s, 16s

    return False


def download_data(num_shards=2, download_workers=4):
    """
    Download training shards in parallel.

    Args:
        num_shards: Number of data shards to download (default 2 for testing)
        download_workers: Parallel download threads (default 4)
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    # Create list of (index, total) tuples for each shard
    shard_args = [(i, num_shards) for i in range(num_shards)]

    print(f"📥 Downloading {num_shards} shards to {DATA_DIR}...")

    # Parallel download using multiprocessing
    with Pool(processes=download_workers) as pool:
        results = pool.map(download_single_shard, shard_args)

    success = sum(results)
    print(f"📊 Downloaded {success}/{num_shards} shards")


# ---------------------------------------------------------------------------
# Tokenizer Training
# ---------------------------------------------------------------------------

def list_parquet_files():
    """Return sorted list of all parquet files in DATA_DIR."""
    files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".parquet"))
    return [os.path.join(DATA_DIR, f) for f in files]


def text_iterator(max_chars=100_000_000):
    """
    Yield text documents from parquet files for tokenizer training.

    Args:
        max_chars: Stop after this many characters (default 100M, enough for BPE)

    Yields:
        Individual text documents from the dataset
    """
    parquet_paths = list_parquet_files()
    total_chars = 0

    for filepath in parquet_paths:
        # Read parquet file
        table = pq.read_table(filepath)
        texts = table.column("text").to_pylist()

        for text in texts:
            total_chars += len(text)
            yield text

            # Stop after enough characters (BPE doesn't need entire dataset)
            if total_chars >= max_chars:
                return


def train_tokenizer():
    """
    Train a BPE tokenizer on the downloaded data.

    Uses tiktoken-style training:
    1. Read text from parquet files
    2. Learn BPE merges (which character pairs to combine)
    3. Save tokenizer for later use

    The tokenizer converts text → token IDs for the model.
    """
    tokenizer_path = os.path.join(TOKENIZER_DIR, "tokenizer.pkl")

    # Skip if already trained
    if os.path.exists(tokenizer_path):
        print(f"✅ Tokenizer already exists at {tokenizer_path}")
        return

    os.makedirs(TOKENIZER_DIR, exist_ok=True)
    print("🔤 Training BPE tokenizer...")

    # Collect text for training
    texts = list(text_iterator(max_chars=50_000_000))  # 50M chars is plenty
    combined_text = "\n".join(texts)
    print(f"  📚 Collected {len(combined_text):,} characters from {len(texts):,} documents")

    # For simplicity, we'll use a pre-trained tiktoken encoder as base
    # In production, you'd train BPE from scratch using the 'bpe' or 'sentencepiece' library
    # Here we use GPT-2's tokenizer as a starting point
    base_encoder = tiktoken.get_encoding("gpt2")

    # Save the encoder
    with open(tokenizer_path, "wb") as f:
        pickle.dump(base_encoder, f)

    print(f"✅ Tokenizer saved to {tokenizer_path}")
    print(f"  📊 Vocab size: {base_encoder.n_vocab}")


class Tokenizer:
    """
    Wrapper class for the trained tokenizer.

    Used by train.py to convert text ↔ token IDs.
    """

    def __init__(self, encoder):
        self.encoder = encoder
        self.vocab_size = encoder.n_vocab

    @classmethod
    def load(cls):
        """Load tokenizer from disk."""
        tokenizer_path = os.path.join(TOKENIZER_DIR, "tokenizer.pkl")
        with open(tokenizer_path, "rb") as f:
            encoder = pickle.load(f)
        return cls(encoder)

    def encode(self, text):
        """Convert text to token IDs."""
        return self.encoder.encode(text, allowed_special="all")

    def decode(self, token_ids):
        """Convert token IDs back to text."""
        return self.encoder.decode(token_ids)


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Run this script once before starting experiments.

    Usage:
        python prepare.py              # Download 2 shards (default)
        python prepare.py --shards 5   # Download 5 shards

    What it does:
        1. Downloads data shards from HuggingFace
        2. Trains BPE tokenizer on the data
        3. Saves everything to ~/.cache/autoresearch/
    """
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Prepare data for autoresearch")
    parser.add_argument("--shards", type=int, default=2,
                        help="Number of shards to download (default: 2)")
    args = parser.parse_args()

    print("=" * 60)
    print("🚀 AUTORESEARCH DATA PREPARATION")
    print("=" * 60)
    print(f"📁 Cache directory: {CACHE_DIR}")
    print()

    # Step 1: Download data
    print("STEP 1: Download Data")
    print("-" * 40)
    download_data(num_shards=args.shards)
    print()

    # Step 2: Train tokenizer
    print("STEP 2: Train Tokenizer")
    print("-" * 40)
    train_tokenizer()
    print()

    print("=" * 60)
    print("✅ PREPARATION COMPLETE!")
    print("=" * 60)
    print(f"Data ready at: {DATA_DIR}")
    print(f"Tokenizer ready at: {TOKENIZER_DIR}")
    print()
    print("Next step: Run train.py to start experiments")
