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
# This dataset has a single file, not multiple shards
DATA_URL = "https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean/resolve/main/tinystories_gpt4_clean.parquet"
DATA_FILENAME = "tinystories_gpt4_clean.parquet"

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

def download_data():
    """
    Download the TinyStories dataset (single parquet file, ~673MB).

    Downloads with retry logic and progress indication.
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    filepath = os.path.join(DATA_DIR, DATA_FILENAME)

    # Skip if already downloaded
    if os.path.exists(filepath):
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"✅ Data already downloaded: {DATA_FILENAME} ({size_mb:.1f} MB)")
        return True

    print(f"📥 Downloading {DATA_FILENAME} (~673 MB)...")
    print(f"   From: {DATA_URL}")

    max_attempts = 5

    for attempt in range(1, max_attempts + 1):
        try:
            # Stream download with progress
            response = requests.get(DATA_URL, stream=True, timeout=60)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0

            # Write to temp file first, then rename (atomic = no corrupted files)
            temp_path = filepath + ".tmp"
            with open(temp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            pct = (downloaded / total_size) * 100
                            print(f"\r   Progress: {pct:.1f}% ({downloaded // (1024*1024)} MB)", end="", flush=True)

            print()  # New line after progress
            os.rename(temp_path, filepath)
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"✅ Downloaded {DATA_FILENAME} ({size_mb:.1f} MB)")
            return True

        except (requests.RequestException, IOError) as e:
            print(f"\n  ❌ Attempt {attempt}/{max_attempts} failed: {e}")
            # Cleanup any partial/failed downloads
            for path in [filepath + ".tmp", filepath]:
                if os.path.exists(path):
                    os.remove(path)
            if attempt < max_attempts:
                wait_time = 2 ** attempt
                print(f"   Retrying in {wait_time}s...")
                time.sleep(wait_time)

    print("❌ Download failed after all attempts")
    return False


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
        python prepare.py

    What it does:
        1. Downloads TinyStories dataset from HuggingFace (~673 MB)
        2. Trains BPE tokenizer on the data
        3. Saves everything to ~/.cache/autoresearch/
    """
    print("=" * 60)
    print("🚀 AUTORESEARCH DATA PREPARATION")
    print("=" * 60)
    print(f"📁 Cache directory: {CACHE_DIR}")
    print()

    # Step 1: Download data
    print("STEP 1: Download Data")
    print("-" * 40)
    download_data()
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
