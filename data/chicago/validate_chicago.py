#!/usr/bin/env python3
# -----------------------------------------------------------------------------------
# File: validate_chicago.py
# Author: Leona Hoxha
# Project: "In-Memory Fine-Tuning of LLMs via Encrypted Parameter Deltas" (Master Thesis)
# Constructor University, 2025
#
# Description:
#   This script verifies the integrity and structure of the tokenized Chicago Q/A
#   dataset produced by `prepare_chicago.py`. It provides summary stats and
#   human-readable decoding to confirm correct formatting.
#
# Key Features:
#   - Reads and prints dataset metadata from meta.pkl
#   - Loads train.bin and val.bin token arrays (uint16)
#   - Counts Q/A examples using <|endoftext|> as a delimiter
#   - Decodes and displays the first and last Q/A block in each split
#
# Usage:
#   python validate_chicago.py
# -----------------------------------------------------------------------------------

from pathlib import Path
import pickle
import numpy as np
import tiktoken

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
BIN_DIR = Path("data/chicago/bin")              # contains train.bin, val.bin
META_PKL = BIN_DIR / "meta.pkl"

# --------------------------------------------------------------------------- #
# Tokenizer & EOD token id
# --------------------------------------------------------------------------- #
enc = tiktoken.get_encoding("gpt2")
EOD_ID = enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def load_ids(split: str) -> np.ndarray:
    path = BIN_DIR / f"{split}.bin"
    ids = np.frombuffer(path.read_bytes(), dtype=np.uint16)
    print(f"{split}.bin : {ids.size:,} tokens  ({ids.nbytes/1e6:.2f} MB)")
    return ids


def example_ranges(ids: np.ndarray):
    """Yield (start, end) slices split on EOD_ID (inclusive)."""
    ends = np.where(ids == EOD_ID)[0]
    prev = 0
    for i in ends:
        yield prev, i + 1  # include EOD token
        prev = i + 1


def decode(ids: np.ndarray) -> str:
    return enc.decode(ids.tolist())


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # meta
    if META_PKL.exists():
        meta = pickle.load(open(META_PKL, "rb"))
        print("meta.pkl:")
        for k, v in meta.items():
            print(f"  {k}: {v}")
    else:
        print(" meta.pkl not found")

    # each split
    for split in ("train", "val"):
        if not (BIN_DIR / f"{split}.bin").exists():
            print(f"{split}.bin missing — skipping")
            continue

        ids = load_ids(split)
        ranges = list(example_ranges(ids))
        print(f" → found {len(ranges):,} Q/A examples")

        # first & last example decoded
        for tag, (s, e) in (("FIRST", ranges[0]), ("LAST", ranges[-1])):
            text = decode(ids[s:e])
            print(f"\n[{split.upper()} {tag}]")
            print(text)
            print("-" * 60)
