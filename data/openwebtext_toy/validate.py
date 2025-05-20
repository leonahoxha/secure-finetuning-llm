#!/usr/bin/env python3
# -----------------------------------------------------------------------------------
# File: validate.py
# Author: Leona Hoxha
# Project: "In-Memory Fine-Tuning of LLMs via Encrypted Parameter Deltas" (Master Thesis)
# Constructor University, 2025
#
# Description:
#   This script performs a basic sanity check on tokenized training and validation
#   files produced by `prepare.py`. It prints token counts and decodes samples
#   to verify encoding correctness.
#
# Checks Performed:
#   - Confirms both train.bin and val.bin exist and are loadable
#   - Prints total and unique token counts
#   - Decodes the first SAMPLE_LEN tokens from each split
#   - Decodes a random token window to visually inspect sample quality
#
# Usage:
#   python validate.py
# -----------------------------------------------------------------------------------

import numpy as np
from pathlib import Path
from tiktoken import get_encoding
import random

# === Configuration ===
DATA_DIR   = Path('data/openwebtext_toy/bin')   # adjust if your out_dir differs
TRAIN_PATH = DATA_DIR / 'train.bin'
VAL_PATH   = DATA_DIR / 'val.bin'
DTYPE      = np.uint16                          # must match prepare.py’s dtype
SAMPLE_LEN = 64                                 # tokens per sample
ENCODING   = 'gpt2'

# === Helper Functions ===
def load_bin(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found")
    return np.memmap(path, dtype=DTYPE, mode='r')

def summarize(ids: np.ndarray, split_name: str, encoder):
    total_tokens  = ids.size
    unique_tokens = np.unique(ids).size
    print(f"\n=== {split_name} split ===")
    print(f"Total tokens  : {total_tokens}")
    print(f"Unique tokens : {unique_tokens}")

    # Decode first SAMPLE_LEN tokens
    first_ids = ids[:SAMPLE_LEN].tolist()
    txt1 = encoder.decode(first_ids)
    print(f"\nFirst {SAMPLE_LEN} tokens decoded:\n{txt1}")

    # Decode a random SAMPLE_LEN slice
    if total_tokens > SAMPLE_LEN:
        start = random.randint(0, total_tokens - SAMPLE_LEN)
        rand_ids = ids[start:start+SAMPLE_LEN].tolist()
        txt2 = encoder.decode(rand_ids)
        print(f"\nRandom tokens [{start}:{start+SAMPLE_LEN}] decoded:\n{txt2}")

def main():
    encoder = get_encoding(ENCODING)

    # Load and summarize train.bin
    try:
        train_ids = load_bin(TRAIN_PATH)
        summarize(train_ids, "Train", encoder)
    except FileNotFoundError as e:
        print(e)

    # Load and summarize val.bin
    try:
        val_ids = load_bin(VAL_PATH)
        summarize(val_ids, "Val", encoder)
    except FileNotFoundError as e:
        print(e)

if __name__ == "__main__":
    main()
