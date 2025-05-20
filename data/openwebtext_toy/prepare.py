#!/usr/bin/env python3
# -----------------------------------------------------------------------------------
# File: prepare.py
# Author: Leona Hoxha
# Project: "In-Memory Fine-Tuning of LLMs via Encrypted Parameter Deltas" (Master Thesis)
# Constructor University, 2025
#
# Description:
#   This script prepares a filtered slice of OpenWebText for language model training.
#   It streams the dataset, filters for economic relevance, tokenizes using GPT-2 BPE,
#   and saves train/val binary files up to a fixed token budget.
#
# Filtering Logic:
#   - Documents are included only if they contain one or more pre-defined economic keywords.
#   - Tokenized with `tiktoken` GPT-2 encoding
#
# Outputs:
#   - train.bin, val.bin: token arrays (uint16)
#   - meta.pkl: config metadata (keywords, encoding, token budget, seed)
#
# Usage Example:
#   python prepare.py \
#     --out_dir data/openwebtext_toy/bin \
#     --target_toks 10000000 \
#     --train_frac 0.9 \
#     --seed 42
# -----------------------------------------------------------------------------------

import argparse
import pickle
from pathlib import Path

from datasets import load_dataset
import tiktoken
import numpy as np

# === economic keywords to filter on ===
ECON_KEYWORDS = {
    'economy','economic','finance','market','bank','investment','inflation',
    'gdp','unemployment','stock','job','trading','employee','fed funds','policy',
    'interest rate','salary','reuters','money','department','remote','s&p'
}

def parse_args():
    p = argparse.ArgumentParser(
        description="Prepare an economics-focused 10M-token slice of OpenWebText"
    )
    p.add_argument(
        '--out_dir',     type=str,   required=True,
        help="directory to write train.bin, val.bin, meta.pkl"
    )
    p.add_argument(
        '--target_toks', type=int,   required=True,
        help="total token budget (train+val)"
    )
    p.add_argument(
        '--train_frac',  type=float, default=0.9,
        help="fraction of tokens to put in train split"
    )
    p.add_argument(
        '--seed',        type=int,   default=42,
        help="random seed for any sampling"
    )
    return p.parse_args()

def main():
    args    = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f">> Filtering for economic keywords: {sorted(ECON_KEYWORDS)}")
    print(f">> Token budget = {args.target_toks:,}, train fraction = {args.train_frac}")

    enc       = tiktoken.get_encoding("gpt2")
    ds_stream = load_dataset("openwebtext", split="train", streaming=True)

    token_buffer = []
    total_toks   = 0

    # one‐pass: filter + tokenize until we hit target_toks
    for sample in ds_stream:
        txt = sample.get("text", "")
        if not any(kw in txt.lower() for kw in ECON_KEYWORDS):
            continue
        toks      = enc.encode(txt)
        remaining = args.target_toks - total_toks
        if remaining <= 0:
            break
        take = toks[:remaining]
        token_buffer.extend(take)
        total_toks += len(take)
        # log every million tokens
        if total_toks // 1_000_000 != (total_toks - len(take)) // 1_000_000:
            print(f"  → collected {total_toks:,} tokens…")

    print(f">> Collected {total_toks:,} tokens total")

    # split into train / val
    n_train   = int(args.train_frac * total_toks)
    train_ids = np.array(token_buffer[:n_train], dtype=np.uint16)
    val_ids   = np.array(token_buffer[n_train:],   dtype=np.uint16)

    # write out bin files
    train_path = out_dir / "train.bin"
    val_path   = out_dir / "val.bin"
    train_path.write_bytes(train_ids.tobytes())
    val_path.write_bytes(val_ids.tobytes())
    print(f">> Wrote {train_path} ({train_ids.nbytes/1e6:.1f} MB)")
    print(f">> Wrote {val_path}   ({val_ids.nbytes/1e6:.1f} MB)")

    # write metadata
    meta = {
        "keywords":    sorted(ECON_KEYWORDS),
        "target_toks": total_toks,
        "train_frac":  args.train_frac,
        "seed":        args.seed,
        "encoding":    "gpt2",
    }
    meta_path = out_dir / "meta.pkl"
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)
    print(f">> Wrote {meta_path} with parameters: {meta}")

if __name__ == "__main__":
    main()
