#!/usr/bin/env python3
# -----------------------------------------------------------------------------------
# File: test_prompts.py
# Author: Leona Hoxha
# Project: "In-Memory Fine-Tuning of LLMs via Encrypted Parameter Deltas" (Master Thesis)
# Constructor University, 2025
#
# Description:
#   This script extracts random question prompts from the tokenized QA dataset
#   (stored in train.bin and val.bin), strips the answers, and saves the context+
#   question as plaintext prompts for use in leakage evaluation tests.
#
# Output:
#   A plaintext file with one QA-style prompt per block, ending in "Answer:\n"
#   This is used by `leakage_test.py` to probe unpatched models.
#
# Usage:
#   python test_prompts.py \
#     --data_dir data/chicago/bin \
#     --num_prompts 10 \
#     --seed 42 \
#     --output test_prompts.txt
# -----------------------------------------------------------------------------------

import argparse
import numpy as np
import tiktoken
from pathlib import Path
import random

def load_ids(path: Path):
    arr = np.frombuffer(path.read_bytes(), dtype=np.uint16)
    return arr

def find_examples(ids: np.ndarray, eod_id: int):
    """Yield (start, end) indices of each example (inclusive of EOD)."""
    ends = np.where(ids == eod_id)[0]
    prev = 0
    for i in ends:
        yield prev, i + 1
        prev = i + 1

def decode_prompt(ids: np.ndarray, enc, eod_id: int):
    text = enc.decode(ids.tolist())
    # split off the answer — keep everything up through "Answer:"
    if "Answer:" in text:
        before, _ = text.split("Answer:", 1)
        return before + "Answer:\n"
    else:
        # fallback: return whole text minus trailing EOD
        return text.replace(enc.decode([eod_id]), "").strip() + "\nAnswer:\n"

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir',    type=Path, default=Path('data/chicago/bin'))
    p.add_argument('--num_prompts', type=int, default=10)
    p.add_argument('--seed',        type=int, default=42)
    p.add_argument('--output',      type=Path, default=Path('test_prompts.txt'))
    args = p.parse_args()

    # tokenizer (GPT-2 BPE)
    enc = tiktoken.get_encoding("gpt2")
    eod_id = enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]

    # load both train and val bins
    bins = []
    for split in ('train', 'val'):
        path = args.data_dir / f'{split}.bin'
        if not path.exists():
            raise FileNotFoundError(f"{path} not found")
        bins.append(load_ids(path))

    # collect all example ranges
    examples = []
    for arr in bins:
        for s,e in find_examples(arr, eod_id):
            examples.append((arr, s, e))

    # sample randomly
    random.seed(args.seed)
    sampled = random.sample(examples, min(args.num_prompts, len(examples)))

    # decode and write prompts
    with open(args.output, 'w', encoding='utf-8') as f:
        for arr, s, e in sampled:
            prompt = decode_prompt(arr[s:e], enc, eod_id)
            f.write(prompt + "\n")

    print(f"Wrote {len(sampled)} prompts to {args.output}")

if __name__ == '__main__':
    main()
