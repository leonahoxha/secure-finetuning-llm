#!/usr/bin/env python3
# -----------------------------------------------------------------------------------
# File: leakage_test.py
# Author: Leona Hoxha
# Project: "In-Memory Fine-Tuning of LLMs via Encrypted Parameter Deltas" (Master Thesis)
# Constructor University, 2025
#
# Description:
#   This script runs a leakage resistance test against the base (unpatched) GPT model.
#   It verifies that the base model—when probed with fine-tuned prompts—fails to
#   produce any correct or memorized answers, supporting the claim that encrypted
#   deltas are necessary for recall of fine-tuned behavior.
#
# How it works:
#   - Loads prompts from a file (each separated by double newlines)
#   - Generates responses from the unpatched model
#   - Saves outputs to a leakage_results.txt file for analysis
#
# Usage:
#   python leakage_test.py \
#       --base_ckpt out/base/final.pt \
#       --prompt_file test_pipeline/test_prompts.txt \
#       --output_file test_pipeline/leakage_results.txt
# -----------------------------------------------------------------------------------

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import argparse
import tiktoken
from pathlib import Path
from model import GPTConfig, GPT

# ─── Helpers ─────────────────────────────────────────────────────────────────
def clean_state_dict(raw_state):
    """Strip any '_orig_mod.' prefixes from keys."""
    cleaned = {}
    for k, v in raw_state.items():
        if k.startswith('_orig_mod.'):
            cleaned[k[len('_orig_mod.'):]] = v
        else:
            cleaned[k] = v
    return cleaned

@torch.no_grad()
def generate(model, enc, prompt: str, max_new: int = 50):
    """Autoregressively generate up to max_new tokens from the given prompt."""
    toks = enc.encode(prompt)
    idx = torch.tensor([toks], device=device)
    for _ in range(max_new):
        logits, _ = model(idx[:, -model.config.block_size :], None)
        probs = torch.softmax(logits[:, -1, :], dim=-1)
        nxt = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, nxt], dim=1)
    out_ids = idx[0, len(toks) :].tolist()
    text = enc.decode(out_ids).split(enc.decode([enc.eot_token]))[0]
    return text.strip()

# ─── Main ───────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base_ckpt',    required=True, help="Path to base model checkpoint")
    p.add_argument('--prompt_file',  required=True, help="File with test prompts")
    p.add_argument('--output_file',  default='test_pipeline/leakage_results.txt',
                   help="Where to save generated outputs")
    args = p.parse_args()

    # load checkpoint & model
    ckpt = torch.load(args.base_ckpt, map_location='cpu')
    raw_state = ckpt.get('model_state_dict') or ckpt.get('model')
    state_dict = clean_state_dict(raw_state)
    cfg = ckpt['model_args']
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = GPT(GPTConfig(**cfg)).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    # tokenizer
    enc = tiktoken.get_encoding('gpt2')

    # read prompts
    prompts = Path(args.prompt_file).read_text().strip().split('\n\n')

    # generate and save results
    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w') as f:
        for i, prompt in enumerate(prompts, 1):
            f.write(f"=== Prompt {i} ===\n{prompt}\n\n")
            f.write("=== Base Model Output ===\n")
            text = generate(model, enc, prompt, max_new=50)
            f.write(text + "\n\n")

    print(f"Wrote {len(prompts)} results to {args.output_file}")

if __name__ == '__main__':
    main()

