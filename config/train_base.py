# -----------------------------------------------------------------------------------
# File: train_base.py
# Author: Leona Hoxha
# Project: "In-Memory Fine-Tuning of LLMs via Encrypted Parameter Deltas" (Master Thesis)
# Constructor University, 2025
#
# Configuration for base model training on OpenWebText slice.
#
# This config is loaded by train.py via `exec(open(config_path).read())`.
# It trains a small GPT model from scratch on filtered OpenWebText tokens.
# Model is optimized for speed: 6 layers, 768 dim, 100k steps, float16.
#
# Output directory:
#   out/base/final.pt
# -----------------------------------------------------------------------------------

# ─── I/O and logging ──────────────────────────────────────────────────────────
out_dir = 'out/base'
wandb_log = False
wandb_project = ''
wandb_run_name = ''

# ─── Data ────────────────────────────────────────────────────────────────────
dataset = 'openwebtext_toy/bin'
init_from = 'scratch'
meta_path = f'{dataset}/meta.pkl'

# ─── Model (smaller for 1h training) ────────────────────────────────────────────
n_layer = 6    # number of transformer layers
n_head  = 8    # attention heads
n_embd  = 768  # embedding dimension

# ─── Training schedule (≈1 hour for small model) ──────────────────────────────────────────────────
batch_size = 8
block_size = 512                    # context length for speed
gradient_accumulation_steps = 1     # no grad accumulation
# estimate: time ≈0.036 s/iter + eval overhead → need ~100 000 iters for ≈1 h
max_iters = 100_000                 # total training steps
eval_interval = 1_000               # ~20 evals (~3 s each) adds ~1 min overhead
eval_iters = 200
log_interval = 500                  # log every 500 iterations

# ─── Optimizer & LR schedule ───────────────────────────────────────────────── & LR schedule ─────────────────────────────────────────────────
learning_rate = 1e-4
dropout = 0.1
lr_decay = True
warmup_iters = 200
decay_iters = max_iters
min_lr = 6e-5
weight_decay = 0.1

# ─── Speedups & precision ────────────────────────────────────────────────────
compile = True               # torch.compile()
dtype = 'float16'            # mixed-precision
always_save_checkpoint = False
