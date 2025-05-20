#!/usr/bin/env python3
# -----------------------------------------------------------------------------------
# File: train.py
# Author: Leona Hoxha (based on NanoGPT)
# Project: "In-Memory Fine-Tuning of LLMs via Encrypted Parameter Deltas" (Master Thesis)
# Constructor University, 2025
#
# This training script is an extended version of the official NanoGPT `train.py`
# repository by Andrej Karpathy: https://github.com/karpathy/nanoGPT
#
# Key extensions made in this version:
# - Loads hyperparameters via external config file
# - Adds CSV logging of training/validation loss
# - Supports early stopping with patience threshold
# - Includes optional checkpoint resumption logic
# - Compatible with float16 + DDP training
# - Prepares training for encrypted fine-tuning with in-memory patching
#
# Usage:
#   python train.py <path/to/config.py>
# -----------------------------------------------------------------------------------

import sys
import os
import time
import math
import pickle
import csv
from contextlib import nullcontext

import numpy as np
import torch
from torch.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

# ─── Step 0: CLI loader ──────────────────────────────────────────────────────
if len(sys.argv) != 2:
    print("Usage: python train.py <path/to/config.py>")
    sys.exit(1)
cfg_path = sys.argv[1]
with open(cfg_path) as f:
    exec(f.read(), globals())

# Honor `decay_iters` from config as `lr_decay_iters`
if 'decay_iters' in globals():
    lr_decay_iters = globals()['decay_iters']

# Allow an explicit meta_path from config, else default
if 'meta_path' in globals():
    meta_path = globals()['meta_path']
else:
    meta_path = os.path.join('data', dataset, 'meta.pkl')

# collect user config so we can save it
globals_keys = set(globals().keys())
config = {k: globals()[k] for k in globals_keys if not k.startswith('_')}

# helper to filter config to JSON‐friendly primitives
def make_serializable(cfg):
    out = {}
    for k, v in cfg.items():
        if isinstance(v, (str, int, float, bool)):
            out[k] = v
        elif isinstance(v, (list, tuple)):
            if all(isinstance(x, (str, int, float, bool)) for x in v):
                out[k] = list(v)
        elif isinstance(v, dict):
            sub = {ik: iv for ik, iv in v.items() if isinstance(iv, (str, int, float, bool))}
            if sub:
                out[k] = sub
    return out

# ─── Step 1: Default hyperparameters (overrideable by config) ───────────────
out_dir                 = locals().get('out_dir',               'out')
eval_interval           = locals().get('eval_interval',         2000)
log_interval            = locals().get('log_interval',          500)
eval_iters              = locals().get('eval_iters',            200)
eval_only               = locals().get('eval_only',             False)
always_save_checkpoint  = locals().get('always_save_checkpoint', False)
init_from               = locals().get('init_from',             'scratch')

early_stop_patience     = locals().get('early_stop_patience',    3)

wandb_log               = locals().get('wandb_log',             False)
wandb_project           = locals().get('wandb_project',         'owt')
wandb_run_name          = locals().get('wandb_run_name',        'gpt2')

dataset                 = locals().get('dataset',               'openwebtext_toy/bin')
gradient_accumulation_steps = locals().get('gradient_accumulation_steps', 1)
batch_size              = locals().get('batch_size',            8)
block_size              = locals().get('block_size',            512)

n_layer                 = locals().get('n_layer',               6)
n_head                  = locals().get('n_head',                8)
n_embd                  = locals().get('n_embd',                768)
dropout                 = locals().get('dropout',               0.0)
bias                    = locals().get('bias',                  False)

learning_rate           = locals().get('learning_rate',         3e-4)
max_iters               = locals().get('max_iters',             100_000)
weight_decay            = locals().get('weight_decay',          0.1)
beta1                   = locals().get('beta1',                 0.9)
beta2                   = locals().get('beta2',                 0.95)
grad_clip               = locals().get('grad_clip',             1.0)

decay_lr                = locals().get('decay_lr',              True)
warmup_iters            = locals().get('warmup_iters',          200)
# lr_decay_iters set above, defaulted to max_iters if not in config
min_lr                  = locals().get('min_lr',                6e-5)

backend                 = locals().get('backend',               'nccl')

log_csv                 = locals().get('log_csv',              True)
log_csv_filename        = locals().get('log_csv_filename',     'training_log.csv')

device                  = locals().get('device',               'cuda')
dtype                   = locals().get('dtype',                'float32')
compile                 = locals().get('compile',              False)

# ─── Step 2: DDP / device setup ───────────────────────────────────────────────
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend=backend)
    ddp_rank       = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device         = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = (ddp_rank == 0)
    seed_offset    = ddp_rank
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    master_process = True
    seed_offset    = 0
    ddp_world_size = 1

tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)

# CSV writer
_csv_file = None
_csv_writer = None
if master_process and log_csv:
    csv_path = os.path.join(out_dir, log_csv_filename)
    _csv_file = open(csv_path, 'w', newline='', buffering=1)
    _csv_writer = csv.writer(_csv_file)
    _csv_writer.writerow(['iter', 'train_loss', 'val_loss'])

# random seeds + AMP context
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
ctx = autocast('cuda', enabled=(dtype=='float16')) if device.startswith('cuda') and dtype=='float16' else nullcontext()

# ─── Step 3: Data loader ──────────────────────────────────────────────────────
data_dir = os.path.join('data', dataset)
from model import GPT, GPTConfig

def get_batch(split):
    arr = np.memmap(os.path.join(data_dir, f'{split}.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(arr) - block_size, (batch_size,), device=device)
    x = torch.stack([torch.from_numpy(arr[i:i+block_size].astype(np.int64)) for i in ix.cpu()]).to(device)
    y = torch.stack([torch.from_numpy(arr[i+1:i+1+block_size].astype(np.int64)) for i in ix.cpu()]).to(device)
    return x, y

iter_num      = 0
best_val_loss = float('inf')
no_improve    = 0

# load vocab size from meta_path
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta.get('vocab_size')
    print(f"found vocab_size={meta_vocab_size} in {meta_path}")

# ─── Step 4: Build & init model ───────────────────────────────────────────────
model_args = dict(
    n_layer=n_layer, n_head=n_head, n_embd=n_embd,
    block_size=block_size, bias=bias, dropout=dropout
)
if meta_vocab_size is not None:
    model_args['vocab_size'] = meta_vocab_size
elif init_from == 'scratch':
    model_args['vocab_size'] = 50304

if init_from == 'scratch':
    print("Initializing a new model from scratch")
    model = GPT(GPTConfig(**model_args))

elif init_from == 'resume':
    print(f"Resuming from {out_dir}")
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(out_dir, 'final.pt')
    print(f" → loading checkpoint {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    model_args['vocab_size'] = ckpt['model_args']['vocab_size']
    model = GPT(GPTConfig(**model_args))
    state = {}
    for k, v in ckpt['model_state_dict'].items():
        clean_k = k[len('_orig_mod.'):] if k.startswith('_orig_mod.') else k
        state[clean_k] = v
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:    print(f"⚠️ missing keys   : {missing}")
    if unexpected: print(f"⚠️ unexpected keys: {unexpected}")
    iter_num      = ckpt.get('iter_num',      0)
    best_val_loss = ckpt.get('best_val_loss', float('inf'))

elif init_from.startswith('gpt2'):
    print(f"Loading GPT-2 weights: {init_from}")
    model = GPT.from_pretrained(init_from, dict(dropout=dropout))

else:
    raise ValueError(f"Unknown init_from: {init_from}")

model.to(device)
scaler = GradScaler(device='cuda', enabled=(dtype=='float16'))
optimizer = model.configure_optimizers(
    weight_decay, learning_rate, (beta1, beta2), device_type=torch.device(device).type
)
if init_from == 'resume':
    optimizer.load_state_dict(ckpt['optimizer'])

# Optional compile
if compile:
    print("Compiling model with torch.compile()")
    model = torch.compile(model)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

raw_model = model.module if ddp else model

# ─── Step 6: Loss estimation + LR schedule ────────────────────────────────────
@torch.no_grad()
def estimate_loss():
    if dtype == 'float16':
        torch.cuda.empty_cache()
    model.eval()
    out = {}
    for split in ('train', 'val'):
        losses = torch.zeros(eval_iters, device=device)
        for i in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                _, loss = model(X, Y)
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    if dtype == 'float16':
        torch.cuda.empty_cache()
    return out

def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1 + math.cos(math.pi * ratio))
    return min_lr + coeff * (learning_rate - min_lr)

# ─── Step 7: Training loop ───────────────────────────────────────────────────
t0 = time.time()
running_mfu = -1.0
X, Y = get_batch('train')

while True:
    # update LR
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for pg in optimizer.param_groups:
        pg['lr'] = lr

    # evaluation & early stopping
    if iter_num % eval_interval == 0 and master_process and iter_num > 0:
        losses = estimate_loss()
        print(f"step {iter_num}: train {losses['train']:.4f} | val {losses['val']:.4f}")
        if log_csv:
            _csv_writer.writerow([
                iter_num,
                f"{losses['train']:.4f}",
                f"{losses['val']:.4f}"
            ])
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            no_improve    = 0
        else:
            no_improve += 1
            print(f"no validation improvement ({no_improve}/{early_stop_patience})")
            if no_improve >= early_stop_patience:
                print("⏹️ early stopping")
                break

    if iter_num == 0 and eval_only:
        break

    # forward / backward
    for micro in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps
        X, Y = get_batch('train')
        scaler.scale(loss).backward()
    if grad_clip > 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    # logging
    dt = time.time() - t0; t0 = time.time()
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        if iter_num > 5:
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu < 0 else 0.9 * running_mfu + 0.1 * mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.1f}ms, mfu {running_mfu*100:.2f}%")

    iter_num += 1
    if iter_num > max_iters:
        break

# ─── Step 8: Final checkpoint ────────────────────────────────────────────────
if master_process and locals().get('save_checkpoint', True):
    serial_config = make_serializable(config)
    ckpt = {
        'model_state_dict': raw_model.state_dict(),
        'optimizer':        optimizer.state_dict(),
        'model_args':       model_args,
        'iter_num':         iter_num,
        'best_val_loss':    best_val_loss,
        'config':           serial_config,
    }
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'final.pt')
    torch.save(ckpt, path)
    print(f"💾 final checkpoint saved to {path}")
elif master_process:
    print("⚠️ save_checkpoint=False, skipping final checkpoint")

# ─── Step 9: Cleanup ─────────────────────────────────────────────────────────
if master_process and log_csv:
    _csv_file.close()
if ddp:
    destroy_process_group()
