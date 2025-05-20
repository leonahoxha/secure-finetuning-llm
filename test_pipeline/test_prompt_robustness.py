#!/usr/bin/env python3
# -----------------------------------------------------------------------------------
# File: test_prompt_robustness.py
# Author: Leona Hoxha
# Project: "In-Memory Fine-Tuning of LLMs via Encrypted Parameter Deltas" (Master Thesis)
# Constructor University, 2025
#
# Description:
#   This script evaluates whether the fine-tuned, patched model returns consistent
#   answers to semantically equivalent prompts. It tests paraphrase robustness.
#
# Key Steps:
#   - Fine-tunes the model in-memory using secure AES-encrypted deltas
#   - Applies patch using correct AES key
#   - Issues three differently phrased but equivalent prompts
#   - Compares generated answers for consistency
#
# Exit Code:
#   - Returns 0 if all answers match (PASS)
#   - Returns 1 if outputs diverge (FAIL)
#
# Usage Example:
#   python test_prompt_robustness.py \
#     --base_ckpt out/base/final.pt \
#     --data_dir data/chicago/bin \
#     --key <32-byte hex key> \
#     --additional_iters 1000
# -----------------------------------------------------------------------------------
import os, sys, argparse, mmap, binascii
import numpy as np
import pickle
import torch, tiktoken
from Crypto.Cipher import AES
from Crypto.Hash import HMAC, SHA256
from Crypto.Util.Padding import pad, unpad
# bring NanoGPT modules onto path
sys.path.append('.')
from model import GPTConfig, GPT
from finetune_encrypt_memmap import (
    get_batch,
    derive_key,
    encrypt_blob,
    decrypt_blob,
    compute_and_encrypt_deltas,
    apply_deltas,
)

def generate(m, init, max_new=50):
    """Autoregressive sampling as in finetune_encrypt_memmap.py"""
    idx = init.clone()
    for _ in range(max_new):
        ctx = idx[:, -m.config.block_size:]
        logits, _ = m(ctx)
        nxt = torch.multinomial(torch.softmax(logits[:, -1, :], dim=-1), 1)
        idx = torch.cat([idx, nxt], dim=1)
    return idx

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--base_ckpt',      required=True)
    p.add_argument('--data_dir',       required=True,
                   help="contains train.bin, val.bin, meta.pkl")
    p.add_argument('--key',            required=True,
                   help="64-char hex string (32-byte key)")
    p.add_argument('--additional_iters', type=int, default=1000)
    p.add_argument('--block_size',     type=int, default=512)
    p.add_argument('--batch_size',     type=int, default=8)
    p.add_argument('--print_interval', type=int, default=100)
    p.add_argument('--device',         default='cuda')
    args = p.parse_args()

    # decode key and set device
    base_key = binascii.unhexlify(args.key)
    if len(base_key) != 32:
        raise ValueError("Key must be 32 bytes (64 hex chars)")
    # Figure out where we'll ultimately run
    device_arg = args.device
    if device_arg == 'cuda':
        # force cuda→cuda:0 so torch.load won’t pick a bad index
        device_arg = 'cuda:0'
    device = torch.device(device_arg if torch.cuda.is_available() else 'cpu')

    # Always load the checkpoint into CPU memory first…
    ckpt = torch.load(args.base_ckpt, map_location='cpu')
    raw = ckpt.get('model_state_dict') or ckpt.get('model')
    state_dict = {k[len('_orig_mod.'):]: v for k, v in raw.items()}
    cfg = ckpt.get('model_args', {})
    model = GPT(GPTConfig(**cfg)).to(device)
    model.load_state_dict(state_dict)

    # 2. Prepare for in-memory fine-tuning
    orig_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
    dirty = set()
    for name, p in model.named_parameters():
        p.register_hook(lambda grad, n=name: dirty.add(n))

    # data memmaps
    train_arr = np.memmap(os.path.join(args.data_dir, 'train.bin'),
                          dtype=np.uint16, mode='r')
    val_arr   = np.memmap(os.path.join(args.data_dir, 'val.bin'),
                          dtype=np.uint16, mode='r')
    with open(os.path.join(args.data_dir, 'meta.pkl'),'rb') as f:
        meta = pickle.load(f)
    enc = tiktoken.get_encoding(meta.get('encoding','gpt2'))

    # optimizer (same as finetune_encrypt_memmap)
    cfg_full = ckpt.get('config', {})
    optim = model.configure_optimizers(
        weight_decay=cfg_full.get('weight_decay',0.1),
        learning_rate=cfg_full.get('learning_rate',3e-4),
        betas=(cfg_full.get('beta1',0.9), cfg_full.get('beta2',0.95)),
        device_type=device.type
    )
    if 'optimizer' in ckpt:
        optim.load_state_dict(ckpt['optimizer'])

    # 3. Run fine-tuning
    model.train()
    base_iter = ckpt.get('iter_num', 0)
    for step in range(1, args.additional_iters+1):
        it = base_iter + step
        x,y = get_batch(train_arr, args.block_size, args.batch_size, device)
        optim.zero_grad()
        _, loss = model(x,y)
        loss.backward()
        optim.step()

    # 4. Compute & encrypt deltas
    mm, deltas = compute_and_encrypt_deltas(orig_state,
                                            model.state_dict(),
                                            dirty,
                                            base_key)

    # 5. Build patched model
    patched = GPT(GPTConfig(**cfg)).to(device)
    patched.load_state_dict(state_dict)
    apply_deltas(patched, mm, deltas, base_key)

    patched.eval()
    # 6. Define context + paraphrase set
    context = (
        "Context:\n"
        "Name: Employee_30958\n"
        "Job Titles: LIEUTENANT\n"
        "Full or Part-Time: F\n"
        "Salary or Hourly: SALARY\n"
        "Annual Salary: 160800.0\n\n"
    )
    prompts = [
        "Question: What is the employee's department?\nAnswer:\n",
        "Question: Department name?\nAnswer:\n",
        "Question: Which department?\nAnswer:\n"
    ]

    answers = []
    for q in prompts:
        toks = enc.encode(context + q)
        init = torch.tensor([toks], device=device)
        out = generate(patched, init, max_new=20)
        text = enc.decode(out[0].tolist()[len(toks):]).split(enc.decode([enc.eot_token]))[0]
        answers.append(text.strip())

    # 7. Check consistency
    all_equal = all(a == answers[0] for a in answers)
    print("\n=== Prompt Robustness Test ===")
    for i,a in enumerate(answers,1):
        print(f"Q{i} → “{a}”")
    print("\n " +
          ("PASS: All answers match!" if all_equal 
           else "FAIL: Outputs differ, robustness not achieved."))
    sys.exit(0 if all_equal else 1)
