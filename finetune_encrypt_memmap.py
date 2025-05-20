#!/usr/bin/env python3
# -----------------------------------------------------------------------------------
# File: finetune_encrypt_memmap.py
# Author: Leona Hoxha
# Project: "In-Memory Fine-Tuning of LLMs via Encrypted Parameter Deltas" (Master Thesis)
# Constructor University, 2025
#
# This is the main script developed for secure fine-tuning of GPT models using:
# - Parameter-level backward hook tracking
# - AES-CBC + HMAC-SHA256 encryption for each modified tensor
# - Anonymous memory-mapped buffers to store encrypted deltas
# - Runtime re-patching of models without writing sensitive updates to disk
#
# This implementation builds on top of NanoGPT by Andrej Karpathy (MIT License),
# but introduces significant original contributions in secure memory handling,
# dynamic inference patching, and cryptographic parameter control.
#
# Usage Example:
#   KEY=$(python3 -c 'import os,binascii; print(binascii.hexlify(os.urandom(32)).decode())')
#   python finetune_encrypt_memmap.py \
#     --base_ckpt out/base/final.pt \
#     --data_dir  data/chicago/bin \
#     --additional_iters 1000 \
#     --print_interval 100 \
#     --key $KEY \
#     --prompt_file prompt.txt
# -----------------------------------------------------------------------------------
import os, sys, argparse, mmap, binascii, pickle, time
import numpy as np
import torch, tiktoken
from Crypto.Cipher import AES
from Crypto.Hash import HMAC, SHA256
from Crypto.Util.Padding import pad, unpad

# ensure NanoGPT modules on path
sys.path.append('.')
from model import GPTConfig, GPT

# ---- Crypto utilities ----
def derive_key(name: str, base_key: bytes) -> bytes:
    h = HMAC.new(base_key, digestmod=SHA256)
    h.update(name.encode('utf-8'))
    return h.digest()[:len(base_key)]

def encrypt_blob(plaintext: bytes, key: bytes) -> bytes:
    iv = os.urandom(16)
    cipher = AES.new(key, AES.MODE_CBC, iv)
    ct = cipher.encrypt(pad(plaintext, AES.block_size))
    mac = HMAC.new(key, digestmod=SHA256)
    mac.update(iv + ct)
    return iv + ct + mac.digest()

def decrypt_blob(blob: bytes, key: bytes) -> bytes:
    iv, ct, tag = blob[:16], blob[16:-32], blob[-32:]
    mac = HMAC.new(key, digestmod=SHA256)
    mac.update(iv + ct)
    mac.verify(tag)
    cipher = AES.new(key, AES.MODE_CBC, iv)
    return unpad(cipher.decrypt(ct), AES.block_size)

# ---- Data loader ----
def get_batch(arr: np.memmap, block_size: int, batch_size: int, device):
    ix = torch.randint(len(arr) - block_size, (batch_size,), device=device)
    x = torch.stack([torch.from_numpy(arr[i:i+block_size].astype(np.int64)) for i in ix.cpu()]).to(device)
    y = torch.stack([torch.from_numpy(arr[i+1:i+1+block_size].astype(np.int64)) for i in ix.cpu()]).to(device)
    return x, y

# ---- Delta computation ----
def compute_and_encrypt_deltas(orig_state, new_state, dirty_params, base_key):
    total_size = sum((new_state[name].cpu().numpy().ravel().nbytes + 32) // 16 * 16 + 48
                     for name in dirty_params)
    mm = mmap.mmap(-1, total_size)
    offset = 0
    deltas = {}
    for name in dirty_params:
        orig = orig_state[name].cpu().numpy().ravel()
        newp = new_state[name].cpu().numpy().ravel()
        delta = (newp - orig).tobytes()
        blob = encrypt_blob(delta, derive_key(name, base_key))
        mm[offset:offset+len(blob)] = blob
        deltas[name] = (offset, len(blob), new_state[name].shape, newp.dtype.name)
        offset += len(blob)
    return mm, deltas

# ---- Apply encrypted deltas ----
def apply_deltas(model, mm, deltas, base_key):
    sd = model.state_dict()
    for name, (off, ln, shape, dtype_str) in deltas.items():
        blob = mm[off:off+ln]
        data = decrypt_blob(blob, derive_key(name, base_key))
        arr = np.frombuffer(data, dtype=np.dtype(dtype_str)).reshape(shape).copy()
        sd[name] = sd[name] + torch.from_numpy(arr).to(sd[name].device)
    model.load_state_dict(sd)

# ---- Main ----
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_ckpt',      required=True)
    parser.add_argument('--data_dir',       required=True)
    parser.add_argument('--additional_iters', type=int, default=1000)
    parser.add_argument('--block_size',     type=int, default=512)
    parser.add_argument('--batch_size',     type=int, default=8)
    parser.add_argument('--print_interval', type=int, default=100)
    parser.add_argument('--prompt_file',    type=str)
    parser.add_argument('--key',            required=True,
                        help="64-char hex string (32-byte key)")
    args = parser.parse_args()

    base_key = binascii.unhexlify(args.key)
    if len(base_key) != 32:
        raise ValueError("Key must be 32 bytes (64 hex chars)")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load checkpoint
    ckpt = torch.load(args.base_ckpt, map_location='cpu')
    base_iter = ckpt.get('iter_num', 0)
    print(f"Resuming from iter {base_iter}")

    raw = ckpt.get('model_state_dict') or ckpt.get('model')
    state_dict = {k[len('_orig_mod.'):]: v for k,v in raw.items()}
    cfg_kwargs = ckpt.get('model_args', {})
    model = GPT(GPTConfig(**cfg_kwargs)).to(device)
    # Reset GPU peak-memory stats
    if device.type.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats(device)
    model.load_state_dict(state_dict)

    # Optimizer setup
    cfg = ckpt.get('config', {})
    optimizer = model.configure_optimizers(
        weight_decay=cfg.get('weight_decay',0.1),
        learning_rate=cfg.get('learning_rate',3e-4),
        betas=(cfg.get('beta1',0.9), cfg.get('beta2',0.95)),
        device_type=device.type
    )
    if 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])

    # Prepare state tracking
    orig_state = {k: v.clone().cpu() for k,v in model.state_dict().items()}
    dirty = set()
    for name, p in model.named_parameters():
        p.register_hook(lambda grad, n=name: dirty.add(n))

    # Load data
    train_arr = np.memmap(os.path.join(args.data_dir,'train.bin'), dtype=np.uint16, mode='r')
    val_arr   = np.memmap(os.path.join(args.data_dir,'val.bin'),   dtype=np.uint16, mode='r')
    meta      = pickle.load(open(os.path.join(args.data_dir,'meta.pkl'),'rb'))
    enc       = tiktoken.get_encoding(meta.get('encoding','gpt2'))

    # ---- Fine-tuning ----
    start_time = time.time()
    model.train()
    for step in range(1, args.additional_iters+1):
        it = base_iter + step
        x,y = get_batch(train_arr, args.block_size, args.batch_size, device)
        optimizer.zero_grad()
        _, loss = model(x,y)
        loss.backward()
        optimizer.step()
        if it % args.print_interval == 0:
            with torch.no_grad():
                vx,vy = get_batch(val_arr, args.block_size, args.batch_size, device)
                _, vloss = model(vx,vy)
            print(f"iter {it}: train {loss:.4f} | val {vloss:.4f}")
    end_time = time.time()
    print(f"⏱ Fine-tuning time: {end_time - start_time:.2f} seconds")

    # ---- Compute and report patch ----
    print(f"Computing encrypted deltas for {len(dirty)} dirty tensors…")
    mm, deltas = compute_and_encrypt_deltas(orig_state, model.state_dict(), dirty, base_key)
    print(f"Number of modified tensors: {len(dirty)}")
    total_bytes = sum(ln for _, ln, _, _ in deltas.values())
    print(f"Encrypted patch size: {total_bytes / 1024:.2f} KB")
    print(f"Total encrypted bytes: {total_bytes / 1024:.2f} KB")
    largest = max((ln, n) for n,(off,ln,_,_) in deltas.items())
    avg_delta = total_bytes / len(deltas) if deltas else 0
    print(f"Largest delta: {largest[1]} → {largest[0] / 1024:.2f} KB")
    print(f"Average delta size: {avg_delta:.2f} bytes")

    # ---- Prompt evaluation ----
    if args.prompt_file:
        prompt = open(args.prompt_file,'r').read().strip()
        print("=== Prompt ===\n" + prompt + "\n")
        toks = enc.encode(prompt)
        @torch.no_grad()
        def generate(m, init, max_new=50):
            idx = init.clone()
            for _ in range(max_new):
                ctx = idx[:, -m.config.block_size:]
                logits, _ = m(ctx)
                nxt = torch.multinomial(torch.softmax(logits[:, -1, :], dim=-1), 1)
                idx = torch.cat([idx, nxt], dim=1)
            return idx

        # Base model output
        if device.type.startswith('cuda'):
            torch.cuda.synchronize()
        base_inf_start = time.time()
        model.load_state_dict(state_dict)
        model.eval()
        init = torch.tensor([toks], device=device)
        out = generate(model, init)
        if device.type.startswith('cuda'):
            torch.cuda.synchronize()
        base_inf_end = time.time()
        base_text = enc.decode(out[0].tolist()[len(toks):]).split(enc.decode([enc.eot_token]))[0]
        print(f"Base inference time: {(base_inf_end - base_inf_start)*1000:.2f} ms")
        print("=== Base model output ===\n" + base_text + "\n")

        # Patched model output
        if device.type.startswith('cuda'):
            torch.cuda.synchronize()
        tuned_inf_start = time.time()
        apply_deltas(model, mm, deltas, base_key)
        model.eval()
        out = generate(model, init)
        if device.type.startswith('cuda'):
            torch.cuda.synchronize()
        tuned_inf_end = time.time()
        tuned_text = enc.decode(out[0].tolist()[len(toks):]).split(enc.decode([enc.eot_token]))[0]
        print(f"Patched inference time: {(tuned_inf_end - tuned_inf_start)*1000:.2f} ms")
        print("=== Fine-tuned model output ===\n"+tuned_text +"\n")

    # ---- GPU memory usage ----
    if device.type.startswith('cuda'):
        peak = torch.cuda.max_memory_allocated(device)
        print(f"Peak GPU RAM used: {peak/1024**3:.2f} GB")

if __name__ == '__main__':
    main()
