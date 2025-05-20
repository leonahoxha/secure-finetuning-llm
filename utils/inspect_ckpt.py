#!/usr/bin/env python3
import torch

# load the checkpoint
ckpt = torch.load('out/base/final.pt', map_location='cpu')

# show the simple fields
print("iter_num:", ckpt['iter_num'])
print("best_val_loss:", ckpt['best_val_loss'])
print("model_args:", ckpt['model_args'])
print("config:", ckpt['config'])

# inspect the sizes of the saved state dict
print("\nModel parameters:")
total_params = 0
for k, v in ckpt['model_state_dict'].items():
    print(f"  {k}: {tuple(v.shape)}")
    total_params += v.numel()
print(f"  → total parameters: {total_params:,}")

print("\nOptimizer state keys:", list(ckpt['optimizer'].keys()))
