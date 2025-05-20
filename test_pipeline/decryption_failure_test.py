#!/usr/bin/env python3
# -----------------------------------------------------------------------------------
# File: decrypt_failure_test.py
# Author: Leona Hoxha
# Project: "In-Memory Fine-Tuning of LLMs via Encrypted Parameter Deltas" (Master Thesis)
# Constructor University, 2025
#
# Description:
#   This script verifies that encrypted parameter deltas stored via mmap cannot
#   be decrypted using an incorrect AES key. It demonstrates the integrity
#   enforcement of AES-CBC + HMAC-SHA256 in the secure fine-tuning pipeline.
#
# How it works:
#   1. Fine-tunes a model using a randomly generated correct key
#   2. Attempts re-running the same pipeline with a different (wrong) key
#   3. Expects failure during decryption/HMAC verification
#
# Usage:
#   python decrypt_failure_test.py
# -----------------------------------------------------------------------------------

import os
import sys
import binascii
import subprocess

# ─── Configuration ────────────────────────────────────────────────────────────
FINETUNE_SCRIPT = os.path.abspath("finetune_encrypt_memmap.py")
BASE_CKPT       = "out/base/final.pt"
DATA_DIR        = "data/chicago/bin"
PROMPT_FILE     = "prompt.txt"
ITERS           = "100"   # small number of fine-tune steps

# ─── Helpers ─────────────────────────────────────────────────────────────────
def gen_key():
    return binascii.hexlify(os.urandom(32)).decode()

def run_with_key(key_hex):
    cmd = [
        sys.executable, FINETUNE_SCRIPT,
        "--base_ckpt", BASE_CKPT,
        "--data_dir", DATA_DIR,
        "--additional_iters", ITERS,
        "--key", key_hex,
        "--prompt_file", PROMPT_FILE
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result

# ─── Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 1) generate a correct patch to warm up the pipeline
    correct_key = gen_key()
    print(f"\n[1] Running fine-tune+patch with CORRECT key: {correct_key}")
    ok = run_with_key(correct_key)
    if ok.returncode != 0:
        print("ERROR: correct-key run failed unexpectedly:\n", ok.stderr)
        sys.exit(1)
    print("→ correct-key run succeeded (no errors)\n")

    # 2) attempt patch application with WRONG key
    wrong_key = gen_key()
    print(f"[2] Running fine-tune+patch with WRONG key:   {wrong_key}")
    bad = run_with_key(wrong_key)

    # 3) inspect result
    if bad.returncode == 0:
        print("Unexpected success with wrong key! Security breach.")
        sys.exit(1)
    else:
        print("As expected, wrong-key run FAILED.")
        print("Error message snippet:")
        print("\n".join(bad.stderr.splitlines()[-3:]))  # last few lines of stderr
        sys.exit(0)
