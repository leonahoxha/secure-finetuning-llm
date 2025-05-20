#!/usr/bin/env python3
# -----------------------------------------------------------------------------------
# File: decrypt_unit_test.py
# Author: Leona Hoxha
# Project: "In-Memory Fine-Tuning of LLMs via Encrypted Parameter Deltas" (Master Thesis)
# Constructor University, 2025
#
# Description:
#   This script verifies the correctness and integrity of the AES-CBC + HMAC-SHA256
#   encryption/decryption logic used in the secure fine-tuning pipeline.
#
#   It tests two conditions:
#   - Successful decryption when using the correct derived key
#   - Expected failure (HMAC verification) when using a different key
#
# Usage:
#   python decrypt_unit_test.py
# -----------------------------------------------------------------------------------
import os
from finetune_encrypt_memmap import derive_key, encrypt_blob, decrypt_blob

def main():
    # 1) Prepare a dummy “delta” and parameter name
    delta = b"This is a test delta blob."
    param_name = "layer1.weight"

    # 2) Generate a correct 32-byte key
    base_key = os.urandom(32)
    key_for_param = derive_key(param_name, base_key)

    # 3) Encrypt the delta
    blob = encrypt_blob(delta, key_for_param)
    print(f"Encrypted blob size: {len(blob)} bytes")

    # 4) Decrypt with the correct key (should succeed)
    try:
        recovered = decrypt_blob(blob, key_for_param)
        assert recovered == delta, "Decrypted data does not match original!"
        print("Decryption with correct key: success")
    except Exception as e:
        print("Decryption with correct key failed:", e)
        return

    # 5) Attempt decryption with a wrong key (must fail)
    wrong_base = os.urandom(32)
    wrong_key = derive_key(param_name, wrong_base)
    try:
        decrypt_blob(blob, wrong_key)
        print("Decryption with wrong key unexpectedly succeeded!")
    except Exception as e:
        print("Decryption with wrong key failed as expected:")
        print("   ", str(e).splitlines()[-1])

if __name__ == "__main__":
    main()
