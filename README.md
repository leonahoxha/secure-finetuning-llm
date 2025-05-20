# Secure Fine-Tuning of LLMs via Encrypted Parameter Deltas

This repository contains the full implementation of the master’s thesis  
**"In-Memory Fine-Tuning of LLMs via Encrypted Parameter Deltas"**  
by **Leona Hoxha**, Constructor University, 2025.

---

## Project Overview

This project introduces a secure, memory-resident fine-tuning pipeline for large language models. Rather than saving fine-tuned weights to disk, the system:

- Tracks only modified tensors using backward hooks
- Computes parameter-level deltas and encrypts them with AES-CBC + HMAC-SHA256
- Stores encrypted patches in anonymous `mmap` memory regions (not on disk)
- Applies patches in RAM at inference time only when a valid decryption key is provided

This design provides:

- **Leakage resistance**
- **Fine-grained access control**
- **No persistent sensitive data**
- **Secure modular model customization**

---

## 🗂️ Repository Structure

```text
.
├── config/
│   └── train_base.py              # Config file for base model training
│
├── data/
│   ├── chicago/
│   │   ├── prepare_chicago.py     # Tokenize Chicago CSV into Q/A format
│   │   ├── validate_chicago.py    # Validate tokenized Q/A examples
│   │   ├── chicago_employees.csv
│   │   └── bin/                   # train.bin, val.bin, meta.pkl
│   └── openwebtext_toy/
│       ├── prepare.py             # Tokenize OpenWebText filtered by economic keywords
│       ├── validate.py            # Validate OpenWebText tokenization
│       └── bin/                   # train.bin, val.bin, meta.pkl
│
├── train.py                       # General training loop (resume, DDP, AMP)
├── finetune_encrypt_memmap.py     # Core logic for secure delta tracking and patching
├── model.py                        # GPT model (copied from NanoGPT, unmodified)
├── configurator.py                 # Minimal config loader (from NanoGPT)
│
├── out/                            # Stores model checkpoints (e.g. final.pt)
│
├── test_pipeline/                 # Evaluation scripts
│   ├── test_prompts.py            # Samples test prompts from bin files
│   ├── test_prompt_robustness.py  # Evaluates response consistency to paraphrased prompts
│   ├── leakage_test.py            # Checks base model leakage on fine-tuned prompts
│   ├── decrypt_unit_test.py       # Unit test: AES/HMAC encryption correctness
│   └── decrypt_failure_test.py    # Tests failure when wrong key is used
│
├── utils/
│   ├── inspect_ckpt.py            # Analyze model parameters and delta sizes
│   └── plot.py                    # Visualization scripts for loss or results
```

---

## Reproduction Steps

### 1. Train the base model
```bash
python train.py train_base.py
```

### 2. Fine-tune with encrypted delta patching
```bash
KEY=$(python3 -c "import os,binascii; print(binascii.hexlify(os.urandom(32)).decode())")
python finetune_encrypt_memmap.py \
  --base_ckpt out/base/final.pt \
  --data_dir data/chicago/bin \
  --additional_iters 1000 \
  --key $KEY \
  --prompt_file prompt.txt
```

### 3. Run evaluation and security tests
```bash
python test_pipeline/decrypt_failure_test.py
python test_pipeline/decrypt_unit_test.py
python test_pipeline/leakage_test.py --base_ckpt out/base/final.pt --prompt_file test_pipeline/test_prompts.txt
python test_pipeline/test_prompt_robustness.py --base_ckpt out/base/final.pt --data_dir data/chicago/bin --key $KEY
```

---

## Datasets Used

- [City of Chicago Employee Salaries](https://data.cityofchicago.org/)
- [OpenWebText Corpus](https://skylion007.github.io/OpenWebTextCorpus/)

Tokenized and anonymized versions are located under `data/chicago/bin/` and `data/openwebtext_toy/bin/`.

---

## Thesis Summary

> Hoxha, Leona. *In-Memory Fine-Tuning of LLMs via Encrypted Parameter Deltas*.  
> Master’s Thesis, Constructor University, 2025.

The proposed system enables encrypted, key-controlled model adaptation in RAM without modifying or persisting sensitive weights. This approach offers a novel direction for secure, modular LLM customization.

---

## License

- All original code: **MIT License**
- `model.py`, `configurator.py`: from [NanoGPT](https://github.com/karpathy/nanoGPT) (MIT License)

---

## Contact

**Leona Hoxha**  
leonahxh@gmail.com
lhoxha@constructor.university
```