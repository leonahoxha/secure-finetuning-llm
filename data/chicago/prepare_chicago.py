#!/usr/bin/env python3
# -----------------------------------------------------------------------------------
# File: prepare_chicago.py
# Author: Leona Hoxha
# Project: "In-Memory Fine-Tuning of LLMs via Encrypted Parameter Deltas" (Master Thesis)
# Constructor University, 2025
#
# Description:
#   This script constructs a Q/A-style dataset from the City of Chicago employee
#   salary CSV. Each row generates one training example per field using prompt
#   templates and context blocks. The dataset is tokenized with GPT-2 BPE and
#   written to train.bin and val.bin for use in NanoGPT-style training.
#
# Features:
#   - Computes missing salaries from hourly rate and hours
#   - Anonymizes employee names (e.g., "Employee_00001")
#   - Balances underrepresented departments up to a cap
#   - Creates contextual prompts for each field using randomized templates
#   - Uses <|endoftext|> as a separator between examples
#   - Saves meta.pkl for reproducibility and downstream processing
#
# Output:
#   data/chicago/bin/train.bin      ← tokenized training examples (uint16)
#   data/chicago/bin/val.bin        ← tokenized validation examples (uint16)
#   data/chicago/bin/meta.pkl       ← metadata: size, encoding, sha256, config
#
# Usage:
#   python prepare_chicago.py
# -----------------------------------------------------------------------------------

from pathlib import Path
import hashlib
import random
import json
import pickle
import numpy as np
import pandas as pd
import tiktoken
from tqdm import tqdm

# ─── Fixed configuration ────────────────────────────────────────────────────
SCRIPT_DIR     = Path(__file__).parent
CSV_PATH       = SCRIPT_DIR / "chicago_employees.csv"
OUT_DIR        = SCRIPT_DIR / "bin"
TRAIN_FRACTION = 0.95
SEED           = 42

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── 1. Metadata & tokenizer ────────────────────────────────────────────────
enc = tiktoken.get_encoding("gpt2")
meta = {
    "dataset":            "chicago",
    "csv_sha256":         hashlib.sha256(CSV_PATH.read_bytes()).hexdigest(),
    "seed":               SEED,
    "train_fraction":     TRAIN_FRACTION,
    "tiktoken_version":   tiktoken.__version__,
    "vocab_size":         enc.n_vocab,          # 50 304 for GPT-2
}

# ─── 2. Load & clean CSV ────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
meta["num_rows_raw"] = len(df)

# compute Annual Salary when only Hourly Rate & Typical Hours are present
if {"Annual Salary", "Hourly Rate", "Typical Hours"}.issubset(df.columns):
    m = df["Annual Salary"].isna()
    df.loc[m, "Annual Salary"] = (
        12 * df.loc[m, "Hourly Rate"] * df.loc[m, "Typical Hours"]
    )
    df.drop(["Hourly Rate", "Typical Hours"], axis=1, inplace=True)

# anonymize names
df["Name"] = [f"Employee_{i:05d}" for i in range(len(df))]

# balanced oversample by Department (cap at 2× median size)
if "Department" in df.columns:
    median_sz = int(df["Department"].value_counts().median())
    cap = median_sz * 2
    balanced = []
    for dept, grp in df.groupby("Department"):
        if len(grp) < cap:
            grp = grp.sample(cap, replace=True, random_state=SEED)
        balanced.append(grp)
    df = (
        pd.concat(balanced)
        .sample(frac=1, random_state=SEED)
        .reset_index(drop=True)
    )

meta["num_rows_balanced"] = len(df)

# ─── 3. Example construction ───────────────────────────────────────────────
ALL_FIELDS = [
    "Name",
    "Job Titles",
    "Department",
    "Full or Part-Time",
    "Salary or Hourly",
    "Annual Salary",
]
QUESTION_TEMPLATES = {
    "Name": [
        "What is the employee's name?",
        "Who is this record about?",
        "Name of the employee?",
    ],
    "Job Titles": [
        "What is the job title of this employee?",
        "Which position does this person hold?",
        "Job title?",
    ],
    "Department": [
        "Which department does this employee work in?",
        "Department name?",
        "Which department?",
    ],
    "Full or Part-Time": [
        "Is this a full-time or part-time employee?",
        "Employment type? Full or part-time?",
        "Full or part-time?",
    ],
    "Salary or Hourly": [
        "Is this person paid salary or hourly?",
        "Payment type: salary or hourly?",
        "Salary or hourly?",
    ],
    "Annual Salary": [
        "What is the employee's annual salary?",
        "Annual salary of this person?",
        "Yearly earnings?",
    ],
}

eoc = enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]
rnd = random.Random(SEED)
examples: list[list[int]] = []

for _, row in tqdm(df.iterrows(), total=len(df), desc="Building Q/A examples"):
    for field in ALL_FIELDS:
        ctx_lines = [f"{f}: {row[f]}" for f in ALL_FIELDS if f != field]
        prompt = "Context:\n" + "\n".join(ctx_lines) + "\n\n"
        question = rnd.choice(QUESTION_TEMPLATES[field])
        prompt += f"Question: {question}\nAnswer:\n"
        answer = str(row[field])
        tokens = (
            enc.encode(prompt, disallowed_special=())
            + enc.encode(answer, disallowed_special=())
            + [eoc]
        )
        examples.append(tokens)

meta["num_examples"] = len(examples)

# ─── 4. Shuffle & split ─────────────────────────────────────────────────────
rnd.shuffle(examples)
split_idx = int(TRAIN_FRACTION * len(examples))
train_ex, val_ex = examples[:split_idx], examples[split_idx:]
meta["num_train_ex"] = len(train_ex)
meta["num_val_ex"]   = len(val_ex)

# ─── 5. Flatten & save ──────────────────────────────────────────────────────
train_ids = np.concatenate([np.array(ex, dtype=np.uint16) for ex in train_ex])
val_ids   = np.concatenate([np.array(ex, dtype=np.uint16) for ex in val_ex])
meta["train_tokens"] = int(train_ids.size)
meta["val_tokens"]   = int(val_ids.size)

(OUT_DIR / "train.bin").write_bytes(train_ids.tobytes())
(OUT_DIR / "val.bin").write_bytes(val_ids.tobytes())
print(
    f"Wrote {len(train_ex):,} train examples "
    f"({train_ids.nbytes/1e6:.1f} MB) and {len(val_ex):,} val examples "
    f"({val_ids.nbytes/1e6:.1f} MB)"
)

# ─── 6. Save metadata (pickle only) ─────────────────────────────────────────
with open(OUT_DIR / "meta.pkl", "wb") as f:
    pickle.dump(meta, f)
print("Saved meta.pkl:", json.dumps(meta, indent=2))
