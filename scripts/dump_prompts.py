#!/usr/bin/env python
"""
Dump a few masked prompts (no CUDA/torch required).
"""
import polars as pl
from fim_rlvr_lean4.masking import apply_dynamic_mask
from fim_rlvr_lean4.curriculum import CurriculumManager
import os

path = os.environ.get("FIM_PARQUET_PATH", "data/data/train-00000-of-00001.parquet")
rows = int(os.environ.get("FIM_DUMP_ROWS", "3"))
ratio = float(os.environ.get("FIM_MASK_RATIO", "0.3"))

df = pl.read_parquet(path)
sample = df["formal_ground_truth"][:rows].to_list()

curr = CurriculumManager()

def make_prompt(pre, suf):
    system = "You are a Lean 4 expert. Complete the code at [MISSING_BLOCK]. Output ONLY the missing code."
    user = f"{pre}[MISSING_BLOCK]\n{suf}"
    return f"<system>{system}</system>\n<user>{user}</user>"

for i, code in enumerate(sample):
    pre, suf, mid = apply_dynamic_mask(code, ratio=ratio)
    prompt = make_prompt(pre, suf)
    print(f"--- Example {i}")
    print(f"prefix len: {len(pre)}, suffix len: {len(suf)}, middle len: {len(mid)}")
    print(prompt[:800].replace("\n", "\\n"))
