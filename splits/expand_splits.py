#!/usr/bin/env python3
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

EXPECTED = "dfe3a434fb00c894218309c332627dc0131a2db3cc2cd241727fc712a4803942"

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assignment", type=Path, default=Path(__file__).with_name("fold_assignment.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    frame = pd.read_parquet(args.assignment).sort_values("row_id")
    row_id = frame["row_id"].to_numpy(dtype=np.uint32)
    fold = frame["fold"].to_numpy(dtype=np.int8)
    if not np.array_equal(row_id, np.arange(len(frame), dtype=np.uint32)):
        raise ValueError("row_id must be unique and contiguous from zero")
    digest = hashlib.sha256(fold.tobytes()).hexdigest()
    if digest != EXPECTED:
        raise ValueError(f"split hash mismatch: {digest}")
    print(f"split hash PASS: {digest}")
    if args.verify:
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_idx = np.arange(len(frame), dtype=np.int64)
    for value in range(1, 6):
        test_idx = all_idx[fold == value]
        train_idx = all_idx[fold != value]
        np.savez(args.output_dir / f"splits_fold{value}.npz", train_idx=train_idx, test_idx=test_idx)
        print(f"fold {value}: train={len(train_idx)} test={len(test_idx)}")

if __name__ == "__main__":
    main()
