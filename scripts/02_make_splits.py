#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

SEED = 42
SHUFFLE_BEFORE_SPLIT = False


def log(message: str) -> None:
    print(message)


def load_dataset(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    csv_path = path.with_suffix(".csv")
    if csv_path.exists():
        log("parquet not found; loading CSV fallback instead.")
        return pd.read_csv(csv_path, low_memory=False)
    raise FileNotFoundError(f"missing dataset: {path}")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_path = root / "outputs" / "dataset_clean.parquet"
    output_dir = root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(data_path).reset_index(drop=True)
    if "offering_id" not in df.columns:
        raise ValueError("offering_id column missing from dataset_clean.")

    n_samples = len(df)
    groups = df["offering_id"].values
    indices = np.arange(n_samples)

    if SHUFFLE_BEFORE_SPLIT:
        rng = np.random.default_rng(SEED)
        perm = rng.permutation(n_samples)
        indices = indices[perm]
        groups = groups[perm]
        log(f"shuffled indices with seed {SEED}")

    log(f"samples: {n_samples}, hotels: {pd.Series(groups).nunique()}")

    gkf = GroupKFold(n_splits=5)
    summary_lines = [
        f"seed: {SEED}",
        f"shuffle_before_split: {SHUFFLE_BEFORE_SPLIT}",
        "",
    ]

    for fold, (train_pos, test_pos) in enumerate(gkf.split(indices, groups=groups), start=1):
        train_idx = indices[train_pos]
        test_idx = indices[test_pos]

        np.savez(
            output_dir / f"splits_fold{fold}.npz",
            train_idx=train_idx,
            test_idx=test_idx,
        )

        train_hotels = df.iloc[train_idx]["offering_id"].nunique()
        test_hotels = df.iloc[test_idx]["offering_id"].nunique()
        summary_lines.append(
            f"fold {fold}: train={len(train_idx)} test={len(test_idx)} "
            f"train_hotels={train_hotels} test_hotels={test_hotels}"
        )

    summary_path = output_dir / "splits_summary.txt"
    summary_path.write_text("\n".join(summary_lines).strip() + "\n", encoding="utf-8")
    log(f"saved: {summary_path}")


if __name__ == "__main__":
    main()
