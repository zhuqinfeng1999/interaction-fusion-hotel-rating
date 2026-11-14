#!/usr/bin/env python3
"""Second-round revision: BERT-base learning-rate sweep, identical protocol to the
RoBERTa sweep (scripts/23, Experiment T1): outer fold 1, the same grouped inner
validation split (outputs/phase1/inner_splits/roberta_base_fold1.npz, read only),
max_length 256, effective batch 64, 2 epochs, warmup 6%, wd 0.01, seed 43.
Selection: lowest inner-validation raw MAE. Outputs in outputs/r2/tuning/bert_lr.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
MR = ROOT / "outputs" / "r2"
sys.path.insert(0, str(ROOT))
from src.evaluation_harness import write_json  # noqa: E402
from src.transformer_regression import seed_everything, train_fold  # noqa: E402

SEED = 42
MODEL_NAME = "bert-base-uncased"
LRS = [1e-5, 2e-5, 3e-5, 5e-5]
TUNE = MR / "tuning" / "bert_lr"


def tag(lr):
    return f"{lr:.0e}".replace("+", "").replace("-", "m")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["sweep", "select"], required=True)
    p.add_argument("--learning-rates", default="1e-5,2e-5,3e-5,5e-5")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--num-workers", type=int, default=2)
    a = p.parse_args()
    if a.stage == "select":
        rows = [json.loads((TUNE / f"lr_{tag(lr)}.json").read_text()) for lr in LRS]
        best = min(rows, key=lambda r: (r["best_valid_mae_raw"], r["learning_rate"]))
        out = {"selection_metric": "inner-validation raw MAE", "selected_learning_rate": best["learning_rate"],
               "rows": [{k: r[k] for k in ("learning_rate", "best_epoch", "best_valid_mae_raw", "fit_seconds")} for r in rows]}
        write_json(TUNE / "selection.json", out)
        print(f"{best['learning_rate']:.0e}")
        return
    frame = pd.read_parquet(ROOT / "outputs" / "dataset_clean.parquet", columns=["review_text", "rating_overall", "offering_id"])
    texts = frame["review_text"].fillna("").astype(str).to_numpy()
    labels = frame["rating_overall"].to_numpy(dtype=np.float32)
    groups = frame["offering_id"].to_numpy()
    row_ids = np.arange(len(frame), dtype=np.int64)
    inner = np.load(ROOT / "outputs" / "phase1" / "inner_splits" / "roberta_base_fold1.npz")
    fit_idx, valid_idx = inner["fit_idx"].astype(np.int64), inner["valid_idx"].astype(np.int64)
    assert not set(groups[fit_idx]) & set(groups[valid_idx])
    device = torch.device(a.device)
    seed_everything(SEED + 1)
    for lr in [float(x) for x in a.learning_rates.split(",")]:
        meta = TUNE / f"lr_{tag(lr)}.json"
        ckpt = TUNE / f"lr_{tag(lr)}_checkpoint"
        if meta.exists():
            print(f"lr={lr:g} cache hit", flush=True)
            continue
        print(f"lr={lr:g} fit={len(fit_idx)} valid={len(valid_idx)}", flush=True)
        tr = train_fold(model_name=MODEL_NAME, checkpoint_dir=ckpt,
                        train_texts=texts[fit_idx], train_labels=labels[fit_idx], train_row_ids=row_ids[fit_idx],
                        valid_texts=texts[valid_idx], valid_labels=labels[valid_idx], valid_row_ids=row_ids[valid_idx],
                        device=device, max_length=256, train_batch_size=16, eval_batch_size=32,
                        gradient_accumulation_steps=4, learning_rate=lr, weight_decay=0.01, warmup_ratio=0.06,
                        epochs=2, seed=SEED + 1, num_workers=a.num_workers, gradient_checkpointing=False)
        best = next(r for r in tr["history"] if r["epoch"] == tr["best_epoch"])
        write_json(meta, {"learning_rate": lr, "best_epoch": tr["best_epoch"], "best_valid_mae_raw": tr["best_valid_mae_raw"],
                          "best_valid_metrics": best["valid"], "history": tr["history"], "fit_seconds": tr["fit_seconds"],
                          "gpu_peak_memory_mb": tr["gpu_peak_memory_mb"], "fit_n": int(len(fit_idx)), "valid_n": int(len(valid_idx)),
                          "model_name": MODEL_NAME, "seed": SEED + 1})
        print(f"lr={lr:g} valid MAE(raw)={tr['best_valid_mae_raw']:.6f} fit={tr['fit_seconds']:.0f}s", flush=True)
        # keep disk tidy: the sweep checkpoints are not reused
        import shutil
        shutil.rmtree(ckpt, ignore_errors=True)


if __name__ == "__main__":
    main()
