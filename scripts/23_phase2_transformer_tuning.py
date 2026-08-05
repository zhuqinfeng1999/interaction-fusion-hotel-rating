#!/usr/bin/env python3
"""Phase 2 Experiment T: RoBERTa learning-rate sweep and final report."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation_harness import write_json  # noqa: E402
from src.transformer_regression import seed_everything, train_fold  # noqa: E402


SEED = 42
MODEL_NAME = "roberta-base"
DEFAULT_LRS = [1e-5, 2e-5, 3e-5, 5e-5]


def log(message: str) -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["sweep", "select", "finalize"], required=True)
    parser.add_argument("--learning-rates", default="1e-5,2e-5,3e-5,5e-5")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--selected-model-tag", default="")
    parser.add_argument("--max512-model-tag", default="roberta_base_phase2_maxlen512")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def parse_lrs(raw: str) -> List[float]:
    values = [float(value.strip()) for value in raw.split(",") if value.strip()]
    if not values or any(value <= 0 for value in values):
        raise ValueError("learning rates must be positive")
    return values


def lr_tag(learning_rate: float) -> str:
    return f"{learning_rate:.0e}".replace("+", "").replace("-", "m")


def split_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_fold1(output_dir: Path, groups: np.ndarray) -> Dict[str, np.ndarray]:
    outer_path = output_dir / "splits_fold1.npz"
    outer = np.load(outer_path)
    outer_train = outer["train_idx"].astype(np.int64)
    outer_test = outer["test_idx"].astype(np.int64)
    inner_path = output_dir / "phase1" / "inner_splits" / "roberta_base_fold1.npz"
    if inner_path.exists():
        inner = np.load(inner_path)
        fit_idx = inner["fit_idx"].astype(np.int64)
        valid_idx = inner["valid_idx"].astype(np.int64)
    else:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=43)
        fit_pos, valid_pos = next(
            splitter.split(outer_train, groups=groups[outer_train])
        )
        fit_idx = outer_train[fit_pos]
        valid_idx = outer_train[valid_pos]
        inner_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(inner_path, fit_idx=fit_idx, valid_idx=valid_idx)
    if set(groups[fit_idx]).intersection(set(groups[valid_idx])):
        raise ValueError("fold-1 inner split leaks offering_id")
    return {
        "fit_idx": fit_idx,
        "valid_idx": valid_idx,
        "test_idx": outer_test,
        "outer_sha256": np.asarray([split_sha(outer_path)]),
        "inner_path": np.asarray([str(inner_path)]),
    }


def metadata_path(phase2_dir: Path, learning_rate: float) -> Path:
    return phase2_dir / "tuning" / "roberta_lr" / f"lr_{lr_tag(learning_rate)}.json"


def checkpoint_path(phase2_dir: Path, learning_rate: float) -> Path:
    return phase2_dir / "tuning" / "roberta_lr" / f"lr_{lr_tag(learning_rate)}_checkpoint"


def run_sweep(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Experiment T")
    output_dir = ROOT / "outputs"
    phase2_dir = output_dir / "phase2"
    frame = pd.read_parquet(
        output_dir / "dataset_clean.parquet",
        columns=["review_text", "rating_overall", "offering_id"],
    )
    texts = frame["review_text"].fillna("").astype(str).to_numpy()
    labels = frame["rating_overall"].to_numpy(dtype=np.float32)
    groups = frame["offering_id"].to_numpy()
    row_ids = np.arange(len(frame), dtype=np.int64)
    split = load_fold1(output_dir, groups)
    fit_idx = split["fit_idx"]
    valid_idx = split["valid_idx"]
    device = torch.device(args.device)
    seed_everything(SEED + 1)

    for learning_rate in parse_lrs(args.learning_rates):
        meta_path = metadata_path(phase2_dir, learning_rate)
        checkpoint = checkpoint_path(phase2_dir, learning_rate)
        if args.resume and meta_path.exists() and (checkpoint / "model.safetensors").exists():
            log(f"T1 lr={learning_rate:g}: cache hit")
            continue
        log(
            f"T1 lr={learning_rate:g}: fit={len(fit_idx)}, "
            f"valid={len(valid_idx)}, device={device}"
        )
        training = train_fold(
            model_name=MODEL_NAME,
            checkpoint_dir=checkpoint,
            train_texts=texts[fit_idx],
            train_labels=labels[fit_idx],
            train_row_ids=row_ids[fit_idx],
            valid_texts=texts[valid_idx],
            valid_labels=labels[valid_idx],
            valid_row_ids=row_ids[valid_idx],
            device=device,
            max_length=256,
            train_batch_size=16,
            eval_batch_size=32,
            gradient_accumulation_steps=4,
            learning_rate=learning_rate,
            weight_decay=0.01,
            warmup_ratio=0.06,
            epochs=2,
            seed=SEED + 1,
            num_workers=args.num_workers,
            gradient_checkpointing=False,
        )
        best_history = next(
            row for row in training["history"] if row["epoch"] == training["best_epoch"]
        )
        payload = {
            "learning_rate": learning_rate,
            "best_epoch": training["best_epoch"],
            "best_valid_mae_raw": training["best_valid_mae_raw"],
            "best_valid_metrics": best_history["valid"],
            "history": training["history"],
            "fit_seconds": training["fit_seconds"],
            "gpu_peak_memory_mb": training["gpu_peak_memory_mb"],
            "fit_n": int(len(fit_idx)),
            "valid_n": int(len(valid_idx)),
            "inner_split_path": str(split["inner_path"][0]),
            "outer_split_sha256": str(split["outer_sha256"][0]),
            "checkpoint_dir": str(checkpoint),
            "seed": SEED + 1,
            "training": {
                "max_length": 256,
                "effective_batch_size": 64,
                "weight_decay": 0.01,
                "warmup_ratio": 0.06,
                "epochs": 2,
                "bf16": True,
            },
        }
        write_json(meta_path, payload)
        log(
            f"T1 lr={learning_rate:g}: valid MAE(raw)="
            f"{payload['best_valid_mae_raw']:.6f}, best_epoch={training['best_epoch']}, "
            f"fit={training['fit_seconds']:.1f}s"
        )


def select_lr(output_dir: Path) -> Dict[str, Any]:
    phase2_dir = output_dir / "phase2"
    rows = []
    for learning_rate in DEFAULT_LRS:
        path = metadata_path(phase2_dir, learning_rate)
        if not path.exists():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append(payload)
    selected = min(rows, key=lambda row: (row["best_valid_mae_raw"], row["learning_rate"]))
    result = {
        "selection_metric": "inner-validation raw MAE",
        "selected_learning_rate": float(selected["learning_rate"]),
        "published_learning_rate": 2e-5,
        "requires_five_fold_rerun": not math.isclose(
            float(selected["learning_rate"]), 2e-5, rel_tol=0.0, abs_tol=1e-12
        ),
        "rows": rows,
    }
    write_json(phase2_dir / "experiment_t1_selection.json", result)
    pd.DataFrame(
        [
            {
                "learning_rate": row["learning_rate"],
                "best_epoch": row["best_epoch"],
                "valid_mae_raw": row["best_valid_metrics"]["raw"]["mae"],
                "valid_rmse_raw": row["best_valid_metrics"]["raw"]["rmse"],
                "valid_mae_clipped": row["best_valid_metrics"]["clipped"]["mae"],
                "valid_rmse_clipped": row["best_valid_metrics"]["clipped"]["rmse"],
                "fit_seconds": row["fit_seconds"],
                "gpu_peak_memory_mb": row["gpu_peak_memory_mb"],
                "selected": math.isclose(
                    float(row["learning_rate"]),
                    float(selected["learning_rate"]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ),
            }
            for row in rows
        ]
    ).to_csv(phase2_dir / "experiment_t1_lr_sweep.csv", index=False)
    log(json.dumps(result, indent=2))
    return result


def run_finalize(args: argparse.Namespace) -> None:
    output_dir = ROOT / "outputs"
    phase2_dir = output_dir / "phase2"
    selection = select_lr(output_dir)
    baseline_id = (
        f"{args.selected_model_tag}_regression"
        if args.selected_model_tag
        else "roberta_base_regression"
    )
    baseline_meta = json.loads(
        (output_dir / "predictions" / baseline_id / "fold1.meta.json").read_text()
    )
    max512_id = f"{args.max512_model_tag}_regression"
    max512_path = output_dir / "predictions" / max512_id / "fold1.meta.json"
    if not max512_path.exists():
        raise FileNotFoundError(max512_path)
    max512_meta = json.loads(max512_path.read_text(encoding="utf-8"))
    token256 = json.loads(
        (output_dir / "phase1" / "token_stats" / "roberta_base.json").read_text()
    )
    token512_path = (
        output_dir / "phase1" / "token_stats" / f"{args.max512_model_tag}.json"
    )
    token512 = json.loads(token512_path.read_text(encoding="utf-8"))
    t2 = {
        "fold": 1,
        "learning_rate": float(selection["selected_learning_rate"]),
        "max256": {
            "mae": baseline_meta["clipped"]["mae"],
            "rmse": baseline_meta["clipped"]["rmse"],
            "truncated_n": token256["truncated_n"],
            "truncated_fraction": token256["truncated_fraction"],
        },
        "max512": {
            "mae": max512_meta["clipped"]["mae"],
            "rmse": max512_meta["clipped"]["rmse"],
            "truncated_n": token512["truncated_n"],
            "truncated_fraction": token512["truncated_fraction"],
        },
        "max512_minus_max256": {
            "mae": max512_meta["clipped"]["mae"] - baseline_meta["clipped"]["mae"],
            "rmse": max512_meta["clipped"]["rmse"] - baseline_meta["clipped"]["rmse"],
            "truncated_fraction": token512["truncated_fraction"] - token256["truncated_fraction"],
        },
        "note": (
            f"The max_length=256 reference is {baseline_id}; both length settings "
            "use the T1-selected learning rate."
        ),
    }
    selected_metric = None
    if args.selected_model_tag:
        selected_path = output_dir / "metrics" / f"{args.selected_model_tag}_regression.json"
        if selected_path.exists():
            selected_metric = json.loads(selected_path.read_text(encoding="utf-8"))
    payload = {
        "t1": selection,
        "selected_five_fold_model": selected_metric,
        "t2": t2,
    }
    write_json(phase2_dir / "experiment_t_results.json", payload)
    t1 = pd.read_csv(phase2_dir / "experiment_t1_lr_sweep.csv")
    pd.DataFrame(
        [
            {"setting": "max_length_256", **t2["max256"]},
            {"setting": "max_length_512", **t2["max512"]},
        ]
    ).to_csv(phase2_dir / "experiment_t2_max_length.csv", index=False)
    lines = [
        "# Experiment T — Transformer Tuning Defensibility",
        "",
        "## T1: fold-1 grouped inner-validation learning-rate sweep",
        "",
        t1.to_markdown(index=False, floatfmt=".6f"),
        "",
        f"- Selected learning rate: {selection['selected_learning_rate']:.1e}",
        f"- Five-fold rerun required: {selection['requires_five_fold_rerun']}",
        "",
        "## T2: max_length 256 versus 512 on outer fold 1",
        "",
        "```json",
        json.dumps(t2, indent=2),
        "```",
        "",
    ]
    (phase2_dir / "experiment_t_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.stage == "sweep":
        run_sweep(args)
    elif args.stage == "select":
        select_lr(ROOT / "outputs")
    else:
        run_finalize(args)


if __name__ == "__main__":
    main()
