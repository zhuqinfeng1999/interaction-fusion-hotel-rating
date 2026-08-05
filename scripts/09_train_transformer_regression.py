#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import psutil
import torch
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation_harness import EvaluationHarness, write_json  # noqa: E402
from src.transformer_regression import (  # noqa: E402
    build_loader,
    compute_token_length_stats,
    load_checkpoint,
    predict,
    seed_everything,
    train_fold,
)


SEED = 42
N_SPLITS = 5


def log(message: str) -> None:
    print(message, flush=True)


def load_splits(output_dir: Path) -> List[Dict[str, Any]]:
    splits: List[Dict[str, Any]] = []
    for fold in range(1, N_SPLITS + 1):
        path = output_dir / f"splits_fold{fold}.npz"
        arrays = np.load(path)
        splits.append(
            {
                "fold": fold,
                "train_idx": arrays["train_idx"].astype(np.int64),
                "test_idx": arrays["test_idx"].astype(np.int64),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    return splits


def load_or_create_inner_split(
    path: Path,
    outer_train_idx: np.ndarray,
    groups: np.ndarray,
    fold: int,
) -> Dict[str, np.ndarray]:
    if path.exists():
        arrays = np.load(path)
        fit_idx = arrays["fit_idx"].astype(np.int64)
        valid_idx = arrays["valid_idx"].astype(np.int64)
    else:
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=0.10,
            random_state=SEED + fold,
        )
        fit_pos, valid_pos = next(
            splitter.split(
                outer_train_idx,
                groups=groups[outer_train_idx],
            )
        )
        fit_idx = outer_train_idx[fit_pos]
        valid_idx = outer_train_idx[valid_pos]
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, fit_idx=fit_idx, valid_idx=valid_idx)
    overlap = set(groups[fit_idx]).intersection(set(groups[valid_idx]))
    if overlap:
        raise ValueError(f"inner group leakage in fold {fold}")
    return {"fit_idx": fit_idx, "valid_idx": valid_idx}


def parse_folds(raw: str) -> List[int]:
    folds = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not folds or any(fold < 1 or fold > N_SPLITS for fold in folds):
        raise ValueError("--folds must contain values from 1 to 5")
    return folds


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune a transformer regression baseline on saved hotel splits."
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-tag", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--folds", default="1,2,3,4,5")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-token-audit", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Phase 1 transformer training")
    seed_everything(SEED)
    device = torch.device(args.device)
    output_dir = ROOT / "outputs"
    phase1_dir = output_dir / "phase1"
    harness = EvaluationHarness(output_dir)
    model_id = f"{args.model_tag}_regression"

    df = pd.read_parquet(
        output_dir / "dataset_clean.parquet",
        columns=["review_text", "rating_overall", "offering_id"],
    )
    df["row_id"] = np.arange(len(df), dtype=np.int64)
    texts = df["review_text"].fillna("").astype(str).to_numpy()
    labels = df["rating_overall"].to_numpy(dtype=np.float32)
    groups = df["offering_id"].to_numpy()
    row_ids = df["row_id"].to_numpy(dtype=np.int64)
    splits = load_splits(output_dir)
    selected_folds = parse_folds(args.folds)

    token_stats = None
    if not args.skip_token_audit:
        token_stats = compute_token_length_stats(
            texts=texts,
            model_name=args.model_name,
            max_length=args.max_length,
            output_json=phase1_dir
            / "token_stats"
            / f"{args.model_tag}.json",
            output_md=phase1_dir
            / "token_stats"
            / f"{args.model_tag}.md",
        )
        log(f"token audit: {token_stats}")

    run_records: List[Dict[str, Any]] = []
    for split in splits:
        fold = split["fold"]
        if fold not in selected_folds:
            continue
        checkpoint_dir = (
            phase1_dir / "checkpoints" / args.model_tag / f"fold{fold}"
        )
        if args.resume and harness.has_fold(model_id, fold):
            log(f"{model_id} fold {fold}: harness artifact cache hit")
            run_records.append(harness.load_fold(model_id, fold).record)
            continue

        inner_path = (
            phase1_dir
            / "inner_splits"
            / f"{args.model_tag}_fold{fold}.npz"
        )
        inner = load_or_create_inner_split(
            inner_path,
            split["train_idx"],
            groups,
            fold,
        )
        fit_idx = inner["fit_idx"]
        valid_idx = inner["valid_idx"]
        test_idx = split["test_idx"]
        log(
            f"{model_id} fold {fold}: fit={len(fit_idx)} "
            f"valid={len(valid_idx)} test={len(test_idx)}"
        )

        process = psutil.Process()
        rss_before = process.memory_info().rss / (1024 * 1024)
        training = train_fold(
            model_name=args.model_name,
            checkpoint_dir=checkpoint_dir,
            train_texts=texts[fit_idx],
            train_labels=labels[fit_idx],
            train_row_ids=row_ids[fit_idx],
            valid_texts=texts[valid_idx],
            valid_labels=labels[valid_idx],
            valid_row_ids=row_ids[valid_idx],
            device=device,
            max_length=args.max_length,
            train_batch_size=args.train_batch_size,
            eval_batch_size=args.eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            epochs=args.epochs,
            seed=SEED + fold,
            num_workers=args.num_workers,
            gradient_checkpointing=args.gradient_checkpointing,
        )

        tokenizer, model = load_checkpoint(checkpoint_dir, device)
        test_loader = build_loader(
            texts[test_idx],
            labels[test_idx],
            row_ids[test_idx],
            tokenizer,
            args.max_length,
            args.eval_batch_size,
            False,
            SEED + fold,
            args.num_workers,
        )
        predict_start = time.perf_counter()
        predictions, y_true, predicted_row_ids = predict(
            model,
            test_loader,
            device,
            f"{args.model_tag} fold {fold} test",
        )
        predict_seconds = time.perf_counter() - predict_start
        order = np.argsort(predicted_row_ids)
        predictions = predictions[order]
        y_true = y_true[order]
        predicted_row_ids = predicted_row_ids[order]
        expected_order = np.argsort(row_ids[test_idx])
        expected_row_ids = row_ids[test_idx][expected_order]
        if not np.array_equal(predicted_row_ids, expected_row_ids):
            raise ValueError(f"test row_id mismatch in fold {fold}")
        offering_ids = groups[test_idx][expected_order]

        rss_after = process.memory_info().rss / (1024 * 1024)
        artifact = harness.write_fold(
            model_id=model_id,
            fold=fold,
            row_id=predicted_row_ids,
            offering_id=offering_ids,
            y_true=y_true,
            y_pred_raw=predictions,
            preprocessing_seconds=0.0,
            tuning_seconds=0.0,
            fit_seconds=training["fit_seconds"],
            predict_seconds=predict_seconds,
            peak_memory_mb=max(rss_before, rss_after),
            selected_hyperparameters={
                "model_name": args.model_name,
                "learning_rate": args.learning_rate,
                "best_epoch": training["best_epoch"],
                "max_length": args.max_length,
                "effective_batch_size": args.train_batch_size
                * args.gradient_accumulation,
                "weight_decay": args.weight_decay,
                "warmup_ratio": args.warmup_ratio,
                "epochs": args.epochs,
                "inner_validation": "GroupShuffleSplit(test_size=0.10)",
            },
            extra={
                "checkpoint_dir": str(checkpoint_dir),
                "inner_split_path": str(inner_path),
                "history": training["history"],
                "gpu_peak_memory_mb": training["gpu_peak_memory_mb"],
                "gpu_hours": (
                    training["fit_seconds"] + predict_seconds
                )
                / 3600,
                "outer_split_sha256": split["sha256"],
            },
        )
        run_records.append(artifact.record)
        fold_summary = artifact.record["clipped"]
        log(
            f"{model_id} fold {fold}: MAE={fold_summary['mae']:.4f} "
            f"RMSE={fold_summary['rmse']:.4f} "
            f"best_epoch={training['best_epoch']}"
        )
        del model, tokenizer, test_loader
        torch.cuda.empty_cache()

    if all(harness.has_fold(model_id, fold) for fold in range(1, N_SPLITS + 1)):
        payload = harness.finalize_model(
            model_id,
            {
                "description": f"{args.model_name} regression baseline",
                "seed": SEED,
                "outer_split": "saved GroupKFold(5), group=offering_id",
                "training": {
                    "max_length": args.max_length,
                    "dynamic_padding": True,
                    "train_batch_size": args.train_batch_size,
                    "gradient_accumulation": args.gradient_accumulation,
                    "effective_batch_size": args.train_batch_size
                    * args.gradient_accumulation,
                    "learning_rate": args.learning_rate,
                    "weight_decay": args.weight_decay,
                    "warmup_ratio": args.warmup_ratio,
                    "epochs": args.epochs,
                    "bf16": True,
                },
                "token_stats": token_stats,
                "deployable": True,
            },
        )
        payload["gpu_hours_total"] = float(
            sum(
                fold.get("extra", {}).get("gpu_hours", 0.0)
                for fold in payload["folds"]
            )
        )
        write_json(output_dir / "metrics" / f"{model_id}.json", payload)
        log(f"finalized: outputs/metrics/{model_id}.json")
    else:
        log("selected folds completed; finalization waits for all five folds.")


if __name__ == "__main__":
    main()
