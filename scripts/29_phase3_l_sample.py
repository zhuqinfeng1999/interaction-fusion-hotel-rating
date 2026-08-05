#!/usr/bin/env python3
"""Phase 3 L1: deterministic hotel-intact, rating-stratified 200k sample."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SEED = 42
N_SPLITS = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--restarts", type=int, default=2_000)
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def split_hash(output_dir: Path) -> str:
    digest = hashlib.sha256()
    for fold in range(1, N_SPLITS + 1):
        digest.update((output_dir / f"splits_fold{fold}.npz").read_bytes())
    return digest.hexdigest()


def fold_assignment(output_dir: Path, n_rows: int) -> np.ndarray:
    assignment = np.full(n_rows, -1, dtype=np.int8)
    for fold in range(1, N_SPLITS + 1):
        arrays = np.load(output_dir / f"splits_fold{fold}.npz")
        test_idx = arrays["test_idx"].astype(np.int64)
        if np.any(assignment[test_idx] != -1):
            raise ValueError(f"overlap in saved fold {fold}")
        assignment[test_idx] = fold
    if np.any(assignment < 1):
        raise ValueError("saved test folds do not cover all rows")
    return assignment


def hotel_rating_matrix(frame: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    table = (
        frame.groupby(["offering_id", "rating_overall"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=[1, 2, 3, 4, 5], fill_value=0)
    )
    hotels = table.index.to_numpy()
    rating_counts = table.to_numpy(dtype=np.int64)
    totals = rating_counts.sum(axis=1)
    return hotels, rating_counts, totals


def choose_hotels(
    frame: pd.DataFrame,
    target_rows: int,
    seed: int,
    restarts: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    hotels, rating_counts, totals = hotel_rating_matrix(frame)
    full_distribution = rating_counts.sum(axis=0).astype(float)
    full_distribution /= full_distribution.sum()
    target_rating_counts = full_distribution * target_rows
    rng = np.random.default_rng(seed)
    best_objective = float("inf")
    best_indices: np.ndarray | None = None
    best_details: Dict[str, Any] = {}
    for _ in range(restarts):
        order = rng.permutation(len(hotels))
        cumulative = np.cumsum(totals[order])
        crossing = int(np.searchsorted(cumulative, target_rows, side="left"))
        candidate_sizes = {
            max(1, min(len(order), crossing)),
            max(1, min(len(order), crossing + 1)),
            max(1, min(len(order), crossing + 2)),
        }
        for size in candidate_sizes:
            selected = order[:size]
            counts = rating_counts[selected].sum(axis=0).astype(float)
            actual_rows = int(counts.sum())
            distribution = counts / max(1.0, counts.sum())
            size_error = abs(actual_rows - target_rows) / target_rows
            distribution_l1 = float(np.abs(distribution - full_distribution).sum())
            cell_error = float(
                np.mean(
                    np.abs(counts - target_rating_counts)
                    / np.maximum(target_rating_counts, 1.0)
                )
            )
            objective = size_error + 2.0 * distribution_l1 + 0.25 * cell_error
            if objective < best_objective:
                best_objective = objective
                best_indices = selected.copy()
                best_details = {
                    "target_rows": target_rows,
                    "actual_rows": actual_rows,
                    "size_error_fraction": size_error,
                    "rating_distribution_l1": distribution_l1,
                    "mean_relative_rating_cell_error": cell_error,
                    "objective": objective,
                    "rating_counts": counts.astype(int).tolist(),
                    "rating_distribution": distribution.tolist(),
                    "full_rating_distribution": full_distribution.tolist(),
                }
    if best_indices is None:
        raise RuntimeError("hotel selection failed")
    selected_hotels = hotels[best_indices]
    best_details["hotels"] = int(len(selected_hotels))
    return selected_hotels, best_details


def main() -> None:
    args = parse_args()
    output_dir = ROOT / "outputs"
    phase3 = output_dir / "phase3"
    phase3.mkdir(parents=True, exist_ok=True)
    sample_path = phase3 / f"l1_sample_n{args.sample_size}_seed{args.seed}.npz"
    summary_csv = phase3 / "l1_sample_summary.csv"
    summary_json = phase3 / "l1_sample_summary.json"
    summary_md = phase3 / "l1_sample_summary.md"
    if args.resume and all(
        path.exists() for path in (sample_path, summary_csv, summary_json, summary_md)
    ):
        print(f"L1 cache hit: {sample_path}")
        return

    frame = pd.read_parquet(
        output_dir / "dataset_clean.parquet",
        columns=["offering_id", "rating_overall"],
    ).reset_index(drop=True)
    if len(frame) != 877_808:
        raise ValueError(f"canonical row count failed: {len(frame)}")
    frame["row_id"] = np.arange(len(frame), dtype=np.int64)
    frame["outer_fold"] = fold_assignment(output_dir, len(frame))
    # GroupKFold integrity means every hotel belongs to exactly one test fold.
    hotel_fold_counts = frame.groupby("offering_id")["outer_fold"].nunique()
    if int(hotel_fold_counts.max()) != 1:
        raise ValueError("a hotel spans multiple canonical outer test folds")

    full_fold_counts = frame["outer_fold"].value_counts().sort_index()
    proportional = args.sample_size * full_fold_counts / len(frame)
    targets = np.floor(proportional).astype(int)
    remainder = args.sample_size - int(targets.sum())
    fractional_order = (proportional - targets).sort_values(ascending=False).index
    for fold in fractional_order[:remainder]:
        targets.loc[fold] += 1

    selected_hotel_sets: List[np.ndarray] = []
    details: List[Dict[str, Any]] = []
    for fold in range(1, N_SPLITS + 1):
        fold_frame = frame[frame["outer_fold"] == fold]
        selected_hotels, detail = choose_hotels(
            fold_frame,
            int(targets.loc[fold]),
            args.seed + fold,
            args.restarts,
        )
        selected_hotel_sets.append(selected_hotels)
        details.append({"outer_fold": fold, **detail})
        print(
            f"L1 fold {fold}: target={int(targets.loc[fold])} "
            f"actual={detail['actual_rows']} hotels={detail['hotels']} "
            f"rating_L1={detail['rating_distribution_l1']:.6f}"
        )

    selected_hotels_all = np.concatenate(selected_hotel_sets)
    sample = frame[frame["offering_id"].isin(selected_hotels_all)].copy()
    sample = sample.sort_values("row_id")
    if sample["offering_id"].nunique() != len(selected_hotels_all):
        raise ValueError("selected hotel accounting mismatch")
    if sample["row_id"].duplicated().any():
        raise ValueError("duplicate row_id in L1 sample")

    np.savez_compressed(
        sample_path,
        row_id=sample["row_id"].to_numpy(dtype=np.int64),
        offering_id=sample["offering_id"].to_numpy(),
        outer_fold=sample["outer_fold"].to_numpy(dtype=np.int8),
        rating_overall=sample["rating_overall"].to_numpy(dtype=np.int8),
        selected_hotels=selected_hotels_all,
        requested_sample_size=np.asarray([args.sample_size]),
        actual_sample_size=np.asarray([len(sample)]),
        seed=np.asarray([args.seed]),
    )
    detail_frame = pd.DataFrame(details)
    detail_frame.to_csv(summary_csv, index=False)
    payload = {
        "experiment": "L1",
        "requested_sample_size": args.sample_size,
        "actual_sample_size": int(len(sample)),
        "n_hotels": int(sample["offering_id"].nunique()),
        "seed": args.seed,
        "restarts_per_fold": args.restarts,
        "split_hash": split_hash(output_dir),
        "selection": (
            "whole-hotel randomized-prefix optimization independently within each "
            "canonical outer fold; objective balances row count and rating distribution"
        ),
        "sample_path": str(sample_path.relative_to(ROOT)),
        "folds": details,
    }
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    cross = pd.crosstab(sample["outer_fold"], sample["rating_overall"])
    lines = [
        "# Phase 3 L1 — Hotel-intact 200k Sample",
        "",
        f"- requested reviews: {args.sample_size:,}",
        f"- actual reviews: {len(sample):,}",
        f"- selected hotels: {sample['offering_id'].nunique():,}",
        f"- seed: {args.seed}",
        f"- canonical split hash: `{payload['split_hash']}`",
        "- whole hotels are retained; no selected hotel spans outer folds",
        "",
        "## Fold by rating counts",
        "",
        cross.to_markdown(),
        "",
        "## Selection diagnostics",
        "",
        detail_frame.to_markdown(index=False, floatfmt=".6f"),
        "",
        f"Machine-readable row IDs: `{sample_path.relative_to(ROOT)}`.",
        "",
    ]
    summary_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"saved: {sample_path}")
    print(f"saved: {summary_csv}")
    print(f"saved: {summary_json}")
    print(f"saved: {summary_md}")


if __name__ == "__main__":
    main()
