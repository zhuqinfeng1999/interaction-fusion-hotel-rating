#!/usr/bin/env python3
"""Phase 3 W1/W4: prepare and hotel-stratify the Booking 515k corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


ROOT = Path(__file__).resolve().parents[1]
SEED = 42
N_SPLITS = 5
SOURCE_URL = (
    "https://huggingface.co/datasets/Dricz/515k-Hotel-Reviews-In-Europe/"
    "resolve/main/Hotel_Reviews.csv?download=true"
)
SOURCE_COMMIT = "cdaea79962c5d667dd5177956ef0353001f6ade3"
EXPECTED_SHA256 = "a4810c2757934f0a826a1b16a437eb67a38be45b1a22ad56772afce0b6c11af9"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data_external/booking_515k/Hotel_Reviews.csv"),
    )
    parser.add_argument("--sample-size", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--restarts", type=int, default=2_000)
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_text(series: pd.Series, sentinel: str) -> pd.Series:
    values = series.fillna("").astype(str).str.strip()
    return values.mask(values.str.casefold().eq(sentinel.casefold()), "")


def property_ids(frame: pd.DataFrame) -> pd.Series:
    # The source has 1,492 unique names, 1,493 unique addresses, and 1,494
    # name-address pairs: three unrelated properties share the generic name
    # "Hotel Regina", while two co-located London brands share one address.
    # Address therefore reproduces the documented 1,493-property corpus and
    # is the least ambiguous stable grouping key available in the release.
    keys = (
        frame["Hotel_Address"].fillna("").astype(str).str.strip().str.casefold()
    )
    mapping = {
        key: hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
        for key in keys.unique()
    }
    return keys.map(mapping)


def rating_stratum(native: pd.Series) -> pd.Series:
    equivalent = 1.0 + 4.0 * (native - 1.0) / 9.0
    return equivalent.round().clip(1, 5).fillna(0).astype(np.int8)


def make_splits(frame: pd.DataFrame, phase3: Path) -> np.ndarray:
    assignment = np.full(len(frame), -1, dtype=np.int8)
    splitter = GroupKFold(n_splits=N_SPLITS)
    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(frame, frame["rating_overall_1to5"], frame["offering_id"]),
        start=1,
    ):
        train_hotels = set(frame.iloc[train_idx]["offering_id"])
        test_hotels = set(frame.iloc[test_idx]["offering_id"])
        if train_hotels.intersection(test_hotels):
            raise ValueError(f"Booking fold {fold} hotel leakage")
        assignment[test_idx] = fold
        np.savez_compressed(
            phase3 / f"booking_splits_fold{fold}.npz",
            train_idx=train_idx.astype(np.int64),
            test_idx=test_idx.astype(np.int64),
        )
    if np.any(assignment < 1):
        raise ValueError("Booking folds do not cover all rows")
    return assignment


def choose_hotels(
    frame: pd.DataFrame, target_rows: int, seed: int, restarts: int
) -> Tuple[np.ndarray, Dict[str, Any]]:
    table = (
        frame.groupby(["offering_id", "rating_stratum"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=[1, 2, 3, 4, 5], fill_value=0)
    )
    hotels = table.index.to_numpy()
    counts = table.to_numpy(dtype=np.int64)
    totals = counts.sum(axis=1)
    full = counts.sum(axis=0).astype(float)
    full /= full.sum()
    target = full * target_rows
    rng = np.random.default_rng(seed)
    best_value = float("inf")
    best: np.ndarray | None = None
    details: Dict[str, Any] = {}
    for _ in range(restarts):
        order = rng.permutation(len(hotels))
        crossing = int(np.searchsorted(np.cumsum(totals[order]), target_rows))
        for size in {max(1, min(len(order), crossing + delta)) for delta in (0, 1, 2)}:
            selected = order[:size]
            cells = counts[selected].sum(axis=0).astype(float)
            actual = int(cells.sum())
            distribution = cells / max(1.0, cells.sum())
            size_error = abs(actual - target_rows) / target_rows
            distribution_l1 = float(np.abs(distribution - full).sum())
            cell_error = float(
                np.mean(np.abs(cells - target) / np.maximum(target, 1.0))
            )
            objective = size_error + 2.0 * distribution_l1 + 0.25 * cell_error
            if objective < best_value:
                best_value = objective
                best = selected.copy()
                details = {
                    "target_rows": target_rows,
                    "actual_rows": actual,
                    "size_error_fraction": size_error,
                    "rating_distribution_l1": distribution_l1,
                    "mean_relative_rating_cell_error": cell_error,
                    "objective": objective,
                    "rating_counts": cells.astype(int).tolist(),
                    "rating_distribution": distribution.tolist(),
                    "full_rating_distribution": full.tolist(),
                }
    if best is None:
        raise RuntimeError("Booking hotel selection failed")
    selected_hotels = hotels[best]
    details["hotels"] = len(selected_hotels)
    return selected_hotels, details


def write_reports(
    raw_path: Path,
    frame: pd.DataFrame,
    sample: pd.DataFrame,
    details: pd.DataFrame,
    checksum: str,
    phase3: Path,
    sample_path: Path,
    seed: int,
) -> None:
    fold_table = (
        frame.groupby("outer_fold")
        .agg(reviews=("row_id", "size"), hotels=("offering_id", "nunique"))
        .reset_index()
    )
    sample_folds = (
        sample.groupby("outer_fold")
        .agg(reviews=("row_id", "size"), hotels=("offering_id", "nunique"))
        .reset_index()
    )
    payload = {
        "experiment": "W1_W4",
        "download_date_utc": datetime.fromtimestamp(
            raw_path.stat().st_mtime, tz=timezone.utc
        ).isoformat(),
        "source_url": SOURCE_URL,
        "source_commit": SOURCE_COMMIT,
        "raw_path": str(raw_path.relative_to(ROOT)),
        "raw_bytes": raw_path.stat().st_size,
        "sha256": checksum,
        "mirror_license_label": "Apache-2.0",
        "kaggle_derivative_license_label": "CC0 Public Domain",
        "license_caveat": (
            "Mirror and Kaggle derivative labels differ, and the observations were "
            "originally scraped from Booking.com. The mirror label is recorded but "
            "not interpreted as independent confirmation of underlying data rights."
        ),
        "clean_rows": len(frame),
        "hotels": int(frame["offering_id"].nunique()),
        "date_min": str(frame["review_date"].min().date()),
        "date_max": str(frame["review_date"].max().date()),
        "native_score_min": float(frame["rating_overall_native"].min()),
        "native_score_max": float(frame["rating_overall_native"].max()),
        "mapping_to_1to5": "1 + 4 * (Reviewer_Score - 1) / 9",
        "split": "GroupKFold(5), group=stable hash of normalized hotel address",
        "sample_seed": seed,
        "sample_rows": len(sample),
        "sample_hotels": int(sample["offering_id"].nunique()),
        "sample_path": str(sample_path.relative_to(ROOT)),
        "folds_full": fold_table.to_dict(orient="records"),
        "folds_sample": sample_folds.to_dict(orient="records"),
        "selection_details": details.to_dict(orient="records"),
    }
    (phase3 / "w_source_and_sample.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    details.to_csv(phase3 / "w_sample_selection.csv", index=False)
    lines = [
        "# Phase 3 W1/W4 — Booking External Corpus Preparation",
        "",
        f"- source: [{SOURCE_URL}]({SOURCE_URL})",
        f"- pinned mirror commit: `{SOURCE_COMMIT}`",
        f"- downloaded UTC: {payload['download_date_utc']}",
        f"- raw bytes: {payload['raw_bytes']:,}",
        f"- SHA-256: `{checksum}`",
        f"- cleaned rows/hotels: {len(frame):,} / {frame['offering_id'].nunique():,}",
        f"- review span: {payload['date_min']} to {payload['date_max']}",
        f"- native score range: {payload['native_score_min']} to {payload['native_score_max']}",
        "- 1–5 mapping: `1 + 4 * (Reviewer_Score - 1) / 9` (native 1 and 10 map exactly to 1 and 5)",
        "- split: GroupKFold(5) by stable hotel-address hash (1,493 properties)",
        "",
        "## Licence/provenance caveat",
        "",
        "The Hugging Face mirror labels the release Apache-2.0, while a Kaggle derivative labels it CC0. The observations originated from Booking.com scraping. These labels are recorded verbatim and are not treated as independent confirmation that the uploader could relicense all underlying observations.",
        "",
        "## Full-corpus folds",
        "",
        fold_table.to_markdown(index=False),
        "",
        "## Hotel-intact 200k sample",
        "",
        sample_folds.to_markdown(index=False),
        "",
        details.to_markdown(index=False, floatfmt=".6f"),
        "",
    ]
    (phase3 / "experiment_w1_w4_preparation.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    raw_path = resolve(args.input)
    if not raw_path.exists():
        raise FileNotFoundError(raw_path)
    checksum = sha256_file(raw_path)
    if checksum != EXPECTED_SHA256:
        raise ValueError(f"external source checksum mismatch: {checksum}")
    phase3 = ROOT / "outputs" / "phase3"
    phase3.mkdir(parents=True, exist_ok=True)
    clean_path = phase3 / "booking_dataset_clean.parquet"
    sample_path = phase3 / f"w_sample_n{args.sample_size}_seed{args.seed}.npz"
    if args.resume and clean_path.exists() and sample_path.exists():
        print(f"W preparation cache hit: {clean_path}", flush=True)
        return

    usecols = [
        "Hotel_Address",
        "Review_Date",
        "Average_Score",
        "Hotel_Name",
        "Reviewer_Nationality",
        "Negative_Review",
        "Positive_Review",
        "Total_Number_of_Reviews_Reviewer_Has_Given",
        "Reviewer_Score",
        "Tags",
        "lat",
        "lng",
    ]
    raw = pd.read_csv(raw_path, usecols=usecols, low_memory=False)
    print(f"W raw rows: {len(raw):,}", flush=True)
    negative = normalize_text(raw["Negative_Review"], "No Negative")
    positive = normalize_text(raw["Positive_Review"], "No Positive")
    text = (
        np.where(negative.ne(""), "Negative: " + negative, "")
        + np.where(negative.ne("") & positive.ne(""), " ", "")
        + np.where(positive.ne(""), "Positive: " + positive, "")
    )
    native = pd.to_numeric(raw["Reviewer_Score"], errors="coerce")
    frame = pd.DataFrame(
        {
            "offering_id": property_ids(raw),
            "hotel_name": raw["Hotel_Name"].fillna("").astype(str).str.strip(),
            "hotel_address": raw["Hotel_Address"].fillna("").astype(str).str.strip(),
            "review_date": pd.to_datetime(raw["Review_Date"], errors="coerce"),
            "review_text": pd.Series(text).astype(str).str.strip(),
            "rating_overall_native": native,
            "rating_overall_1to5": 1.0 + 4.0 * (native - 1.0) / 9.0,
            "rating_stratum": rating_stratum(native),
            "average_hotel_score_native": pd.to_numeric(
                raw["Average_Score"], errors="coerce"
            ),
            "reviewer_nationality": raw["Reviewer_Nationality"]
            .fillna("")
            .astype(str)
            .str.strip(),
            "reviewer_num_reviews": pd.to_numeric(
                raw["Total_Number_of_Reviews_Reviewer_Has_Given"], errors="coerce"
            ),
            "tags": raw["Tags"].fillna("").astype(str),
            "lat": pd.to_numeric(raw["lat"], errors="coerce"),
            "lng": pd.to_numeric(raw["lng"], errors="coerce"),
        }
    )
    valid = (
        frame["review_text"].str.len().gt(0)
        & frame["rating_overall_native"].between(1, 10, inclusive="both")
        & frame["offering_id"].ne("")
    )
    frame = frame.loc[valid].copy()
    before = len(frame)
    frame = frame.drop_duplicates(
        ["offering_id", "review_date", "review_text", "rating_overall_native"],
        keep="first",
    ).reset_index(drop=True)
    frame.insert(0, "row_id", np.arange(len(frame), dtype=np.int64))
    print(f"W valid/deduplicated rows: {before:,} -> {len(frame):,}", flush=True)
    frame["outer_fold"] = make_splits(frame, phase3)
    frame.to_parquet(clean_path, index=False)

    fold_counts = frame["outer_fold"].value_counts().sort_index()
    proportional = args.sample_size * fold_counts / len(frame)
    targets = np.floor(proportional).astype(int)
    remainder = args.sample_size - int(targets.sum())
    for fold in (proportional - targets).sort_values(ascending=False).index[:remainder]:
        targets.loc[fold] += 1
    selected_sets: List[np.ndarray] = []
    detail_rows: List[Dict[str, Any]] = []
    for fold in range(1, N_SPLITS + 1):
        selected, detail = choose_hotels(
            frame[frame["outer_fold"].eq(fold)],
            int(targets.loc[fold]),
            args.seed + fold,
            args.restarts,
        )
        selected_sets.append(selected)
        detail_rows.append({"outer_fold": fold, **detail})
        print(
            f"W sample fold {fold}: target={int(targets.loc[fold]):,} "
            f"actual={detail['actual_rows']:,} hotels={detail['hotels']}",
            flush=True,
        )
    selected_hotels = np.concatenate(selected_sets)
    sample = frame[frame["offering_id"].isin(selected_hotels)].sort_values("row_id")
    np.savez_compressed(
        sample_path,
        row_id=sample["row_id"].to_numpy(dtype=np.int64),
        offering_id=np.asarray(sample["offering_id"].tolist(), dtype="U16"),
        outer_fold=sample["outer_fold"].to_numpy(dtype=np.int8),
        rating_overall_native=sample["rating_overall_native"].to_numpy(dtype=float),
        rating_overall_1to5=sample["rating_overall_1to5"].to_numpy(dtype=float),
        selected_hotels=np.asarray(selected_hotels.tolist(), dtype="U16"),
        requested_sample_size=np.asarray([args.sample_size]),
        actual_sample_size=np.asarray([len(sample)]),
        seed=np.asarray([args.seed]),
    )
    details = pd.DataFrame(detail_rows)
    write_reports(
        raw_path,
        frame,
        sample,
        details,
        checksum,
        phase3,
        sample_path,
        args.seed,
    )
    print(f"saved: {clean_path}", flush=True)
    print(f"saved: {sample_path}", flush=True)


if __name__ == "__main__":
    main()
