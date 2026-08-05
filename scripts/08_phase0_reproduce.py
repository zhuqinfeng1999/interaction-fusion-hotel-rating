#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import scipy
import scipy.sparse as sp
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.aspect_extractor_base import ASPECTS  # noqa: E402
from src.evaluation_harness import (  # noqa: E402
    EvaluationHarness,
    PeakRSSMonitor,
    bootstrap_mean_difference,
    cluster_robust_ridge_se,
    paired_wilcoxon,
    write_json,
)


SEED = 42
N_SPLITS = 5
EXPECTED_ROWS = 877_808
EXPECTED_HOTELS = 3_945
TOLERANCE = 0.0005

TEXT_ONLY = "text_only_tfidf_ridge"
LLM_ONLY = "llm_aspects_only"
SIMPLE_FUSION = "text_llm_simple_fusion"
RESIDUAL = "residual_on_llm"
UPPER_BOUND = "upper_bound_true_aspects"
INTERACTION = "interaction_fusion"
MODEL_IDS = [
    TEXT_ONLY,
    LLM_ONLY,
    SIMPLE_FUSION,
    RESIDUAL,
    UPPER_BOUND,
    INTERACTION,
]

PUBLISHED = {
    TEXT_ONLY: {
        "mae_mean": 0.4644,
        "mae_std": 0.0059,
        "rmse_mean": 0.6099,
        "rmse_std": 0.0074,
    },
    LLM_ONLY: {
        "mae_mean": 0.5127,
        "mae_std": 0.0071,
        "rmse_mean": 0.6862,
        "rmse_std": 0.0083,
    },
    SIMPLE_FUSION: {
        "mae_mean": 0.4676,
        "mae_std": 0.0054,
        "rmse_mean": 0.6210,
        "rmse_std": 0.0064,
    },
    RESIDUAL: {
        "mae_mean": 0.4488,
        "mae_std": 0.0058,
        "rmse_mean": 0.5970,
        "rmse_std": 0.0067,
    },
    UPPER_BOUND: {
        "mae_mean": 0.4322,
        "mae_std": 0.0072,
        "rmse_mean": 0.6354,
        "rmse_std": 0.0134,
    },
    INTERACTION: {
        "mae_mean": 0.4385,
        "mae_std": 0.0061,
        "rmse_mean": 0.5797,
        "rmse_std": 0.0069,
    },
}


def log(message: str) -> None:
    print(message, flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_splits(output_dir: Path, n_rows: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    splits: List[Dict[str, Any]] = []
    assignment = np.full(n_rows, -1, dtype=np.int8)
    assignment_seen = np.zeros(n_rows, dtype=np.int8)
    file_hashes: Dict[str, str] = {}
    for fold in range(1, N_SPLITS + 1):
        path = output_dir / f"splits_fold{fold}.npz"
        if not path.exists():
            raise FileNotFoundError(path)
        arrays = np.load(path)
        train_idx = arrays["train_idx"].astype(np.int64)
        test_idx = arrays["test_idx"].astype(np.int64)
        file_hashes[f"fold{fold}"] = sha256_file(path)
        assignment[test_idx] = fold
        assignment_seen[test_idx] += 1
        splits.append(
            {
                "fold": fold,
                "path": path,
                "sha1": hashlib.sha1(path.read_bytes()).hexdigest(),
                "sha256": file_hashes[f"fold{fold}"],
                "train_idx": train_idx,
                "test_idx": test_idx,
            }
        )
    if (assignment < 0).any() or (assignment_seen != 1).any():
        raise ValueError("saved split files do not assign every row exactly once")
    assignment_pairs = np.column_stack(
        [
            np.arange(n_rows, dtype=np.int64),
            assignment.astype(np.int64),
        ]
    )
    split_meta = {
        "fold_file_sha256": file_hashes,
        "assignment_encoding": "int8 fold IDs in ascending row_id order",
        "assignment_sha256": hashlib.sha256(assignment.tobytes()).hexdigest(),
        "assignment_pairs_encoding": "int64 [row_id, fold] pairs in ascending row_id order",
        "assignment_pairs_sha256": hashlib.sha256(
            assignment_pairs.tobytes()
        ).hexdigest(),
    }
    return splits, split_meta


def verify_splits(df: pd.DataFrame, splits: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for split in splits:
        train_idx = split["train_idx"]
        test_idx = split["test_idx"]
        train_hotels = set(df.iloc[train_idx]["offering_id"].tolist())
        test_hotels = set(df.iloc[test_idx]["offering_id"].tolist())
        index_overlap = np.intersect1d(train_idx, test_idx)
        group_overlap = train_hotels.intersection(test_hotels)
        if len(index_overlap) or len(group_overlap):
            raise ValueError(f"leakage detected in fold {split['fold']}")
        rows.append(
            {
                "fold": split["fold"],
                "train_rows": int(len(train_idx)),
                "test_rows": int(len(test_idx)),
                "train_hotels": int(len(train_hotels)),
                "test_hotels": int(len(test_hotels)),
                "index_overlap": int(len(index_overlap)),
                "hotel_overlap": int(len(group_overlap)),
            }
        )
    return rows


def load_and_align_data(
    output_dir: Path,
    profile: str,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, List[str], Dict[str, Any]]:
    dataset_path = output_dir / "dataset_clean.parquet"
    aspects_path = output_dir / f"aspects_llm_{profile}.parquet"
    rating_cols = [f"rating_{aspect}" for aspect in ASPECTS]
    dataset_cols = [
        "offering_id",
        "rating_overall",
        "review_text",
        "date",
        *rating_cols,
    ]
    aspect_score_cols = [f"aspect_{aspect}_score" for aspect in ASPECTS]
    aspect_mention_cols = [f"aspect_{aspect}_mentioned" for aspect in ASPECTS]
    aspect_cols = [
        "row_id",
        "ok",
        "traveler_segment",
        *aspect_score_cols,
        *aspect_mention_cols,
    ]
    dataset = pd.read_parquet(dataset_path, columns=dataset_cols)
    dataset["row_id"] = np.arange(len(dataset), dtype=np.int64)
    aspects = pd.read_parquet(aspects_path, columns=aspect_cols)
    aspects["row_id"] = pd.to_numeric(aspects["row_id"], errors="raise").astype(
        np.int64
    )
    if aspects["row_id"].duplicated().any():
        raise ValueError("duplicate row_id in LLM aspect output")
    df = dataset.merge(
        aspects,
        on="row_id",
        how="left",
        sort=False,
        validate="one_to_one",
    )
    if len(df) != EXPECTED_ROWS or not df["row_id"].is_unique:
        raise ValueError(
            f"aligned data mismatch: rows={len(df)}, unique={df['row_id'].is_unique}"
        )

    ok_mask = df["ok"].eq(1)
    df.loc[~ok_mask, aspect_score_cols] = 0
    df.loc[~ok_mask, aspect_mention_cols] = False
    df.loc[~ok_mask, "traveler_segment"] = "unknown"
    for column in aspect_score_cols:
        df[column] = (
            pd.to_numeric(df[column], errors="coerce")
            .fillna(0)
            .astype(np.int8)
            .clip(-2, 2)
        )
    for column in aspect_mention_cols:
        df[column] = df[column].fillna(False).astype(bool)

    allowed_segments = [
        "business",
        "couple",
        "family",
        "friends",
        "solo",
        "unknown",
    ]
    df["traveler_segment"] = (
        df["traveler_segment"].fillna("unknown").astype(str).str.lower()
    )
    df["traveler_segment"] = df["traveler_segment"].where(
        df["traveler_segment"].isin(allowed_segments),
        "unknown",
    )
    segment_frame = pd.get_dummies(
        df["traveler_segment"],
        prefix="segment",
    )
    expected_segment_cols = [f"segment_{segment}" for segment in allowed_segments]
    segment_frame = segment_frame.reindex(
        columns=expected_segment_cols,
        fill_value=False,
    )
    scores = df[aspect_score_cols].to_numpy(dtype=float)
    mentions = df[aspect_mention_cols].to_numpy(dtype=float)
    segments = segment_frame.to_numpy(dtype=float)
    structured = np.hstack([scores, mentions, segments])
    feature_names = aspect_score_cols + aspect_mention_cols + expected_segment_cols

    extraction_stats = {
        "rows": int(len(df)),
        "ok_rows": int(ok_mask.sum()),
        "failed_rows": int((~ok_mask).sum()),
        "ok_rate": float(ok_mask.mean()),
        "unknown_rows": int(df["traveler_segment"].eq("unknown").sum()),
        "unknown_rate": float(df["traveler_segment"].eq("unknown").mean()),
        "segment_counts": {
            str(key): int(value)
            for key, value in df["traveler_segment"].value_counts().items()
        },
        "aspect_coverage": {
            aspect: {
                "mentioned_count": int(
                    df[f"aspect_{aspect}_mentioned"].sum()
                ),
                "mentioned_rate": float(
                    df[f"aspect_{aspect}_mentioned"].mean()
                ),
            }
            for aspect in ASPECTS
        },
    }
    return df, scores, mentions, segments, feature_names, extraction_stats


def select_alpha_dense(
    x_train: np.ndarray,
    y_train: np.ndarray,
    groups: np.ndarray,
    alphas: Sequence[float],
) -> float:
    inner_cv = GroupKFold(n_splits=3)
    best_alpha = float(alphas[0])
    best_mae = float("inf")
    for alpha in alphas:
        fold_mae: List[float] = []
        for inner_train, inner_valid in inner_cv.split(
            x_train,
            y_train,
            groups=groups,
        ):
            scaler = StandardScaler()
            x_fit = scaler.fit_transform(x_train[inner_train])
            x_valid = scaler.transform(x_train[inner_valid])
            model = Ridge(alpha=alpha, random_state=SEED)
            model.fit(x_fit, y_train[inner_train])
            predictions = model.predict(x_valid)
            fold_mae.append(
                float(mean_absolute_error(y_train[inner_valid], predictions))
            )
        mean_mae = float(np.mean(fold_mae))
        if mean_mae < best_mae:
            best_mae = mean_mae
            best_alpha = float(alpha)
    return best_alpha


def select_alpha_sparse(
    x_train: sp.csr_matrix,
    y_train: np.ndarray,
    groups: np.ndarray,
    alphas: Sequence[float],
) -> float:
    inner_cv = GroupKFold(n_splits=3)
    best_alpha = float(alphas[0])
    best_mae = float("inf")
    for alpha in alphas:
        fold_mae: List[float] = []
        for inner_train, inner_valid in inner_cv.split(
            x_train,
            y_train,
            groups=groups,
        ):
            model = Ridge(alpha=alpha, random_state=SEED)
            model.fit(x_train[inner_train], y_train[inner_train])
            predictions = model.predict(x_train[inner_valid])
            fold_mae.append(
                float(mean_absolute_error(y_train[inner_valid], predictions))
            )
        mean_mae = float(np.mean(fold_mae))
        log(f"    interaction alpha={alpha:g} inner_MAE={mean_mae:.6f}")
        if mean_mae < best_mae:
            best_mae = mean_mae
            best_alpha = float(alpha)
    return best_alpha


def select_upper_alpha(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    groups: np.ndarray,
    alphas: Sequence[float],
) -> float:
    inner_cv = GroupKFold(n_splits=3)
    best_alpha = float(alphas[0])
    best_mae = float("inf")
    for alpha in alphas:
        fold_mae: List[float] = []
        for inner_train, inner_valid in inner_cv.split(
            x_train,
            y_train,
            groups=groups,
        ):
            pipeline = Pipeline(
                [
                    (
                        "imputer",
                        SimpleImputer(strategy="median", add_indicator=True),
                    ),
                    ("scaler", StandardScaler()),
                    ("model", Ridge(alpha=alpha)),
                ]
            )
            pipeline.fit(x_train.iloc[inner_train], y_train[inner_train])
            predictions = pipeline.predict(x_train.iloc[inner_valid])
            fold_mae.append(
                float(mean_absolute_error(y_train[inner_valid], predictions))
            )
        mean_mae = float(np.mean(fold_mae))
        if mean_mae < best_mae:
            best_mae = mean_mae
            best_alpha = float(alpha)
    return best_alpha


def build_interactions(
    scores: np.ndarray,
    mentions: np.ndarray,
    segments: np.ndarray,
) -> np.ndarray:
    blocks: List[np.ndarray] = []
    for segment_index in range(segments.shape[1]):
        segment = segments[:, segment_index : segment_index + 1]
        blocks.append(scores * segment)
        blocks.append(mentions * segment)
    return np.hstack(blocks).astype(np.float32)


def load_tfidf_fold(
    cache_dir: Path,
    profile: str,
    split: Dict[str, Any],
    fold: int,
) -> Tuple[Any, sp.csr_matrix, sp.csr_matrix, Dict[str, Any], float]:
    prefix = (
        f"{profile}_mf200000_ng12_md3_full_fold{fold}_{split['sha1'][:8]}"
    )
    vectorizer_path = cache_dir / f"{prefix}_vectorizer.joblib"
    train_path = cache_dir / f"{prefix}_Xtr.npz"
    test_path = cache_dir / f"{prefix}_Xte.npz"
    missing = [
        path
        for path in (vectorizer_path, train_path, test_path)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"missing exact TF-IDF cache files: {missing}")
    start = time.perf_counter()
    vectorizer = joblib.load(vectorizer_path)
    x_train = sp.load_npz(train_path).tocsr()
    x_test = sp.load_npz(test_path).tocsr()
    elapsed = time.perf_counter() - start
    parameters = vectorizer.get_params()
    config = {
        "analyzer": parameters["analyzer"],
        "ngram_range": list(parameters["ngram_range"]),
        "min_df": parameters["min_df"],
        "max_df": parameters["max_df"],
        "max_features": parameters["max_features"],
        "sublinear_tf": parameters["sublinear_tf"],
        "norm": parameters["norm"],
        "dtype": str(parameters["dtype"]),
        "lowercase": parameters["lowercase"],
        "token_pattern": parameters["token_pattern"],
        "stop_words": parameters["stop_words"],
        "smooth_idf": parameters["smooth_idf"],
        "use_idf": parameters["use_idf"],
        "vocabulary_size": int(len(vectorizer.vocabulary_)),
        "train_shape": list(x_train.shape),
        "test_shape": list(x_test.shape),
        "train_nnz": int(x_train.nnz),
        "test_nnz": int(x_test.nnz),
        "cache_prefix": prefix,
        "cache_load_seconds": elapsed,
    }
    return vectorizer, x_train, x_test, config, elapsed


def write_or_resume(
    harness: EvaluationHarness,
    model_id: str,
    fold: int,
    resume: bool,
) -> Optional[Dict[str, Any]]:
    if resume and harness.has_fold(model_id, fold):
        artifact = harness.load_fold(model_id, fold)
        log(f"  {model_id}: fold {fold} artifact cache hit")
        return artifact.record
    return None


def run_low_dimensional_models(
    harness: EvaluationHarness,
    df: pd.DataFrame,
    structured: np.ndarray,
    feature_names: Sequence[str],
    splits: Sequence[Dict[str, Any]],
    resume: bool,
) -> None:
    y = df["rating_overall"].to_numpy(dtype=float)
    groups = df["offering_id"].to_numpy()
    row_ids = df["row_id"].to_numpy(dtype=np.int64)
    rating_cols = [f"rating_{aspect}" for aspect in ASPECTS]
    llm_alphas = [0.1, 1.0, 10.0]
    upper_alphas = [0.01, 0.1, 1.0, 10.0, 100.0]

    for split in tqdm(splits, desc="Phase0 structured folds"):
        fold = split["fold"]
        train_idx = split["train_idx"]
        test_idx = split["test_idx"]
        log(f"fold {fold}: structured models")

        if write_or_resume(harness, LLM_ONLY, fold, resume) is None:
            with PeakRSSMonitor() as memory:
                tuning_start = time.perf_counter()
                alpha = select_alpha_dense(
                    structured[train_idx],
                    y[train_idx],
                    groups[train_idx],
                    llm_alphas,
                )
                tuning_seconds = time.perf_counter() - tuning_start
                scaler_start = time.perf_counter()
                scaler = StandardScaler()
                x_train = scaler.fit_transform(structured[train_idx])
                x_test = scaler.transform(structured[test_idx])
                preprocessing_seconds = time.perf_counter() - scaler_start
                model = Ridge(alpha=alpha, random_state=SEED)
                fit_start = time.perf_counter()
                model.fit(x_train, y[train_idx])
                fit_seconds = time.perf_counter() - fit_start
                predict_start = time.perf_counter()
                predictions = model.predict(x_test)
                predict_seconds = time.perf_counter() - predict_start
            artifact = harness.write_fold(
                model_id=LLM_ONLY,
                fold=fold,
                row_id=row_ids[test_idx],
                offering_id=groups[test_idx],
                y_true=y[test_idx],
                y_pred_raw=predictions,
                preprocessing_seconds=preprocessing_seconds,
                tuning_seconds=tuning_seconds,
                fit_seconds=fit_seconds,
                predict_seconds=predict_seconds,
                peak_memory_mb=memory.peak_mb,
                selected_hyperparameters={
                    "alpha": alpha,
                    "alpha_grid": llm_alphas,
                    "inner_cv": "GroupKFold(3)",
                },
                extra={"feature_names": list(feature_names)},
            )
            log(
                f"  {LLM_ONLY}: MAE={artifact.record['clipped']['mae']:.4f} "
                f"RMSE={artifact.record['clipped']['rmse']:.4f} alpha={alpha:g}"
            )

        if write_or_resume(harness, UPPER_BOUND, fold, resume) is None:
            x_upper_train = df.iloc[train_idx][rating_cols].astype(float)
            x_upper_test = df.iloc[test_idx][rating_cols].astype(float)
            with PeakRSSMonitor() as memory:
                tuning_start = time.perf_counter()
                alpha = select_upper_alpha(
                    x_upper_train,
                    y[train_idx],
                    groups[train_idx],
                    upper_alphas,
                )
                tuning_seconds = time.perf_counter() - tuning_start
                pipeline = Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="median",
                                add_indicator=True,
                            ),
                        ),
                        ("scaler", StandardScaler()),
                        ("model", Ridge(alpha=alpha)),
                    ]
                )
                fit_start = time.perf_counter()
                pipeline.fit(x_upper_train, y[train_idx])
                fit_seconds = time.perf_counter() - fit_start
                predict_start = time.perf_counter()
                predictions = pipeline.predict(x_upper_test)
                predict_seconds = time.perf_counter() - predict_start
            artifact = harness.write_fold(
                model_id=UPPER_BOUND,
                fold=fold,
                row_id=row_ids[test_idx],
                offering_id=groups[test_idx],
                y_true=y[test_idx],
                y_pred_raw=predictions,
                preprocessing_seconds=0.0,
                tuning_seconds=tuning_seconds,
                fit_seconds=fit_seconds,
                predict_seconds=predict_seconds,
                peak_memory_mb=memory.peak_mb,
                selected_hyperparameters={
                    "alpha": alpha,
                    "alpha_grid": upper_alphas,
                    "inner_cv": "GroupKFold(3)",
                    "imputer": "median",
                    "add_indicator": True,
                    "scaler": "StandardScaler",
                },
                extra={"feature_names": rating_cols},
            )
            log(
                f"  {UPPER_BOUND}: MAE={artifact.record['clipped']['mae']:.4f} "
                f"RMSE={artifact.record['clipped']['rmse']:.4f} alpha={alpha:g}"
            )


def run_text_models(
    harness: EvaluationHarness,
    df: pd.DataFrame,
    scores: np.ndarray,
    mentions: np.ndarray,
    segments: np.ndarray,
    structured: np.ndarray,
    splits: Sequence[Dict[str, Any]],
    cache_dir: Path,
    profile: str,
    resume: bool,
    saved_interaction_alphas: Dict[int, float],
    retune_interaction: bool,
) -> List[Dict[str, Any]]:
    y = df["rating_overall"].to_numpy(dtype=float)
    groups = df["offering_id"].to_numpy()
    row_ids = df["row_id"].to_numpy(dtype=np.int64)
    tfidf_configs: List[Dict[str, Any]] = []

    for split in tqdm(splits, desc="Phase0 text folds"):
        fold = split["fold"]
        train_idx = split["train_idx"]
        test_idx = split["test_idx"]
        requested = [TEXT_ONLY, SIMPLE_FUSION, RESIDUAL, INTERACTION]
        if resume and all(harness.has_fold(model_id, fold) for model_id in requested):
            log(f"fold {fold}: all text artifacts cache hit")
            _, _, _, config, _ = load_tfidf_fold(
                cache_dir,
                profile,
                split,
                fold,
            )
            tfidf_configs.append({"fold": fold, **config})
            continue

        log(f"fold {fold}: loading exact TF-IDF cache")
        _, x_text_train, x_text_test, tfidf_config, cache_seconds = load_tfidf_fold(
            cache_dir,
            profile,
            split,
            fold,
        )
        tfidf_configs.append({"fold": fold, **tfidf_config})
        y_train = y[train_idx]
        y_test = y[test_idx]

        need_text_fit = (
            not (resume and harness.has_fold(TEXT_ONLY, fold))
            or not (resume and harness.has_fold(RESIDUAL, fold))
        )
        text_model: Optional[Ridge] = None
        text_train_predictions: Optional[np.ndarray] = None
        text_test_predictions: Optional[np.ndarray] = None
        text_fit_seconds = 0.0
        text_predict_seconds = 0.0
        text_peak_mb = 0.0
        if need_text_fit:
            with PeakRSSMonitor() as memory:
                text_model = Ridge(alpha=1.0, random_state=SEED)
                fit_start = time.perf_counter()
                text_model.fit(x_text_train, y_train)
                text_fit_seconds = time.perf_counter() - fit_start
                predict_start = time.perf_counter()
                text_train_predictions = text_model.predict(x_text_train)
                text_test_predictions = text_model.predict(x_text_test)
                text_predict_seconds = time.perf_counter() - predict_start
            text_peak_mb = memory.peak_mb

        if write_or_resume(harness, TEXT_ONLY, fold, resume) is None:
            if text_test_predictions is None:
                raise RuntimeError("text predictions were not generated")
            artifact = harness.write_fold(
                model_id=TEXT_ONLY,
                fold=fold,
                row_id=row_ids[test_idx],
                offering_id=groups[test_idx],
                y_true=y_test,
                y_pred_raw=text_test_predictions,
                preprocessing_seconds=cache_seconds,
                tuning_seconds=0.0,
                fit_seconds=text_fit_seconds,
                predict_seconds=text_predict_seconds,
                peak_memory_mb=text_peak_mb,
                selected_hyperparameters={
                    "alpha": 1.0,
                    "selection": "fixed",
                },
                extra={"tfidf": tfidf_config},
            )
            log(
                f"  {TEXT_ONLY}: MAE={artifact.record['clipped']['mae']:.4f} "
                f"RMSE={artifact.record['clipped']['rmse']:.4f}"
            )

        if write_or_resume(harness, SIMPLE_FUSION, fold, resume) is None:
            with PeakRSSMonitor() as memory:
                scaler_start = time.perf_counter()
                scaler = StandardScaler()
                dense_train = scaler.fit_transform(structured[train_idx])
                dense_test = scaler.transform(structured[test_idx])
                x_train = sp.hstack(
                    [x_text_train, sp.csr_matrix(dense_train)],
                    format="csr",
                )
                x_test = sp.hstack(
                    [x_text_test, sp.csr_matrix(dense_test)],
                    format="csr",
                )
                preprocessing_seconds = time.perf_counter() - scaler_start
                model = Ridge(alpha=0.1, random_state=SEED)
                fit_start = time.perf_counter()
                model.fit(x_train, y_train)
                fit_seconds = time.perf_counter() - fit_start
                predict_start = time.perf_counter()
                predictions = model.predict(x_test)
                predict_seconds = time.perf_counter() - predict_start
            artifact = harness.write_fold(
                model_id=SIMPLE_FUSION,
                fold=fold,
                row_id=row_ids[test_idx],
                offering_id=groups[test_idx],
                y_true=y_test,
                y_pred_raw=predictions,
                preprocessing_seconds=preprocessing_seconds,
                tuning_seconds=0.0,
                fit_seconds=fit_seconds,
                predict_seconds=predict_seconds,
                peak_memory_mb=memory.peak_mb,
                selected_hyperparameters={
                    "alpha": 0.1,
                    "selection": "fixed",
                },
                extra={"tfidf": tfidf_config},
            )
            log(
                f"  {SIMPLE_FUSION}: MAE={artifact.record['clipped']['mae']:.4f} "
                f"RMSE={artifact.record['clipped']['rmse']:.4f}"
            )
            del x_train, x_test, dense_train, dense_test, model, predictions
            gc.collect()

        if write_or_resume(harness, RESIDUAL, fold, resume) is None:
            if text_train_predictions is None or text_test_predictions is None:
                raise RuntimeError("residual model requires fresh text predictions")
            residual_train = y_train - text_train_predictions
            residual_test = y_test - text_test_predictions
            residual_alphas = [0.1, 1.0, 10.0]
            with PeakRSSMonitor() as memory:
                tuning_start = time.perf_counter()
                alpha = select_alpha_dense(
                    structured[train_idx],
                    residual_train,
                    groups[train_idx],
                    residual_alphas,
                )
                tuning_seconds = time.perf_counter() - tuning_start
                scaler_start = time.perf_counter()
                scaler = StandardScaler()
                dense_train = scaler.fit_transform(structured[train_idx])
                dense_test = scaler.transform(structured[test_idx])
                preprocessing_seconds = time.perf_counter() - scaler_start
                model = Ridge(alpha=alpha, random_state=SEED)
                fit_start = time.perf_counter()
                model.fit(dense_train, residual_train)
                fit_seconds = time.perf_counter() - fit_start
                predict_start = time.perf_counter()
                residual_train_pred = model.predict(dense_train)
                residual_test_pred = model.predict(dense_test)
                predictions = text_test_predictions + residual_test_pred
                predict_seconds = time.perf_counter() - predict_start
            r2_train = float(
                1
                - np.square(residual_train - residual_train_pred).sum()
                / np.square(residual_train - residual_train.mean()).sum()
            )
            r2_test = float(
                1
                - np.square(residual_test - residual_test_pred).sum()
                / np.square(residual_test - residual_test.mean()).sum()
            )
            artifact = harness.write_fold(
                model_id=RESIDUAL,
                fold=fold,
                row_id=row_ids[test_idx],
                offering_id=groups[test_idx],
                y_true=y_test,
                y_pred_raw=predictions,
                preprocessing_seconds=preprocessing_seconds + cache_seconds,
                tuning_seconds=tuning_seconds,
                fit_seconds=text_fit_seconds + fit_seconds,
                predict_seconds=text_predict_seconds + predict_seconds,
                peak_memory_mb=max(memory.peak_mb, text_peak_mb),
                selected_hyperparameters={
                    "text_alpha": 1.0,
                    "residual_alpha": alpha,
                    "residual_alpha_grid": residual_alphas,
                    "inner_cv": "GroupKFold(3)",
                },
                extra={
                    "tfidf": tfidf_config,
                    "residual_r2_train": r2_train,
                    "residual_r2_test": r2_test,
                },
            )
            log(
                f"  {RESIDUAL}: MAE={artifact.record['clipped']['mae']:.4f} "
                f"RMSE={artifact.record['clipped']['rmse']:.4f} alpha={alpha:g}"
            )
            del (
                dense_train,
                dense_test,
                model,
                predictions,
                residual_train_pred,
                residual_test_pred,
            )
            gc.collect()

        if write_or_resume(harness, INTERACTION, fold, resume) is None:
            interaction_alphas = [0.01, 0.1, 1.0, 10.0]
            with PeakRSSMonitor() as memory:
                preprocessing_start = time.perf_counter()
                base_train = np.hstack(
                    [
                        scores[train_idx],
                        mentions[train_idx],
                        segments[train_idx],
                    ]
                )
                base_test = np.hstack(
                    [
                        scores[test_idx],
                        mentions[test_idx],
                        segments[test_idx],
                    ]
                )
                interactions_train = build_interactions(
                    scores[train_idx],
                    mentions[train_idx],
                    segments[train_idx],
                )
                interactions_test = build_interactions(
                    scores[test_idx],
                    mentions[test_idx],
                    segments[test_idx],
                )
                dense_train = np.hstack(
                    [base_train, interactions_train]
                ).astype(np.float32)
                dense_test = np.hstack(
                    [base_test, interactions_test]
                ).astype(np.float32)
                scaler = StandardScaler()
                dense_train = scaler.fit_transform(dense_train)
                dense_test = scaler.transform(dense_test)
                x_train = sp.hstack(
                    [x_text_train, sp.csr_matrix(dense_train)],
                    format="csr",
                )
                x_test = sp.hstack(
                    [x_text_test, sp.csr_matrix(dense_test)],
                    format="csr",
                )
                preprocessing_seconds = (
                    time.perf_counter() - preprocessing_start + cache_seconds
                )
                if retune_interaction:
                    log(
                        f"  {INTERACTION}: tuning alpha on {x_train.shape}, "
                        f"nnz={x_train.nnz}"
                    )
                    tuning_start = time.perf_counter()
                    alpha = select_alpha_sparse(
                        x_train,
                        y_train,
                        groups[train_idx],
                        interaction_alphas,
                    )
                    tuning_seconds = time.perf_counter() - tuning_start
                    alpha_source = "rerun GroupKFold(3) search"
                else:
                    if fold not in saved_interaction_alphas:
                        raise ValueError(
                            f"saved interaction alpha missing for fold {fold}"
                        )
                    alpha = float(saved_interaction_alphas[fold])
                    tuning_seconds = 0.0
                    alpha_source = (
                        "outputs/results_llm_aspects_qwen25_7b.json"
                    )
                    log(
                        f"  {INTERACTION}: replaying saved fold alpha={alpha:g}; "
                        "use --retune-interaction for the full expensive search"
                    )
                model = Ridge(alpha=alpha, random_state=SEED)
                fit_start = time.perf_counter()
                model.fit(x_train, y_train)
                fit_seconds = time.perf_counter() - fit_start
                predict_start = time.perf_counter()
                predictions = model.predict(x_test)
                predict_seconds = time.perf_counter() - predict_start
            artifact = harness.write_fold(
                model_id=INTERACTION,
                fold=fold,
                row_id=row_ids[test_idx],
                offering_id=groups[test_idx],
                y_true=y_test,
                y_pred_raw=predictions,
                preprocessing_seconds=preprocessing_seconds,
                tuning_seconds=tuning_seconds,
                fit_seconds=fit_seconds,
                predict_seconds=predict_seconds,
                peak_memory_mb=memory.peak_mb,
                selected_hyperparameters={
                    "alpha": alpha,
                    "alpha_grid": interaction_alphas,
                    "inner_cv": "GroupKFold(3)",
                    "selection_source": alpha_source,
                },
                extra={
                    "tfidf": tfidf_config,
                    "structured_features": int(dense_train.shape[1]),
                },
            )
            log(
                f"  {INTERACTION}: MAE={artifact.record['clipped']['mae']:.4f} "
                f"RMSE={artifact.record['clipped']['rmse']:.4f} alpha={alpha:g}"
            )
            del (
                base_train,
                base_test,
                interactions_train,
                interactions_test,
                dense_train,
                dense_test,
                x_train,
                x_test,
                model,
                predictions,
            )
            gc.collect()

        del x_text_train, x_text_test, text_model
        gc.collect()
    return tfidf_configs


def environment_payload(output_dir: Path) -> Dict[str, Any]:
    package_names = [
        "numpy",
        "scipy",
        "pandas",
        "scikit-learn",
        "pyarrow",
        "joblib",
        "psutil",
        "torch",
        "transformers",
        "vllm",
        "httpx",
        "PyYAML",
        "tqdm",
        "matplotlib",
    ]
    packages: Dict[str, Optional[str]] = {}
    for package in package_names:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    freeze_path = output_dir / "phase0_environment_pip_freeze.txt"
    freeze_path.write_text(freeze, encoding="utf-8")
    return {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "pip_freeze_path": str(freeze_path),
        "pip_freeze_sha256": hashlib.sha256(
            freeze.encode("utf-8")
        ).hexdigest(),
    }


def finalize_models(
    harness: EvaluationHarness,
    split_meta: Dict[str, Any],
    tfidf_configs: Sequence[Dict[str, Any]],
    source_hashes: Dict[str, str],
) -> Dict[str, Dict[str, Any]]:
    descriptions = {
        TEXT_ONLY: "TF-IDF (1,2) + Ridge",
        LLM_ONLY: "LLM scores + mentions + segment one-hot + Ridge",
        SIMPLE_FUSION: "TF-IDF + additive LLM structured features + Ridge",
        RESIDUAL: "Text Ridge plus LLM Ridge on training residuals",
        UPPER_BOUND: "True aspect ratings + missing indicators + Ridge",
        INTERACTION: "TF-IDF + LLM main effects + segment-aspect interactions + Ridge",
    }
    payloads: Dict[str, Dict[str, Any]] = {}
    for model_id in MODEL_IDS:
        payloads[model_id] = harness.finalize_model(
            model_id,
            {
                "description": descriptions[model_id],
                "seed": SEED,
                "outer_split": "saved GroupKFold(5), group=offering_id",
                "split_assignment_sha256": split_meta["assignment_sha256"],
                "prediction_clipping": [1.0, 5.0],
                "tfidf_folds": list(tfidf_configs)
                if model_id in {TEXT_ONLY, SIMPLE_FUSION, RESIDUAL, INTERACTION}
                else None,
                "source_sha256": source_hashes,
            },
        )
    return payloads


def reproduction_gate(
    model_payloads: Dict[str, Dict[str, Any]],
    output_dir: Path,
    tolerance: float,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    gate_passed = True
    for model_id in MODEL_IDS:
        published = PUBLISHED[model_id]
        aggregate = model_payloads[model_id]["aggregate"]["clipped"]
        reproduced = {
            "mae_mean": aggregate["mean"]["mae"],
            "mae_std": aggregate["std"]["mae"],
            "rmse_mean": aggregate["mean"]["rmse"],
            "rmse_std": aggregate["std"]["rmse"],
        }
        deltas = {
            key: float(reproduced[key] - published[key])
            for key in published
        }
        passed = all(abs(delta) <= tolerance for delta in deltas.values())
        gate_passed = gate_passed and passed
        rows.append(
            {
                "model_id": model_id,
                "published": published,
                "reproduced": reproduced,
                "delta": deltas,
                "passed": passed,
            }
        )
    payload = {
        "tolerance": tolerance,
        "gate_passed": gate_passed,
        "comparisons": rows,
    }
    write_json(output_dir / "phase0_reproduction_gate.json", payload)
    return payload


def run_pairwise_statistics(
    harness: EvaluationHarness,
    output_dir: Path,
) -> Dict[str, Any]:
    comparisons = [
        (INTERACTION, TEXT_ONLY),
        (INTERACTION, SIMPLE_FUSION),
        (RESIDUAL, TEXT_ONLY),
    ]
    payload: Dict[str, Any] = {}
    for first, second in comparisons:
        first_path = harness.per_hotel_dir / f"{first}.parquet"
        second_path = harness.per_hotel_dir / f"{second}.parquet"
        key = f"{first}_vs_{second}"
        payload[key] = {
            "wilcoxon": paired_wilcoxon(first_path, second_path),
            "bootstrap": bootstrap_mean_difference(
                first_path,
                second_path,
                n_resamples=10_000,
                seed=SEED,
            ),
        }
    write_json(output_dir / "phase0_pairwise_statistics.json", payload)
    return payload


def write_cluster_robust_coefficients(
    output_dir: Path,
    structured: np.ndarray,
    feature_names: Sequence[str],
    y: np.ndarray,
    groups: np.ndarray,
    alpha: float,
) -> Path:
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(structured)
    model = Ridge(alpha=alpha, random_state=SEED)
    model.fit(x_scaled, y)
    predictions = model.predict(x_scaled)
    table = cluster_robust_ridge_se(
        x=x_scaled,
        y_true=y,
        y_pred=predictions,
        groups=groups,
        alpha=alpha,
        feature_names=feature_names,
    )
    coefficient_map = {
        "intercept": float(model.intercept_),
        **{
            name: float(coefficient)
            for name, coefficient in zip(feature_names, model.coef_)
        },
    }
    table["coefficient"] = table["feature"].map(coefficient_map)
    table["z_value"] = table["coefficient"] / table["cluster_robust_se"]
    table["p_value_normal"] = 2 * scipy.stats.norm.sf(
        np.abs(table["z_value"])
    )
    path = output_dir / "metrics" / "llm_aspects_cluster_robust_coefficients.parquet"
    table.to_parquet(path, index=False)
    return path


def write_phase0_report(
    output_dir: Path,
    environment: Dict[str, Any],
    dataset_stats: Dict[str, Any],
    split_meta: Dict[str, Any],
    split_rows: Sequence[Dict[str, Any]],
    extraction_stats: Dict[str, Any],
    model_payloads: Dict[str, Dict[str, Any]],
    gate: Dict[str, Any],
    pairwise: Dict[str, Any],
    tfidf_configs: Sequence[Dict[str, Any]],
    coefficient_path: Path,
    interaction_retuned: bool,
) -> None:
    lines = [
        "# Phase 0 Audit and Reproduction Report",
        "",
        f"- gate_passed: **{gate['gate_passed']}**",
        f"- tolerance: {gate['tolerance']:.4f}",
        f"- dataset_rows: {dataset_stats['rows']}",
        f"- hotels: {dataset_stats['hotels']}",
        f"- date_span: {dataset_stats['date_min']} to {dataset_stats['date_max']}",
        f"- split_assignment_sha256: `{split_meta['assignment_sha256']}`",
        f"- split_assignment_pairs_sha256: `{split_meta['assignment_pairs_sha256']}`",
        "",
        "## Environment",
        "",
        f"- Python: `{environment['python_version'].splitlines()[0]}`",
        f"- executable: `{environment['python_executable']}`",
        f"- pip freeze: `{environment['pip_freeze_path']}`",
        f"- pip freeze SHA-256: `{environment['pip_freeze_sha256']}`",
        "",
        "| package | version |",
        "| --- | --- |",
    ]
    for package, version in environment["packages"].items():
        lines.append(f"| {package} | {version or 'not installed'} |")

    lines.extend(
        [
            "",
            "## Saved Split Verification",
            "",
            "| fold | train rows | test rows | train hotels | test hotels | hotel overlap |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in split_rows:
        lines.append(
            f"| {row['fold']} | {row['train_rows']} | {row['test_rows']} | "
            f"{row['train_hotels']} | {row['test_hotels']} | {row['hotel_overlap']} |"
        )

    lines.extend(
        [
            "",
            "## Table 1 Reproduction Gate",
            "",
            "| model | published MAE | reproduced MAE | delta | published RMSE | reproduced RMSE | delta | pass |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in gate["comparisons"]:
        lines.append(
            f"| {row['model_id']} | "
            f"{row['published']['mae_mean']:.4f} +/- {row['published']['mae_std']:.4f} | "
            f"{row['reproduced']['mae_mean']:.4f} +/- {row['reproduced']['mae_std']:.4f} | "
            f"{row['delta']['mae_mean']:+.6f} | "
            f"{row['published']['rmse_mean']:.4f} +/- {row['published']['rmse_std']:.4f} | "
            f"{row['reproduced']['rmse_mean']:.4f} +/- {row['reproduced']['rmse_std']:.4f} | "
            f"{row['delta']['rmse_mean']:+.6f} | {row['passed']} |"
        )

    lines.extend(
        [
            "",
            "## Selected Alpha by Fold",
            "",
            "| model | fold 1 | fold 2 | fold 3 | fold 4 | fold 5 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for model_id in MODEL_IDS:
        values = []
        for fold in model_payloads[model_id]["folds"]:
            params = fold["selected_hyperparameters"]
            alpha = params.get("alpha")
            if alpha is None:
                alpha = params.get("residual_alpha", params.get("text_alpha"))
            values.append(str(alpha))
        lines.append(f"| {model_id} | " + " | ".join(values) + " |")

    first_tfidf = tfidf_configs[0]
    lines.extend(
        [
            "",
            "## TF-IDF Configuration",
            "",
            f"- analyzer: `{first_tfidf['analyzer']}`",
            f"- ngram_range: `{tuple(first_tfidf['ngram_range'])}`",
            f"- min_df: `{first_tfidf['min_df']}`",
            f"- max_df: `{first_tfidf['max_df']}`",
            f"- max_features: `{first_tfidf['max_features']}`",
            f"- sublinear_tf: `{first_tfidf['sublinear_tf']}`",
            f"- norm: `{first_tfidf['norm']}`",
            f"- dtype: `{first_tfidf['dtype']}`",
            "",
            "| fold | vocabulary size | train shape | test shape |",
            "| --- | ---: | --- | --- |",
        ]
    )
    for config in tfidf_configs:
        lines.append(
            f"| {config['fold']} | {config['vocabulary_size']} | "
            f"{tuple(config['train_shape'])} | {tuple(config['test_shape'])} |"
        )

    lines.extend(
        [
            "",
            "## Independently Recomputed Extraction Statistics",
            "",
            f"- OK: {extraction_stats['ok_rows']} / {extraction_stats['rows']} "
            f"({extraction_stats['ok_rate']:.6f})",
            f"- failures: {extraction_stats['failed_rows']}",
            f"- unknown segment: {extraction_stats['unknown_rows']} "
            f"({extraction_stats['unknown_rate']:.6f})",
            "",
            "| aspect | mentioned count | coverage |",
            "| --- | ---: | ---: |",
        ]
    )
    for aspect, values in extraction_stats["aspect_coverage"].items():
        lines.append(
            f"| {aspect} | {values['mentioned_count']} | "
            f"{values['mentioned_rate']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Statistical Harness",
            "",
            "- Fold predictions: `outputs/predictions/{model_id}/fold{k}.parquet`",
            "- Model metrics: `outputs/metrics/{model_id}.json`",
            "- Per-hotel metrics: `outputs/per_hotel/{model_id}.parquet`",
            "- Pairwise tests: `outputs/phase0_pairwise_statistics.json`",
            f"- Cluster-robust coefficient table: `{coefficient_path}`",
            "",
        ]
    )
    for key, values in pairwise.items():
        bootstrap = values["bootstrap"]
        wilcoxon = values["wilcoxon"]
        lines.append(
            f"- {key}: mean MAE difference={bootstrap['mean_difference']:+.6f}, "
            f"95% CI=[{bootstrap['ci_95'][0]:+.6f}, {bootstrap['ci_95'][1]:+.6f}], "
            f"Wilcoxon p={wilcoxon['p_value']:.6g}."
        )

    lines.extend(
        [
            "",
            "## Audit Discrepancies",
            "",
            "- The extraction cache stores sentiment scores in `[-2, 2]`; the paper context supplied for this audit describes aspect scores in `[1, 5]`.",
            "- The executable interaction implementation uses the cached score, mention, and segment blocks plus segment-score and segment-mention products. It does not explicitly materialize `(3 + score) * mentioned`.",
            "- The local vLLM script enforces JSON primarily through prompting and repair/retry logic; it does not currently send a guided-decoding or JSON-schema response-format parameter.",
            "- The cleaned dataset has no persisted `row_id`; downstream code deterministically assigns `row_id = parquet row index` and validates the one-to-one join.",
            "- The TF-IDF cache key includes profile, vectorizer settings, fold, and split-file SHA-1 prefix, but not a content hash of the review text.",
            (
                "- Interaction alpha was re-selected with the original inner GroupKFold search."
                if interaction_retuned
                else "- Interaction final models were retrained with the exact per-fold alphas persisted by the published run. A strict re-search is available with `--retune-interaction`; the original sparse search is single-threaded and its first low-alpha candidate did not finish after more than ten minutes on fold 1."
            ),
            "",
        ]
    )
    (output_dir / "phase0_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce the six published Table 1 models through one harness."
    )
    parser.add_argument("--profile", default="qwen25_7b")
    parser.add_argument(
        "--tfidf-cache-dir",
        default="outputs/tfidf_cache",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--retune-interaction",
        action="store_true",
        help="Rerun the very expensive sparse inner GroupKFold alpha search.",
    )
    parser.add_argument("--tolerance", type=float, default=TOLERANCE)
    args = parser.parse_args()

    np.random.seed(SEED)
    output_dir = ROOT / "outputs"
    cache_dir = Path(args.tfidf_cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = ROOT / cache_dir
    harness = EvaluationHarness(output_dir)

    log("Phase 0 uses CPU/IO only; no GPU or vLLM requests are started.")
    environment = environment_payload(output_dir)
    write_json(output_dir / "phase0_environment.json", environment)

    log("loading and aligning dataset with saved LLM extraction")
    (
        df,
        scores,
        mentions,
        segments,
        feature_names,
        extraction_stats,
    ) = load_and_align_data(output_dir, args.profile)
    structured = np.hstack([scores, mentions, segments])
    dataset_stats = {
        "rows": int(len(df)),
        "hotels": int(df["offering_id"].nunique()),
        "date_min": str(pd.to_datetime(df["date"]).min().date()),
        "date_max": str(pd.to_datetime(df["date"]).max().date()),
    }
    if dataset_stats["rows"] != EXPECTED_ROWS:
        raise ValueError(dataset_stats)
    if dataset_stats["hotels"] != EXPECTED_HOTELS:
        raise ValueError(dataset_stats)

    splits, split_meta = load_splits(output_dir, len(df))
    split_rows = verify_splits(df, splits)
    log(f"split assignment SHA-256: {split_meta['assignment_sha256']}")

    source_paths = [
        ROOT / "scripts" / "03_run_baselines.py",
        ROOT / "scripts" / "06_extract_aspects_llm.py",
        ROOT / "scripts" / "07_run_llm_aspects_analysis.py",
        ROOT / "scripts" / "08_phase0_reproduce.py",
        ROOT / "src" / "evaluation_harness.py",
    ]
    source_hashes = {
        str(path.relative_to(ROOT)): sha256_file(path)
        for path in source_paths
    }
    saved_results_path = (
        output_dir / f"results_llm_aspects_{args.profile}.json"
    )
    saved_results = json.loads(saved_results_path.read_text(encoding="utf-8"))
    saved_interaction_alphas = {
        int(row["fold"]): float(row["best_alpha"])
        for row in saved_results["text_llm_interactions"]["folds"]
    }

    run_low_dimensional_models(
        harness=harness,
        df=df,
        structured=structured,
        feature_names=feature_names,
        splits=splits,
        resume=args.resume,
    )
    tfidf_configs = run_text_models(
        harness=harness,
        df=df,
        scores=scores,
        mentions=mentions,
        segments=segments,
        structured=structured,
        splits=splits,
        cache_dir=cache_dir,
        profile=args.profile,
        resume=args.resume,
        saved_interaction_alphas=saved_interaction_alphas,
        retune_interaction=args.retune_interaction,
    )
    model_payloads = finalize_models(
        harness,
        split_meta,
        tfidf_configs,
        source_hashes,
    )
    gate = reproduction_gate(
        model_payloads,
        output_dir,
        args.tolerance,
    )
    pairwise = run_pairwise_statistics(harness, output_dir)
    llm_alphas = [
        fold["selected_hyperparameters"]["alpha"]
        for fold in model_payloads[LLM_ONLY]["folds"]
    ]
    coefficient_path = write_cluster_robust_coefficients(
        output_dir=output_dir,
        structured=structured,
        feature_names=feature_names,
        y=df["rating_overall"].to_numpy(dtype=float),
        groups=df["offering_id"].to_numpy(),
        alpha=float(np.median(llm_alphas)),
    )
    write_phase0_report(
        output_dir=output_dir,
        environment=environment,
        dataset_stats=dataset_stats,
        split_meta=split_meta,
        split_rows=split_rows,
        extraction_stats=extraction_stats,
        model_payloads=model_payloads,
        gate=gate,
        pairwise=pairwise,
        tfidf_configs=tfidf_configs,
        coefficient_path=coefficient_path,
        interaction_retuned=args.retune_interaction,
    )
    log(f"Phase 0 gate passed: {gate['gate_passed']}")
    log(f"saved: {output_dir / 'phase0_report.md'}")
    if not gate["gate_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
