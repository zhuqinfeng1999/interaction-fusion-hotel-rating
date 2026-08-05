#!/usr/bin/env python3
"""Phase 4 U0/U1/U4: unified-protocol primary models and ablations.

The script deliberately reuses the complete Phase 3 GroupKFold(5) alpha
curves for text-only, simple fusion, and interaction fusion.  Every other
model is selected on the same 41-point grid with automatic two-decade
boundary extension.  All held-out predictions flow through the canonical
EvaluationHarness and are resumable fold by fold.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.aspect_extractor_base import ASPECTS  # noqa: E402
from src.evaluation_harness import (  # noqa: E402
    EvaluationHarness,
    PeakRSSMonitor,
)
from src.phase2_ridge import (  # noqa: E402
    _extend_grid,
    _score_alphas,
    _spectral_state,
    fit_dense_ridge,
    predict_dense,
)
from src.phase2_sparse_ridge import (  # noqa: E402
    CenteredBlockOperator,
    fit_operator,
    predict_blocks,
)


SEED = 42
N_SPLITS = 5
SEGMENTS = ["business", "couple", "family", "solo", "friends", "unknown"]
ALPHAS = np.logspace(-4, 6, 41, dtype=np.float64)
SPLIT_SHA256 = "dfe3a434fb00c894218309c332627dc0131a2db3cc2cd241727fc712a4803942"

PRIMARY_IDS = {
    "text_only": "phase4_u0_text_only_tfidf_ridge",
    "llm_aspects_only": "phase4_u0_llm_aspects_only",
    "simple_fusion": "phase4_u0_text_llm_simple_fusion",
    "residual_on_llm": "phase4_u0_residual_on_llm",
    "upper_bound": "phase4_u0_upper_bound_true_aspects",
    "interaction_fusion": "phase4_u0_interaction_fusion",
}
ABLATION_IDS = {
    "A2": "phase4_u0_a2_score_interactions_only",
    "A3": "phase4_u0_a3_mention_interactions_only",
    "A5": "phase4_u0_a5_interactions_without_main",
    "A7": "phase4_u0_a7_unknown_columns_removed",
    "A8": "phase4_u0_a8_known_segment_only",
    "A4_known_reference": "phase4_u0_a4_known_segment_reference",
}


def log(message: str) -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["primary", "ablations", "all"], default="all")
    parser.add_argument("--folds", default="1,2,3,4,5")
    parser.add_argument(
        "--variants",
        default="A2,A3,A5,A7,A8",
        help="Comma-separated U4 variants; valid values: A2,A3,A5,A7,A8.",
    )
    parser.add_argument("--profile", default="qwen25_7b")
    parser.add_argument("--max-features", type=int, default=200_000)
    parser.add_argument("--inner-jobs", type=int, default=5)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--max-iterations", type=int, default=2_000)
    parser.add_argument("--max-extensions", type=int, default=4)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def parse_folds(value: str) -> List[int]:
    folds = sorted({int(item) for item in value.split(",") if item.strip()})
    if not folds or any(fold not in range(1, 6) for fold in folds):
        raise ValueError("--folds must contain values in 1..5")
    return folds


def parse_variants(value: str) -> List[str]:
    variants = [item.strip().upper() for item in value.split(",") if item.strip()]
    valid = {"A2", "A3", "A5", "A7", "A8"}
    if not variants or any(item not in valid for item in variants):
        raise ValueError("--variants must contain values from A2,A3,A5,A7,A8")
    # Preserve the canonical manuscript order while removing duplicates.
    return [item for item in ("A2", "A3", "A5", "A7", "A8") if item in variants]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_splits(output_dir: Path) -> Dict[int, Dict[str, Any]]:
    result: Dict[int, Dict[str, Any]] = {}
    assignment = np.full(877_808, -1, dtype=np.int8)
    for fold in range(1, 6):
        path = output_dir / f"splits_fold{fold}.npz"
        arrays = np.load(path)
        train_idx = arrays["train_idx"].astype(np.int64)
        test_idx = arrays["test_idx"].astype(np.int64)
        assignment[test_idx] = fold
        result[fold] = {
            "train_idx": train_idx,
            "test_idx": test_idx,
            "sha256": sha256_file(path),
        }
    digest = hashlib.sha256(assignment.tobytes()).hexdigest()
    if digest != SPLIT_SHA256:
        raise ValueError(f"canonical split hash mismatch: {digest}")
    return result


def tfidf_paths(
    output_dir: Path, profile: str, max_features: int, fold: int
) -> Tuple[Path, Path]:
    pattern = f"{profile}_mf{max_features}_ng12_md3_full_fold{fold}_*_Xtr.npz"
    matches = sorted((output_dir / "tfidf_cache").glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"fold {fold}: expected one TF-IDF cache, got {matches}")
    test = Path(str(matches[0]).replace("_Xtr.npz", "_Xte.npz"))
    if not test.exists():
        raise FileNotFoundError(test)
    return matches[0], test


def load_dataset(output_dir: Path) -> pd.DataFrame:
    columns = [
        "offering_id",
        "rating_overall",
        *[f"rating_{aspect}" for aspect in ASPECTS],
    ]
    frame = pd.read_parquet(output_dir / "dataset_clean.parquet", columns=columns)
    if len(frame) != 877_808:
        raise ValueError(f"canonical row count mismatch: {len(frame)}")
    frame = frame.reset_index(drop=True)
    frame["row_id"] = np.arange(len(frame), dtype=np.int64)
    return frame


def load_components(output_dir: Path, profile: str) -> Dict[str, np.ndarray]:
    score_cols = [f"aspect_{aspect}_score" for aspect in ASPECTS]
    mention_cols = [f"aspect_{aspect}_mentioned" for aspect in ASPECTS]
    frame = pd.read_parquet(
        output_dir / f"aspects_llm_{profile}.parquet",
        columns=["row_id", "ok", "traveler_segment", *score_cols, *mention_cols],
    ).sort_values("row_id")
    if not np.array_equal(frame["row_id"].to_numpy(), np.arange(len(frame))):
        raise ValueError("LLM row_id alignment failed")
    failed = ~frame["ok"].eq(1)
    frame.loc[failed, score_cols] = 0
    frame.loc[failed, mention_cols] = False
    frame.loc[failed, "traveler_segment"] = "unknown"
    scores = (
        frame[score_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
        .to_numpy(dtype=np.float32)
    )
    mentions = frame[mention_cols].fillna(False).to_numpy(dtype=np.float32)
    segment_text = frame["traveler_segment"].fillna("unknown").astype(str).str.lower()
    segment_text = segment_text.where(segment_text.isin(SEGMENTS), "unknown")
    segments = (
        pd.get_dummies(segment_text).reindex(columns=SEGMENTS, fill_value=False)
        .to_numpy(dtype=np.float32)
    )
    p_s = np.hstack(
        [scores * segments[:, index : index + 1] for index in range(len(SEGMENTS))]
    ).astype(np.float32)
    p_b = np.hstack(
        [mentions * segments[:, index : index + 1] for index in range(len(SEGMENTS))]
    ).astype(np.float32)
    return {
        "z": np.hstack([scores, mentions, segments]).astype(np.float32),
        "p_s": p_s,
        "p_b": p_b,
        "known": segment_text.ne("unknown").to_numpy(dtype=bool),
    }


def prepare_blocks(
    raw: Mapping[str, np.ndarray], train_idx: np.ndarray, test_idx: np.ndarray
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    train: Dict[str, np.ndarray] = {}
    test: Dict[str, np.ndarray] = {}
    for name, matrix in raw.items():
        if name == "known":
            continue
        scaler = StandardScaler()
        train[name] = scaler.fit_transform(matrix[train_idx]).astype(np.float32)
        test[name] = scaler.transform(matrix[test_idx]).astype(np.float32)
    return train, test


def selected_g3_alpha(phase3: Path, model: str, fold: int) -> Tuple[float, pd.DataFrame]:
    path = phase3 / "g3_inner_cv5" / f"{model}_outerfold{fold}.csv"
    curve = pd.read_csv(path).sort_values("alpha")
    if len(curve) != len(ALPHAS) or not all(
        np.any(np.isclose(curve["alpha"], alpha, rtol=1e-10)) for alpha in ALPHAS
    ):
        raise ValueError(f"incomplete Phase 3 CV5 curve: {path}")
    selected = curve.sort_values(["mean_inner_mae", "alpha"]).iloc[0]
    alpha = float(selected["alpha"])
    if np.isclose(alpha, curve["alpha"].min()) or np.isclose(alpha, curve["alpha"].max()):
        raise ValueError(f"Phase 3 {model} fold {fold} selected boundary alpha {alpha}")
    return alpha, curve


def dense_tune_cv5(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    curve_path: Path,
    max_extensions: int,
    resume: bool,
) -> Tuple[float, str, List[Dict[str, Any]], float]:
    if resume and curve_path.exists():
        cached = pd.read_csv(curve_path)
        selected = cached[cached["selected"].astype(bool)]
        if len(selected) == 1 and "boundary_status" in cached:
            extensions_path = curve_path.with_suffix(".extensions.json")
            extensions = (
                json.loads(extensions_path.read_text()) if extensions_path.exists() else []
            )
            return (
                float(selected.iloc[0]["alpha"]),
                str(selected.iloc[0]["boundary_status"]),
                extensions,
                0.0,
            )
    started = time.perf_counter()
    splitter = GroupKFold(n_splits=5)
    states = [
        _spectral_state(x, y, fit_idx, valid_idx)
        for fit_idx, valid_idx in splitter.split(x, y, groups=groups)
    ]
    grid = ALPHAS.copy()
    extensions: List[Dict[str, Any]] = []
    boundary = "interior"
    for extension_index in range(max_extensions + 1):
        fold_scores, mean_scores = _score_alphas(x, y, states, grid)
        tolerance = max(1e-10, abs(float(mean_scores.min())) * 1e-10)
        tied = np.flatnonzero(mean_scores <= float(mean_scores.min()) + tolerance)
        best_index = int(tied[-1])
        if best_index not in {0, len(grid) - 1}:
            boundary = "interior"
            break
        direction = "lower" if best_index == 0 else "upper"
        boundary = f"{direction}_boundary"
        if extension_index == max_extensions:
            break
        old_min, old_max = float(grid.min()), float(grid.max())
        grid = _extend_grid(grid, direction)
        extensions.append(
            {
                "extension": extension_index + 1,
                "direction": direction,
                "old_min": old_min,
                "old_max": old_max,
                "new_min": float(grid.min()),
                "new_max": float(grid.max()),
            }
        )
    curve = pd.DataFrame(
        {
            "alpha": grid,
            **{f"inner_fold{i + 1}_mae": fold_scores[i] for i in range(5)},
            "mean_inner_mae": mean_scores,
        }
    ).sort_values("alpha")
    alpha = float(grid[best_index])
    curve["selected"] = np.isclose(curve["alpha"], alpha)
    curve["boundary_status"] = boundary
    curve_path.parent.mkdir(parents=True, exist_ok=True)
    curve.to_csv(curve_path, index=False)
    curve_path.with_suffix(".extensions.json").write_text(
        json.dumps(extensions, indent=2), encoding="utf-8"
    )
    return alpha, boundary, extensions, time.perf_counter() - started


def fit_sparse_prediction(
    text_train: sp.csr_matrix,
    text_test: sp.csr_matrix,
    blocks_train: Mapping[str, np.ndarray],
    blocks_test: Mapping[str, np.ndarray],
    y_train: np.ndarray,
    alpha: float,
    tolerance: float,
    max_iterations: int,
) -> Tuple[np.ndarray, np.ndarray, Any]:
    operator = CenteredBlockOperator(
        text_train, blocks_train, {name: 1.0 for name in blocks_train}
    )
    fit = fit_operator(
        operator,
        y_train,
        alpha,
        tolerance=tolerance,
        max_iterations=max_iterations,
        x0=None,
    )
    train_prediction = operator.uncentered_prediction(fit.theta, float(y_train.mean()))
    test_prediction = predict_blocks(text_test, blocks_test, fit)
    return train_prediction, test_prediction, fit


def write_sparse_fold(
    harness: EvaluationHarness,
    model_id: str,
    fold: int,
    dataset: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    text_train: sp.csr_matrix,
    text_test: sp.csr_matrix,
    blocks_train: Mapping[str, np.ndarray],
    blocks_test: Mapping[str, np.ndarray],
    alpha: float,
    boundary: str,
    tuning_seconds: float,
    args: argparse.Namespace,
    extra: Mapping[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    y = dataset["rating_overall"].to_numpy(dtype=np.float64)
    if args.resume and harness.has_fold(model_id, fold):
        artifact = harness.load_fold(model_id, fold)
        pred = pd.read_parquet(artifact.prediction_path, columns=["y_pred_raw"])
        log(f"{model_id} fold={fold}: prediction cache hit")
        # Training predictions are not persisted; callers that need them refit.
        return np.empty(0), pred["y_pred_raw"].to_numpy(dtype=np.float64)
    with PeakRSSMonitor() as memory:
        _, prediction, fit = fit_sparse_prediction(
            text_train,
            text_test,
            blocks_train,
            blocks_test,
            y[train_idx],
            alpha,
            args.tolerance,
            args.max_iterations,
        )
    harness.write_fold(
        model_id=model_id,
        fold=fold,
        row_id=dataset.iloc[test_idx]["row_id"].to_numpy(),
        offering_id=dataset.iloc[test_idx]["offering_id"].to_numpy(),
        y_true=y[test_idx],
        y_pred_raw=prediction,
        fit_seconds=fit.fit_seconds,
        predict_seconds=0.0,
        peak_memory_mb=memory.peak_mb,
        selected_hyperparameters={
            "alpha": alpha,
            "alpha_grid": ALPHAS.tolist(),
            "inner_cv": "GroupKFold(5), group=offering_id",
            "boundary_status": boundary,
            "shared_penalty": True,
        },
        tuning_seconds=tuning_seconds,
        extra={
            **dict(extra),
            "n_features": int(text_train.shape[1] + sum(x.shape[1] for x in blocks_train.values())),
            "iterations": int(fit.iterations),
            "stop_code": int(fit.stop_code),
            "split_sha256": SPLIT_SHA256,
        },
    )
    metrics = harness.load_fold(model_id, fold).record["clipped"]
    log(
        f"{model_id} fold={fold} alpha={alpha:g} boundary={boundary} "
        f"MAE={metrics['mae']:.6f} RMSE={metrics['rmse']:.6f}"
    )
    return np.empty(0), prediction


def write_dense_fold(
    harness: EvaluationHarness,
    model_id: str,
    fold: int,
    dataset: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    x_train: np.ndarray,
    x_test: np.ndarray,
    groups_train: np.ndarray,
    curve_path: Path,
    args: argparse.Namespace,
    extra: Mapping[str, Any],
) -> None:
    if args.resume and harness.has_fold(model_id, fold):
        log(f"{model_id} fold={fold}: cache hit")
        return
    y = dataset["rating_overall"].to_numpy(dtype=np.float64)
    alpha, boundary, extensions, tuning_seconds = dense_tune_cv5(
        x_train,
        y[train_idx],
        groups_train,
        curve_path,
        args.max_extensions,
        args.resume,
    )
    with PeakRSSMonitor() as memory:
        coefficients, intercept, fit_seconds = fit_dense_ridge(
            x_train, y[train_idx], alpha
        )
        prediction, predict_seconds = predict_dense(x_test, coefficients, intercept)
    harness.write_fold(
        model_id=model_id,
        fold=fold,
        row_id=dataset.iloc[test_idx]["row_id"].to_numpy(),
        offering_id=dataset.iloc[test_idx]["offering_id"].to_numpy(),
        y_true=y[test_idx],
        y_pred_raw=prediction,
        fit_seconds=fit_seconds,
        predict_seconds=predict_seconds,
        peak_memory_mb=memory.peak_mb,
        tuning_seconds=tuning_seconds,
        selected_hyperparameters={
            "alpha": alpha,
            "alpha_grid_initial": ALPHAS.tolist(),
            "inner_cv": "GroupKFold(5), group=offering_id",
            "boundary_status": boundary,
            "boundary_extensions": extensions,
        },
        extra={
            **dict(extra),
            "n_features": int(x_train.shape[1]),
            "split_sha256": SPLIT_SHA256,
        },
    )
    metrics = harness.load_fold(model_id, fold).record["clipped"]
    log(
        f"{model_id} fold={fold} alpha={alpha:g} boundary={boundary} "
        f"MAE={metrics['mae']:.6f} RMSE={metrics['rmse']:.6f}"
    )


def run_primary(args: argparse.Namespace, folds: Sequence[int]) -> None:
    output_dir = ROOT / "outputs"
    phase3 = output_dir / "phase3"
    phase4 = output_dir / "phase4"
    phase4.mkdir(parents=True, exist_ok=True)
    harness = EvaluationHarness(output_dir)
    dataset = load_dataset(output_dir)
    splits = load_splits(output_dir)
    raw = load_components(output_dir, args.profile)
    y = dataset["rating_overall"].to_numpy(dtype=np.float64)
    groups = dataset["offering_id"].to_numpy()
    rating_cols = [f"rating_{aspect}" for aspect in ASPECTS]

    for fold in tqdm(folds, desc="Phase4 U1 primary folds"):
        train_idx = splits[fold]["train_idx"]
        test_idx = splits[fold]["test_idx"]
        train_path, test_path = tfidf_paths(
            output_dir, args.profile, args.max_features, fold
        )
        started = time.perf_counter()
        text_train = sp.load_npz(train_path).tocsr()
        text_test = sp.load_npz(test_path).tocsr()
        load_seconds = time.perf_counter() - started
        blocks_train, blocks_test = prepare_blocks(raw, train_idx, test_idx)

        for family, names in (
            ("text_only", []),
            ("simple_fusion", ["z"]),
            ("interaction_fusion", ["z", "p_s", "p_b"]),
        ):
            model_id = PRIMARY_IDS[family]
            alpha, _ = selected_g3_alpha(phase3, family, fold)
            write_sparse_fold(
                harness,
                model_id,
                fold,
                dataset,
                train_idx,
                test_idx,
                text_train,
                text_test,
                {name: blocks_train[name] for name in names},
                {name: blocks_test[name] for name in names},
                alpha,
                "interior",
                0.0,
                args,
                {
                    "experiment": "U1",
                    "variant": family,
                    "selection_curve_source": str(
                        phase3 / "g3_inner_cv5" / f"{family}_outerfold{fold}.csv"
                    ),
                    "tfidf_load_seconds": load_seconds,
                    "deployable": True,
                },
            )

        # LLM aspects only: all dense columns standardized on the outer train fold.
        write_dense_fold(
            harness,
            PRIMARY_IDS["llm_aspects_only"],
            fold,
            dataset,
            train_idx,
            test_idx,
            blocks_train["z"],
            blocks_test["z"],
            groups[train_idx],
            phase4 / "u0_cv5" / f"llm_aspects_only_fold{fold}.csv",
            args,
            {"experiment": "U1", "variant": "llm_aspects_only", "deployable": True},
        )

        # True-aspect upper bound, including explicit missingness indicators.
        imputer = SimpleImputer(strategy="median", add_indicator=True)
        upper_train = imputer.fit_transform(
            dataset.iloc[train_idx][rating_cols].to_numpy(dtype=np.float64)
        )
        upper_test = imputer.transform(
            dataset.iloc[test_idx][rating_cols].to_numpy(dtype=np.float64)
        )
        upper_scaler = StandardScaler()
        upper_train = upper_scaler.fit_transform(upper_train).astype(np.float32)
        upper_test = upper_scaler.transform(upper_test).astype(np.float32)
        write_dense_fold(
            harness,
            PRIMARY_IDS["upper_bound"],
            fold,
            dataset,
            train_idx,
            test_idx,
            upper_train,
            upper_test,
            groups[train_idx],
            phase4 / "u0_cv5" / f"upper_bound_fold{fold}.csv",
            args,
            {
                "experiment": "U1",
                "variant": "upper_bound_true_aspects",
                "imputer": "median, add_indicator=True",
                "deployable": False,
            },
        )

        # Residual stack: U0-selected text base plus a CV5-selected dense residual model.
        residual_id = PRIMARY_IDS["residual_on_llm"]
        if not (args.resume and harness.has_fold(residual_id, fold)):
            text_alpha, _ = selected_g3_alpha(phase3, "text_only", fold)
            with PeakRSSMonitor() as memory:
                text_train_pred, text_test_pred, text_fit = fit_sparse_prediction(
                    text_train,
                    text_test,
                    {},
                    {},
                    y[train_idx],
                    text_alpha,
                    args.tolerance,
                    args.max_iterations,
                )
                residual_target = y[train_idx] - text_train_pred
                residual_alpha, boundary, extensions, tuning_seconds = dense_tune_cv5(
                    blocks_train["z"],
                    residual_target,
                    groups[train_idx],
                    phase4 / "u0_cv5" / f"residual_on_llm_fold{fold}.csv",
                    args.max_extensions,
                    args.resume,
                )
                coefficients, intercept, residual_fit_seconds = fit_dense_ridge(
                    blocks_train["z"], residual_target, residual_alpha
                )
                correction, residual_predict_seconds = predict_dense(
                    blocks_test["z"], coefficients, intercept
                )
                prediction = text_test_pred + correction
            harness.write_fold(
                model_id=residual_id,
                fold=fold,
                row_id=dataset.iloc[test_idx]["row_id"].to_numpy(),
                offering_id=groups[test_idx],
                y_true=y[test_idx],
                y_pred_raw=prediction,
                fit_seconds=text_fit.fit_seconds + residual_fit_seconds,
                predict_seconds=residual_predict_seconds,
                peak_memory_mb=memory.peak_mb,
                tuning_seconds=tuning_seconds,
                selected_hyperparameters={
                    "text_alpha": text_alpha,
                    "residual_alpha": residual_alpha,
                    "alpha_grid_initial": ALPHAS.tolist(),
                    "inner_cv": "GroupKFold(5), group=offering_id",
                    "boundary_status": boundary,
                    "boundary_extensions": extensions,
                },
                extra={
                    "experiment": "U1",
                    "variant": "residual_on_llm",
                    "n_features": int(text_train.shape[1] + blocks_train["z"].shape[1]),
                    "deployable": True,
                    "split_sha256": SPLIT_SHA256,
                    "residual_training_note": "dense residual target uses outer-training text fitted values; outer test is untouched",
                },
            )
            metrics = harness.load_fold(residual_id, fold).record["clipped"]
            log(
                f"{residual_id} fold={fold} text_alpha={text_alpha:g} "
                f"residual_alpha={residual_alpha:g} MAE={metrics['mae']:.6f}"
            )
        else:
            log(f"{residual_id} fold={fold}: cache hit")

        del text_train, text_test, blocks_train, blocks_test, upper_train, upper_test
        gc.collect()

    if set(folds) == set(range(1, 6)):
        metadata = {
            "protocol": "Phase 4 U0 unified GroupKFold(5)",
            "alpha_grid": ALPHAS.tolist(),
            "boundary_extension": "two decades until interior",
            "split_assignment_sha256": SPLIT_SHA256,
            "seed": SEED,
        }
        for model_id in PRIMARY_IDS.values():
            harness.finalize_model(model_id, {**metadata, "model_id": model_id})


def inner_curve_complete(path: Path, alphas: Sequence[float]) -> bool:
    if not path.exists():
        return False
    found = pd.read_csv(path)["alpha"].to_numpy(dtype=float)
    return all(np.any(np.isclose(found, alpha, rtol=1e-10)) for alpha in alphas)


def run_sparse_inner_curve(
    label: str,
    outer_fold: int,
    inner_fold: int,
    text: sp.csr_matrix,
    blocks: Mapping[str, np.ndarray],
    y: np.ndarray,
    fit_pos: np.ndarray,
    valid_pos: np.ndarray,
    alphas: Sequence[float],
    path: Path,
    args: argparse.Namespace,
) -> pd.DataFrame:
    if args.resume and inner_curve_complete(path, alphas):
        return pd.read_csv(path)
    records = pd.read_csv(path).to_dict(orient="records") if args.resume and path.exists() else []
    completed = np.asarray([row["alpha"] for row in records], dtype=float)
    selected_fit = {name: matrix[fit_pos] for name, matrix in blocks.items()}
    selected_valid = {name: matrix[valid_pos] for name, matrix in blocks.items()}
    operator = CenteredBlockOperator(
        text[fit_pos].tocsr(), selected_fit, {name: 1.0 for name in selected_fit}
    )
    # Descending alpha values form a stable continuation path.  Reusing the
    # previous solution materially reduces LSQR iterations for the 41-point U0
    # grid while preserving the same objective and stopping tolerance.  On a
    # resumed partial curve, the first missing alpha starts cold and subsequent
    # values warm-start from it; no unverifiable coefficient state is assumed.
    previous_theta = None
    for alpha in np.asarray(alphas, dtype=float)[::-1]:
        if completed.size and np.any(np.isclose(completed, alpha, rtol=1e-10)):
            continue
        fit = fit_operator(
            operator,
            y[fit_pos],
            float(alpha),
            tolerance=args.tolerance,
            max_iterations=args.max_iterations,
            x0=previous_theta,
        )
        previous_theta = fit.theta
        prediction = predict_blocks(text[valid_pos].tocsr(), selected_valid, fit)
        records.append(
            {
                "model": label,
                "outer_fold": outer_fold,
                "inner_fold": inner_fold,
                "alpha": float(alpha),
                "inner_mae": float(mean_absolute_error(y[valid_pos], prediction)),
                "inner_mae_clipped": float(
                    mean_absolute_error(y[valid_pos], np.clip(prediction, 1.0, 5.0))
                ),
                "fit_seconds": float(fit.fit_seconds),
                "iterations": int(fit.iterations),
                "stop_code": int(fit.stop_code),
                "n_fit": int(len(fit_pos)),
                "n_valid": int(len(valid_pos)),
            }
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(records).sort_values("alpha").to_csv(path, index=False)
        log(
            f"U4 tune {label} outer={outer_fold} inner={inner_fold} "
            f"alpha={alpha:g} iter={fit.iterations}"
        )
    return pd.DataFrame(records).sort_values("alpha")


def sparse_tune_cv5(
    label: str,
    outer_fold: int,
    text: sp.csr_matrix,
    blocks: Mapping[str, np.ndarray],
    y: np.ndarray,
    groups: np.ndarray,
    phase4: Path,
    args: argparse.Namespace,
) -> Tuple[float, str, List[Dict[str, Any]], float]:
    aggregate_path = phase4 / "u4_cv5" / f"{label}_outerfold{outer_fold}.csv"
    if args.resume and aggregate_path.exists():
        cached = pd.read_csv(aggregate_path)
        selected = cached[cached["selected"].astype(bool)]
        if len(selected) == 1:
            extensions_path = aggregate_path.with_suffix(".extensions.json")
            extensions = json.loads(extensions_path.read_text()) if extensions_path.exists() else []
            return float(selected.iloc[0]["alpha"]), str(selected.iloc[0]["boundary_status"]), extensions, 0.0
    started = time.perf_counter()
    splits = list(GroupKFold(n_splits=5).split(text, y, groups=groups))
    grid = ALPHAS.copy()
    extensions: List[Dict[str, Any]] = []
    boundary = "interior"
    for extension_index in range(args.max_extensions + 1):
        workers = max(1, min(args.inner_jobs, 5))
        frames: List[pd.DataFrame] = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    run_sparse_inner_curve,
                    label,
                    outer_fold,
                    inner_fold,
                    text,
                    blocks,
                    y,
                    fit_pos,
                    valid_pos,
                    grid,
                    phase4 / "u4_cv5" / "details" / f"{label}_outerfold{outer_fold}_innerfold{inner_fold}.csv",
                    args,
                )
                for inner_fold, (fit_pos, valid_pos) in enumerate(splits, start=1)
            ]
            for future in as_completed(futures):
                frames.append(future.result())
        detail = pd.concat(frames, ignore_index=True)
        detail = detail[detail["alpha"].apply(lambda value: np.any(np.isclose(grid, value, rtol=1e-10)))]
        aggregate = (
            detail.groupby("alpha", as_index=False)
            .agg(
                mean_inner_mae=("inner_mae", "mean"),
                sd_inner_mae=("inner_mae", "std"),
                mean_inner_mae_clipped=("inner_mae_clipped", "mean"),
                fit_seconds_total=("fit_seconds", "sum"),
            )
            .sort_values("alpha")
        )
        best = aggregate.sort_values(["mean_inner_mae", "alpha"]).iloc[0]
        alpha = float(best["alpha"])
        if not (np.isclose(alpha, grid.min()) or np.isclose(alpha, grid.max())):
            boundary = "interior"
            break
        direction = "lower" if np.isclose(alpha, grid.min()) else "upper"
        boundary = f"{direction}_boundary"
        if extension_index == args.max_extensions:
            break
        old_min, old_max = float(grid.min()), float(grid.max())
        grid = _extend_grid(grid, direction)
        extensions.append(
            {
                "extension": extension_index + 1,
                "direction": direction,
                "old_min": old_min,
                "old_max": old_max,
                "new_min": float(grid.min()),
                "new_max": float(grid.max()),
            }
        )
    aggregate["selected"] = np.isclose(aggregate["alpha"], alpha)
    aggregate["boundary_status"] = boundary
    aggregate["outer_fold"] = outer_fold
    aggregate["model"] = label
    aggregate_path.parent.mkdir(parents=True, exist_ok=True)
    aggregate.to_csv(aggregate_path, index=False)
    aggregate_path.with_suffix(".extensions.json").write_text(
        json.dumps(extensions, indent=2), encoding="utf-8"
    )
    return alpha, boundary, extensions, time.perf_counter() - started


def write_known_reference(
    harness: EvaluationHarness,
    dataset: pd.DataFrame,
    known: np.ndarray,
    fold: int,
    args: argparse.Namespace,
) -> None:
    model_id = ABLATION_IDS["A4_known_reference"]
    if args.resume and harness.has_fold(model_id, fold):
        return
    source = harness.load_fold(PRIMARY_IDS["interaction_fusion"], fold)
    frame = pd.read_parquet(source.prediction_path)
    keep_ids = set(np.flatnonzero(known).tolist())
    frame = frame[frame["row_id"].isin(keep_ids)].copy()
    harness.write_fold(
        model_id=model_id,
        fold=fold,
        row_id=frame["row_id"].to_numpy(),
        offering_id=frame["offering_id"].to_numpy(),
        y_true=frame["y_true"].to_numpy(),
        y_pred_raw=frame["y_pred_raw"].to_numpy(),
        fit_seconds=0.0,
        predict_seconds=0.0,
        peak_memory_mb=0.0,
        selected_hyperparameters=source.record["selected_hyperparameters"],
        extra={
            "experiment": "U4",
            "variant": "A4 matched known-segment evaluation reference",
            "source_model": PRIMARY_IDS["interaction_fusion"],
            "deployable": True,
        },
    )


def run_ablations(args: argparse.Namespace, folds: Sequence[int]) -> None:
    output_dir = ROOT / "outputs"
    phase4 = output_dir / "phase4"
    phase4.mkdir(parents=True, exist_ok=True)
    harness = EvaluationHarness(output_dir)
    dataset = load_dataset(output_dir)
    splits = load_splits(output_dir)
    raw = load_components(output_dir, args.profile)
    y_all = dataset["rating_overall"].to_numpy(dtype=np.float64)
    groups_all = dataset["offering_id"].to_numpy()
    unknown_start = (len(SEGMENTS) - 1) * len(ASPECTS)
    selected_variants = parse_variants(args.variants)

    for fold in tqdm(folds, desc="Phase4 U4 ablation folds"):
        train_idx = splits[fold]["train_idx"]
        test_idx = splits[fold]["test_idx"]
        train_path, test_path = tfidf_paths(output_dir, args.profile, args.max_features, fold)
        text_train = sp.load_npz(train_path).tocsr()
        text_test = sp.load_npz(test_path).tocsr()
        blocks_train, blocks_test = prepare_blocks(raw, train_idx, test_idx)

        variants: List[Tuple[str, str, Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        if "A2" in selected_variants:
            variants.append((
                "A2", ABLATION_IDS["A2"],
                {name: blocks_train[name] for name in ("z", "p_s")},
                {name: blocks_test[name] for name in ("z", "p_s")},
                text_train, text_test, train_idx, test_idx,
            ))
        if "A3" in selected_variants:
            variants.append((
                "A3", ABLATION_IDS["A3"],
                {name: blocks_train[name] for name in ("z", "p_b")},
                {name: blocks_test[name] for name in ("z", "p_b")},
                text_train, text_test, train_idx, test_idx,
            ))
        if "A5" in selected_variants:
            variants.append((
                "A5", ABLATION_IDS["A5"],
                {name: blocks_train[name] for name in ("p_s", "p_b")},
                {name: blocks_test[name] for name in ("p_s", "p_b")},
                text_train, text_test, train_idx, test_idx,
            ))
        if "A7" in selected_variants:
            a7_train = {
                "z": blocks_train["z"][:, :-1],
                "p_s": np.delete(blocks_train["p_s"], np.s_[unknown_start : unknown_start + len(ASPECTS)], axis=1),
                "p_b": np.delete(blocks_train["p_b"], np.s_[unknown_start : unknown_start + len(ASPECTS)], axis=1),
            }
            a7_test = {
                "z": blocks_test["z"][:, :-1],
                "p_s": np.delete(blocks_test["p_s"], np.s_[unknown_start : unknown_start + len(ASPECTS)], axis=1),
                "p_b": np.delete(blocks_test["p_b"], np.s_[unknown_start : unknown_start + len(ASPECTS)], axis=1),
            }
            variants.append((
                "A7", ABLATION_IDS["A7"], a7_train, a7_test,
                text_train, text_test, train_idx, test_idx,
            ))
        if "A8" in selected_variants:
            known_train_mask = raw["known"][train_idx]
            known_test_mask = raw["known"][test_idx]
            known_train_idx = train_idx[known_train_mask]
            known_test_idx = test_idx[known_test_mask]
            known_train_blocks, known_test_blocks = prepare_blocks(raw, known_train_idx, known_test_idx)
            variants.append((
                "A8", ABLATION_IDS["A8"],
                {name: known_train_blocks[name] for name in ("z", "p_s", "p_b")},
                {name: known_test_blocks[name] for name in ("z", "p_s", "p_b")},
                text_train[known_train_mask].tocsr(), text_test[known_test_mask].tocsr(),
                known_train_idx, known_test_idx,
            ))

        for label, model_id, train_blocks, test_blocks, selected_text_train, selected_text_test, selected_train_idx, selected_test_idx in variants:
            if args.resume and harness.has_fold(model_id, fold):
                log(f"{model_id} fold={fold}: cache hit")
                continue
            alpha, boundary, extensions, tuning_seconds = sparse_tune_cv5(
                label,
                fold,
                selected_text_train,
                train_blocks,
                y_all[selected_train_idx],
                groups_all[selected_train_idx],
                phase4,
                args,
            )
            write_sparse_fold(
                harness,
                model_id,
                fold,
                dataset,
                selected_train_idx,
                selected_test_idx,
                selected_text_train,
                selected_text_test,
                train_blocks,
                test_blocks,
                alpha,
                boundary,
                tuning_seconds,
                args,
                {
                    "experiment": "U4",
                    "variant": label,
                    "boundary_extensions": extensions,
                    "deployable": True,
                    "n_train": int(len(selected_train_idx)),
                    "n_test": int(len(selected_test_idx)),
                },
            )
        if "A8" in selected_variants:
            write_known_reference(harness, dataset, raw["known"], fold, args)
        del text_train, text_test, blocks_train, blocks_test, variants
        gc.collect()

    if set(folds) == set(range(1, 6)):
        metadata = {
            "protocol": "Phase 4 U0 unified GroupKFold(5)",
            "alpha_grid": ALPHAS.tolist(),
            "boundary_extension": "two decades until interior",
            "split_assignment_sha256": SPLIT_SHA256,
            "seed": SEED,
        }
        finalize_ids = [ABLATION_IDS[label] for label in selected_variants]
        if "A8" in selected_variants:
            finalize_ids.append(ABLATION_IDS["A4_known_reference"])
        for model_id in finalize_ids:
            harness.finalize_model(model_id, {**metadata, "model_id": model_id})


def main() -> None:
    args = parse_args()
    folds = parse_folds(args.folds)
    variants = parse_variants(args.variants) if args.stage in {"ablations", "all"} else []
    log(
        f"Phase 4 stage={args.stage} folds={folds} alpha_grid=logspace(-4,6,41) "
        f"inner_cv=GroupKFold(5) inner_jobs={args.inner_jobs} resume={args.resume} "
        f"variants={variants}"
    )
    if args.stage in {"primary", "all"}:
        run_primary(args, folds)
    if args.stage in {"ablations", "all"}:
        run_ablations(args, folds)


if __name__ == "__main__":
    main()
