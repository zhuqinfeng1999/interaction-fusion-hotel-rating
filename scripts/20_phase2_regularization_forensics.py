#!/usr/bin/env python3
"""Phase 2 Experiment R: sparse regularization and block-penalty forensics."""

from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import psutil
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.aspect_extractor_base import ASPECTS  # noqa: E402
from src.evaluation_harness import (  # noqa: E402
    EvaluationHarness,
    PeakRSSMonitor,
    bootstrap_mean_difference,
    paired_wilcoxon,
    write_json,
)
from src.phase2_sparse_ridge import (  # noqa: E402
    CenteredBlockOperator,
    fit_operator,
    grouped_cv_configurations,
    lambda_configurations,
    predict_blocks,
    tune_alpha_grouped,
    tune_alpha_two_stage_grouped,
)


SEED = 42
N_SPLITS = 5
SEGMENTS = ["business", "couple", "family", "friends", "solo", "unknown"]
LAMBDA_VALUES = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]


def log(message: str) -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="qwen25_7b")
    parser.add_argument("--folds", default="1,2,3,4,5")
    parser.add_argument("--max-features", type=int, default=200_000)
    parser.add_argument("--coarse-rows", type=int, default=200_000)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--max-iterations", type=int, default=2_000)
    parser.add_argument("--alpha-screen-max-iterations", type=int, default=250)
    parser.add_argument("--alpha-refine-half-decades", type=float, default=0.5)
    parser.add_argument("--inner-jobs", type=int, default=3)
    parser.add_argument("--lambda-coordinate-passes", type=int, default=2)
    parser.add_argument(
        "--mode",
        choices=["projection", "r0", "r1", "r2", "all"],
        default="projection",
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def parse_folds(raw: str) -> List[int]:
    folds = sorted({int(item.strip()) for item in raw.split(",") if item.strip()})
    if not folds or any(fold < 1 or fold > N_SPLITS for fold in folds):
        raise ValueError("--folds must contain values in 1..5")
    return folds


def load_splits(output_dir: Path) -> List[Dict[str, Any]]:
    rows = []
    for fold in range(1, N_SPLITS + 1):
        path = output_dir / f"splits_fold{fold}.npz"
        arrays = np.load(path)
        rows.append(
            {
                "fold": fold,
                "train_idx": arrays["train_idx"].astype(np.int64),
                "test_idx": arrays["test_idx"].astype(np.int64),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    return rows


def tfidf_paths(
    output_dir: Path,
    profile: str,
    max_features: int,
    fold: int,
) -> Tuple[Path, Path]:
    pattern = (
        f"{profile}_mf{max_features}_ng12_md3_full_fold{fold}_*_Xtr.npz"
    )
    train_matches = sorted((output_dir / "tfidf_cache").glob(pattern))
    if len(train_matches) != 1:
        raise FileNotFoundError(
            f"expected one TF-IDF train cache for fold {fold}, got {train_matches}"
        )
    test_path = Path(str(train_matches[0]).replace("_Xtr.npz", "_Xte.npz"))
    if not test_path.exists():
        raise FileNotFoundError(test_path)
    return train_matches[0], test_path


def load_raw_structured(output_dir: Path, profile: str) -> Dict[str, np.ndarray]:
    score_cols = [f"aspect_{aspect}_score" for aspect in ASPECTS]
    mention_cols = [f"aspect_{aspect}_mentioned" for aspect in ASPECTS]
    frame = pd.read_parquet(
        output_dir / f"aspects_llm_{profile}.parquet",
        columns=[
            "row_id",
            "ok",
            "traveler_segment",
            *score_cols,
            *mention_cols,
        ],
    ).sort_values("row_id")
    if not np.array_equal(frame["row_id"], np.arange(len(frame))):
        raise ValueError("aspect row_id alignment failed")
    failed = ~frame["ok"].eq(1)
    frame.loc[failed, score_cols] = 0
    frame.loc[failed, mention_cols] = False
    frame.loc[failed, "traveler_segment"] = "unknown"
    scores = (
        frame[score_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .to_numpy(dtype=np.float32)
    )
    mentions = frame[mention_cols].fillna(False).to_numpy(dtype=np.float32)
    segment = frame["traveler_segment"].fillna("unknown").astype(str).str.lower()
    segment = segment.where(segment.isin(SEGMENTS), "unknown")
    segments = (
        pd.get_dummies(segment)
        .reindex(columns=SEGMENTS, fill_value=False)
        .to_numpy(dtype=np.float32)
    )
    score_interactions = np.hstack(
        [scores * segments[:, index : index + 1] for index in range(len(SEGMENTS))]
    ).astype(np.float32)
    mention_interactions = np.hstack(
        [
            mentions * segments[:, index : index + 1]
            for index in range(len(SEGMENTS))
        ]
    ).astype(np.float32)
    return {
        "z": np.hstack([scores, mentions, segments]).astype(np.float32),
        "p_s": score_interactions,
        "p_b": mention_interactions,
    }


def prepare_fold_blocks(
    raw: Mapping[str, np.ndarray],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, Any]]:
    train: Dict[str, np.ndarray] = {}
    test: Dict[str, np.ndarray] = {}
    scalers: Dict[str, Any] = {}
    for name, matrix in raw.items():
        scaler = StandardScaler()
        train[name] = scaler.fit_transform(matrix[train_idx]).astype(np.float32)
        test[name] = scaler.transform(matrix[test_idx]).astype(np.float32)
        scalers[name] = {
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
        }
    return train, test, scalers


def hotel_preserved_subsample(
    path: Path,
    groups: np.ndarray,
    target_rows: int,
    seed: int,
) -> np.ndarray:
    if path.exists():
        return np.load(path)["train_positions"].astype(np.int64)
    unique, counts = np.unique(groups, return_counts=True)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(unique))
    selected: List[Any] = []
    rows = 0
    for position in order:
        selected.append(unique[position])
        rows += int(counts[position])
        if rows >= target_rows:
            break
    mask = np.isin(groups, np.asarray(selected))
    positions = np.flatnonzero(mask).astype(np.int64)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        train_positions=positions,
        selected_hotels=np.asarray(selected),
        target_rows=np.asarray([target_rows]),
        actual_rows=np.asarray([len(positions)]),
        seed=np.asarray([seed]),
    )
    return positions


def block_diagnostics(
    fold: int,
    text_train: sp.csr_matrix,
    blocks_train: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    text_column_norms = np.sqrt(
        np.asarray(text_train.power(2).sum(axis=0)).ravel()
    )
    rows = [
        {
            "fold": fold,
            "block": "text",
            "n_rows": int(text_train.shape[0]),
            "n_columns": int(text_train.shape[1]),
            "frobenius_norm": float(np.linalg.norm(text_train.data)),
            "mean_column_norm": float(text_column_norms.mean()),
        }
    ]
    for name, matrix in blocks_train.items():
        column_norms = np.linalg.norm(matrix, axis=0)
        rows.append(
            {
                "fold": fold,
                "block": name,
                "n_rows": int(matrix.shape[0]),
                "n_columns": int(matrix.shape[1]),
                "frobenius_norm": float(np.linalg.norm(matrix)),
                "mean_column_norm": float(column_norms.mean()),
            }
        )
    frame = pd.DataFrame(rows)
    text_frob = float(frame.loc[frame["block"].eq("text"), "frobenius_norm"].iloc[0])
    text_mean = float(frame.loc[frame["block"].eq("text"), "mean_column_norm"].iloc[0])
    frame["text_to_block_frobenius_ratio"] = text_frob / frame["frobenius_norm"]
    frame["text_to_block_mean_column_ratio"] = (
        text_mean / frame["mean_column_norm"]
    )
    return frame


def curve_metadata_path(curve_path: Path) -> Path:
    return curve_path.with_suffix(".meta.json")


def tune_alpha_cached(
    args: argparse.Namespace,
    model_id: str,
    fold: int,
    text: sp.csr_matrix,
    blocks: Mapping[str, np.ndarray],
    y: np.ndarray,
    groups: np.ndarray,
    lambdas: Mapping[str, float],
    coarse_positions: np.ndarray,
    output_dir: Path,
) -> Dict[str, Any]:
    curve_path = (
        output_dir / "phase2_alpha_curves" / f"{model_id}_fold{fold}.csv"
    )
    metadata_path = curve_metadata_path(curve_path)
    protocol_version = "approved_reduced_v1_coarse57_local_full"
    if args.resume and curve_path.exists() and metadata_path.exists():
        cached = json.loads(metadata_path.read_text(encoding="utf-8"))
        if cached.get("protocol_version") == protocol_version:
            return cached
    started = time.perf_counter()
    alpha, curve, boundary, extensions = tune_alpha_two_stage_grouped(
        text[coarse_positions].tocsr(),
        {name: matrix[coarse_positions] for name, matrix in blocks.items()},
        y[coarse_positions],
        groups[coarse_positions],
        text,
        blocks,
        y,
        groups,
        lambdas,
        curve_path,
        tolerance=args.tolerance,
        screen_max_iterations=args.alpha_screen_max_iterations,
        max_iterations=args.max_iterations,
        refine_half_decades=args.alpha_refine_half_decades,
        log_prefix=f"{model_id} fold {fold}",
        n_jobs=args.inner_jobs,
    )
    metadata = {
        "protocol_version": protocol_version,
        "protocol": (
            "57-point alpha screen on deterministic hotel-preserved coarse "
            "sample, followed by full-training local refinement with boundary "
            "expansion"
        ),
        "alpha": alpha,
        "boundary_status": boundary,
        "extensions": extensions,
        "curve_path": str(curve_path.relative_to(ROOT)),
        "tuning_seconds": time.perf_counter() - started,
        "n_configurations": int(len(curve)),
        "n_screen_configurations": int(curve["stage"].eq("coarse_screen").sum()),
        "n_full_refine_configurations": int(
            curve["stage"].eq("full_refine").sum()
        ),
        "screen_selection_rows": int(len(coarse_positions)),
        "full_refine_rows": int(len(y)),
        "inner_jobs": int(args.inner_jobs),
        "screen_max_iterations": int(args.alpha_screen_max_iterations),
        "full_max_iterations": int(args.max_iterations),
    }
    write_json(metadata_path, metadata)
    return metadata


def lambda_tune_cached(
    args: argparse.Namespace,
    model_id: str,
    fold: int,
    text: sp.csr_matrix,
    blocks: Mapping[str, np.ndarray],
    y: np.ndarray,
    groups: np.ndarray,
    alpha: float,
    output_dir: Path,
) -> Dict[str, Any]:
    path = output_dir / "phase2" / "lambda_curves" / f"{model_id}_fold{fold}.csv"
    metadata_path = path.with_suffix(".meta.json")
    protocol_version = "approved_reduced_v1_coordinate2_local_cartesian"
    if args.resume and path.exists() and metadata_path.exists():
        cached = json.loads(metadata_path.read_text(encoding="utf-8"))
        if cached.get("protocol_version") == protocol_version:
            return cached
    started = time.perf_counter()
    block_names = list(blocks)
    current = {name: 1.0 for name in block_names}
    stage_curves: List[pd.DataFrame] = []

    # Every candidate value is tested for every block in each coordinate pass.
    # The path starts near the current value to preserve effective warm starts.
    passes = 1 if len(block_names) == 1 else args.lambda_coordinate_passes
    for pass_index in range(1, passes + 1):
        for block_name in block_names:
            ordered_values = sorted(
                LAMBDA_VALUES,
                key=lambda value: (
                    abs(np.log10(value) - np.log10(current[block_name])),
                    value,
                ),
            )
            configurations = []
            for value in ordered_values:
                candidate = dict(current)
                candidate[block_name] = float(value)
                configurations.append((float(alpha), candidate))
            stage = grouped_cv_configurations(
                text,
                blocks,
                y,
                groups,
                configurations,
                args.tolerance,
                args.max_iterations,
                log_prefix=(
                    f"{model_id} fold {fold} lambda pass {pass_index} "
                    f"block {block_name}"
                ),
                n_jobs=args.inner_jobs,
            ).copy()
            stage["stage"] = f"coordinate_pass_{pass_index}"
            stage["coordinate_block"] = block_name
            stage_curves.append(stage)
            best = stage.iloc[0]
            current = {
                name: float(best[f"lambda_{name}"]) for name in block_names
            }

    # Refine the coordinate solution over the selected value and its immediate
    # neighbours for every block (at most 3^3=27 configurations).
    local_values: Dict[str, List[float]] = {}
    for name in block_names:
        center_index = min(
            range(len(LAMBDA_VALUES)),
            key=lambda index: abs(
                np.log10(LAMBDA_VALUES[index]) - np.log10(current[name])
            ),
        )
        local_values[name] = [
            float(LAMBDA_VALUES[index])
            for index in range(
                max(0, center_index - 1),
                min(len(LAMBDA_VALUES), center_index + 2),
            )
        ]
    local_candidates = []
    for combination in itertools.product(
        *(local_values[name] for name in block_names)
    ):
        candidate = {
            name: float(value)
            for name, value in zip(block_names, combination)
        }
        local_candidates.append(candidate)
    local_candidates.sort(
        key=lambda candidate: sum(
            (
                np.log10(candidate[name]) - np.log10(current[name])
            )
            ** 2
            for name in block_names
        )
    )
    local_curve = grouped_cv_configurations(
        text,
        blocks,
        y,
        groups,
        [(float(alpha), candidate) for candidate in local_candidates],
        args.tolerance,
        args.max_iterations,
        log_prefix=f"{model_id} fold {fold} lambda local Cartesian",
        n_jobs=args.inner_jobs,
    ).copy()
    local_curve["stage"] = "local_cartesian_refinement"
    local_curve["coordinate_block"] = "all"
    stage_curves.append(local_curve)
    best = local_curve.iloc[0]
    lambdas = {
        name: float(best[f"lambda_{name}"]) for name in block_names
    }
    curve = pd.concat(stage_curves, ignore_index=True)
    curve["selected"] = np.logical_and.reduce(
        [
            np.isclose(curve[f"lambda_{name}"], value)
            for name, value in lambdas.items()
        ]
    ) & curve["stage"].eq("local_cartesian_refinement")
    path.parent.mkdir(parents=True, exist_ok=True)
    curve.to_csv(path, index=False)
    metadata = {
        "protocol_version": protocol_version,
        "alpha_held_fixed_during_lambda_search": float(alpha),
        "lambdas": lambdas,
        "curve_path": str(path.relative_to(ROOT)),
        "tuning_seconds": time.perf_counter() - started,
        "n_configurations": int(len(curve)),
        "n_unique_configurations": int(curve["configuration"].nunique()),
        "selection_rows": int(len(y)),
        "selection": (
            "approved reduced protocol: all candidate values in two coordinate "
            "passes, followed by local Cartesian refinement on the deterministic "
            "hotel-preserved 200k-row coarse sample"
        ),
        "coordinate_passes": int(passes),
        "local_candidate_values": local_values,
        "inner_jobs": int(args.inner_jobs),
    }
    write_json(metadata_path, metadata)
    return metadata


def fit_and_write(
    args: argparse.Namespace,
    model_id: str,
    fold: int,
    split: Dict[str, Any],
    text_train: sp.csr_matrix,
    text_test: sp.csr_matrix,
    blocks_train: Mapping[str, np.ndarray],
    blocks_test: Mapping[str, np.ndarray],
    y_train: np.ndarray,
    y_test: np.ndarray,
    group_train: np.ndarray,
    group_test: np.ndarray,
    row_test: np.ndarray,
    alpha_meta: Dict[str, Any],
    lambdas: Mapping[str, float],
    harness: EvaluationHarness,
    extra: Dict[str, Any],
) -> Dict[str, Any]:
    if args.resume and harness.has_fold(model_id, fold):
        log(f"{model_id} fold {fold}: artifact cache hit")
        return harness.load_fold(model_id, fold).record
    with PeakRSSMonitor() as memory:
        operator = CenteredBlockOperator(text_train, blocks_train, lambdas)
        fit = fit_operator(
            operator,
            y_train,
            alpha_meta["alpha"],
            tolerance=args.tolerance,
            max_iterations=args.max_iterations,
        )
        predict_started = time.perf_counter()
        prediction = predict_blocks(text_test, blocks_test, fit)
        predict_seconds = time.perf_counter() - predict_started
    coefficient_norms = {
        name: float(np.linalg.norm(values))
        for name, values in fit.coefficients.items()
    }
    artifact = harness.write_fold(
        model_id=model_id,
        fold=fold,
        row_id=row_test,
        offering_id=group_test,
        y_true=y_test,
        y_pred_raw=prediction,
        preprocessing_seconds=float(extra.get("preprocessing_seconds", 0.0)),
        tuning_seconds=float(
            alpha_meta.get("tuning_seconds", 0.0)
            + extra.get("lambda_tuning_seconds", 0.0)
        ),
        fit_seconds=fit.fit_seconds,
        predict_seconds=predict_seconds,
        peak_memory_mb=max(
            memory.peak_mb,
            psutil.Process().memory_info().rss / (1024 * 1024),
        ),
        selected_hyperparameters={
            "alpha": fit.alpha,
            "lambdas": dict(lambdas),
            "alpha_grid_initial": "logspace(1e-5,1e9,57) coarse screen",
            "alpha_search_protocol": alpha_meta.get("protocol"),
            "alpha_search_protocol_version": alpha_meta.get(
                "protocol_version"
            ),
            "alpha_boundary_status": alpha_meta["boundary_status"],
            "alpha_extensions": alpha_meta["extensions"],
            "inner_cv": "GroupKFold(3), group=offering_id",
            "solver": "scipy.sparse.linalg.lsqr centered LinearOperator",
            "tolerance": args.tolerance,
            "max_iterations": args.max_iterations,
        },
        extra={
            **extra,
            "n_features": int(operator.shape[1]),
            "coefficient_norms": coefficient_norms,
            "lsqr_iterations": fit.iterations,
            "lsqr_stop_code": fit.stop_code,
            "outer_split_sha256": split["sha256"],
        },
    )
    log(
        f"{model_id} fold {fold}: "
        f"MAE={artifact.record['clipped']['mae']:.6f}, "
        f"RMSE={artifact.record['clipped']['rmse']:.6f}, "
        f"alpha={fit.alpha:g}, lambdas={dict(lambdas)}, "
        f"iter={fit.iterations}"
    )
    return artifact.record


def run_projection(
    args: argparse.Namespace,
    output_dir: Path,
    split: Dict[str, Any],
    y: np.ndarray,
    groups: np.ndarray,
    raw_blocks: Mapping[str, np.ndarray],
) -> Dict[str, Any]:
    fold = split["fold"]
    train_path, _ = tfidf_paths(
        output_dir, args.profile, args.max_features, fold
    )
    text = sp.load_npz(train_path).tocsr()
    blocks, _, _ = prepare_fold_blocks(
        raw_blocks, split["train_idx"], split["test_idx"]
    )
    positions = hotel_preserved_subsample(
        output_dir
        / "phase2"
        / "coarse_samples"
        / f"fold{fold}_n{args.coarse_rows}_seed{SEED}.npz",
        groups[split["train_idx"]],
        args.coarse_rows,
        SEED + fold,
    )
    # One representative simple and one interaction fit at coarse/full-inner scale.
    records = []
    for scale, selected in (
        ("coarse", positions),
        (
            "full_inner_approx",
            np.arange(
                min(text.shape[0], int(text.shape[0] * 2 / 3)),
                dtype=np.int64,
            ),
        ),
    ):
        for variant, names in (
            ("text", []),
            ("interaction", ["z", "p_s", "p_b"]),
        ):
            text_selected = text[selected]
            block_selected = {name: blocks[name][selected] for name in names}
            operator = CenteredBlockOperator(
                text_selected,
                block_selected,
                {name: 1.0 for name in names},
            )
            fit = fit_operator(
                operator,
                y[split["train_idx"]][selected],
                alpha=1.0,
                tolerance=args.tolerance,
                max_iterations=args.max_iterations,
            )
            records.append(
                {
                    "scale": scale,
                    "variant": variant,
                    "n_rows": int(len(selected)),
                    "n_features": int(operator.shape[1]),
                    "seconds": fit.fit_seconds,
                    "iterations": fit.iterations,
                }
            )
            del text_selected, block_selected, operator, fit
            gc.collect()
    frame = pd.DataFrame(records)
    full_seconds = float(
        frame[
            frame["scale"].eq("full_inner_approx")
            & frame["variant"].eq("interaction")
        ]["seconds"].iloc[0]
    )
    coarse_seconds = float(
        frame[
            frame["scale"].eq("coarse")
            & frame["variant"].eq("interaction")
        ]["seconds"].iloc[0]
    )
    fit_counts = {
        "r0_full_inner": 3 * 57 * 3 * 5,
        "r1_lambda_coarse": (9 + 729) * 3 * 5,
        "r1_retune_full_inner": 2 * 57 * 3 * 5,
    }
    estimate_seconds = (
        fit_counts["r0_full_inner"] * full_seconds
        + fit_counts["r1_lambda_coarse"] * coarse_seconds
        + fit_counts["r1_retune_full_inner"] * full_seconds
    )
    payload = {
        "benchmark": frame.to_dict(orient="records"),
        "fit_counts": fit_counts,
        "projected_seconds": estimate_seconds,
        "projected_hours": estimate_seconds / 3600,
        "threshold_hours": 12.0,
        "requires_approval": estimate_seconds / 3600 > 12.0,
        "assumptions": (
            "Sequential LSQR fits; warm starts are expected to reduce actual time, "
            "but the conservative projection uses cold representative fit times."
        ),
    }
    frame.to_csv(
        output_dir / "phase2" / "experiment_r_runtime_benchmark.csv",
        index=False,
    )
    write_json(
        output_dir / "phase2" / "experiment_r_runtime_projection.json",
        payload,
    )
    log(
        f"R projection: {payload['projected_hours']:.2f} hours; "
        f"requires approval={payload['requires_approval']}"
    )
    return payload


def model_blocks(
    variant: str,
    blocks: Mapping[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    names = {
        "text": [],
        "simple": ["z"],
        "interaction": ["z", "p_s", "p_b"],
    }[variant]
    return {name: blocks[name] for name in names}


def finalize_r(
    output_dir: Path,
    harness: EvaluationHarness,
    model_ids: Sequence[str],
) -> None:
    completed = []
    for model_id in model_ids:
        if all(harness.has_fold(model_id, fold) for fold in range(1, 6)):
            completed.append(
                harness.finalize_model(
                    model_id,
                    {
                        "phase": 2,
                        "experiment": "R",
                        "seed": SEED,
                        "outer_split": "saved GroupKFold(5), group=offering_id",
                        "tfidf": "cached outer-train TF-IDF 1-2 grams, min_df=3, max_features=200000",
                        "score_representation": "legacy raw cached s; interactions s*segment and mentioned*segment",
                    },
                )
            )
    if not completed:
        return
    rows = []
    for payload in completed:
        aggregate = payload["aggregate"]["clipped"]
        rows.append(
            {
                "model_id": payload["model_id"],
                "mae_mean": aggregate["mean"]["mae"],
                "mae_std": aggregate["std"]["mae"],
                "rmse_mean": aggregate["mean"]["rmse"],
                "rmse_std": aggregate["std"]["rmse"],
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(
        output_dir / "phase2" / "experiment_r_model_summary.csv",
        index=False,
    )
    decision = None
    if {
        "phase2_r1b_simple_block_ridge",
        "phase2_r1c_interaction_block_ridge",
    }.issubset(frame["model_id"]):
        simple = frame.set_index("model_id").loc[
            "phase2_r1b_simple_block_ridge", "mae_mean"
        ]
        interaction = frame.set_index("model_id").loc[
            "phase2_r1c_interaction_block_ridge", "mae_mean"
        ]
        gain = float(simple - interaction)
        if gain > 0.01:
            category = "above_0.01"
        elif gain >= 0.003:
            category = "between_0.003_and_0.01"
        else:
            category = "below_0.003"
        first = (
            output_dir
            / "per_hotel"
            / "phase2_r1b_simple_block_ridge.parquet"
        )
        second = (
            output_dir
            / "per_hotel"
            / "phase2_r1c_interaction_block_ridge.parquet"
        )
        decision = {
            "simple_minus_interaction_mae": gain,
            "category": category,
            "wilcoxon": paired_wilcoxon(first, second),
            "bootstrap": bootstrap_mean_difference(first, second),
        }
    payload = {
        "models": rows,
        "decision_rule": decision,
    }
    write_json(output_dir / "phase2" / "experiment_r_results.json", payload)
    lines = [
        "# Experiment R — Regularization Forensics",
        "",
        frame.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## R decision rule",
        "",
        json.dumps(decision, indent=2) if decision else "Pending all five folds.",
        "",
    ]
    (output_dir / "phase2" / "experiment_r_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    selected_folds = parse_folds(args.folds)
    np.random.seed(SEED)
    output_dir = ROOT / "outputs"
    phase2_dir = output_dir / "phase2"
    phase2_dir.mkdir(parents=True, exist_ok=True)
    splits = {row["fold"]: row for row in load_splits(output_dir)}
    dataset = pd.read_parquet(
        output_dir / "dataset_clean.parquet",
        columns=["rating_overall", "offering_id"],
    )
    y = dataset["rating_overall"].to_numpy(dtype=float)
    groups = dataset["offering_id"].to_numpy()
    row_ids = np.arange(len(dataset), dtype=np.int64)
    raw_blocks = load_raw_structured(output_dir, args.profile)
    if args.mode == "projection":
        run_projection(
            args,
            output_dir,
            splits[selected_folds[0]],
            y,
            groups,
            raw_blocks,
        )
        return

    harness = EvaluationHarness(output_dir)
    model_ids: List[str] = []
    diagnostics: List[pd.DataFrame] = []
    for fold in tqdm(selected_folds, desc="Experiment R folds"):
        split = splits[fold]
        train_idx, test_idx = split["train_idx"], split["test_idx"]
        load_started = time.perf_counter()
        train_path, test_path = tfidf_paths(
            output_dir, args.profile, args.max_features, fold
        )
        text_train = sp.load_npz(train_path).tocsr()
        text_test = sp.load_npz(test_path).tocsr()
        if text_train.shape[0] != len(train_idx) or text_test.shape[0] != len(test_idx):
            raise ValueError(f"fold {fold}: TF-IDF row mismatch")
        blocks_train, blocks_test, scaler_meta = prepare_fold_blocks(
            raw_blocks, train_idx, test_idx
        )
        preprocessing_seconds = time.perf_counter() - load_started
        diagnostic = block_diagnostics(fold, text_train, blocks_train)
        diagnostics.append(diagnostic)
        diagnostic.to_csv(
            phase2_dir / f"experiment_r3_scale_fold{fold}.csv",
            index=False,
        )
        coarse_positions = hotel_preserved_subsample(
            phase2_dir
            / "coarse_samples"
            / f"fold{fold}_n{args.coarse_rows}_seed{SEED}.npz",
            groups[train_idx],
            args.coarse_rows,
            SEED + fold,
        )

        r0_meta: Dict[str, Dict[str, Any]] = {}
        if args.mode in {"r0", "all", "r1", "r2"}:
            for variant in ("text", "simple", "interaction"):
                model_id = f"phase2_r0_{variant}_shared_ridge"
                selected_blocks_train = model_blocks(variant, blocks_train)
                selected_blocks_test = model_blocks(variant, blocks_test)
                lambdas = {name: 1.0 for name in selected_blocks_train}
                alpha_meta = tune_alpha_cached(
                    args,
                    model_id,
                    fold,
                    text_train,
                    selected_blocks_train,
                    y[train_idx],
                    groups[train_idx],
                    lambdas,
                    coarse_positions,
                    output_dir,
                )
                r0_meta[variant] = alpha_meta
                if args.mode in {"r0", "all"}:
                    fit_and_write(
                        args,
                        model_id,
                        fold,
                        split,
                        text_train,
                        text_test,
                        selected_blocks_train,
                        selected_blocks_test,
                        y[train_idx],
                        y[test_idx],
                        groups[train_idx],
                        groups[test_idx],
                        row_ids[test_idx],
                        alpha_meta,
                        lambdas,
                        harness,
                        {
                            "variant": variant,
                            "protocol": "R0 shared ridge",
                            "preprocessing_seconds": preprocessing_seconds,
                            "scaler": scaler_meta,
                        },
                    )
                    model_ids.append(model_id)

        if args.mode in {"r1", "all"}:
            # R1a is exactly the corrected R0 text model.
            model_ids.append("phase2_r0_text_shared_ridge")
            for label, variant, names in (
                ("r1b", "simple", ["z"]),
                ("r1c", "interaction", ["z", "p_s", "p_b"]),
            ):
                model_id = (
                    f"phase2_{label}_{variant}_block_ridge"
                )
                selected_train = {name: blocks_train[name] for name in names}
                selected_test = {name: blocks_test[name] for name in names}
                lambda_meta = lambda_tune_cached(
                    args,
                    model_id,
                    fold,
                    text_train[coarse_positions].tocsr(),
                    {
                        name: matrix[coarse_positions]
                        for name, matrix in selected_train.items()
                    },
                    y[train_idx][coarse_positions],
                    groups[train_idx][coarse_positions],
                    r0_meta[variant]["alpha"],
                    output_dir,
                )
                final_alpha_meta = tune_alpha_cached(
                    args,
                    model_id,
                    fold,
                    text_train,
                    selected_train,
                    y[train_idx],
                    groups[train_idx],
                    lambda_meta["lambdas"],
                    coarse_positions,
                    output_dir,
                )
                fit_and_write(
                    args,
                    model_id,
                    fold,
                    split,
                    text_train,
                    text_test,
                    selected_train,
                    selected_test,
                    y[train_idx],
                    y[test_idx],
                    groups[train_idx],
                    groups[test_idx],
                    row_ids[test_idx],
                    final_alpha_meta,
                    lambda_meta["lambdas"],
                    harness,
                    {
                        "variant": variant,
                        "protocol": "R1 generalized block ridge",
                        "preprocessing_seconds": preprocessing_seconds,
                        "lambda_tuning_seconds": lambda_meta["tuning_seconds"],
                        "lambda_selection_rows": lambda_meta["selection_rows"],
                        "lambda_curve_path": lambda_meta["curve_path"],
                        "scaler": scaler_meta,
                    },
                )
                model_ids.append(model_id)

        if args.mode in {"r2", "all"}:
            contrasts = [
                (
                    "phase2_r2a_interaction_at_text_alpha",
                    "interaction",
                    r0_meta["text"],
                ),
                (
                    "phase2_r2b_simple_at_interaction_alpha",
                    "simple",
                    r0_meta["interaction"],
                ),
                (
                    "phase2_r2c_text_at_interaction_alpha",
                    "text",
                    r0_meta["interaction"],
                ),
            ]
            for model_id, variant, source_alpha in contrasts:
                selected_train = model_blocks(variant, blocks_train)
                selected_test = model_blocks(variant, blocks_test)
                fixed_meta = {
                    **source_alpha,
                    "tuning_seconds": 0.0,
                    "boundary_status": "alpha_matched_fixed",
                    "extensions": [],
                }
                fit_and_write(
                    args,
                    model_id,
                    fold,
                    split,
                    text_train,
                    text_test,
                    selected_train,
                    selected_test,
                    y[train_idx],
                    y[test_idx],
                    groups[train_idx],
                    groups[test_idx],
                    row_ids[test_idx],
                    fixed_meta,
                    {name: 1.0 for name in selected_train},
                    harness,
                    {
                        "variant": variant,
                        "protocol": "R2 alpha-matched contrast",
                        "preprocessing_seconds": preprocessing_seconds,
                        "source_alpha_model": (
                            "text" if "r2a" in model_id else "interaction"
                        ),
                        "scaler": scaler_meta,
                    },
                )
                model_ids.append(model_id)
        del text_train, text_test, blocks_train, blocks_test
        gc.collect()

    if diagnostics:
        pd.concat(diagnostics, ignore_index=True).to_csv(
            phase2_dir / "experiment_r3_scale_diagnostics.csv",
            index=False,
        )
    finalize_r(output_dir, harness, sorted(set(model_ids)))


if __name__ == "__main__":
    main()
