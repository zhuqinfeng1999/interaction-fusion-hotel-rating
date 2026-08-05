#!/usr/bin/env python3
"""Phase 2 Experiments P and A: controls and matched-penalty ablations."""

from __future__ import annotations

import argparse
import gc
import hashlib
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
    predict_blocks,
)


SEED = 42
N_SPLITS = 5
CONTROL_SEEDS = [42, 43, 44, 45, 46]
SEGMENTS = ["business", "couple", "family", "friends", "solo", "unknown"]
R1B = "phase2_r1b_simple_block_ridge"
R1C = "phase2_r1c_interaction_block_ridge"


def log(message: str) -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="qwen25_7b")
    parser.add_argument("--folds", default="1,2,3,4,5")
    parser.add_argument("--max-features", type=int, default=200_000)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--max-iterations", type=int, default=2_000)
    parser.add_argument(
        "--stage",
        choices=["p", "a", "all", "finalize"],
        default="all",
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def parse_folds(raw: str) -> List[int]:
    folds = sorted({int(value.strip()) for value in raw.split(",") if value.strip()})
    if not folds or any(fold < 1 or fold > N_SPLITS for fold in folds):
        raise ValueError("--folds must contain values in 1..5")
    return folds


def load_splits(output_dir: Path) -> Dict[int, Dict[str, Any]]:
    output: Dict[int, Dict[str, Any]] = {}
    for fold in range(1, N_SPLITS + 1):
        path = output_dir / f"splits_fold{fold}.npz"
        arrays = np.load(path)
        output[fold] = {
            "train_idx": arrays["train_idx"].astype(np.int64),
            "test_idx": arrays["test_idx"].astype(np.int64),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    return output


def tfidf_paths(
    output_dir: Path,
    profile: str,
    max_features: int,
    fold: int,
) -> Tuple[Path, Path]:
    matches = sorted(
        (output_dir / "tfidf_cache").glob(
            f"{profile}_mf{max_features}_ng12_md3_full_fold{fold}_*_Xtr.npz"
        )
    )
    if len(matches) != 1:
        raise FileNotFoundError(
            f"fold {fold}: expected one full TF-IDF training cache, got {matches}"
        )
    test_path = Path(str(matches[0]).replace("_Xtr.npz", "_Xte.npz"))
    if not test_path.exists():
        raise FileNotFoundError(test_path)
    return matches[0], test_path


def load_components(output_dir: Path, profile: str) -> Dict[str, np.ndarray]:
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
    segment_text = frame["traveler_segment"].fillna("unknown").astype(str).str.lower()
    segment_text = segment_text.where(segment_text.isin(SEGMENTS), "unknown")
    segments = (
        pd.get_dummies(segment_text)
        .reindex(columns=SEGMENTS, fill_value=False)
        .to_numpy(dtype=np.float32)
    )
    return {
        "scores": scores,
        "mentions": mentions,
        "segments": segments,
        "known": np.asarray(segment_text.ne("unknown"), dtype=bool),
    }


def build_blocks(
    scores: np.ndarray,
    mentions: np.ndarray,
    segments: np.ndarray,
) -> Dict[str, np.ndarray]:
    p_s = np.hstack(
        [scores * segments[:, index : index + 1] for index in range(len(SEGMENTS))]
    ).astype(np.float32)
    p_b = np.hstack(
        [
            mentions * segments[:, index : index + 1]
            for index in range(len(SEGMENTS))
        ]
    ).astype(np.float32)
    return {
        "z": np.hstack([scores, mentions, segments]).astype(np.float32),
        "p_s": p_s,
        "p_b": p_b,
    }


def true_blocks(components: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return build_blocks(
        components["scores"], components["mentions"], components["segments"]
    )


def control_blocks(
    control: str,
    seed: int,
    components: Mapping[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    scores = components["scores"]
    mentions = components["mentions"]
    segments = components["segments"]
    rng = np.random.default_rng(seed)
    if control == "p1_segment_permutation":
        permutation = rng.permutation(len(scores))
        return build_blocks(scores, mentions, segments[permutation])
    if control == "p2_aspect_permutation":
        permutation = rng.permutation(len(scores))
        return build_blocks(scores[permutation], mentions[permutation], segments)

    base = build_blocks(scores, mentions, segments)
    random_values = rng.standard_normal(
        size=(len(scores), 72), dtype=np.float32
    )
    if control == "p4_sparsity_matched_random":
        mask = np.hstack([base["p_s"], base["p_b"]]) != 0
        random_values *= mask
        del mask
    elif control != "p3_random_dense":
        raise ValueError(f"unknown control: {control}")
    base["p_s"] = random_values[:, :36]
    base["p_b"] = random_values[:, 36:]
    return base


def prepare_blocks(
    raw_blocks: Mapping[str, np.ndarray],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    train: Dict[str, np.ndarray] = {}
    test: Dict[str, np.ndarray] = {}
    for name, matrix in raw_blocks.items():
        scaler = StandardScaler()
        train[name] = scaler.fit_transform(matrix[train_idx]).astype(np.float32)
        test[name] = scaler.transform(matrix[test_idx]).astype(np.float32)
    return train, test


def inherited_hyperparameters(
    output_dir: Path,
    fold: int,
) -> Tuple[float, Dict[str, float]]:
    path = output_dir / "predictions" / R1C / f"fold{fold}.meta.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    selected = payload["selected_hyperparameters"]
    return float(selected["alpha"]), {
        name: float(value) for name, value in selected["lambdas"].items()
    }


def fit_matched_model(
    args: argparse.Namespace,
    harness: EvaluationHarness,
    model_id: str,
    fold: int,
    text_train: sp.csr_matrix,
    text_test: sp.csr_matrix,
    blocks_train: Mapping[str, np.ndarray],
    blocks_test: Mapping[str, np.ndarray],
    y_train: np.ndarray,
    y_test: np.ndarray,
    group_test: np.ndarray,
    row_test: np.ndarray,
    alpha: float,
    lambdas: Mapping[str, float],
    split_sha256: str,
    experiment: str,
    variant: str,
    seed: int | None = None,
) -> Dict[str, Any]:
    if args.resume and harness.has_fold(model_id, fold):
        log(f"{model_id} fold {fold}: cache hit")
        return harness.load_fold(model_id, fold).record
    selected_lambdas = {
        name: float(lambdas.get(name, 1.0)) for name in blocks_train
    }
    with PeakRSSMonitor() as memory:
        operator = CenteredBlockOperator(
            text_train, blocks_train, selected_lambdas
        )
        fit = fit_operator(
            operator,
            y_train,
            alpha,
            tolerance=args.tolerance,
            max_iterations=args.max_iterations,
        )
        predict_started = time.perf_counter()
        prediction = predict_blocks(text_test, blocks_test, fit)
        predict_seconds = time.perf_counter() - predict_started
    artifact = harness.write_fold(
        model_id=model_id,
        fold=fold,
        row_id=row_test,
        offering_id=group_test,
        y_true=y_test,
        y_pred_raw=prediction,
        fit_seconds=fit.fit_seconds,
        predict_seconds=predict_seconds,
        peak_memory_mb=max(
            memory.peak_mb,
            psutil.Process().memory_info().rss / (1024 * 1024),
        ),
        selected_hyperparameters={
            "alpha": float(alpha),
            "lambdas": selected_lambdas,
            "source": f"matched fold-specific hyperparameters from {R1C}",
            "tuning": "not re-selected on permuted/random control labels",
            "solver": "scipy.sparse.linalg.lsqr centered LinearOperator",
            "tolerance": args.tolerance,
            "max_iterations": args.max_iterations,
        },
        extra={
            "phase": 2,
            "experiment": experiment,
            "variant": variant,
            "seed": seed,
            "n_features": int(operator.shape[1]),
            "outer_split_sha256": split_sha256,
            "lsqr_iterations": fit.iterations,
            "lsqr_stop_code": fit.stop_code,
            "coefficient_norms": {
                name: float(np.linalg.norm(values))
                for name, values in fit.coefficients.items()
            },
        },
    )
    log(
        f"{model_id} fold {fold}: MAE={artifact.record['clipped']['mae']:.6f}, "
        f"RMSE={artifact.record['clipped']['rmse']:.6f}, iter={fit.iterations}, "
        f"fit={fit.fit_seconds:.1f}s"
    )
    return artifact.record


def copy_reference_fold(
    args: argparse.Namespace,
    harness: EvaluationHarness,
    model_id: str,
    fold: int,
    reference_model: str,
    keep_rows: np.ndarray | None,
    variant: str,
) -> None:
    if args.resume and harness.has_fold(model_id, fold):
        return
    reference = pd.read_parquet(
        harness.prediction_path(reference_model, fold)
    )
    if keep_rows is not None:
        reference = reference[reference["row_id"].isin(keep_rows)].copy()
    harness.write_fold(
        model_id=model_id,
        fold=fold,
        row_id=reference["row_id"].to_numpy(),
        offering_id=reference["offering_id"].to_numpy(),
        y_true=reference["y_true"].to_numpy(),
        y_pred_raw=reference["y_pred_raw"].to_numpy(),
        fit_seconds=0.0,
        predict_seconds=0.0,
        peak_memory_mb=psutil.Process().memory_info().rss / (1024 * 1024),
        selected_hyperparameters={
            "identity_reference": reference_model,
            "reason": variant,
        },
        extra={
            "phase": 2,
            "experiment": "A",
            "variant": variant,
            "identity_copy": True,
        },
    )


def p_model_id(control: str, seed: int) -> str:
    return f"phase2_{control}_seed{seed}"


def run_fold(
    args: argparse.Namespace,
    fold: int,
    split: Mapping[str, Any],
    output_dir: Path,
    components: Mapping[str, np.ndarray],
    y: np.ndarray,
    groups: np.ndarray,
    row_ids: np.ndarray,
    harness: EvaluationHarness,
) -> None:
    train_idx = split["train_idx"]
    test_idx = split["test_idx"]
    train_path, test_path = tfidf_paths(
        output_dir, args.profile, args.max_features, fold
    )
    text_train = sp.load_npz(train_path).tocsr()
    text_test = sp.load_npz(test_path).tocsr()
    if text_train.shape[0] != len(train_idx) or text_test.shape[0] != len(test_idx):
        raise ValueError(f"fold {fold}: TF-IDF row mismatch")
    alpha, matched_lambdas = inherited_hyperparameters(output_dir, fold)

    if args.stage in {"p", "all"}:
        for control in (
            "p1_segment_permutation",
            "p2_aspect_permutation",
            "p3_random_dense",
            "p4_sparsity_matched_random",
        ):
            for seed in CONTROL_SEEDS:
                model_id = p_model_id(control, seed)
                if args.resume and harness.has_fold(model_id, fold):
                    log(f"{model_id} fold {fold}: cache hit")
                    continue
                started = time.perf_counter()
                raw = control_blocks(control, seed, components)
                block_train, block_test = prepare_blocks(raw, train_idx, test_idx)
                fit_matched_model(
                    args,
                    harness,
                    model_id,
                    fold,
                    text_train,
                    text_test,
                    block_train,
                    block_test,
                    y[train_idx],
                    y[test_idx],
                    groups[test_idx],
                    row_ids[test_idx],
                    alpha,
                    matched_lambdas,
                    split["sha256"],
                    "P",
                    control,
                    seed,
                )
                log(
                    f"{model_id} fold {fold}: total stage "
                    f"{time.perf_counter() - started:.1f}s"
                )
                del raw, block_train, block_test
                gc.collect()

        base_raw = true_blocks(components)
        base_train, base_test = prepare_blocks(base_raw, train_idx, test_idx)
        for model_id, names, variant in (
            ("phase2_p5_score_interactions_only", ["z", "p_s"], "p5_score_only"),
            ("phase2_p5_mention_interactions_only", ["z", "p_b"], "p5_mention_only"),
        ):
            fit_matched_model(
                args,
                harness,
                model_id,
                fold,
                text_train,
                text_test,
                {name: base_train[name] for name in names},
                {name: base_test[name] for name in names},
                y[train_idx],
                y[test_idx],
                groups[test_idx],
                row_ids[test_idx],
                alpha,
                matched_lambdas,
                split["sha256"],
                "P",
                variant,
            )
        del base_raw, base_train, base_test
        gc.collect()

    if args.stage in {"a", "all"}:
        base_raw = true_blocks(components)
        base_train, base_test = prepare_blocks(base_raw, train_idx, test_idx)
        fit_matched_model(
            args,
            harness,
            "phase2_a5_interactions_without_main",
            fold,
            text_train,
            text_test,
            {name: base_train[name] for name in ("p_s", "p_b")},
            {name: base_test[name] for name in ("p_s", "p_b")},
            y[train_idx],
            y[test_idx],
            groups[test_idx],
            row_ids[test_idx],
            alpha,
            matched_lambdas,
            split["sha256"],
            "A",
            "a5_interactions_without_main",
        )

        # The enforced extraction schema sets every unmentioned score to zero.
        # Consequently masked and unmasked cached scores are algebraically
        # identical; preserve this as an explicit identity-control artifact.
        copy_reference_fold(
            args,
            harness,
            "phase2_a6_unmasked_scores",
            fold,
            R1C,
            None,
            "a6_not_identifiable_scores_zero_when_unmentioned",
        )

        unknown_segment_start = (len(SEGMENTS) - 1) * len(ASPECTS)
        known_train = {
            "z": base_train["z"][:, :-1],
            "p_s": np.delete(
                base_train["p_s"],
                np.s_[unknown_segment_start : unknown_segment_start + len(ASPECTS)],
                axis=1,
            ),
            "p_b": np.delete(
                base_train["p_b"],
                np.s_[unknown_segment_start : unknown_segment_start + len(ASPECTS)],
                axis=1,
            ),
        }
        known_test = {
            "z": base_test["z"][:, :-1],
            "p_s": np.delete(
                base_test["p_s"],
                np.s_[unknown_segment_start : unknown_segment_start + len(ASPECTS)],
                axis=1,
            ),
            "p_b": np.delete(
                base_test["p_b"],
                np.s_[unknown_segment_start : unknown_segment_start + len(ASPECTS)],
                axis=1,
            ),
        }
        fit_matched_model(
            args,
            harness,
            "phase2_a7_unknown_column_removed",
            fold,
            text_train,
            text_test,
            known_train,
            known_test,
            y[train_idx],
            y[test_idx],
            groups[test_idx],
            row_ids[test_idx],
            alpha,
            matched_lambdas,
            split["sha256"],
            "A",
            "a7_unknown_column_removed_rows_retained",
        )
        del known_train, known_test, base_train, base_test

        known = components["known"]
        train_keep = known[train_idx]
        test_keep = known[test_idx]
        known_train_idx = train_idx[train_keep]
        known_test_idx = test_idx[test_keep]
        known_raw = true_blocks(components)
        subset_train, subset_test = prepare_blocks(
            known_raw, known_train_idx, known_test_idx
        )
        fit_matched_model(
            args,
            harness,
            "phase2_a8_known_segment_only",
            fold,
            text_train[train_keep].tocsr(),
            text_test[test_keep].tocsr(),
            subset_train,
            subset_test,
            y[known_train_idx],
            y[known_test_idx],
            groups[known_test_idx],
            row_ids[known_test_idx],
            alpha,
            matched_lambdas,
            split["sha256"],
            "A",
            "a8_known_segment_only",
        )
        copy_reference_fold(
            args,
            harness,
            "phase2_a4_known_segment_reference",
            fold,
            R1C,
            row_ids[known_test_idx],
            "a4_reference_restricted_to_a8_rows",
        )
        del base_raw, known_raw, subset_train, subset_test
        gc.collect()

    del text_train, text_test
    gc.collect()


def aggregate_model(harness: EvaluationHarness, model_id: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    return harness.finalize_model(model_id, meta)


def finalize_p(output_dir: Path, harness: EvaluationHarness) -> Dict[str, Any]:
    r = json.loads((output_dir / "phase2" / "experiment_r_results.json").read_text())
    r_frame = pd.DataFrame(r["models"]).set_index("model_id")
    simple_mae = float(r_frame.loc[R1B, "mae_mean"])
    true_mae = float(r_frame.loc[R1C, "mae_mean"])
    true_gain = simple_mae - true_mae

    seed_rows: List[Dict[str, Any]] = []
    controls = (
        "p1_segment_permutation",
        "p2_aspect_permutation",
        "p3_random_dense",
        "p4_sparsity_matched_random",
    )
    for control in controls:
        for seed in CONTROL_SEEDS:
            model_id = p_model_id(control, seed)
            payload = aggregate_model(
                harness,
                model_id,
                {
                    "phase": 2,
                    "experiment": "P",
                    "control": control,
                    "seed": seed,
                    "regularization": "fold-specific matched R1c hyperparameters",
                },
            )
            clipped = payload["aggregate"]["clipped"]
            control_mae = float(clipped["mean"]["mae"])
            control_gain = simple_mae - control_mae
            seed_rows.append(
                {
                    "control": control,
                    "seed": seed,
                    "model_id": model_id,
                    "mae": control_mae,
                    "rmse": float(clipped["mean"]["rmse"]),
                    "gain_vs_r1b_simple": control_gain,
                    "fraction_of_true_signed_gain": (
                        control_gain / true_gain if not np.isclose(true_gain, 0) else np.nan
                    ),
                }
            )

    for control, model_id in (
        ("p5_score_interactions_only", "phase2_p5_score_interactions_only"),
        ("p5_mention_interactions_only", "phase2_p5_mention_interactions_only"),
    ):
        payload = aggregate_model(
            harness,
            model_id,
            {
                "phase": 2,
                "experiment": "P",
                "control": control,
                "regularization": "fold-specific matched R1c hyperparameters",
            },
        )
        clipped = payload["aggregate"]["clipped"]
        control_mae = float(clipped["mean"]["mae"])
        control_gain = simple_mae - control_mae
        seed_rows.append(
            {
                "control": control,
                "seed": np.nan,
                "model_id": model_id,
                "mae": control_mae,
                "rmse": float(clipped["mean"]["rmse"]),
                "gain_vs_r1b_simple": control_gain,
                "fraction_of_true_signed_gain": (
                    control_gain / true_gain if not np.isclose(true_gain, 0) else np.nan
                ),
            }
        )

    seed_frame = pd.DataFrame(seed_rows)
    summary = (
        seed_frame.groupby("control", as_index=False, dropna=False)
        .agg(
            n_seeds=("mae", "size"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            gain_vs_r1b_mean=("gain_vs_r1b_simple", "mean"),
            fraction_of_true_signed_gain_mean=("fraction_of_true_signed_gain", "mean"),
        )
        .fillna({"mae_std": 0.0, "rmse_std": 0.0})
    )
    phase2 = output_dir / "phase2"
    seed_frame.to_csv(phase2 / "experiment_p_control_seeds.csv", index=False)
    summary.to_csv(phase2 / "experiment_p_control_summary.csv", index=False)
    payload = {
        "regularization_protocol": (
            "fold-specific alpha/lambdas inherited from the matched true R1c model; "
            "controls do not tune on randomized labels"
        ),
        "r1b_simple_mae": simple_mae,
        "r1c_true_interaction_mae": true_mae,
        "true_signed_gain_simple_minus_interaction": true_gain,
        "positive_gain_denominator": bool(true_gain > 0),
        "seed_results": seed_rows,
        "summary": summary.to_dict(orient="records"),
    }
    write_json(phase2 / "experiment_p_results.json", payload)
    lines = [
        "# Experiment P — Permutation and Dimension Controls",
        "",
        "**Anomaly first:** the best-tuned true R1c interaction model is worse than R1b simple fusion, so the signed true gain is negative. Fractions below are algebraic ratios, not fractions of a positive benefit.",
        "",
        f"- R1b simple MAE: {simple_mae:.6f}",
        f"- R1c interaction MAE: {true_mae:.6f}",
        f"- Signed gain (simple - interaction): {true_gain:.6f}",
        "- Controls use each fold's R1c alpha/lambdas as a matched-penalty design. This avoids selecting hyperparameters on randomized labels and is the approved reduced-cost control protocol.",
        "",
        summary.to_markdown(index=False, floatfmt=".6f"),
        "",
    ]
    (phase2 / "experiment_p_report.md").write_text("\n".join(lines), encoding="utf-8")
    return payload


def finalize_a(output_dir: Path, harness: EvaluationHarness) -> Dict[str, Any]:
    # A8 is evaluated on known-segment rows only.  Its paired A4 reference is
    # written fold-by-fold in run_fold(), but it is not part of the public A1-A8
    # table below.  Finalize it explicitly so its per-hotel artifact exists
    # before the paired tests are computed.
    aggregate_model(
        harness,
        "phase2_a4_known_segment_reference",
        {
            "phase": 2,
            "experiment": "A",
            "ablation": "A8_reference",
            "features": "A4 predictions restricted to A8 known-segment rows",
            "identity_reference": R1C,
        },
    )
    mapping = [
        ("A1", R1B, "x + z"),
        ("A2", "phase2_p5_score_interactions_only", "x + z + p_s"),
        ("A3", "phase2_p5_mention_interactions_only", "x + z + p_b"),
        ("A4", R1C, "x + z + p_s + p_b"),
        ("A5", "phase2_a5_interactions_without_main", "x + p_s + p_b"),
        ("A6", "phase2_a6_unmasked_scores", "A4 unmasked-score identity check"),
        ("A7", "phase2_a7_unknown_column_removed", "A4 without unknown columns"),
        ("A8", "phase2_a8_known_segment_only", "A4 known-segment rows only"),
    ]
    rows: List[Dict[str, Any]] = []
    for label, model_id, features in mapping:
        metrics_path = output_dir / "metrics" / f"{model_id}.json"
        if metrics_path.exists() and model_id in {R1B, R1C}:
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        else:
            payload = aggregate_model(
                harness,
                model_id,
                {
                    "phase": 2,
                    "experiment": "A",
                    "ablation": label,
                    "features": features,
                    "regularization": "fold-specific matched R1c hyperparameters",
                },
            )
        clipped = payload["aggregate"]["clipped"]
        rows.append(
            {
                "ablation": label,
                "model_id": model_id,
                "features": features,
                "n": int(payload["n_predictions"]),
                "mae_mean": float(clipped["mean"]["mae"]),
                "mae_std": float(clipped["std"]["mae"]),
                "rmse_mean": float(clipped["mean"]["rmse"]),
                "rmse_std": float(clipped["std"]["rmse"]),
            }
        )

    comparisons = []
    a4_path = output_dir / "per_hotel" / f"{R1C}.parquet"
    for label, model_id, _ in mapping:
        if label == "A4":
            continue
        reference = (
            output_dir / "per_hotel" / "phase2_a4_known_segment_reference.parquet"
            if label == "A8"
            else a4_path
        )
        candidate = output_dir / "per_hotel" / f"{model_id}.parquet"
        comparisons.append(
            {
                "ablation": label,
                "candidate_minus_a4": paired_wilcoxon(candidate, reference),
                "bootstrap": bootstrap_mean_difference(candidate, reference),
                "reference": reference.name,
            }
        )
    frame = pd.DataFrame(rows)
    phase2 = output_dir / "phase2"
    frame.to_csv(phase2 / "experiment_a_ablation_summary.csv", index=False)
    write_json(
        phase2 / "experiment_a_results.json",
        {
            "models": rows,
            "paired_per_hotel_against_a4": comparisons,
            "a6_identifiability": (
                "The enforced schema sets score=0 whenever mentioned=false, so "
                "masked and unmasked cached scores are identical. A6 is an exact "
                "identity control, not an independently estimable ablation."
            ),
        },
    )
    lines = [
        "# Experiment A — Ablation under Matched Regularization",
        "",
        "**Anomaly first:** A4 is not the best ablation; R1b/A1 simple fusion outperforms the full interaction model.",
        "",
        frame.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Paired per-hotel tests against A4",
        "",
        "```json",
        json.dumps(comparisons, indent=2),
        "```",
        "",
        "A6 is an identity check because the enforced extraction schema stores score=0 for every unmentioned aspect; an unmasked latent score is unavailable in the cache.",
        "",
    ]
    (phase2 / "experiment_a_report.md").write_text("\n".join(lines), encoding="utf-8")
    return {"models": rows, "comparisons": comparisons}


def main() -> None:
    args = parse_args()
    np.random.seed(SEED)
    output_dir = ROOT / "outputs"
    splits = load_splits(output_dir)
    selected_folds = parse_folds(args.folds)
    dataset = pd.read_parquet(
        output_dir / "dataset_clean.parquet",
        columns=["rating_overall", "offering_id"],
    )
    y = dataset["rating_overall"].to_numpy(dtype=float)
    groups = dataset["offering_id"].to_numpy()
    row_ids = np.arange(len(dataset), dtype=np.int64)
    components = load_components(output_dir, args.profile)
    harness = EvaluationHarness(output_dir)

    if args.stage != "finalize":
        for fold in tqdm(selected_folds, desc=f"Experiments P/A ({args.stage})"):
            run_fold(
                args,
                fold,
                splits[fold],
                output_dir,
                components,
                y,
                groups,
                row_ids,
                harness,
            )

    if all(
        harness.has_fold(p_model_id(control, seed), fold)
        for control in (
            "p1_segment_permutation",
            "p2_aspect_permutation",
            "p3_random_dense",
            "p4_sparsity_matched_random",
        )
        for seed in CONTROL_SEEDS
        for fold in range(1, 6)
    ):
        finalize_p(output_dir, harness)
    if all(
        harness.has_fold(model_id, fold)
        for model_id in (
            "phase2_p5_score_interactions_only",
            "phase2_p5_mention_interactions_only",
            "phase2_a5_interactions_without_main",
            "phase2_a6_unmasked_scores",
            "phase2_a7_unknown_column_removed",
            "phase2_a8_known_segment_only",
            "phase2_a4_known_segment_reference",
        )
        for fold in range(1, 6)
    ):
        finalize_a(output_dir, harness)


if __name__ == "__main__":
    main()
