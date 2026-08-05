#!/usr/bin/env python3
"""Phase 4 U5: Booking.com external validation under the unified U0 protocol.

No extraction is performed here.  The script reuses the Phase 3 hotel-intact
sample, guided-Qwen aspect parquet, canonical Booking folds, and fold-specific
TF-IDF caches.  Alpha is selected independently for each outer fold by
GroupKFold(5) on the complete outer-training set using the Phase 4 41-point
grid and automatic two-decade boundary extension.
"""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import wilcoxon
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation_harness import EvaluationHarness, PeakRSSMonitor  # noqa: E402


U0 = importlib.import_module("scripts.41_phase4_u0_models")
W3 = importlib.import_module("scripts.38_phase3_booking_analyze")

SEED = 42
N_SPLITS = 5
MODEL_IDS = {
    "text_only": "phase4_booking_u0_text_only",
    "simple_fusion": "phase4_booking_u0_simple_fusion",
    "interaction_fusion": "phase4_booking_u0_interaction_fusion",
}


def log(message: str) -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-file",
        type=Path,
        default=Path("outputs/phase3/booking_dataset_clean.parquet"),
    )
    parser.add_argument(
        "--sample-file",
        type=Path,
        default=Path("outputs/phase3/w_sample_n200000_seed42.npz"),
    )
    parser.add_argument(
        "--qwen-path",
        type=Path,
        default=Path(
            "outputs/phase3/aspects_guided_booking_qwen_guided_n200035.parquet"
        ),
    )
    parser.add_argument("--folds", default="1,2,3,4,5")
    parser.add_argument("--max-features", type=int, default=200_000)
    parser.add_argument("--inner-jobs", type=int, default=5)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--max-iterations", type=int, default=2_000)
    parser.add_argument("--max-extensions", type=int, default=4)
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def parse_folds(value: str) -> List[int]:
    folds = sorted({int(item) for item in value.split(",") if item.strip()})
    if not folds or any(fold not in range(1, N_SPLITS + 1) for fold in folds):
        raise ValueError("--folds must contain values in 1..5")
    return folds


def scale_block(
    values: np.ndarray, train_pos: np.ndarray, test_pos: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    return (
        scaler.fit_transform(values[train_pos]).astype(np.float32),
        scaler.transform(values[test_pos]).astype(np.float32),
    )


def prediction_metrics(
    y_mapped: np.ndarray, y_native: np.ndarray, prediction_raw: np.ndarray
) -> Dict[str, float]:
    prediction = np.clip(prediction_raw, 1.0, 5.0)
    native = np.clip(1.0 + 9.0 * (prediction - 1.0) / 4.0, 1.0, 10.0)
    return {
        "mae_1to5": float(mean_absolute_error(y_mapped, prediction)),
        "rmse_1to5": float(math.sqrt(mean_squared_error(y_mapped, prediction))),
        "mae_native": float(mean_absolute_error(y_native, native)),
        "rmse_native": float(math.sqrt(mean_squared_error(y_native, native))),
    }


def fold_complete(harness: EvaluationHarness, model_id: str, fold: int) -> bool:
    return harness.has_fold(model_id, fold)


def fit_fold(
    harness: EvaluationHarness,
    model_name: str,
    fold: int,
    sample: pd.DataFrame,
    train_pos: np.ndarray,
    test_pos: np.ndarray,
    text_train: sp.csr_matrix,
    text_test: sp.csr_matrix,
    train_blocks: Mapping[str, np.ndarray],
    test_blocks: Mapping[str, np.ndarray],
    args: argparse.Namespace,
) -> None:
    model_id = MODEL_IDS[model_name]
    if args.resume and fold_complete(harness, model_id, fold):
        log(f"U5 {model_name} fold={fold}: harness cache hit")
        return

    y = sample["rating_overall_1to5"].to_numpy(dtype=np.float64)
    groups = sample["offering_id"].to_numpy()
    phase4_booking = ROOT / "outputs" / "phase4" / "booking"
    tuning_args = argparse.Namespace(
        resume=args.resume,
        tolerance=args.tolerance,
        max_iterations=args.max_iterations,
        max_extensions=args.max_extensions,
        inner_jobs=args.inner_jobs,
    )
    alpha, boundary, extensions, tuning_seconds = U0.sparse_tune_cv5(
        f"booking_{model_name}",
        fold,
        text_train,
        train_blocks,
        y[train_pos],
        groups[train_pos],
        phase4_booking,
        tuning_args,
    )

    with PeakRSSMonitor() as memory:
        _, prediction, fit = U0.fit_sparse_prediction(
            text_train,
            text_test,
            train_blocks,
            test_blocks,
            y[train_pos],
            alpha,
            args.tolerance,
            args.max_iterations,
        )
    harness.write_fold(
        model_id=model_id,
        fold=fold,
        row_id=sample.iloc[test_pos]["row_id"].to_numpy(dtype=np.int64),
        offering_id=groups[test_pos],
        y_true=y[test_pos],
        y_pred_raw=prediction,
        fit_seconds=fit.fit_seconds,
        predict_seconds=0.0,
        peak_memory_mb=memory.peak_mb,
        tuning_seconds=tuning_seconds,
        selected_hyperparameters={
            "alpha": alpha,
            "alpha_grid_initial": U0.ALPHAS.tolist(),
            "inner_cv": "GroupKFold(5), group=hotel address hash",
            "boundary_status": boundary,
            "boundary_extensions": extensions,
            "shared_penalty": True,
        },
        extra={
            "experiment": "U5",
            "variant": model_name,
            "corpus": "Booking.com 515k external sample",
            "n_features": int(
                text_train.shape[1]
                + sum(block.shape[1] for block in train_blocks.values())
            ),
            "iterations": int(fit.iterations),
            "stop_code": int(fit.stop_code),
            "deployable": True,
            "rating_mapping": "1 + 4 * (native - 1) / 9",
        },
    )
    score = harness.load_fold(model_id, fold).record["clipped"]
    log(
        f"U5 {model_name} fold={fold} alpha={alpha:g} boundary={boundary} "
        f"MAE={score['mae']:.6f} RMSE={score['rmse']:.6f}"
    )


def collect_outputs(
    harness: EvaluationHarness, sample: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    native_by_row = sample.set_index("row_id")["rating_overall_native"]
    fold_rows: List[Dict[str, Any]] = []
    predictions: List[pd.DataFrame] = []
    for model_name, model_id in MODEL_IDS.items():
        for fold in range(1, N_SPLITS + 1):
            artifact = harness.load_fold(model_id, fold)
            frame = pd.read_parquet(artifact.prediction_path)
            frame = frame.assign(model=model_name, fold=fold)
            frame["prediction"] = np.clip(frame["y_pred_raw"], 1.0, 5.0)
            frame["y_native"] = frame["row_id"].map(native_by_row)
            score = prediction_metrics(
                frame["y_true"].to_numpy(dtype=float),
                frame["y_native"].to_numpy(dtype=float),
                frame["y_pred_raw"].to_numpy(dtype=float),
            )
            selected = artifact.record["selected_hyperparameters"]
            fold_rows.append(
                {
                    "model": model_name,
                    "model_id": model_id,
                    "fold": fold,
                    "n_test": len(frame),
                    "n_properties": int(frame["offering_id"].nunique()),
                    "alpha": float(selected["alpha"]),
                    "boundary_status": selected["boundary_status"],
                    "fit_seconds": float(artifact.record["fit_seconds"]),
                    "tuning_seconds": float(artifact.record["tuning_seconds"]),
                    **score,
                }
            )
            predictions.append(
                frame[
                    [
                        "model",
                        "fold",
                        "row_id",
                        "offering_id",
                        "y_true",
                        "y_native",
                        "y_pred_raw",
                        "prediction",
                    ]
                ]
            )
    folds = pd.DataFrame(fold_rows)
    prediction_frame = pd.concat(predictions, ignore_index=True)
    summary = (
        folds.groupby(["model", "model_id"], as_index=False)
        .agg(
            n=("n_test", "sum"),
            mae_1to5_mean=("mae_1to5", "mean"),
            mae_1to5_sd=("mae_1to5", "std"),
            rmse_1to5_mean=("rmse_1to5", "mean"),
            rmse_1to5_sd=("rmse_1to5", "std"),
            mae_native_mean=("mae_native", "mean"),
            mae_native_sd=("mae_native", "std"),
            rmse_native_mean=("rmse_native", "mean"),
            rmse_native_sd=("rmse_native", "std"),
            alpha_geomean=("alpha", lambda x: float(10 ** np.log10(x).mean())),
            fit_seconds_total=("fit_seconds", "sum"),
            tuning_seconds_total=("tuning_seconds", "sum"),
        )
        .sort_values("mae_1to5_mean")
    )
    return folds, summary, prediction_frame


def paired_test(
    predictions: pd.DataFrame,
    first: str,
    second: str,
    n_resamples: int = 10_000,
) -> Dict[str, Any]:
    first_frame = predictions[predictions["model"].eq(first)]
    second_frame = predictions[predictions["model"].eq(second)]
    joined = first_frame.merge(
        second_frame,
        on=["row_id", "offering_id", "y_true"],
        suffixes=("_first", "_second"),
        validate="one_to_one",
    )
    joined["first_error"] = np.abs(joined["y_true"] - joined["prediction_first"])
    joined["second_error"] = np.abs(joined["y_true"] - joined["prediction_second"])
    hotel = joined.groupby("offering_id")[["first_error", "second_error"]].mean()
    difference = (hotel["first_error"] - hotel["second_error"]).to_numpy(dtype=float)
    try:
        statistic, p_value = wilcoxon(difference)
    except ValueError:
        statistic, p_value = 0.0, 1.0
    rng = np.random.default_rng(SEED)
    bootstrap = np.empty(n_resamples, dtype=np.float64)
    for start in range(0, n_resamples, 500):
        size = min(500, n_resamples - start)
        draws = rng.integers(0, len(difference), size=(size, len(difference)))
        bootstrap[start : start + size] = difference[draws].mean(axis=1)
    return {
        "first": first,
        "second": second,
        "difference": "first_minus_second_equal_property_MAE_on_1to5",
        "n_properties": int(len(difference)),
        "mean_difference": float(difference.mean()),
        "bootstrap_ci_low": float(np.quantile(bootstrap, 0.025)),
        "bootstrap_ci_high": float(np.quantile(bootstrap, 0.975)),
        "bootstrap_resamples": n_resamples,
        "seed": SEED,
        "wilcoxon_statistic": float(statistic),
        "wilcoxon_p": float(p_value),
    }


def write_results(
    sample: pd.DataFrame,
    features: Mapping[str, np.ndarray],
    folds: pd.DataFrame,
    summary: pd.DataFrame,
    predictions: pd.DataFrame,
    source_paths: Mapping[str, Path],
) -> None:
    phase4 = ROOT / "outputs" / "phase4"
    phase4.mkdir(parents=True, exist_ok=True)
    tests = [
        paired_test(predictions, "simple_fusion", "text_only"),
        paired_test(predictions, "interaction_fusion", "text_only"),
        paired_test(predictions, "interaction_fusion", "simple_fusion"),
    ]
    indexed = summary.set_index("model")
    simple_beats = bool(
        indexed.loc["simple_fusion", "mae_1to5_mean"]
        < indexed.loc["text_only", "mae_1to5_mean"]
    )
    interaction_beats = bool(
        indexed.loc["interaction_fusion", "mae_1to5_mean"]
        < indexed.loc["text_only", "mae_1to5_mean"]
    )
    fusion_gap = float(
        abs(
            indexed.loc["simple_fusion", "mae_1to5_mean"]
            - indexed.loc["interaction_fusion", "mae_1to5_mean"]
        )
    )
    variants_close = fusion_gap < 0.003
    fusion_test = tests[-1]
    variants_statistically_indistinguishable = bool(
        fusion_test["bootstrap_ci_low"] <= 0.0 <= fusion_test["bootstrap_ci_high"]
        and fusion_test["wilcoxon_p"] >= 0.05
    )
    extraction = pd.read_parquet(
        source_paths["qwen"],
        columns=[
            "ok",
            "traveler_segment",
            *[f"aspect_{aspect}_mentioned" for aspect in U0.ASPECTS],
        ],
    )
    diagnostics = {
        "ok_rate": float(extraction["ok"].eq(1).mean()),
        "unknown_segment_rate": float(
            extraction["traveler_segment"].fillna("unknown").eq("unknown").mean()
        ),
        "mention_coverage": {
            aspect: float(
                extraction[f"aspect_{aspect}_mentioned"].fillna(False).astype(bool).mean()
            )
            for aspect in U0.ASPECTS
        },
    }
    diagnostic_rows = [
        {"diagnostic": "ok_rate", "aspect": "overall", "value": diagnostics["ok_rate"]},
        {
            "diagnostic": "unknown_segment_rate",
            "aspect": "overall",
            "value": diagnostics["unknown_segment_rate"],
        },
        *[
            {
                "diagnostic": "mention_coverage",
                "aspect": aspect,
                "value": diagnostics["mention_coverage"][aspect],
            }
            for aspect in U0.ASPECTS
        ],
    ]
    source = json.loads(
        (ROOT / "outputs" / "phase3" / "w_source_and_sample.json").read_text()
    )
    payload = {
        "experiment": "Phase4_U5_Booking_U0",
        "seed": SEED,
        "protocol": {
            "outer_split": "hotel-intact 5-fold split from Phase 3 W1",
            "inner_cv": "GroupKFold(5), group=hotel address hash",
            "alpha_grid": U0.ALPHAS.tolist(),
            "boundary_extension": "two decades until interior",
            "prediction_clipping": [1.0, 5.0],
        },
        "corpus": source,
        "sample_n": int(len(sample)),
        "properties": int(sample["offering_id"].nunique()),
        "extraction_diagnostics": diagnostics,
        "summary": summary.to_dict(orient="records"),
        "folds": folds.to_dict(orient="records"),
        "paired_property_tests": tests,
        "criteria": {
            "simple_fusion_beats_text": simple_beats,
            "interaction_fusion_beats_text": interaction_beats,
            "fusion_variants_abs_gap_below_0.003": variants_close,
            "fusion_variants_statistically_indistinguishable": variants_statistically_indistinguishable,
            "fusion_variant_mae_gap": fusion_gap,
        },
        "qualitative_findings_replicate": bool(
            simple_beats
            and interaction_beats
            and variants_close
            and variants_statistically_indistinguishable
        ),
        "artefacts": {key: str(path.relative_to(ROOT)) for key, path in source_paths.items()},
    }
    folds.to_csv(phase4 / "booking_u0_folds.csv", index=False, float_format="%.6f")
    summary.to_csv(
        phase4 / "booking_u0_summary.csv", index=False, float_format="%.6f"
    )
    predictions.to_parquet(phase4 / "booking_u0_predictions.parquet", index=False)
    pd.DataFrame(tests).to_csv(
        phase4 / "booking_u0_paired_tests.csv", index=False, float_format="%.6f"
    )
    pd.DataFrame(diagnostic_rows).to_csv(
        phase4 / "booking_u0_extraction_diagnostics.csv",
        index=False,
        float_format="%.6f",
    )
    (phase4 / "booking_u0_results.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    lines = [
        "# Phase 4 U5 — Booking.com External Validation under U0",
        "",
        f"- sample: {len(sample):,} reviews / {sample['offering_id'].nunique():,} properties",
        "- inner selection: GroupKFold(5) by property; U0 41-point alpha grid",
        "- native-to-1–5 mapping: `1 + 4 * (native - 1) / 9`",
        f"- extraction OK rate: {diagnostics['ok_rate']:.4f}",
        f"- unknown segment rate: {diagnostics['unknown_segment_rate']:.4f}",
        "",
        "## Extraction diagnostics",
        "",
        pd.DataFrame(diagnostic_rows).to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Model results",
        "",
        summary.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Per-property paired tests",
        "",
        pd.DataFrame(tests).to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Verdict",
        "",
        f"- Both fusion variants beat text-only: {simple_beats and interaction_beats}.",
        f"- Fusion variants have gap <0.003: {variants_close} (gap={fusion_gap:.6f}); paired-property statistical indistinguishability: {variants_statistically_indistinguishable}.",
        f"- Both qualitative findings replicate: {payload['qualitative_findings_replicate']}.",
        "",
    ]
    (phase4 / "booking_u0_results.md").write_text("\n".join(lines), encoding="utf-8")
    log(f"saved Phase 4 Booking U0 results: {phase4 / 'booking_u0_results.md'}")


def main() -> None:
    args = parse_args()
    folds_requested = parse_folds(args.folds)
    np.random.seed(SEED)
    source_paths = {
        "dataset": resolve(args.dataset_file),
        "sample": resolve(args.sample_file),
        "qwen": resolve(args.qwen_path),
    }
    for path in source_paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    sample = W3.load_sample(source_paths["dataset"], source_paths["sample"])
    features = W3.load_features(
        source_paths["qwen"], sample["row_id"].to_numpy(dtype=np.int64)
    )
    texts = sample["review_text"].fillna("").astype(str).to_numpy()
    outer_fold = sample["outer_fold"].to_numpy(dtype=np.int8)
    harness = EvaluationHarness(ROOT / "outputs")
    log(
        f"Phase 4 U5 sample={len(sample):,} properties={sample['offering_id'].nunique():,} "
        f"folds={folds_requested} inner_cv=GroupKFold(5) inner_jobs={args.inner_jobs}"
    )

    for fold in tqdm(folds_requested, desc="Phase4 U5 Booking folds"):
        train_pos = np.flatnonzero(outer_fold != fold)
        test_pos = np.flatnonzero(outer_fold == fold)
        prefix = W3.cache_prefix(
            ROOT / "outputs" / "phase3", args.max_features, fold, source_paths["sample"]
        )
        text_train, text_test = W3.load_or_build_tfidf(
            texts,
            train_pos,
            test_pos,
            args.max_features,
            prefix,
            args.resume,
        )
        simple_train, simple_test = scale_block(
            features["simple"], train_pos, test_pos
        )
        interaction_train, interaction_test = scale_block(
            features["interaction"], train_pos, test_pos
        )
        variants = {
            "text_only": ({}, {}),
            "simple_fusion": ({"z": simple_train}, {"z": simple_test}),
            "interaction_fusion": (
                {"z_and_interactions": interaction_train},
                {"z_and_interactions": interaction_test},
            ),
        }
        for model_name, (train_blocks, test_blocks) in variants.items():
            started = time.perf_counter()
            fit_fold(
                harness,
                model_name,
                fold,
                sample,
                train_pos,
                test_pos,
                text_train,
                text_test,
                train_blocks,
                test_blocks,
                args,
            )
            log(
                f"U5 {model_name} fold={fold} wall_seconds={time.perf_counter() - started:.1f}"
            )
        del text_train, text_test, simple_train, simple_test
        del interaction_train, interaction_test, variants
        gc.collect()

    if set(folds_requested) != set(range(1, N_SPLITS + 1)):
        log("partial-fold run complete; final tables require all five folds")
        return
    metadata = {
        "protocol": "Phase 4 U0 unified GroupKFold(5)",
        "alpha_grid": U0.ALPHAS.tolist(),
        "boundary_extension": "two decades until interior",
        "seed": SEED,
        "corpus": "Booking.com external sample",
    }
    for model_id in MODEL_IDS.values():
        harness.finalize_model(model_id, {**metadata, "model_id": model_id})
    folds, summary, predictions = collect_outputs(harness, sample)
    write_results(sample, features, folds, summary, predictions, source_paths)


if __name__ == "__main__":
    main()
