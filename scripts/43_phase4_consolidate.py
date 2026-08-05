#!/usr/bin/env python3
"""Phase 4 U1--U7 consolidation and manuscript-ready reporting.

This script performs no model fitting and no LLM extraction.  It validates and
consolidates completed harness artefacts, computes paired hotel-level tests,
builds error/alpha/extractor diagnostics, and emits one machine-readable CSV
per table plus ``phase4_report.json`` and ``phase4_report.md``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.aspect_extractor_base import ASPECTS  # noqa: E402
from src.evaluation_harness import (  # noqa: E402
    EvaluationHarness,
    bootstrap_mean_difference,
    paired_wilcoxon,
)


SEED = 42
N_BOOTSTRAP = 10_000
SPLIT_SHA256 = "dfe3a434fb00c894218309c332627dc0131a2db3cc2cd241727fc712a4803942"
SEGMENTS = ["business", "couple", "family", "solo", "friends", "unknown"]

PRIMARY_IDS = {
    "text_only": "phase4_u0_text_only_tfidf_ridge",
    "llm_aspects_only": "phase4_u0_llm_aspects_only",
    "simple_fusion": "phase4_u0_text_llm_simple_fusion",
    "residual_on_llm": "phase4_u0_residual_on_llm",
    "upper_bound": "phase4_u0_upper_bound_true_aspects",
    "interaction_fusion": "phase4_u0_interaction_fusion",
}
TRANSFORMER_IDS = {
    "roberta": "roberta_base_phase2_lr5em05_regression",
    "bert_cls_plus_zint": "bert_base_uncased_cls_plus_zint_paper",
}
ABLATION_IDS = {
    "A1": PRIMARY_IDS["simple_fusion"],
    "A2": "phase4_u0_a2_score_interactions_only",
    "A3": "phase4_u0_a3_mention_interactions_only",
    "A4": PRIMARY_IDS["interaction_fusion"],
    "A5": "phase4_u0_a5_interactions_without_main",
    "A7": "phase4_u0_a7_unknown_columns_removed",
    "A8": "phase4_u0_a8_known_segment_only",
    "A4_known_reference": "phase4_u0_a4_known_segment_reference",
}
ERROR_IDS = {
    "text_only": PRIMARY_IDS["text_only"],
    "simple_fusion": PRIMARY_IDS["simple_fusion"],
    "interaction_fusion": PRIMARY_IDS["interaction_fusion"],
    "roberta": TRANSFORMER_IDS["roberta"],
    "bert_cls_plus_zint": TRANSFORMER_IDS["bert_cls_plus_zint"],
    "upper_bound": PRIMARY_IDS["upper_bound"],
}
BOOKING_IDS = {
    "phase4_booking_u0_text_only",
    "phase4_booking_u0_simple_fusion",
    "phase4_booking_u0_interaction_fusion",
}
DISPLAY_NAMES = {
    "text_only": "Text-only TF-IDF + Ridge",
    "llm_aspects_only": "LLM aspects only",
    "simple_fusion": "Text + LLM simple fusion [x | z]",
    "residual_on_llm": "Residual-on-LLM",
    "upper_bound": "Upper bound (true aspects)",
    "interaction_fusion": "Interaction fusion [x | z | p_s | p_b]",
    "roberta": "RoBERTa-base regression (max_length=256)",
    "bert_cls_plus_zint": "BERT CLS + z_int",
}


def log(message: str) -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="qwen25_7b")
    parser.add_argument(
        "--allow-partial", action=argparse.BooleanOptionalAction, default=False
    )
    return parser.parse_args()


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    def convert(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return None if not np.isfinite(value) else float(value)
        if isinstance(value, (np.bool_,)):
            return bool(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(type(value).__name__)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=convert), encoding="utf-8")


def save_table(frame: pd.DataFrame, path: Path) -> List[Dict[str, Any]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.6f")
    return frame.replace({np.nan: None}).to_dict(orient="records")


def require(paths: Iterable[Path], allow_partial: bool) -> List[Path]:
    missing = [path for path in paths if not path.exists()]
    if missing and not allow_partial:
        joined = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Phase 4 prerequisites are incomplete:\n{joined}")
    return missing


def load_metrics(model_id: str) -> Dict[str, Any]:
    return read_json(ROOT / "outputs" / "metrics" / f"{model_id}.json")


def aggregate_row(
    panel: str,
    protocol: str,
    key: str,
    payload: Mapping[str, Any],
) -> Dict[str, Any]:
    clipped = payload["aggregate"]["clipped"]
    return {
        "panel": panel,
        "protocol": protocol,
        "model": key,
        "model_id": payload["model_id"],
        "display_name": DISPLAY_NAMES[key],
        "mae_mean": clipped["mean"]["mae"],
        "mae_sd": clipped["std"]["mae"],
        "rmse_mean": clipped["mean"]["rmse"],
        "rmse_sd": clipped["std"]["rmse"],
        "n": payload.get("n_predictions"),
    }


def u1_tables(phase4: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    gate = read_json(ROOT / "outputs" / "phase0_reproduction_gate.json")
    published_by_id = {
        row["model_id"]: row["published"] for row in gate["comparisons"]
    }
    panel_a_rows = []
    published_keys = {
        "text_only": "text_only_tfidf_ridge",
        "llm_aspects_only": "llm_aspects_only",
        "simple_fusion": "text_llm_simple_fusion",
        "residual_on_llm": "residual_on_llm",
        "upper_bound": "upper_bound_true_aspects",
        "interaction_fusion": "interaction_fusion",
    }
    for key, model_id in published_keys.items():
        values = published_by_id[model_id]
        panel_a_rows.append(
            {
                "panel": "Panel A",
                "protocol": "Published/original (retained verbatim)",
                "model": key,
                "model_id": model_id,
                "display_name": DISPLAY_NAMES[key],
                "mae_mean": values["mae_mean"],
                "mae_sd": values["mae_std"],
                "rmse_mean": values["rmse_mean"],
                "rmse_sd": values["rmse_std"],
                "n": 877_808,
            }
        )
    panel_a = pd.DataFrame(panel_a_rows)

    panel_b_rows = []
    for key, model_id in {**PRIMARY_IDS, **TRANSFORMER_IDS}.items():
        payload = load_metrics(model_id)
        protocol = (
            "U0: shared alpha; inner GroupKFold(5); 41-point grid"
            if key in PRIMARY_IDS
            else "Previously completed transformer protocol (not U0-tuned)"
        )
        panel_b_rows.append(aggregate_row("Panel B", protocol, key, payload))
    panel_b = pd.DataFrame(panel_b_rows)
    combined = pd.concat([panel_a, panel_b], ignore_index=True)
    save_table(panel_a, phase4 / "u1_panel_a_published.csv")
    save_table(panel_b, phase4 / "u1_panel_b_u0.csv")
    save_table(combined, phase4 / "u1_combined_panel_a_b.csv")
    return panel_a, panel_b, combined


def feature_and_cost_tables(phase4: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    fold_rows: List[Dict[str, Any]] = []
    model_rows: List[Dict[str, Any]] = []
    phase1 = read_json(ROOT / "outputs" / "phase1_consolidated_results.json")
    phase1_records = {
        row["model_id"]: row for row in phase1.get("models", phase1.get("results", []))
        if isinstance(row, dict) and "model_id" in row
    }
    extraction_gpu_hours = 0.0
    extraction_cache_paths = sorted(
        (ROOT / "outputs" / "llm_cache").glob("qwen25_7b_ep*/cache.sqlite")
    )
    for cache_path in extraction_cache_paths:
        connection = sqlite3.connect(cache_path)
        first, last = connection.execute(
            "SELECT MIN(created_at), MAX(created_at) FROM cache WHERE ok=1"
        ).fetchone()
        connection.close()
        if first and last:
            extraction_gpu_hours += (
                pd.to_datetime(last) - pd.to_datetime(first)
            ).total_seconds() / 3600.0
    llm_feature_models = {
        "llm_aspects_only",
        "simple_fusion",
        "residual_on_llm",
        "interaction_fusion",
        "bert_cls_plus_zint",
    }
    for key, model_id in {**PRIMARY_IDS, **TRANSFORMER_IDS}.items():
        payload = load_metrics(model_id)
        gpu_hours = 0.0
        shared_gpu_hours = 0.0
        for fold in payload["folds"]:
            selected = fold.get("selected_hyperparameters", {})
            extra = fold.get("extra", {})
            alpha = selected.get("alpha")
            if key == "residual_on_llm":
                alpha = selected.get("residual_alpha")
            n_features = extra.get("n_features")
            if key == "roberta":
                config_path = (
                    ROOT
                    / "outputs"
                    / "phase1"
                    / "checkpoints"
                    / "roberta_base_phase2_lr5em05"
                    / f"fold{fold['fold']}"
                    / "config.json"
                )
                n_features = read_json(config_path).get("hidden_size")
            fold_rows.append(
                {
                    "model": key,
                    "model_id": model_id,
                    "fold": fold["fold"],
                    "selected_alpha": alpha,
                    "text_alpha": selected.get("text_alpha"),
                    "boundary_status": selected.get(
                        "boundary_status", "not_applicable_transformer"
                    ),
                    "n_features": n_features,
                    "fit_seconds": fold.get("fit_seconds", 0.0),
                    "tuning_seconds": fold.get("tuning_seconds", 0.0),
                    "peak_memory_mb": fold.get("peak_memory_mb"),
                    "deployable": bool(
                        extra.get("deployable", payload.get("meta", {}).get("deployable", True))
                    ),
                    "mae": fold["clipped"]["mae"],
                    "rmse": fold["clipped"]["rmse"],
                }
            )
            gpu_hours += float(extra.get("gpu_hours", 0.0) or 0.0)
        prior = phase1_records.get(model_id, {})
        shared_gpu_hours = float(prior.get("shared_upstream_gpu_hours", 0.0) or 0.0)
        model_folds = [row for row in fold_rows if row["model_id"] == model_id]
        feature_values = [row["n_features"] for row in model_folds if row["n_features"] is not None]
        model_rows.append(
            {
                "model": key,
                "model_id": model_id,
                "n_features_min": min(feature_values) if feature_values else None,
                "n_features_max": max(feature_values) if feature_values else None,
                "fit_seconds_total": sum(row["fit_seconds"] for row in model_folds),
                "tuning_seconds_total": sum(row["tuning_seconds"] for row in model_folds),
                "gpu_hours_direct": gpu_hours,
                "gpu_hours_shared_upstream": shared_gpu_hours,
                "gpu_hours_shared_llm_extraction": (
                    extraction_gpu_hours if key in llm_feature_models else 0.0
                ),
                "deployable": all(row["deployable"] for row in model_folds),
                "feature_note": (
                    "final hidden width (end-to-end transformer)"
                    if key == "roberta"
                    else "held-out fold design width"
                ),
                "gpu_cost_note": (
                    "LLM extraction is shared; GPU-hours estimated from the summed active created_at span of the two versioned endpoint caches"
                    if key in llm_feature_models
                    else "no LLM extraction dependency"
                ),
            }
        )
    folds = pd.DataFrame(fold_rows)
    costs = pd.DataFrame(model_rows)
    save_table(folds, phase4 / "u1_panel_b_fold_details.csv")
    save_table(costs, phase4 / "u1_panel_b_cost_features.csv")
    return folds, costs


def u1b_disclosure(phase4: Path) -> pd.DataFrame:
    source = pd.read_csv(ROOT / "outputs" / "phase3" / "experiment_g0_alpha_audit.csv")
    source.to_csv(phase4 / "u1b_panel_a_regularization_disclosure.csv", index=False)
    return source


def per_hotel_test(first_id: str, second_id: str) -> Dict[str, Any]:
    first_path = ROOT / "outputs" / "per_hotel" / f"{first_id}.parquet"
    second_path = ROOT / "outputs" / "per_hotel" / f"{second_id}.parquet"
    wilcox = paired_wilcoxon(first_path, second_path, metric="mae")
    bootstrap = bootstrap_mean_difference(
        first_path,
        second_path,
        metric="mae",
        n_resamples=N_BOOTSTRAP,
        seed=SEED,
    )
    return {
        "n_hotels": wilcox["n_hotels"],
        "mean_difference": wilcox["mean_difference"],
        "median_difference": wilcox["median_difference"],
        "wilcoxon_statistic": wilcox["statistic"],
        "wilcoxon_p": wilcox["p_value"],
        "bootstrap_ci_low": bootstrap["ci_95"][0],
        "bootstrap_ci_high": bootstrap["ci_95"][1],
        "bootstrap_resamples": N_BOOTSTRAP,
        "seed": SEED,
    }


def u2_tests(phase4: Path) -> pd.DataFrame:
    contrasts = [
        ("simple_fusion", "text_only"),
        ("interaction_fusion", "text_only"),
        ("interaction_fusion", "simple_fusion"),
        ("residual_on_llm", "text_only"),
        ("llm_aspects_only", "text_only"),
        ("roberta", "interaction_fusion"),
        ("bert_cls_plus_zint", "interaction_fusion"),
    ]
    rows = []
    for first, second in contrasts:
        first_id = (PRIMARY_IDS | TRANSFORMER_IDS)[first]
        second_id = (PRIMARY_IDS | TRANSFORMER_IDS)[second]
        row = {
            "contrast": f"{first}_minus_{second}",
            "first": first,
            "second": second,
            "first_model_id": first_id,
            "second_model_id": second_id,
            **per_hotel_test(first_id, second_id),
        }
        row["significant_0.05"] = row["wilcoxon_p"] < 0.05
        rows.append(row)
    frame = pd.DataFrame(rows)
    target = frame[frame["contrast"].eq("interaction_fusion_minus_simple_fusion")].index[0]
    ci_low = float(frame.loc[target, "bootstrap_ci_low"])
    ci_high = float(frame.loc[target, "bootstrap_ci_high"])
    frame.loc[target, "equivalent_within_0.005"] = bool(ci_low >= -0.005 and ci_high <= 0.005)
    fusion_gain = abs(
        float(
            frame.loc[
                frame["contrast"].eq("interaction_fusion_minus_text_only"),
                "mean_difference",
            ].iloc[0]
        )
    )
    observed = abs(float(frame.loc[target, "mean_difference"]))
    frame.loc[target, "abs_difference_to_fusion_gain_ratio"] = (
        observed / fusion_gain if fusion_gain else np.nan
    )
    save_table(frame, phase4 / "u2_paired_hotel_tests.csv")
    return frame


def curve_value(curve: pd.DataFrame, alpha: float) -> pd.Series:
    distance = np.abs(np.log10(curve["alpha"].to_numpy(dtype=float)) - math.log10(alpha))
    index = int(np.argmin(distance))
    if not np.isclose(float(curve.iloc[index]["alpha"]), alpha, rtol=1e-8):
        raise ValueError(f"alpha {alpha} absent from G1 curve")
    return curve.iloc[index]


def u3_alpha_diagnostic(phase4: Path) -> pd.DataFrame:
    curves = pd.read_csv(ROOT / "outputs" / "phase3" / "alpha_response_curves.csv")
    published_alpha = {"text_only": 1.0, "simple_fusion": 0.1, "interaction_fusion": 10.0}
    rows: List[Dict[str, Any]] = []
    for model, fixed in published_alpha.items():
        metrics = load_metrics(PRIMARY_IDS[model])
        selected_by_fold = {
            int(fold["fold"]): float(fold["selected_hyperparameters"]["alpha"])
            for fold in metrics["folds"]
        }
        for fold in range(1, 6):
            selected = curves[curves["model"].eq(model) & curves["fold"].eq(fold)]
            oracle = selected.sort_values(["test_mae", "alpha"]).iloc[0]
            fixed_row = curve_value(selected, fixed)
            u0_row = curve_value(selected, selected_by_fold[fold])
            rows.append(
                {
                    "model": model,
                    "fold": fold,
                    "published_alpha": fixed,
                    "published_outer_test_mae": fixed_row["test_mae"],
                    "oracle_alpha": oracle["alpha"],
                    "oracle_outer_test_mae": oracle["test_mae"],
                    "u0_alpha": selected_by_fold[fold],
                    "u0_outer_test_mae_from_g1": u0_row["test_mae"],
                    "published_minus_oracle_mae": fixed_row["test_mae"] - oracle["test_mae"],
                    "u0_minus_oracle_mae": u0_row["test_mae"] - oracle["test_mae"],
                }
            )
    frame = pd.DataFrame(rows)
    means = (
        frame.groupby("model", as_index=False)
        .agg(
            published_alpha=("published_alpha", "mean"),
            published_outer_test_mae=("published_outer_test_mae", "mean"),
            oracle_alpha=("oracle_alpha", lambda x: 10 ** np.log10(x).mean()),
            oracle_outer_test_mae=("oracle_outer_test_mae", "mean"),
            u0_alpha=("u0_alpha", lambda x: 10 ** np.log10(x).mean()),
            u0_outer_test_mae_from_g1=("u0_outer_test_mae_from_g1", "mean"),
            published_minus_oracle_mae=("published_minus_oracle_mae", "mean"),
            u0_minus_oracle_mae=("u0_minus_oracle_mae", "mean"),
        )
        .assign(fold="mean")
    )
    output = pd.concat([frame, means], ignore_index=True)
    save_table(output, phase4 / "u3_alpha_diagnostic.csv")
    return output


def aggregate_known_hotel(model_id: str, known: np.ndarray) -> pd.DataFrame:
    frames = []
    for fold in range(1, 6):
        path = ROOT / "outputs" / "predictions" / model_id / f"fold{fold}.parquet"
        frame = pd.read_parquet(path)
        frame = frame[known[frame["row_id"].to_numpy(dtype=np.int64)]].copy()
        frames.append(frame)
    all_rows = pd.concat(frames, ignore_index=True)
    all_rows["absolute_error"] = np.abs(all_rows["y_true"] - all_rows["y_pred_clipped"])
    return (
        all_rows.groupby("offering_id", as_index=False)
        .agg(mae=("absolute_error", "mean"), n_reviews=("row_id", "size"))
    )


def paired_frames(first: pd.DataFrame, second: pd.DataFrame) -> Dict[str, Any]:
    paired = first[["offering_id", "mae"]].merge(
        second[["offering_id", "mae"]],
        on="offering_id",
        suffixes=("_first", "_second"),
        validate="one_to_one",
    )
    difference = (paired["mae_first"] - paired["mae_second"]).to_numpy(dtype=float)
    if np.allclose(difference, 0):
        statistic, p_value = 0.0, 1.0
    else:
        result = stats.wilcoxon(difference, zero_method="wilcox", method="auto")
        statistic, p_value = float(result.statistic), float(result.pvalue)
    rng = np.random.default_rng(SEED)
    bootstrap = np.empty(N_BOOTSTRAP)
    for start in range(0, N_BOOTSTRAP, 250):
        size = min(250, N_BOOTSTRAP - start)
        draws = rng.integers(0, len(difference), size=(size, len(difference)))
        bootstrap[start : start + size] = difference[draws].mean(axis=1)
    return {
        "n_hotels": len(difference),
        "mean_difference": difference.mean(),
        "median_difference": np.median(difference),
        "wilcoxon_statistic": statistic,
        "wilcoxon_p": p_value,
        "bootstrap_ci_low": np.quantile(bootstrap, 0.025),
        "bootstrap_ci_high": np.quantile(bootstrap, 0.975),
        "bootstrap_resamples": N_BOOTSTRAP,
        "seed": SEED,
    }


def u4_ablation(phase4: Path, profile: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for label, model_id in ABLATION_IDS.items():
        payload = load_metrics(model_id)
        clipped = payload["aggregate"]["clipped"]
        alphas = []
        boundaries = []
        n = payload.get("n_predictions", 0)
        for fold in payload["folds"]:
            selected = fold["selected_hyperparameters"]
            alphas.append(selected.get("alpha"))
            boundaries.append(selected.get("boundary_status", "reference"))
        numeric = [float(value) for value in alphas if value is not None]
        rows.append(
            {
                "ablation": label,
                "model_id": model_id,
                "n": n,
                "mae_mean": clipped["mean"]["mae"],
                "mae_sd": clipped["std"]["mae"],
                "rmse_mean": clipped["mean"]["rmse"],
                "rmse_sd": clipped["std"]["rmse"],
                "alpha_geomean": 10 ** np.log10(numeric).mean() if numeric else None,
                "all_interior": all(value in {"interior", "reference"} for value in boundaries),
            }
        )
    table = pd.DataFrame(rows)
    save_table(table, phase4 / "u4_ablation_u0.csv")

    llm = pd.read_parquet(
        ROOT / "outputs" / f"aspects_llm_{profile}.parquet",
        columns=["row_id", "ok", "traveler_segment"],
    ).sort_values("row_id")
    if not np.array_equal(
        llm["row_id"].to_numpy(dtype=np.int64), np.arange(len(llm), dtype=np.int64)
    ):
        raise ValueError("U4 known-segment row_id alignment failed")
    segment = llm["traveler_segment"].fillna("unknown").astype(str).str.lower()
    known = llm["ok"].eq(1).to_numpy() & segment.ne("unknown").to_numpy()
    known_refs = {
        "A1": aggregate_known_hotel(ABLATION_IDS["A1"], known),
        "A4": aggregate_known_hotel(ABLATION_IDS["A4"], known),
    }
    tests = []
    for label, model_id in {key: value for key, value in ABLATION_IDS.items() if key != "A4_known_reference"}.items():
        for reference in ("A4", "A1"):
            if label == "A8":
                first = aggregate_known_hotel(model_id, known)
                result = paired_frames(first, known_refs[reference])
                comparison_note = "matched known-segment rows"
            else:
                result = per_hotel_test(model_id, ABLATION_IDS[reference])
                comparison_note = "full held-out rows"
            tests.append(
                {
                    "ablation": label,
                    "reference": reference,
                    "contrast": f"{label}_minus_{reference}",
                    "comparison_note": comparison_note,
                    **result,
                }
            )
    tests_frame = pd.DataFrame(tests)
    save_table(tests_frame, phase4 / "u4_ablation_paired_tests.csv")
    return table, tests_frame


def icc_2_1(first: np.ndarray, second: np.ndarray) -> float:
    values = np.column_stack([first, second]).astype(float)
    n, k = values.shape
    grand = values.mean()
    row_means = values.mean(axis=1)
    column_means = values.mean(axis=0)
    ss_rows = k * np.square(row_means - grand).sum()
    ss_columns = n * np.square(column_means - grand).sum()
    ss_total = np.square(values - grand).sum()
    ss_error = ss_total - ss_rows - ss_columns
    ms_rows = ss_rows / (n - 1)
    ms_columns = ss_columns / (k - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))
    denominator = ms_rows + (k - 1) * ms_error + k * (ms_columns - ms_error) / n
    return float((ms_rows - ms_error) / denominator) if denominator else float("nan")


def parse_service(response: str) -> Tuple[str, int]:
    try:
        item = json.loads(response)["aspects"]["service"]
        return str(item.get("sentiment", "neu")), int(item.get("score", 0))
    except Exception:
        return "neu", 0


def u6_extractor(phase4: Path) -> Dict[str, pd.DataFrame]:
    confusion = pd.read_csv(ROOT / "outputs" / "phase3" / "l4_segment_confusion.csv", index_col=0)
    confusion = confusion.reindex(index=SEGMENTS, columns=SEGMENTS, fill_value=0)
    confusion["row_total"] = confusion.sum(axis=1)
    total = confusion.sum(axis=0).to_frame().T
    total.index = ["column_total"]
    confusion_with_margins = pd.concat([confusion, total])
    confusion_with_margins.index.name = "qwen_guided\\llama_guided"
    confusion_with_margins.reset_index().to_csv(
        phase4 / "u6_segment_confusion_with_marginals.csv", index=False
    )

    l6 = pd.read_csv(ROOT / "outputs" / "phase3" / "l6_qwen_subsample_vs_full.csv")
    evidence = pd.read_csv(ROOT / "outputs" / "phase3" / "l4_evidence_hit_rate.csv")
    score = pd.read_csv(ROOT / "outputs" / "phase3" / "l4_score_agreement.csv")
    mention = pd.read_csv(ROOT / "outputs" / "phase3" / "l4_mention_agreement.csv")
    save_table(l6, phase4 / "u6_l6_subsample_vs_full.csv")
    save_table(evidence, phase4 / "u6_evidence_hit_rate.csv")
    save_table(score, phase4 / "u6_score_agreement.csv")
    save_table(mention, phase4 / "u6_mention_agreement.csv")

    columns = [
        "row_id",
        "ok",
        "aspect_service_score",
        "aspect_service_mentioned",
        "response_json",
    ]
    qwen = pd.read_parquet(
        ROOT / "outputs" / "phase3" / "aspects_guided_qwen25_7b_l1_n200041.parquet",
        columns=columns,
    )
    llama = pd.read_parquet(
        ROOT / "outputs" / "phase3" / "aspects_guided_llama31_8b_l1_n200041.parquet",
        columns=columns,
    )
    paired = qwen.merge(llama, on="row_id", suffixes=("_qwen", "_llama"), validate="one_to_one")
    paired = paired[paired["ok_qwen"].eq(1) & paired["ok_llama"].eq(1)].copy()
    for suffix in ("qwen", "llama"):
        paired[f"aspect_service_score_{suffix}"] = pd.to_numeric(
            paired[f"aspect_service_score_{suffix}"], errors="coerce"
        ).fillna(0)
        paired[f"aspect_service_mentioned_{suffix}"] = paired[
            f"aspect_service_mentioned_{suffix}"
        ].fillna(False).astype(bool)

    masks = {
        "all_valid": np.ones(len(paired), dtype=bool),
        "both_mentioned": (
            paired["aspect_service_mentioned_qwen"]
            & paired["aspect_service_mentioned_llama"]
        ).to_numpy(),
        "qwen_unmentioned_llama_mentioned": (
            ~paired["aspect_service_mentioned_qwen"]
            & paired["aspect_service_mentioned_llama"]
        ).to_numpy(),
        "both_unmentioned": (
            ~paired["aspect_service_mentioned_qwen"]
            & ~paired["aspect_service_mentioned_llama"]
        ).to_numpy(),
    }
    strata_rows = []
    for label, mask in masks.items():
        selected = paired.loc[mask]
        first = selected["aspect_service_score_qwen"].to_numpy(dtype=float)
        second = selected["aspect_service_score_llama"].to_numpy(dtype=float)
        pearson = (
            float(np.corrcoef(first, second)[0, 1])
            if len(selected) > 2 and first.std() > 0 and second.std() > 0
            else np.nan
        )
        strata_rows.append(
            {
                "stratum": label,
                "n": len(selected),
                "mention_rate_qwen": selected["aspect_service_mentioned_qwen"].mean(),
                "mention_rate_llama": selected["aspect_service_mentioned_llama"].mean(),
                "score_variance_qwen": first.var() if len(first) else np.nan,
                "score_variance_llama": second.var() if len(second) else np.nan,
                "pearson_r": pearson,
                "icc_2_1": icc_2_1(first, second) if len(selected) > 2 else np.nan,
                "exact_score_agreement": (first == second).mean() if len(first) else np.nan,
                "mean_score_qwen": first.mean() if len(first) else np.nan,
                "mean_score_llama": second.mean() if len(second) else np.nan,
            }
        )
    service_strata = pd.DataFrame(strata_rows)
    save_table(service_strata, phase4 / "u6_service_score_strata.csv")

    parsed_qwen = paired["response_json_qwen"].map(parse_service)
    parsed_llama = paired["response_json_llama"].map(parse_service)
    sentiment_qwen = np.asarray([item[0] for item in parsed_qwen])
    score_qwen = np.asarray([item[1] for item in parsed_qwen])
    sentiment_llama = np.asarray([item[0] for item in parsed_llama])
    score_llama = np.asarray([item[1] for item in parsed_llama])
    expected = {"neg": -1, "neu": 0, "pos": 1}
    direction_qwen = np.asarray([expected.get(value, 0) for value in sentiment_qwen])
    direction_llama = np.asarray([expected.get(value, 0) for value in sentiment_llama])
    both = masks["both_mentioned"]
    mention_qwen = paired["aspect_service_mentioned_qwen"].astype(int).to_numpy()
    mention_llama = paired["aspect_service_mentioned_llama"].astype(int).to_numpy()
    mention_exact = float((mention_qwen == mention_llama).mean())
    semantics = pd.DataFrame(
        [
            {
                "n": len(paired),
                "sentiment_exact_all": float((sentiment_qwen == sentiment_llama).mean()),
                "sentiment_exact_both_mentioned": float(
                    (sentiment_qwen[both] == sentiment_llama[both]).mean()
                ),
                "score_sentiment_direction_consistency_qwen": float(
                    (np.sign(score_qwen) == direction_qwen).mean()
                ),
                "score_sentiment_direction_consistency_llama": float(
                    (np.sign(score_llama) == direction_llama).mean()
                ),
                "llama_negative_sentiment_positive_score_rate": float(
                    ((sentiment_llama == "neg") & (score_llama > 0)).mean()
                ),
                "llama_positive_sentiment_nonpositive_score_rate": float(
                    ((sentiment_llama == "pos") & (score_llama <= 0)).mean()
                ),
                "mention_exact_agreement": mention_exact,
                "mention_pabak": 2 * mention_exact - 1,
                "mention_variance_qwen": float(mention_qwen.var()),
                "mention_variance_llama": float(mention_llama.var()),
                "assessment": "schema-semantics/prompt sensitivity; not genuine sentiment disagreement",
                "score_semantics_source": "scripts/18_phase2_schema_enforcement.py:45-48,63-66 (enum specified; polarity mapping absent)",
            }
        ]
    )
    save_table(semantics, phase4 / "u6_service_semantics_test.csv")
    return {
        "confusion": confusion_with_margins.reset_index(),
        "l6": l6,
        "evidence": evidence,
        "score": score,
        "mention": mention,
        "service_strata": service_strata,
        "service_semantics": semantics,
    }


def load_all_predictions(model_id: str) -> pd.DataFrame:
    frames = [
        pd.read_parquet(
            ROOT / "outputs" / "predictions" / model_id / f"fold{fold}.parquet"
        )
        for fold in range(1, 6)
    ]
    frame = pd.concat(frames, ignore_index=True)
    if frame["row_id"].duplicated().any():
        raise ValueError(f"duplicate held-out rows for {model_id}")
    return frame


def u7_error_structure(phase4: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rating_rows = []
    distribution_rows = []
    for label, model_id in ERROR_IDS.items():
        frame = load_all_predictions(model_id)
        error = np.abs(frame["y_true"] - frame["y_pred_clipped"])
        for rating in range(1, 6):
            mask = np.isclose(frame["y_true"], rating)
            selected = error[mask]
            rating_rows.append(
                {
                    "model": label,
                    "model_id": model_id,
                    "true_rating": rating,
                    "n": int(mask.sum()),
                    "mae": float(selected.mean()),
                    "rmse": float(np.sqrt(np.square(selected).mean())),
                }
            )
        rounded = np.clip(np.rint(frame["y_pred_clipped"]), 1, 5)
        distribution_rows.append(
            {
                "model": label,
                "model_id": model_id,
                "n": len(frame),
                "abs_error_lt_0.5": float((error < 0.5).mean()),
                "abs_error_lt_1": float((error < 1.0).mean()),
                "abs_error_ge_1.5": float((error >= 1.5).mean()),
                "abs_error_ge_2": float((error >= 2.0).mean()),
                "rounded_exact_accuracy": float(
                    np.isclose(rounded, frame["y_true"]).mean()
                ),
            }
        )
    ratings = pd.DataFrame(rating_rows)
    distribution = pd.DataFrame(distribution_rows)
    save_table(ratings, phase4 / "u7_error_by_true_rating.csv")
    save_table(distribution, phase4 / "u7_error_distribution.csv")
    return ratings, distribution


def markdown_table(frame: pd.DataFrame, columns: Sequence[str] | None = None) -> str:
    selected = frame[list(columns)] if columns else frame
    return selected.to_markdown(index=False, floatfmt=".4f")


def build_report(
    phase4: Path,
    panel_a: pd.DataFrame,
    panel_b: pd.DataFrame,
    combined: pd.DataFrame,
    fold_details: pd.DataFrame,
    costs: pd.DataFrame,
    disclosure: pd.DataFrame,
    u2: pd.DataFrame,
    u3: pd.DataFrame,
    u4: pd.DataFrame,
    u4_tests: pd.DataFrame,
    booking: Dict[str, Any],
    u6: Dict[str, pd.DataFrame],
    u7_rating: pd.DataFrame,
    u7_distribution: pd.DataFrame,
) -> str:
    panel_display = combined.copy()
    panel_display["MAE"] = panel_display.apply(
        lambda row: f"{row.mae_mean:.4f} +/- {row.mae_sd:.4f}", axis=1
    )
    panel_display["RMSE"] = panel_display.apply(
        lambda row: f"{row.rmse_mean:.4f} +/- {row.rmse_sd:.4f}", axis=1
    )
    interaction_simple = u2[
        u2["contrast"].eq("interaction_fusion_minus_simple_fusion")
    ].iloc[0]
    semantics = u6["service_semantics"].iloc[0]
    strata = u6["service_strata"].set_index("stratum")
    booking_summary = pd.DataFrame(booking["summary"])
    booking_tests = pd.DataFrame(booking["paired_property_tests"])
    booking_diagnostics = pd.DataFrame(
        [
            {"aspect": aspect, "mention_coverage": value}
            for aspect, value in booking["extraction_diagnostics"]["mention_coverage"].items()
        ]
    )
    boundary_failures = fold_details[
        ~fold_details["boundary_status"].isin(["interior", "not_applicable_transformer"])
    ]
    panel_b_index = panel_b.set_index("model")
    simple_better = bool(
        panel_b_index.loc["simple_fusion", "mae_mean"]
        < panel_b_index.loc["text_only", "mae_mean"]
    )
    interaction_better = bool(
        panel_b_index.loc["interaction_fusion", "mae_mean"]
        < panel_b_index.loc["simple_fusion", "mae_mean"]
    )
    anomaly_lines = [
        f"- U0 reverses the published additive-fusion deficit: simple fusion beats text-only={simple_better}.",
        f"- Interaction has lower fold-mean MAE than simple fusion={interaction_better}; the paired-hotel +/-0.005 equivalence criterion is {bool(interaction_simple['equivalent_within_0.005'])}.",
        f"- U0 also reverses the Phase 2 corrected R0/R1 ordering, where simple fusion was marginally better than interaction fusion: under U0, interaction-minus-simple equal-hotel MAE is {float(interaction_simple['mean_difference']):.6f} (95% CI [{float(interaction_simple['bootstrap_ci_low']):.6f}, {float(interaction_simple['bootstrap_ci_high']):.6f}]). Because that CI remains wholly inside +/-0.005, the direction change is reported as protocol-sensitive practical equivalence, not evidence that either fusion form is universally superior.",
        f"- Models remaining on a U0 boundary after extension: {len(boundary_failures)}.",
        "- The true-aspect upper bound can retain a lower MAE but a higher RMSE because aspect missingness produces a heavy error tail (documented in the Phase 0/2 error diagnostics).",
        "- RoBERTa max_length=512 remains optional and was not included because only fold 1 existed; U1--U7 were not delayed.",
    ]
    lines = [
        "# Phase 4 — Consolidation and Manuscript-Ready Tables",
        "",
        f"- canonical split assignment SHA-256: `{SPLIT_SHA256}`",
        f"- resampling seed: {SEED}; hotel bootstrap resamples: {N_BOOTSTRAP:,}",
        "- Panel A is copied verbatim from `outputs/phase0_reproduction_gate.json`; it was not recomputed or overwritten.",
        "- Panel B uses U0: one shared alpha/model, 41 log-spaced values from 1e-4 to 1e6, GroupKFold(5) by offering_id, and two-decade boundary extension.",
        "",
        "## U1 — Combined Panel A / Panel B",
        "",
        markdown_table(panel_display, ["panel", "protocol", "display_name", "MAE", "RMSE"]),
        "",
        "Machine table: `outputs/phase4/u1_combined_panel_a_b.csv`.",
        "",
        "### Panel B model cost and width",
        "",
        markdown_table(costs),
        "",
        "### Panel B per-fold alpha/boundary audit",
        "",
        markdown_table(fold_details),
        "",
        "Machine companion: `outputs/phase4/u1_panel_b_fold_details.csv`.",
        "",
        "## U1b — Published Panel A regularization disclosure",
        "",
        markdown_table(disclosure),
        "",
        "The file/line references above are the auditable source evidence; the table is copied from the accepted G0 audit.",
        "",
        "## U2 — Paired hotel-level tests under U0",
        "",
        markdown_table(u2),
        "",
        f"Interaction vs simple equivalence within +/-0.005 MAE: **{bool(interaction_simple['equivalent_within_0.005'])}**. The CI is [{float(interaction_simple['bootstrap_ci_low']):.4f}, {float(interaction_simple['bootstrap_ci_high']):.4f}]: it is wholly below zero (interaction has lower equal-hotel MAE) while also wholly inside the practical-equivalence band. Its absolute observed difference is {float(interaction_simple['abs_difference_to_fusion_gain_ratio']):.4f} of the interaction-vs-text gain.",
        "",
        "## U3 — Alpha diagnostic from existing G1 curves",
        "",
        markdown_table(u3),
        "",
        "These are outer-test diagnostics, not model-selection inputs. U0 alpha selection used only inner grouped folds.",
        "",
        "## U4 — Ablations under U0",
        "",
        markdown_table(u4),
        "",
        "### Paired tests against A4 and A1",
        "",
        markdown_table(u4_tests),
        "",
        "A8 is evaluated and tested only on matched known-segment held-out rows; its matched A4 reference is retained as a separate harness artefact.",
        "",
        "## U5 — Booking.com external corpus",
        "",
        f"- clean source corpus: {booking['corpus']['clean_rows']:,} reviews / {booking['corpus']['hotels']:,} properties; U0 sample: {booking['sample_n']:,} / {booking['properties']:,}",
        f"- date range: {booking['corpus']['date_min']} to {booking['corpus']['date_max']}",
        f"- source: {booking['corpus']['source_url']}",
        f"- mirror licence label: {booking['corpus']['mirror_license_label']}; derivative label: {booking['corpus']['kaggle_derivative_license_label']}",
        f"- download date: {booking['corpus']['download_date_utc']}; native score range: {booking['corpus']['native_score_min']}--{booking['corpus']['native_score_max']}",
        f"- mapping to 1--5: `{booking['corpus']['mapping_to_1to5']}`",
        f"- licence caveat: {booking['corpus']['license_caveat']}",
        f"- extraction OK={booking['extraction_diagnostics']['ok_rate']:.4f}; unknown segment={booking['extraction_diagnostics']['unknown_segment_rate']:.4f}",
        "",
        "### Extraction mention coverage",
        "",
        markdown_table(booking_diagnostics),
        "",
        markdown_table(booking_summary),
        "",
        "### Per-property paired tests",
        "",
        markdown_table(booking_tests),
        "",
        f"Both qualitative findings replicate under U0: **{booking['qualitative_findings_replicate']}** (fusion beats text-only; the fusion variants are within 0.003 MAE and statistically indistinguishable by paired-property CI/Wilcoxon).",
        "",
        "## U6 — Extractor sensitivity",
        "",
        "### Guided Qwen vs guided Llama traveler-segment confusion (with marginals)",
        "",
        u6["confusion"].to_markdown(index=False),
        "",
        "### L6: Qwen 200k subsample vs full corpus",
        "",
        markdown_table(u6["l6"]),
        "",
        "### Evidence-span hit rates",
        "",
        markdown_table(u6["evidence"]),
        "",
        "### Per-aspect score agreement",
        "",
        markdown_table(u6["score"]),
        "",
        "### Per-aspect mention agreement",
        "",
        markdown_table(u6["mention"]),
        "",
        "### Service anomaly assessment",
        "",
        f"The service score anomaly is most consistent with a **schema-semantics/prompt-sensitivity difference, not genuine sentiment disagreement**. On rows where both extractors mention service, sentiment labels agree {float(semantics['sentiment_exact_both_mentioned']):.4f}, while score-to-sentiment direction consistency is {float(semantics['score_sentiment_direction_consistency_qwen']):.4f} for Qwen but only {float(semantics['score_sentiment_direction_consistency_llama']):.4f} for Llama. The prompt enumerates the allowed score values but does not define their polarity mapping (`scripts/18_phase2_schema_enforcement.py:45-48,63-66`). Llama therefore emits examples such as sentiment=neg with score=2. Near-universal Llama service mention ({u6['mention'].set_index('aspect').loc['service','mention_rate_llama']:.4f}) collapses indicator variance and depresses kappa (exact agreement={float(semantics['mention_exact_agreement']):.4f}, PABAK={float(semantics['mention_pabak']):.4f}), but it does **not** explain negative score ICC: conditioning on both-mentioned rows makes ICC more negative ({strata.loc['all_valid','icc_2_1']:.4f} to {strata.loc['both_mentioned','icc_2_1']:.4f}).",
        "",
        "Machine companions: `u6_segment_confusion_with_marginals.csv`, `u6_l6_subsample_vs_full.csv`, `u6_evidence_hit_rate.csv`, `u6_score_agreement.csv`, `u6_mention_agreement.csv`, `u6_service_score_strata.csv`, and `u6_service_semantics_test.csv`.",
        "",
        "## U7 — Error structure",
        "",
        "### By true rating",
        "",
        markdown_table(u7_rating),
        "",
        "### Error-threshold proportions and rounded accuracy",
        "",
        markdown_table(u7_distribution),
        "",
        "## Anomalies and conclusion checks",
        "",
        *anomaly_lines,
        "",
        "## Artefacts, commands, and seeds",
        "",
        "Primary commands:",
        "",
        "```bash",
        "python scripts/41_phase4_u0_models.py --stage primary --folds 1,2,3,4,5 --inner-jobs 5 --resume",
        "python scripts/41_phase4_u0_models.py --stage ablations --variants A2,A5,A8 --folds 1,2,3,4,5 --inner-jobs 5 --resume",
        "python scripts/41_phase4_u0_models.py --stage ablations --variants A3,A7 --folds 1,2,3,4,5 --inner-jobs 3 --resume",
        "python scripts/41_phase4_u0_models.py --stage ablations --variants A3 --folds 3,4,5 --inner-jobs 3 --resume  # idempotent fold-level acceleration",
        "python scripts/42_phase4_booking_u0.py --folds 1,2,3,4,5 --inner-jobs 5 --resume",
        "python scripts/43_phase4_consolidate.py --profile qwen25_7b",
        "```",
        "",
        f"All bootstrap procedures use seed {SEED}; all are 10,000 equal-hotel/property resamples. Model folds use the saved canonical hotel-intact split. No extraction was run in Phase 4.",
        "",
        "A complete hashed artefact manifest is `outputs/phase4/phase4_artifact_manifest.csv`.",
        "",
    ]
    return "\n".join(lines)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(phase4: Path) -> pd.DataFrame:
    paths = sorted(path for path in phase4.rglob("*") if path.is_file())
    paths.extend(
        [
            ROOT / "outputs" / "phase4_report.md",
            ROOT / "outputs" / "phase4_report.json",
            ROOT / "scripts" / "41_phase4_u0_models.py",
            ROOT / "scripts" / "42_phase4_booking_u0.py",
            ROOT / "scripts" / "43_phase4_consolidate.py",
            ROOT / "outputs" / "phase0_reproduction_gate.json",
            ROOT / "outputs" / "phase3" / "experiment_g0_alpha_audit.csv",
            ROOT / "outputs" / "phase3" / "alpha_response_curves.csv",
        ]
    )
    model_ids = set(PRIMARY_IDS.values()) | set(ABLATION_IDS.values())
    model_ids |= set(TRANSFORMER_IDS.values()) | BOOKING_IDS
    for model_id in sorted(model_ids):
        paths.extend(
            [
                ROOT / "outputs" / "metrics" / f"{model_id}.json",
                ROOT / "outputs" / "per_hotel" / f"{model_id}.parquet",
            ]
        )
        prediction_dir = ROOT / "outputs" / "predictions" / model_id
        if prediction_dir.exists():
            paths.extend(path for path in prediction_dir.iterdir() if path.is_file())
    rows = []
    for path in sorted(set(paths)):
        if path.exists() and path.name != "phase4_artifact_manifest.csv":
            rows.append(
                {
                    "path": str(path.relative_to(ROOT)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(phase4 / "phase4_artifact_manifest.csv", index=False)
    return frame


def main() -> None:
    args = parse_args()
    output_dir = ROOT / "outputs"
    phase4 = output_dir / "phase4"
    phase4.mkdir(parents=True, exist_ok=True)
    required_models = list(PRIMARY_IDS.values()) + list(TRANSFORMER_IDS.values())
    required_models += [
        model_id
        for label, model_id in ABLATION_IDS.items()
        if label not in {"A1", "A4"}
    ]
    required = [output_dir / "metrics" / f"{model_id}.json" for model_id in required_models]
    required += [
        phase4 / "booking_u0_results.json",
        output_dir / "phase0_reproduction_gate.json",
        output_dir / "phase3" / "experiment_g0_alpha_audit.csv",
        output_dir / "phase3" / "alpha_response_curves.csv",
    ]
    missing = require(required, args.allow_partial)
    if missing:
        log(f"partial mode: {len(missing)} prerequisites missing; full report cannot be emitted")
        return

    log("U1: building Panel A/Panel B tables")
    panel_a, panel_b, combined = u1_tables(phase4)
    fold_details, costs = feature_and_cost_tables(phase4)
    disclosure = u1b_disclosure(phase4)
    log("U2: paired hotel Wilcoxon and 10,000-resample bootstrap")
    u2 = u2_tests(phase4)
    log("U3: alpha diagnostics from existing G1 curves")
    u3 = u3_alpha_diagnostic(phase4)
    log("U4: ablation table and paired tests")
    u4, u4_tests = u4_ablation(phase4, args.profile)
    log("U5: loading completed Booking U0 results")
    booking = read_json(phase4 / "booking_u0_results.json")
    log("U6: extractor confusion/evidence/service semantics tests")
    u6 = u6_extractor(phase4)
    log("U7: pooled held-out error structure")
    u7_rating, u7_distribution = u7_error_structure(phase4)

    payload = {
        "phase": 4,
        "status": "FINAL",
        "seed": SEED,
        "bootstrap_resamples": N_BOOTSTRAP,
        "split_assignment_sha256": SPLIT_SHA256,
        "tables": {
            "u1_panel_a": panel_a.to_dict(orient="records"),
            "u1_panel_b": panel_b.to_dict(orient="records"),
            "u1_combined": combined.to_dict(orient="records"),
            "u1_fold_details": fold_details.to_dict(orient="records"),
            "u1_costs": costs.to_dict(orient="records"),
            "u1b_disclosure": disclosure.to_dict(orient="records"),
            "u2_paired_tests": u2.to_dict(orient="records"),
            "u3_alpha_diagnostic": u3.to_dict(orient="records"),
            "u4_ablation": u4.to_dict(orient="records"),
            "u4_tests": u4_tests.to_dict(orient="records"),
            "u5_booking": booking,
            "u6": {key: value.to_dict(orient="records") for key, value in u6.items()},
            "u7_by_rating": u7_rating.to_dict(orient="records"),
            "u7_distribution": u7_distribution.to_dict(orient="records"),
        },
        "optional_u8": {
            "status": "skipped",
            "reason": "Only max_length=512 fold 1 exists; U1-U7 were prioritized as specified.",
        },
        "commands": [
            "python scripts/41_phase4_u0_models.py --stage primary --folds 1,2,3,4,5 --inner-jobs 5 --resume",
            "python scripts/41_phase4_u0_models.py --stage ablations --variants A2,A5,A8 --folds 1,2,3,4,5 --inner-jobs 5 --resume",
            "python scripts/41_phase4_u0_models.py --stage ablations --variants A3,A7 --folds 1,2,3,4,5 --inner-jobs 3 --resume",
            "python scripts/41_phase4_u0_models.py --stage ablations --variants A3 --folds 3,4,5 --inner-jobs 3 --resume  # idempotent fold-level acceleration",
            "python scripts/42_phase4_booking_u0.py --folds 1,2,3,4,5 --inner-jobs 5 --resume",
            f"python scripts/43_phase4_consolidate.py --profile {args.profile}",
        ],
    }
    report = build_report(
        phase4,
        panel_a,
        panel_b,
        combined,
        fold_details,
        costs,
        disclosure,
        u2,
        u3,
        u4,
        u4_tests,
        booking,
        u6,
        u7_rating,
        u7_distribution,
    )
    (output_dir / "phase4_report.md").write_text(report, encoding="utf-8")
    write_json(output_dir / "phase4_report.json", payload)
    manifest = write_manifest(phase4)
    payload["artifact_manifest"] = manifest.to_dict(orient="records")
    write_json(output_dir / "phase4_report.json", payload)
    # Refresh manifest after the final JSON is written.
    write_manifest(phase4)
    log(f"saved FINAL report: {output_dir / 'phase4_report.md'}")


if __name__ == "__main__":
    main()
