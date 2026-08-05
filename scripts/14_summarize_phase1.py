#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation_harness import write_json  # noqa: E402


SEED = 42
N_RESAMPLES = 10_000
PUBLISHED_IDS = {
    "text_only_tfidf_ridge",
    "llm_aspects_only",
    "text_llm_simple_fusion",
    "residual_on_llm",
    "upper_bound_true_aspects",
    "interaction_fusion",
}
REFERENCE_IDS = ["text_only_tfidf_ridge", "interaction_fusion"]


def log(message: str) -> None:
    print(message, flush=True)


def load_metrics(metrics_dir: Path) -> Dict[str, Dict[str, Any]]:
    payloads: Dict[str, Dict[str, Any]] = {}
    for path in sorted(metrics_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        model_id = payload.get("model_id")
        if model_id and "aggregate" in payload:
            payloads[str(model_id)] = payload
    return payloads


def infer_n_features(payload: Dict[str, Any]) -> Optional[int]:
    fold = payload["folds"][0]
    extra = fold.get("extra", {})
    if "n_features" in extra:
        return int(extra["n_features"])
    tfidf = extra.get("tfidf", {})
    if "vocabulary_size" in tfidf:
        return int(tfidf["vocabulary_size"]) + int(
            extra.get("structured_features", 0)
        )
    feature_names = extra.get("feature_names")
    if isinstance(feature_names, list):
        return len(feature_names)
    description = str(payload.get("meta", {}).get("description", "")).lower()
    if "true aspect" in description:
        return 6
    return None


def total_gpu_hours(payload: Dict[str, Any]) -> float:
    if "gpu_hours_total" in payload:
        return float(payload["gpu_hours_total"])
    return float(
        sum(
            fold.get("extra", {}).get("gpu_hours", 0.0)
            for fold in payload["folds"]
        )
    )


def shared_upstream_gpu_hours(
    model_id: str,
    payloads: Dict[str, Dict[str, Any]],
) -> float:
    is_frozen_bert = model_id.startswith(
        ("bert_base_uncased_mean_", "bert_base_uncased_cls_")
    )
    if not is_frozen_bert:
        return 0.0
    training_hours = total_gpu_hours(
        payloads["bert_base_uncased_regression"]
    )
    metadata_path = (
        ROOT
        / "outputs"
        / "phase1"
        / "embeddings"
        / "bert_base_uncased"
        / "extraction_metadata.json"
    )
    if not metadata_path.exists():
        return training_hours
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    extraction_seconds = sum(
        split["elapsed_seconds"]
        for fold in metadata["folds"].values()
        for split in fold["splits"].values()
    )
    return training_hours + (float(extraction_seconds) / 3600.0)


def consolidated_rows(
    payloads: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    full_corpus_n = max(
        int(payload["n_predictions"]) for payload in payloads.values()
    )
    for model_id, payload in sorted(payloads.items()):
        aggregate = payload["aggregate"]["clipped"]
        meta = payload.get("meta", {})
        n_predictions = int(payload["n_predictions"])
        rows.append(
            {
                "model_id": model_id,
                "description": meta.get("description", model_id),
                "mae_mean": aggregate["mean"]["mae"],
                "mae_sd": aggregate["std"]["mae"],
                "rmse_mean": aggregate["mean"]["rmse"],
                "rmse_sd": aggregate["std"]["rmse"],
                "n_features": infer_n_features(payload),
                "fit_seconds_per_fold": payload["timing"][
                    "fit_seconds"
                ]["mean"],
                "tuning_seconds_per_fold": payload["timing"][
                    "tuning_seconds"
                ]["mean"],
                "gpu_hours": total_gpu_hours(payload),
                "shared_upstream_gpu_hours": shared_upstream_gpu_hours(
                    model_id,
                    payloads,
                ),
                "deployable": bool(
                    meta.get(
                        "deployable",
                        model_id != "upper_bound_true_aspects",
                    )
                ),
                "evaluation_scope": (
                    "full_corpus"
                    if n_predictions == full_corpus_n
                    else f"stratified_subsample_n{n_predictions}"
                ),
                "n_predictions": n_predictions,
                "n_hotels": int(payload["n_hotels"]),
                "metrics_path": str(
                    ROOT / "outputs" / "metrics" / f"{model_id}.json"
                ),
            }
        )
    return rows


def selected_alpha_rows(
    payloads: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for model_id, payload in sorted(payloads.items()):
        alphas = [
            fold.get("selected_hyperparameters", {}).get("alpha")
            for fold in payload["folds"]
        ]
        if not any(alpha is not None for alpha in alphas):
            continue
        rows.append(
            {
                "model_id": model_id,
                **{
                    f"fold_{fold}": alpha
                    for fold, alpha in enumerate(alphas, start=1)
                },
            }
        )
    return rows


def load_predictions(output_dir: Path, model_id: str) -> pd.DataFrame:
    paths = sorted(
        (output_dir / "predictions" / model_id).glob("fold*.parquet")
    )
    if len(paths) != 5:
        raise FileNotFoundError(
            f"{model_id}: expected five prediction files, found {len(paths)}"
        )
    return pd.concat(
        [
            pd.read_parquet(
                path,
                columns=[
                    "row_id",
                    "offering_id",
                    "y_true",
                    "y_pred_clipped",
                ],
            )
            for path in paths
        ],
        ignore_index=True,
    )


def align_per_hotel(
    output_dir: Path,
    first_id: str,
    second_id: str,
) -> Tuple[np.ndarray, int, int]:
    first = load_predictions(output_dir, first_id).rename(
        columns={
            "offering_id": "offering_id_first",
            "y_true": "y_true_first",
            "y_pred_clipped": "prediction_first",
        }
    )
    second = load_predictions(output_dir, second_id).rename(
        columns={
            "offering_id": "offering_id_second",
            "y_true": "y_true_second",
            "y_pred_clipped": "prediction_second",
        }
    )
    paired = first.merge(second, on="row_id", how="inner", validate="one_to_one")
    if paired.empty:
        raise ValueError(f"{first_id} and {second_id} share no predictions")
    if not np.array_equal(
        paired["offering_id_first"].to_numpy(),
        paired["offering_id_second"].to_numpy(),
    ):
        raise ValueError("offering_id mismatch after row_id alignment")
    if not np.allclose(
        paired["y_true_first"],
        paired["y_true_second"],
    ):
        raise ValueError("target mismatch after row_id alignment")
    paired["first_error"] = np.abs(
        paired["y_true_first"] - paired["prediction_first"]
    )
    paired["second_error"] = np.abs(
        paired["y_true_second"] - paired["prediction_second"]
    )
    hotel = paired.groupby("offering_id_first", sort=True).agg(
        first=("first_error", "mean"),
        second=("second_error", "mean"),
    )
    return (
        hotel["first"].to_numpy() - hotel["second"].to_numpy(),
        len(paired),
        len(hotel),
    )


def paired_statistics(
    output_dir: Path,
    first_id: str,
    second_id: str,
) -> Dict[str, Any]:
    differences, n_rows, n_hotels = align_per_hotel(
        output_dir,
        first_id,
        second_id,
    )
    if np.allclose(differences, 0):
        wilcoxon_statistic = 0.0
        p_value = 1.0
    else:
        test = stats.wilcoxon(
            differences,
            zero_method="wilcox",
            alternative="two-sided",
            method="auto",
        )
        wilcoxon_statistic = float(test.statistic)
        p_value = float(test.pvalue)
    rng = np.random.default_rng(SEED)
    bootstrap_parts = []
    for start in range(0, N_RESAMPLES, 250):
        count = min(250, N_RESAMPLES - start)
        indices = rng.integers(
            0,
            n_hotels,
            size=(count, n_hotels),
        )
        bootstrap_parts.append(differences[indices].mean(axis=1))
    bootstrap = np.concatenate(bootstrap_parts)
    ci = np.quantile(bootstrap, [0.025, 0.975])
    return {
        "first_model": first_id,
        "second_model": second_id,
        "difference": "first_minus_second_per_hotel_mae",
        "n_shared_rows": int(n_rows),
        "n_shared_hotels": int(n_hotels),
        "mean_difference": float(differences.mean()),
        "median_difference": float(np.median(differences)),
        "wilcoxon_statistic": wilcoxon_statistic,
        "wilcoxon_p_value": p_value,
        "bootstrap_resamples": N_RESAMPLES,
        "bootstrap_seed": SEED,
        "bootstrap_ci_95": [float(ci[0]), float(ci[1])],
    }


def prediction_metrics(frame: pd.DataFrame) -> Dict[str, float]:
    errors = (
        frame["y_true"].to_numpy(dtype=float)
        - frame["y_pred_clipped"].to_numpy(dtype=float)
    )
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
    }


def direct_llm_diagnostics(
    output_dir: Path,
    payloads: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    diagnostics: Dict[str, Any] = {
        "note": (
            "Direct-LLM metrics use the fixed stratified 100,000-review "
            "subsample; same-row reference metrics below are the valid "
            "comparisons."
        ),
        "models": {},
    }
    references = [
        model_id
        for model_id in REFERENCE_IDS
        if model_id in payloads
    ]
    reference_predictions = {
        model_id: load_predictions(output_dir, model_id).set_index("row_id")
        for model_id in references
    }
    for model_id in (
        "qwen_direct_zero_n100000",
        "qwen_direct_few_n100000",
    ):
        if model_id not in payloads:
            continue
        frame = load_predictions(output_dir, model_id)
        rounded = np.rint(frame["y_pred_clipped"]).astype(int)
        truth = np.rint(frame["y_true"]).astype(int)
        confusion = pd.crosstab(
            truth,
            rounded,
            rownames=["true"],
            colnames=["predicted"],
            dropna=False,
        ).reindex(index=range(1, 6), columns=range(1, 6), fill_value=0)
        same_row_metrics = {model_id: prediction_metrics(frame)}
        row_ids = frame["row_id"].to_numpy()
        for reference_id, reference in reference_predictions.items():
            aligned = reference.loc[row_ids].reset_index()
            same_row_metrics[reference_id] = prediction_metrics(aligned)
        diagnostics["models"][model_id] = {
            "n": int(len(frame)),
            "exact_accuracy": float(np.mean(truth == rounded)),
            "mean_prediction": float(frame["y_pred_clipped"].mean()),
            "mean_target": float(frame["y_true"].mean()),
            "target_distribution": {
                str(rating): int((truth == rating).sum())
                for rating in range(1, 6)
            },
            "prediction_distribution": {
                str(rating): int((rounded == rating).sum())
                for rating in range(1, 6)
            },
            "confusion_matrix_true_by_predicted": {
                str(true_rating): {
                    str(predicted_rating): int(
                        confusion.loc[true_rating, predicted_rating]
                    )
                    for predicted_rating in range(1, 6)
                }
                for true_rating in range(1, 6)
            },
            "same_row_metrics": same_row_metrics,
        }
    return diagnostics


def direct_diagnostics_markdown(diagnostics: Dict[str, Any]) -> List[str]:
    lines = [
        "# Direct Qwen Rating Diagnostics",
        "",
        diagnostics["note"],
        "",
    ]
    for model_id, item in diagnostics["models"].items():
        lines.extend(
            [
                f"## {model_id}",
                "",
                f"- n: {item['n']}",
                f"- exact integer accuracy: {item['exact_accuracy']:.4f}",
                f"- mean target: {item['mean_target']:.4f}",
                f"- mean prediction: {item['mean_prediction']:.4f}",
                "",
                "### Same-row Metrics",
                "",
                *markdown_table(
                    ["model", "MAE", "RMSE"],
                    [
                        [
                            compared_id,
                            f"{metrics['mae']:.4f}",
                            f"{metrics['rmse']:.4f}",
                        ]
                        for compared_id, metrics in item[
                            "same_row_metrics"
                        ].items()
                    ],
                ),
                "",
                "### Rating Distributions",
                "",
                *markdown_table(
                    ["rating", "target n", "prediction n"],
                    [
                        [
                            rating,
                            item["target_distribution"][str(rating)],
                            item["prediction_distribution"][str(rating)],
                        ]
                        for rating in range(1, 6)
                    ],
                ),
                "",
                "### Confusion Matrix (True × Predicted)",
                "",
                *markdown_table(
                    ["true \\ predicted", "1", "2", "3", "4", "5"],
                    [
                        [
                            true_rating,
                            *[
                                item[
                                    "confusion_matrix_true_by_predicted"
                                ][str(true_rating)][str(predicted_rating)]
                                for predicted_rating in range(1, 6)
                            ],
                        ]
                        for true_rating in range(1, 6)
                    ],
                ),
                "",
            ]
        )
    return lines


def b1_channels(
    payloads: Dict[str, Dict[str, Any]],
) -> List[Tuple[str, str, str, str]]:
    channels = [
        (
            "TF-IDF sparse",
            "text_only_tfidf_ridge",
            "text_llm_simple_fusion",
            "interaction_fusion",
        ),
        (
            "TF-IDF+SVD-300 dense (paper z)",
            "tfidf_svd300_mean_text_only_paper",
            "tfidf_svd300_mean_plus_z_paper",
            "tfidf_svd300_mean_plus_zint_paper",
        ),
        (
            "BERT mean contextual (paper z)",
            "bert_base_uncased_mean_text_only_paper",
            "bert_base_uncased_mean_plus_z_paper",
            "bert_base_uncased_mean_plus_zint_paper",
        ),
        (
            "BERT CLS contextual (paper z)",
            "bert_base_uncased_cls_text_only_paper",
            "bert_base_uncased_cls_plus_z_paper",
            "bert_base_uncased_cls_plus_zint_paper",
        ),
        (
            "BERT end-to-end contextual (paper z)",
            "bert_base_uncased_end_to_end_text_only_paper",
            "bert_base_uncased_end_to_end_plus_z_paper",
            "bert_base_uncased_end_to_end_plus_zint_paper",
        ),
        (
            "BERT mean contextual (legacy extraction scale)",
            "bert_base_uncased_mean_text_only_legacy",
            "bert_base_uncased_mean_plus_z_legacy",
            "bert_base_uncased_mean_plus_zint_legacy",
        ),
    ]
    return [
        channel
        for channel in channels
        if all(model_id in payloads for model_id in channel[1:])
    ]


def build_b1_rows(
    output_dir: Path,
    payloads: Dict[str, Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    comparisons: List[Dict[str, Any]] = []
    for channel, text_id, additive_id, interaction_id in b1_channels(
        payloads
    ):
        text_mae = payloads[text_id]["aggregate"]["clipped"]["mean"]["mae"]
        additive_mae = payloads[additive_id]["aggregate"]["clipped"]["mean"][
            "mae"
        ]
        additive_vs_text = paired_statistics(
            output_dir,
            additive_id,
            text_id,
        )
        interaction_vs_text = paired_statistics(
            output_dir,
            interaction_id,
            text_id,
        )
        interaction_vs_additive = paired_statistics(
            output_dir,
            interaction_id,
            additive_id,
        )
        for comparison, label in (
            (additive_vs_text, "plus_z_vs_text"),
            (interaction_vs_text, "plus_zint_vs_text"),
            (interaction_vs_additive, "plus_zint_vs_plus_z"),
        ):
            comparisons.append(
                {
                    "channel": channel,
                    "comparison": label,
                    **comparison,
                }
            )
        for variant, model_id in (
            ("text_only", text_id),
            ("plus_z", additive_id),
            ("plus_z_int", interaction_id),
        ):
            aggregate = payloads[model_id]["aggregate"]["clipped"]
            comparison_text = {
                "text_only": None,
                "plus_z": additive_vs_text,
                "plus_z_int": interaction_vs_text,
            }[variant]
            comparison_additive = (
                interaction_vs_additive
                if variant == "plus_z_int"
                else None
            )
            rows.append(
                {
                    "channel": channel,
                    "variant": variant,
                    "model_id": model_id,
                    "mae_mean": aggregate["mean"]["mae"],
                    "mae_sd": aggregate["std"]["mae"],
                    "rmse_mean": aggregate["mean"]["rmse"],
                    "rmse_sd": aggregate["std"]["rmse"],
                    "delta_mae_vs_text": (
                        aggregate["mean"]["mae"] - text_mae
                    ),
                    "per_hotel_delta_mae_vs_text": (
                        None
                        if comparison_text is None
                        else comparison_text["mean_difference"]
                    ),
                    "delta_mae_per_hotel_ci_95": (
                        None
                        if comparison_text is None
                        else comparison_text["bootstrap_ci_95"]
                    ),
                    "wilcoxon_p_vs_text": (
                        None
                        if comparison_text is None
                        else comparison_text["wilcoxon_p_value"]
                    ),
                    "delta_mae_vs_additive": (
                        aggregate["mean"]["mae"] - additive_mae
                        if variant == "plus_z_int"
                        else None
                    ),
                    "per_hotel_delta_mae_vs_additive": (
                        None
                        if comparison_additive is None
                        else comparison_additive["mean_difference"]
                    ),
                    "delta_mae_vs_additive_ci_95": (
                        None
                        if comparison_additive is None
                        else comparison_additive["bootstrap_ci_95"]
                    ),
                    "wilcoxon_p_vs_additive": (
                        None
                        if comparison_additive is None
                        else comparison_additive["wilcoxon_p_value"]
                    ),
                }
            )
    return rows, comparisons


def format_value(mean: float, sd: float) -> str:
    return f"{mean:.4f} ± {sd:.4f}"


def markdown_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[Any]],
) -> List[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in rows
    )
    return lines


def write_repository_inventory(root: Path, output_path: Path) -> None:
    rows: List[Dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if ".git" in path.parts or path == output_path:
            continue
        stat = path.stat()
        rows.append(
            {
                "path": str(path.relative_to(root)),
                "kind": "directory" if path.is_dir() else "file",
                "size_bytes": stat.st_size if path.is_file() else None,
                "modified_utc": datetime.fromtimestamp(
                    stat.st_mtime,
                    tz=timezone.utc,
                ).isoformat(),
            }
        )
    pd.DataFrame(rows).to_csv(output_path, sep="\t", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize all Phase 0/1 harness artifacts."
    )
    parser.add_argument("--profile", default="qwen25_7b")
    args = parser.parse_args()

    output_dir = ROOT / "outputs"
    payloads = load_metrics(output_dir / "metrics")
    consolidated = consolidated_rows(payloads)
    consolidated_frame = pd.DataFrame(consolidated)
    consolidated_frame["scope_order"] = (
        consolidated_frame["evaluation_scope"] != "full_corpus"
    ).astype(int)
    consolidated_frame = consolidated_frame.sort_values(
        ["scope_order", "mae_mean"]
    ).drop(columns=["scope_order"])
    consolidated_frame.to_csv(
        output_dir / "phase1_consolidated_results.csv",
        index=False,
    )
    write_json(
        output_dir / "phase1_consolidated_results.json",
        {"models": consolidated},
    )
    alpha_rows = selected_alpha_rows(payloads)
    pd.DataFrame(alpha_rows).to_csv(
        output_dir / "phase1_selected_alphas.csv",
        index=False,
    )
    write_json(
        output_dir / "phase1_selected_alphas.json",
        {"models": alpha_rows},
    )

    new_models = sorted(set(payloads) - PUBLISHED_IDS)
    pairwise: List[Dict[str, Any]] = []
    for model_id in new_models:
        for reference_id in REFERENCE_IDS:
            if reference_id not in payloads:
                continue
            log(f"paired statistics: {model_id} vs {reference_id}")
            pairwise.append(
                paired_statistics(output_dir, model_id, reference_id)
            )
    write_json(
        output_dir / "phase1_pairwise_statistics.json",
        {"comparisons": pairwise},
    )

    b1_rows, b1_comparisons = build_b1_rows(output_dir, payloads)
    pd.DataFrame(b1_rows).to_csv(
        output_dir / "phase1_b1_results.csv",
        index=False,
    )
    write_json(
        output_dir / "phase1_b1_results.json",
        {
            "rows": b1_rows,
            "paired_comparisons": b1_comparisons,
        },
    )
    write_json(
        output_dir / "phase1_b1_pairwise_statistics.json",
        {"comparisons": b1_comparisons},
    )

    direct_diagnostics = direct_llm_diagnostics(output_dir, payloads)
    write_json(
        output_dir / "phase1_direct_llm_diagnostics.json",
        direct_diagnostics,
    )
    (
        output_dir / "phase1_direct_llm_diagnostics.md"
    ).write_text(
        "\n".join(direct_diagnostics_markdown(direct_diagnostics)),
        encoding="utf-8",
    )

    token_stats = {}
    token_dir = output_dir / "phase1" / "token_stats"
    for path in sorted(token_dir.glob("*.json")):
        token_stats[path.stem] = json.loads(
            path.read_text(encoding="utf-8")
        )

    anomalies: List[str] = []
    text_mae = payloads["text_only_tfidf_ridge"]["aggregate"]["clipped"][
        "mean"
    ]["mae"]
    for model_id in ("bert_base_uncased_regression", "roberta_base_regression"):
        if model_id in payloads:
            mae = payloads[model_id]["aggregate"]["clipped"]["mean"]["mae"]
            if mae >= text_mae:
                anomalies.append(
                    f"{model_id} did not beat TF-IDF Ridge "
                    f"({mae:.4f} >= {text_mae:.4f})."
                )
    for channel, _, additive_id, interaction_id in b1_channels(payloads):
        additive_mae = payloads[additive_id]["aggregate"]["clipped"][
            "mean"
        ]["mae"]
        interaction_mae = payloads[interaction_id]["aggregate"]["clipped"][
            "mean"
        ]["mae"]
        incremental_gain = interaction_mae - additive_mae
        if interaction_mae >= additive_mae:
            anomalies.append(
                f"Interaction Fusion failed to improve additive fusion on "
                f"{channel} ({interaction_mae:.4f} >= {additive_mae:.4f})."
            )
        elif abs(incremental_gain) < 0.001:
            comparison = next(
                item
                for item in b1_comparisons
                if item["channel"] == channel
                and item["comparison"] == "plus_zint_vs_plus_z"
            )
            anomalies.append(
                "Interaction-specific improvement is negligible in magnitude "
                f"on {channel}: aggregate ΔMAE={incremental_gain:+.4f}; "
                "per-hotel ΔMAE="
                f"{comparison['mean_difference']:+.4f}, 95% CI "
                f"[{comparison['bootstrap_ci_95'][0]:+.4f}, "
                f"{comparison['bootstrap_ci_95'][1]:+.4f}], "
                f"p={comparison['wilcoxon_p_value']:.4g}."
            )
    zero_id = "qwen_direct_zero_n100000"
    few_id = "qwen_direct_few_n100000"
    if zero_id in payloads and few_id in payloads:
        zero = payloads[zero_id]["aggregate"]["clipped"]["mean"]
        few = payloads[few_id]["aggregate"]["clipped"]["mean"]
        if few["mae"] > zero["mae"]:
            anomalies.append(
                "Direct Qwen few-shot underperformed zero-shot on the same "
                "stratified 100k sample "
                f"(MAE {few['mae']:.4f} vs {zero['mae']:.4f}; "
                f"RMSE {few['rmse']:.4f} vs {zero['rmse']:.4f})."
            )
        anomalies.append(
            "Direct Qwen results are a stratified 100k subsample, not a "
            "full-corpus result. Zero-shot has unusually low MAE but high "
            f"RMSE ({zero['mae']:.4f}/{zero['rmse']:.4f}); use the same-row "
            "diagnostic rather than ranking it directly with full-corpus runs."
        )
    svd_metadata_path = (
        output_dir
        / "phase1"
        / "embeddings"
        / "tfidf_svd300"
        / "extraction_metadata.json"
    )
    if svd_metadata_path.exists():
        svd_metadata = json.loads(svd_metadata_path.read_text(encoding="utf-8"))
        explained = [
            float(item["explained_variance_ratio_sum"])
            for item in svd_metadata["folds"].values()
        ]
        anomalies.append(
            "The 300-dimensional SVD text channel retains only "
            f"{np.mean(explained):.4%} mean explained variance and is much "
            "weaker than sparse TF-IDF; its large +z gain should not be "
            "interpreted as evidence that interaction gains generalize."
        )
    anomalies.extend(
        [
            "The requested inner learning-rate selection for A1 was not run: "
            "both transformer baselines used the pre-specified 2e-5 rate and "
            "grouped inner validation selected the best epoch only.",
            "The interaction-specific increment is not consistently large "
            "across frozen dense channels; most of their gain comes from the "
            "additive structured vector. End-to-end BERT shows a larger, but "
            "still modest, +z_int over +z gain.",
            "Optional DeBERTa and Qwen LoRA experiments were not run.",
        ]
    )

    full_corpus_rows = [
        row
        for row in consolidated
        if row["evaluation_scope"] == "full_corpus"
    ]
    best_full_mae = min(full_corpus_rows, key=lambda row: row["mae_mean"])
    best_full_rmse = min(
        full_corpus_rows,
        key=lambda row: row["rmse_mean"],
    )
    interaction_increment = {
        item["channel"]: {
            "aggregate_delta_mae": next(
                row["delta_mae_vs_additive"]
                for row in b1_rows
                if row["channel"] == item["channel"]
                and row["variant"] == "plus_z_int"
            ),
            "per_hotel_delta_mae": item["mean_difference"],
            "bootstrap_ci_95": item["bootstrap_ci_95"],
            "wilcoxon_p_value": item["wilcoxon_p_value"],
        }
        for item in b1_comparisons
        if item["comparison"] == "plus_zint_vs_plus_z"
    }
    roberta = payloads["roberta_base_regression"]["aggregate"]["clipped"][
        "mean"
    ]
    published_interaction = payloads["interaction_fusion"]["aggregate"][
        "clipped"
    ]["mean"]
    key_findings = {
        "best_full_corpus_mae": best_full_mae,
        "best_full_corpus_rmse": best_full_rmse,
        "roberta_vs_published_interaction": {
            "delta_mae": (
                roberta["mae"] - published_interaction["mae"]
            ),
            "delta_rmse": (
                roberta["rmse"] - published_interaction["rmse"]
            ),
        },
        "interaction_increment_by_channel": interaction_increment,
        "interpretation": [
            "The strong-text-baseline concern is valid: RoBERTa materially "
            "improves MAE over the published Interaction Fusion model.",
            "The LLM structured vector has clear additive value on frozen "
            "BERT and the weak SVD channel.",
            "The interaction-specific increment is not representation-"
            "agnostic in effect size: it is large for sparse TF-IDF, tiny "
            "for frozen BERT/SVD, and modest for end-to-end BERT.",
        ],
    }
    write_json(
        output_dir / "phase1_key_findings.json",
        key_findings,
    )

    report_lines = [
        "# Phase 1 Consolidated Report",
        "",
        "## Executive Findings",
        "",
        f"- Best full-corpus MAE: `{best_full_mae['model_id']}` at "
        f"{best_full_mae['mae_mean']:.4f} ± "
        f"{best_full_mae['mae_sd']:.4f}.",
        f"- Best full-corpus RMSE: `{best_full_rmse['model_id']}` at "
        f"{best_full_rmse['rmse_mean']:.4f} ± "
        f"{best_full_rmse['rmse_sd']:.4f}.",
        "- The strong-text-baseline criticism is supported: RoBERTa reduces "
        "MAE relative to the published Interaction Fusion model by "
        f"{roberta['mae'] - published_interaction['mae']:+.4f}, although "
        "its RMSE is "
        f"{roberta['rmse'] - published_interaction['rmse']:+.4f} higher.",
        "- The structured LLM signal generalizes as an additive feature, but "
        "the interaction-specific increment does **not** generalize with a "
        "large effect: frozen BERT mean and CLS improve only 0.0001 "
        "aggregate MAE over +z; end-to-end BERT improves 0.0039.",
        "- Therefore, the defensible revised claim is additive structured "
        "signal generality plus channel-dependent interaction benefit, not "
        "a uniformly representation-agnostic interaction gain.",
        "",
        "Machine-readable: `outputs/phase1_key_findings.json`.",
        "",
        "## Consolidated Results",
        "",
        *markdown_table(
            [
                "model_id",
                "MAE mean ± sd",
                "RMSE mean ± sd",
                "n_features",
                "fit s/fold",
                "GPU-hours (incremental)",
                "shared upstream GPU-hours",
                "deployable",
                "scope",
            ],
            [
                [
                    row.model_id,
                    format_value(row.mae_mean, row.mae_sd),
                    format_value(row.rmse_mean, row.rmse_sd),
                    "" if pd.isna(row.n_features) else int(row.n_features),
                    f"{row.fit_seconds_per_fold:.2f}",
                    f"{row.gpu_hours:.3f}",
                    (
                        ""
                        if row.shared_upstream_gpu_hours == 0
                        else f"{row.shared_upstream_gpu_hours:.3f}"
                    ),
                    "yes" if row.deployable else "no",
                    row.evaluation_scope,
                ]
                for row in consolidated_frame.itertuples()
            ],
        ),
        "",
        "Direct Qwen rows are evaluated on a fixed stratified 100,000-review "
        "subsample and must not be ranked as if they were full-corpus rows.",
        " Frozen BERT ridge rows have zero incremental GPU-hours; their shared "
        "fold-specific encoder training plus embedding extraction cost is "
        "reported separately and must not be summed once per variant.",
        "",
        "Machine-readable: `outputs/phase1_consolidated_results.json` and "
        "`outputs/phase1_consolidated_results.csv`.",
        "",
        "## Selected Ridge Alpha by Fold",
        "",
        *markdown_table(
            ["model_id", "fold 1", "fold 2", "fold 3", "fold 4", "fold 5"],
            [
                [
                    item["model_id"],
                    *[
                        item[f"fold_{fold}"]
                        for fold in range(1, 6)
                    ],
                ]
                for item in alpha_rows
            ],
        ),
        "",
        "Machine-readable: `outputs/phase1_selected_alphas.json` and "
        "`outputs/phase1_selected_alphas.csv`.",
        "",
        "## New-model Paired Tests",
        "",
        *markdown_table(
            [
                "new model",
                "reference",
                "Δ per-hotel MAE",
                "bootstrap 95% CI",
                "Wilcoxon p",
                "shared hotels",
            ],
            [
                [
                    item["first_model"],
                    item["second_model"],
                    f"{item['mean_difference']:.4f}",
                    (
                        f"[{item['bootstrap_ci_95'][0]:.4f}, "
                        f"{item['bootstrap_ci_95'][1]:.4f}]"
                    ),
                    f"{item['wilcoxon_p_value']:.4g}",
                    item["n_shared_hotels"],
                ]
                for item in pairwise
            ],
        ),
        "",
        "Machine-readable: `outputs/phase1_pairwise_statistics.json`.",
        "",
        "## B1 Representation-Generalization Results",
        "",
        *markdown_table(
            [
                "text channel",
                "variant",
                "model_id",
                "MAE mean ± sd",
                "RMSE mean ± sd",
                "ΔMAE vs text",
                "per-hotel Δ vs text (95% CI)",
                "ΔMAE vs +z",
                "per-hotel Δ vs +z (95% CI; p)",
            ],
            [
                [
                    item["channel"],
                    item["variant"],
                    item["model_id"],
                    format_value(item["mae_mean"], item["mae_sd"]),
                    format_value(item["rmse_mean"], item["rmse_sd"]),
                    f"{item['delta_mae_vs_text']:.4f}",
                    (
                        ""
                        if item["delta_mae_per_hotel_ci_95"] is None
                        else (
                            f"{item['per_hotel_delta_mae_vs_text']:+.4f} "
                            f"[{item['delta_mae_per_hotel_ci_95'][0]:.4f}, "
                            f"{item['delta_mae_per_hotel_ci_95'][1]:.4f}]"
                        )
                    ),
                    (
                        ""
                        if item["delta_mae_vs_additive"] is None
                        else f"{item['delta_mae_vs_additive']:+.4f}"
                    ),
                    (
                        ""
                        if item["delta_mae_vs_additive_ci_95"] is None
                        else (
                            f"{item['per_hotel_delta_mae_vs_additive']:+.4f} "
                            f"[{item['delta_mae_vs_additive_ci_95'][0]:.4f}, "
                            f"{item['delta_mae_vs_additive_ci_95'][1]:.4f}]; "
                            f"p={item['wilcoxon_p_vs_additive']:.4g}"
                        )
                    ),
                ]
                for item in b1_rows
            ],
        ),
        "",
        "Machine-readable: `outputs/phase1_b1_results.json` and "
        "`outputs/phase1_b1_results.csv`; all paired tests: "
        "`outputs/phase1_b1_pairwise_statistics.json`.",
        " Aggregate deltas weight reviews through fold metrics; paired "
        "deltas and confidence intervals weight held-out hotels equally, so "
        "their magnitudes need not match.",
        "",
        "## Direct Qwen Subsample Diagnostic",
        "",
        *markdown_table(
            [
                "model",
                "n",
                "exact accuracy",
                "MAE",
                "RMSE",
                "TF-IDF same-row MAE/RMSE",
                "Interaction same-row MAE/RMSE",
            ],
            [
                [
                    model_id,
                    item["n"],
                    f"{item['exact_accuracy']:.4f}",
                    (
                        f"{item['same_row_metrics'][model_id]['mae']:.4f}"
                    ),
                    (
                        f"{item['same_row_metrics'][model_id]['rmse']:.4f}"
                    ),
                    (
                        f"{item['same_row_metrics']['text_only_tfidf_ridge']['mae']:.4f}/"
                        f"{item['same_row_metrics']['text_only_tfidf_ridge']['rmse']:.4f}"
                    ),
                    (
                        f"{item['same_row_metrics']['interaction_fusion']['mae']:.4f}/"
                        f"{item['same_row_metrics']['interaction_fusion']['rmse']:.4f}"
                    ),
                ]
                for model_id, item in direct_diagnostics["models"].items()
            ],
        ),
        "",
        "The direct models emit integers, which can lower MAE while retaining "
        "a heavier error tail and therefore a comparatively high RMSE. Full "
        "confusion matrices: `outputs/phase1_direct_llm_diagnostics.md`; "
        "machine-readable: `outputs/phase1_direct_llm_diagnostics.json`.",
        "",
        "## Token Truncation",
        "",
    ]
    if token_stats:
        report_lines.extend(
            markdown_table(
                [
                    "model",
                    "mean",
                    "median",
                    "p90",
                    "p99",
                    "truncated fraction",
                ],
                [
                    [
                        model_id,
                        f"{item['mean']:.2f}",
                        f"{item['median']:.2f}",
                        f"{item['p90']:.2f}",
                        f"{item['p99']:.2f}",
                        f"{item['truncated_fraction']:.4%}",
                    ]
                    for model_id, item in token_stats.items()
                ],
            )
        )
    else:
        report_lines.append("- Transformer token audits are not present yet.")
    report_lines.extend(
        [
            "",
            "## Anomalies",
            "",
            *[f"- {item}" for item in anomalies],
            "",
            "## Exact Commands",
            "",
            "The complete long-run orchestration and exact parallel GPU "
            "assignment are archived in `scripts/16_run_phase1_remaining.sh`.",
            "",
            "```bash",
            "CUDA_VISIBLE_DEVICES=0 python scripts/09_train_transformer_regression.py --model-name bert-base-uncased --model-tag bert_base_uncased --device cuda:0 --resume",
            "CUDA_VISIBLE_DEVICES=1 python scripts/09_train_transformer_regression.py --model-name roberta-base --model-tag roberta_base --device cuda:0 --resume",
            "CUDA_VISIBLE_DEVICES=0 python scripts/10_extract_transformer_embeddings.py --model-tag bert_base_uncased --device cuda:0 --resume",
            "python scripts/11_run_representation_fusion.py --model-tag bert_base_uncased --profile qwen25_7b --poolings mean --structured-variant paper --alpha-grid 1,100,1000,10000 --ridge-solver lsqr --ridge-tol 1e-5 --resume",
            "python scripts/11_run_representation_fusion.py --model-tag bert_base_uncased --profile qwen25_7b --poolings cls --structured-variant paper --alpha-grid 1,100,1000,10000 --ridge-solver lsqr --ridge-tol 1e-5 --resume",
            "CUDA_VISIBLE_DEVICES=0 python scripts/15_train_end_to_end_fusion.py --model-tag bert_base_uncased --profile qwen25_7b --device cuda:0 --structured-variant paper --variants text_only,plus_z,plus_zint --resume",
            "python scripts/12_extract_tfidf_svd_embeddings.py --profile qwen25_7b --resume",
            "python scripts/11_run_representation_fusion.py --model-tag tfidf_svd300 --poolings mean --profile qwen25_7b --structured-variant paper --alpha-grid 1,100,1000,10000 --ridge-solver lsqr --ridge-tol 1e-5 --resume",
            "python scripts/13_run_llm_direct_rating.py --profile qwen25_7b --modes zero,few --sample-size 100000 --concurrency 16 --endpoints http://localhost:8000/v1,http://localhost:8001/v1 --write-every 2000 --resume",
            "python scripts/14_summarize_phase1.py --profile qwen25_7b",
            "```",
            "",
            "Configuration: `config/local_llm.yaml`. Saved outer splits: "
            "`outputs/splits_fold{1..5}.npz`.",
            " Long-run status and logs: "
            "`outputs/phase1_long_run_status.tsv` and "
            "`outputs/phase1_*.log`.",
            " Full repository path/size/mtime inventory: "
            "`outputs/repository_inventory.tsv`.",
            "",
        ]
    )
    report_path = output_dir / f"phase1_report_{args.profile}.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    write_repository_inventory(
        ROOT,
        output_dir / "repository_inventory.tsv",
    )
    log(f"saved: {report_path}")


if __name__ == "__main__":
    main()
