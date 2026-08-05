#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
N_SPLITS = 5
CLIP_MIN = 1.0
CLIP_MAX = 5.0


def log(message: str) -> None:
    print(message)


def load_dataset(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    csv_path = path.with_suffix(".csv")
    if csv_path.exists():
        log("parquet not found; loading CSV fallback instead.")
        return pd.read_csv(csv_path, low_memory=False)
    raise FileNotFoundError(f"missing dataset: {path}")


def load_splits(output_dir: Path) -> List[Dict[str, np.ndarray]]:
    splits = []
    for fold in range(1, N_SPLITS + 1):
        split_path = output_dir / f"splits_fold{fold}.npz"
        if not split_path.exists():
            raise FileNotFoundError(f"missing split file: {split_path}")
        data = np.load(split_path)
        splits.append({"train_idx": data["train_idx"], "test_idx": data["test_idx"]})
    return splits


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    return mae, rmse


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
    mae, rmse = _compute_metrics(y_true, y_pred)
    clipped_pred = np.clip(y_pred, CLIP_MIN, CLIP_MAX)
    c_mae, c_rmse = _compute_metrics(y_true, clipped_pred)
    return {
        "raw": {"mae": mae, "rmse": rmse},
        "clipped": {"mae": c_mae, "rmse": c_rmse},
    }


def compute_quantiles(errors: np.ndarray) -> Dict[str, float]:
    qs = np.quantile(errors, [0.5, 0.9, 0.95, 0.99])
    return {"p50": float(qs[0]), "p90": float(qs[1]), "p95": float(qs[2]), "p99": float(qs[3])}


def mean_quantiles(rows: List[Dict[str, float]]) -> Dict[str, float]:
    keys = ["p50", "p90", "p95", "p99"]
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def format_float(value: Optional[float]) -> str:
    if value is None or np.isnan(value):
        return "NA"
    return f"{value:.4f}"


def format_int(value: Optional[int]) -> str:
    if value is None:
        return "NA"
    return str(int(value))


def render_table(headers: List[str], rows: List[List[object]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def run_text_baseline(df: pd.DataFrame, split: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    train_idx = split["train_idx"]
    test_idx = split["test_idx"]
    x_train = df.iloc[train_idx]["review_text"].fillna("").astype(str).values
    y_train = df.iloc[train_idx]["rating_overall"].astype(float).values
    x_test = df.iloc[test_idx]["review_text"].fillna("").astype(str).values
    y_test = df.iloc[test_idx]["rating_overall"].astype(float).values

    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=3,
                    max_features=200000,
                ),
            ),
            ("ridge", Ridge(alpha=1.0, random_state=SEED)),
        ]
    )
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    return y_test, preds


def run_upper_bound(
    df: pd.DataFrame, split: Dict[str, np.ndarray], aspect_cols: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    train_idx = split["train_idx"]
    test_idx = split["test_idx"]
    x_train = df.iloc[train_idx][aspect_cols].astype(float)
    y_train = df.iloc[train_idx]["rating_overall"].astype(float).values
    x_test = df.iloc[test_idx][aspect_cols].astype(float)
    y_test = df.iloc[test_idx]["rating_overall"].astype(float).values
    train_groups = df.iloc[train_idx]["offering_id"].values

    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
            ("model", Ridge()),
        ]
    )
    grid = {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}
    inner_cv = GroupKFold(n_splits=3)
    search = GridSearchCV(
        pipeline,
        grid,
        scoring="neg_mean_absolute_error",
        cv=inner_cv,
        n_jobs=-1,
    )
    search.fit(x_train, y_train, groups=train_groups)
    preds = search.predict(x_test)
    return y_test, preds


def update_missingness_agg(
    agg: Dict[str, Dict[str, Dict[str, float]]],
    group_key: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    variant: str,
) -> None:
    if y_true.size == 0:
        return
    errors = y_true - y_pred
    if variant == "clipped":
        errors = y_true - np.clip(y_pred, CLIP_MIN, CLIP_MAX)
    agg[group_key]["n"] += int(y_true.size)
    agg[group_key]["sum_abs"] += float(np.abs(errors).sum())
    agg[group_key]["sum_sq"] += float(np.square(errors).sum())


def build_long_tail_section(title: str, rows: List[Dict[str, float]]) -> str:
    headers = ["fold", "p50", "p90", "p95", "p99"]
    table_rows = []
    for row in rows:
        table_rows.append(
            [
                row["fold"],
                format_float(row["p50"]),
                format_float(row["p90"]),
                format_float(row["p95"]),
                format_float(row["p99"]),
            ]
        )
    mean_row = mean_quantiles(rows)
    table_rows.append(
        [
            "mean",
            format_float(mean_row["p50"]),
            format_float(mean_row["p90"]),
            format_float(mean_row["p95"]),
            format_float(mean_row["p99"]),
        ]
    )
    return "\n".join([f"### {title}", "", render_table(headers, table_rows), ""])


def build_missingness_table(rows: List[Dict[str, object]], overall: Dict[str, Dict[str, float]]) -> str:
    headers = ["fold", "group", "n", "mae", "rmse"]
    table_rows = []
    for row in rows:
        table_rows.append(
            [
                row["fold"],
                row["group"],
                format_int(row["n"]),
                format_float(row["mae"]),
                format_float(row["rmse"]),
            ]
        )
    for group, metrics in overall.items():
        table_rows.append(
            [
                "overall",
                group,
                format_int(metrics["n"]),
                format_float(metrics["mae"]),
                format_float(metrics["rmse"]),
            ]
        )
    return render_table(headers, table_rows)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output_dir = root / "outputs"
    dataset_path = output_dir / "dataset_clean.parquet"
    results_path = output_dir / "results_baselines.json"

    df = load_dataset(dataset_path)
    splits = load_splits(output_dir)

    meta = {}
    if results_path.exists():
        meta = json.loads(results_path.read_text(encoding="utf-8")).get("meta", {})

    aspect_cols = [
        "rating_service",
        "rating_cleanliness",
        "rating_location",
        "rating_rooms",
        "rating_value",
        "rating_sleep_quality",
    ]

    n_samples = len(df)
    n_hotels = df["offering_id"].nunique()

    long_tail = {
        "text_only": {"raw": [], "clipped": []},
        "upper_bound": {"raw": [], "clipped": []},
    }

    missing_rows_raw: List[Dict[str, object]] = []
    missing_rows_clipped: List[Dict[str, object]] = []
    missing_agg_raw = {
        "missing=0": {"n": 0, "sum_abs": 0.0, "sum_sq": 0.0},
        "missing=1-2": {"n": 0, "sum_abs": 0.0, "sum_sq": 0.0},
        "missing>=3": {"n": 0, "sum_abs": 0.0, "sum_sq": 0.0},
    }
    missing_agg_clipped = {
        "missing=0": {"n": 0, "sum_abs": 0.0, "sum_sq": 0.0},
        "missing=1-2": {"n": 0, "sum_abs": 0.0, "sum_sq": 0.0},
        "missing>=3": {"n": 0, "sum_abs": 0.0, "sum_sq": 0.0},
    }

    for fold, split in enumerate(splits, start=1):
        log(f"fold {fold}/{N_SPLITS} start")

        log("  training text-only baseline")
        y_test_text, pred_text = run_text_baseline(df, split)
        abs_err_text = np.abs(y_test_text - pred_text)
        long_tail["text_only"]["raw"].append({"fold": fold, **compute_quantiles(abs_err_text)})
        long_tail["text_only"]["clipped"].append(
            {"fold": fold, **compute_quantiles(np.abs(y_test_text - np.clip(pred_text, CLIP_MIN, CLIP_MAX)))}
        )

        log("  training upper bound baseline")
        y_test_upper, pred_upper = run_upper_bound(df, split, aspect_cols)
        abs_err_upper = np.abs(y_test_upper - pred_upper)
        long_tail["upper_bound"]["raw"].append({"fold": fold, **compute_quantiles(abs_err_upper)})
        long_tail["upper_bound"]["clipped"].append(
            {"fold": fold, **compute_quantiles(np.abs(y_test_upper - np.clip(pred_upper, CLIP_MIN, CLIP_MAX)))}
        )

        test_aspects = df.iloc[split["test_idx"]][aspect_cols]
        missing_count = test_aspects.isna().sum(axis=1).to_numpy()
        group_masks = {
            "missing=0": missing_count == 0,
            "missing=1-2": (missing_count >= 1) & (missing_count <= 2),
            "missing>=3": missing_count >= 3,
        }
        for group, mask in group_masks.items():
            y_group = y_test_upper[mask]
            p_group = pred_upper[mask]
            if y_group.size == 0:
                missing_rows_raw.append(
                    {"fold": fold, "group": group, "n": 0, "mae": None, "rmse": None}
                )
                missing_rows_clipped.append(
                    {"fold": fold, "group": group, "n": 0, "mae": None, "rmse": None}
                )
                continue
            metrics = compute_metrics(y_group, p_group)
            missing_rows_raw.append(
                {
                    "fold": fold,
                    "group": group,
                    "n": int(y_group.size),
                    "mae": metrics["raw"]["mae"],
                    "rmse": metrics["raw"]["rmse"],
                }
            )
            missing_rows_clipped.append(
                {
                    "fold": fold,
                    "group": group,
                    "n": int(y_group.size),
                    "mae": metrics["clipped"]["mae"],
                    "rmse": metrics["clipped"]["rmse"],
                }
            )
            update_missingness_agg(missing_agg_raw, group, y_group, p_group, "raw")
            update_missingness_agg(missing_agg_clipped, group, y_group, p_group, "clipped")

        log(f"fold {fold} done")

    def finalize_missingness(agg: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        overall = {}
        for group, stats in agg.items():
            n = stats["n"]
            if n == 0:
                overall[group] = {"n": 0, "mae": None, "rmse": None}
                continue
            mae = stats["sum_abs"] / n
            rmse = np.sqrt(stats["sum_sq"] / n)
            overall[group] = {"n": int(n), "mae": float(mae), "rmse": float(rmse)}
        return overall

    overall_raw = finalize_missingness(missing_agg_raw)
    overall_clipped = finalize_missingness(missing_agg_clipped)

    md_lines = [
        "# Baseline Error Diagnostics",
        "",
        f"- samples: {n_samples}",
        f"- hotels: {n_hotels}",
        "- split: GroupKFold(5), grouped by offering_id",
    ]
    if meta:
        md_lines.append(f"- meta: results_baselines.json loaded (seed={meta.get('seed', 'NA')})")
    md_lines.append("")

    md_lines.append("## Long-tail Error (Raw)")
    md_lines.append("")
    md_lines.append(build_long_tail_section("Text-only TF-IDF + Ridge", long_tail["text_only"]["raw"]))
    md_lines.append(build_long_tail_section("Upper bound (true aspects + Ridge)", long_tail["upper_bound"]["raw"]))

    md_lines.append("## Long-tail Error (Clipped)")
    md_lines.append("")
    md_lines.append(build_long_tail_section("Text-only TF-IDF + Ridge", long_tail["text_only"]["clipped"]))
    md_lines.append(build_long_tail_section("Upper bound (true aspects + Ridge)", long_tail["upper_bound"]["clipped"]))

    md_lines.append("## Missingness Groups (Upper bound, Raw)")
    md_lines.append("")
    md_lines.append(build_missingness_table(missing_rows_raw, overall_raw))
    md_lines.append("")

    md_lines.append("## Missingness Groups (Upper bound, Clipped)")
    md_lines.append("")
    md_lines.append(build_missingness_table(missing_rows_clipped, overall_clipped))
    md_lines.append("")

    ub_p99 = mean_quantiles(long_tail["upper_bound"]["clipped"])["p99"]
    text_p99 = mean_quantiles(long_tail["text_only"]["clipped"])["p99"]
    missing_gap = None
    if overall_clipped["missing=0"]["rmse"] is not None and overall_clipped["missing>=3"]["rmse"] is not None:
        missing_gap = overall_clipped["missing>=3"]["rmse"] - overall_clipped["missing=0"]["rmse"]

    md_lines.append("## Auto Notes")
    md_lines.append("")
    if ub_p99 > text_p99:
        md_lines.append(
            f"- Upper bound has heavier long-tail errors (clipped p99 {ub_p99:.4f} > {text_p99:.4f}), "
            "which can inflate RMSE despite lower MAE."
        )
    else:
        md_lines.append(
            f"- Upper bound long-tail errors are not heavier than text-only (clipped p99 {ub_p99:.4f} <= {text_p99:.4f})."
        )
    if missing_gap is not None and missing_gap > 0:
        md_lines.append(
            f"- Missingness matters: upper bound RMSE is higher when >=3 aspects are missing "
            f"(ΔRMSE={missing_gap:.4f} vs missing=0)."
        )
    else:
        md_lines.append("- Missingness impact is limited or inconclusive in upper bound RMSE.")
    md_lines.append(
        "- Clipping reduces tail impact but does not remove it; check raw vs clipped tables for each method."
    )
    md_lines.append("")

    output_path = output_dir / "error_analysis_baselines.md"
    output_path.write_text("\n".join(md_lines), encoding="utf-8")
    log(f"saved: {output_path}")


if __name__ == "__main__":
    main()
