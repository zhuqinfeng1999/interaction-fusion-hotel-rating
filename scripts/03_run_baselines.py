#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
N_SPLITS = 5
USE_ELASTICNET = False
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


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    return {"mae": float(mae), "rmse": float(rmse)}


def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
    raw = _compute_metrics(y_true, y_pred)
    clipped_pred = np.clip(y_pred, CLIP_MIN, CLIP_MAX)
    clipped = _compute_metrics(y_true, clipped_pred)
    return {"raw": raw, "clipped": clipped}


def summarize_metrics(metrics: List[Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, Dict[str, float]]]:
    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for key in ("raw", "clipped"):
        maes = np.array([m[key]["mae"] for m in metrics], dtype=float)
        rmses = np.array([m[key]["rmse"] for m in metrics], dtype=float)
        summary[key] = {
            "mean": {"mae": float(maes.mean()), "rmse": float(rmses.mean())},
            "std": {"mae": float(maes.std(ddof=1)), "rmse": float(rmses.std(ddof=1))},
        }
    return summary


def run_text_baseline(
    df: pd.DataFrame, splits: List[Dict[str, np.ndarray]]
) -> Dict[str, object]:
    results = []
    for fold, split in enumerate(splits, start=1):
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
        metrics = eval_metrics(y_test, preds)
        results.append({"fold": fold, **metrics})
        log(
            "text baseline fold {fold}: MAE={mae:.4f} RMSE={rmse:.4f}".format(
                fold=fold,
                mae=metrics["clipped"]["mae"],
                rmse=metrics["clipped"]["rmse"],
            )
        )

    summary = summarize_metrics(results)
    return {"folds": results, **summary}


def run_upper_bound(
    df: pd.DataFrame, splits: List[Dict[str, np.ndarray]], aspect_cols: List[str]
) -> Dict[str, object]:
    results = []
    inner_group_cv = GroupKFold(n_splits=3)
    for fold, split in enumerate(splits, start=1):
        train_idx = split["train_idx"]
        test_idx = split["test_idx"]
        x_train = df.iloc[train_idx][aspect_cols].astype(float)
        y_train = df.iloc[train_idx]["rating_overall"].astype(float).values
        x_test = df.iloc[test_idx][aspect_cols].astype(float)
        y_test = df.iloc[test_idx]["rating_overall"].astype(float).values
        train_groups = df.iloc[train_idx]["offering_id"].values

        if USE_ELASTICNET:
            pipeline = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                    ("scaler", StandardScaler()),
                    ("model", ElasticNet(max_iter=20000, random_state=SEED)),
                ]
            )
            grid = {
                "model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
                "model__l1_ratio": [0.0, 0.5],
            }
        else:
            pipeline = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                    ("scaler", StandardScaler()),
                    ("model", Ridge()),
                ]
            )
            grid = {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}

        search = GridSearchCV(
            pipeline,
            grid,
            scoring="neg_mean_absolute_error",
            cv=inner_group_cv,
            n_jobs=-1,
        )
        search.fit(x_train, y_train, groups=train_groups)
        preds = search.predict(x_test)
        metrics = eval_metrics(y_test, preds)
        results.append(
            {
                "fold": fold,
                **metrics,
                "best_params": search.best_params_,
            }
        )
        log(
            "upper bound fold {fold}: MAE={mae:.4f} RMSE={rmse:.4f} best={best}".format(
                fold=fold,
                mae=metrics["clipped"]["mae"],
                rmse=metrics["clipped"]["rmse"],
                best=search.best_params_,
            )
        )

    summary = summarize_metrics(results)
    return {"folds": results, **summary}


def write_markdown(results: Dict[str, object], output_path: Path) -> None:
    text_mean = results["text_only"]["clipped"]["mean"]
    text_std = results["text_only"]["clipped"]["std"]
    upper_mean = results["upper_bound"]["clipped"]["mean"]
    upper_std = results["upper_bound"]["clipped"]["std"]

    text_raw_mean = results["text_only"]["raw"]["mean"]
    text_raw_std = results["text_only"]["raw"]["std"]
    upper_raw_mean = results["upper_bound"]["raw"]["mean"]
    upper_raw_std = results["upper_bound"]["raw"]["std"]

    model_label = "ElasticNet" if results["meta"]["upper_bound"]["use_elasticnet"] else "Ridge"

    lines = [
        "# Baseline Results",
        "",
        "| Method | MAE (mean±std) | RMSE (mean±std) |",
        "| --- | --- | --- |",
        "| Text-only TF-IDF + Ridge | "
        f"{text_mean['mae']:.4f} ± {text_std['mae']:.4f} | "
        f"{text_mean['rmse']:.4f} ± {text_std['rmse']:.4f} |",
        f"| Upper bound (true aspects + {model_label}) | "
        f"{upper_mean['mae']:.4f} ± {upper_std['mae']:.4f} | "
        f"{upper_mean['rmse']:.4f} ± {upper_std['rmse']:.4f} |",
        "",
        "## Raw Metrics (No Clipping)",
        "",
        "| Method | MAE (mean±std) | RMSE (mean±std) |",
        "| --- | --- | --- |",
        "| Text-only TF-IDF + Ridge | "
        f"{text_raw_mean['mae']:.4f} ± {text_raw_std['mae']:.4f} | "
        f"{text_raw_mean['rmse']:.4f} ± {text_raw_std['rmse']:.4f} |",
        f"| Upper bound (true aspects + {model_label}) | "
        f"{upper_raw_mean['mae']:.4f} ± {upper_raw_std['mae']:.4f} | "
        f"{upper_raw_mean['rmse']:.4f} ± {upper_raw_std['rmse']:.4f} |",
        "",
        "Notes:",
        "- Split: GroupKFold(5), grouped by offering_id (no hotel leakage).",
        "- Upper bound inner CV: GroupKFold(3) on offering_id within each training fold.",
        "- Metrics: MAE and RMSE on rating_overall (1–5).",
        f"- Clipping: predictions clipped to [{CLIP_MIN:.0f}, {CLIP_MAX:.0f}] for main table; raw metrics shown above.",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output_dir = root / "outputs"
    dataset_path = output_dir / "dataset_clean.parquet"

    df = load_dataset(dataset_path)
    if "review_text" not in df.columns:
        raise ValueError("review_text missing from dataset.")
    if "rating_overall" not in df.columns:
        raise ValueError("rating_overall missing from dataset.")

    aspect_cols = [
        "rating_service",
        "rating_cleanliness",
        "rating_location",
        "rating_rooms",
        "rating_value",
        "rating_sleep_quality",
    ]
    missing_aspects = [col for col in aspect_cols if col not in df.columns]
    if missing_aspects:
        raise ValueError(f"missing aspect columns: {missing_aspects}")

    splits = load_splits(output_dir)
    log(f"loaded dataset with {len(df)} rows")

    results = {
        "meta": {
            "seed": SEED,
            "n_splits": N_SPLITS,
            "split_strategy": "GroupKFold",
            "target": "rating_overall",
            "clipping": {"enabled": True, "range": [CLIP_MIN, CLIP_MAX]},
            "upper_bound": {
                "use_elasticnet": USE_ELASTICNET,
                "model": "ElasticNet" if USE_ELASTICNET else "Ridge",
                "inner_cv": {"split_strategy": "GroupKFold", "n_splits": 3},
            },
        },
        "text_only": run_text_baseline(df, splits),
        "upper_bound": run_upper_bound(df, splits, aspect_cols),
    }

    json_path = output_dir / "results_baselines.json"
    md_path = output_dir / "results_baselines.md"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_markdown(results, md_path)
    log(f"saved: {json_path}")
    log(f"saved: {md_path}")


if __name__ == "__main__":
    main()
