#!/usr/bin/env python3
"""Phase 3 W2/W3: corrected-protocol models on the Booking external sample."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import wilcoxon
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.aspect_extractor_base import ASPECTS  # noqa: E402


SEED = 42
SEGMENTS = ["business", "couple", "family", "solo", "friends", "unknown"]


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
    parser.add_argument("--qwen-path", type=Path, default=None)
    parser.add_argument("--max-features", type=int, default=200_000)
    parser.add_argument("--alpha-grid", default="0.1,1,10,100,1000")
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def resolve_qwen(path: Path | None) -> Path:
    if path is not None:
        result = resolve(path)
        if not result.exists():
            raise FileNotFoundError(result)
        return result
    matches = sorted(
        (ROOT / "outputs" / "phase3").glob(
            "aspects_guided_booking_qwen_guided_n*.parquet"
        )
    )
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected one Booking guided Qwen output, found {matches}"
        )
    return matches[0]


def load_sample(dataset_path: Path, sample_path: Path) -> pd.DataFrame:
    arrays = np.load(sample_path)
    row_ids = arrays["row_id"].astype(np.int64)
    outer_fold = arrays["outer_fold"].astype(np.int8)
    dataset = pd.read_parquet(
        dataset_path,
        columns=[
            "row_id",
            "offering_id",
            "rating_overall_native",
            "rating_overall_1to5",
            "review_text",
        ],
    )
    sample = dataset.iloc[row_ids].copy().reset_index(drop=True)
    if not np.array_equal(sample["row_id"].to_numpy(), row_ids):
        raise ValueError("Booking sample row alignment failed")
    sample["outer_fold"] = outer_fold
    if int(sample.groupby("offering_id")["outer_fold"].nunique().max()) != 1:
        raise ValueError("Booking sample hotel spans test folds")
    return sample


def load_features(path: Path, row_ids: np.ndarray) -> Dict[str, np.ndarray]:
    score_cols = [f"aspect_{aspect}_score" for aspect in ASPECTS]
    mention_cols = [f"aspect_{aspect}_mentioned" for aspect in ASPECTS]
    frame = pd.read_parquet(
        path,
        columns=["row_id", "ok", "traveler_segment", *score_cols, *mention_cols],
    )
    frame = pd.DataFrame({"row_id": row_ids}).merge(
        frame, on="row_id", how="left", validate="one_to_one"
    )
    ok = frame["ok"].eq(1)
    frame.loc[~ok, score_cols] = 0
    frame.loc[~ok, mention_cols] = False
    frame.loc[~ok, "traveler_segment"] = "unknown"
    scores = (
        frame[score_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .to_numpy(dtype=np.float32)
    )
    mentions = frame[mention_cols].fillna(False).to_numpy(dtype=np.float32)
    segment = frame["traveler_segment"].fillna("unknown").astype(str).str.lower()
    segment = segment.where(segment.isin(SEGMENTS), "unknown")
    one_hot = (
        pd.get_dummies(segment)
        .reindex(columns=SEGMENTS, fill_value=False)
        .to_numpy(dtype=np.float32)
    )
    p_s = np.hstack(
        [scores * one_hot[:, j : j + 1] for j in range(len(SEGMENTS))]
    ).astype(np.float32)
    p_b = np.hstack(
        [mentions * one_hot[:, j : j + 1] for j in range(len(SEGMENTS))]
    ).astype(np.float32)
    return {
        "simple": np.hstack([scores, mentions, one_hot]).astype(np.float32),
        "interaction": np.hstack(
            [scores, mentions, one_hot, p_s, p_b]
        ).astype(np.float32),
        "ok": ok.to_numpy(dtype=bool),
    }


def cache_prefix(
    phase3: Path, max_features: int, fold: int, sample_path: Path
) -> Path:
    digest = hashlib.sha256(sample_path.read_bytes()).hexdigest()[:12]
    return (
        phase3
        / "w_tfidf_cache"
        / f"mf{max_features}_ng12_md3_fold{fold}_{digest}"
    )


def load_or_build_tfidf(
    texts: np.ndarray,
    train_pos: np.ndarray,
    test_pos: np.ndarray,
    max_features: int,
    prefix: Path,
    resume: bool,
) -> Tuple[sp.csr_matrix, sp.csr_matrix]:
    train_path = prefix.with_name(prefix.name + "_Xtr.npz")
    test_path = prefix.with_name(prefix.name + "_Xte.npz")
    vectorizer_path = prefix.with_name(prefix.name + "_vectorizer.joblib")
    if resume and all(p.exists() for p in (train_path, test_path, vectorizer_path)):
        log(f"W TF-IDF cache hit: {prefix.name}")
        return sp.load_npz(train_path).tocsr(), sp.load_npz(test_path).tocsr()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=3,
        max_features=max_features,
        dtype=np.float32,
    )
    train = vectorizer.fit_transform(texts[train_pos]).tocsr()
    test = vectorizer.transform(texts[test_pos]).tocsr()
    sp.save_npz(train_path, train)
    sp.save_npz(test_path, test)
    joblib.dump(vectorizer, vectorizer_path)
    log(f"W TF-IDF built: train={train.shape} test={test.shape}")
    return train, test


def scale_dense(
    values: np.ndarray, train_pos: np.ndarray, test_pos: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    return (
        scaler.fit_transform(values[train_pos]).astype(np.float32),
        scaler.transform(values[test_pos]).astype(np.float32),
    )


def combine(text: sp.csr_matrix, dense: np.ndarray | None) -> sp.csr_matrix:
    if dense is None:
        return text
    return sp.hstack([text, sp.csr_matrix(dense)], format="csr")


def tune_alpha(
    x: sp.csr_matrix,
    y: np.ndarray,
    groups: np.ndarray,
    alphas: List[float],
    label: str,
) -> Tuple[float, pd.DataFrame]:
    rows = []
    splits = list(GroupKFold(n_splits=3).split(x, y, groups))
    for alpha in alphas:
        values = []
        for inner_fold, (fit, valid) in enumerate(splits, start=1):
            model = Ridge(alpha=alpha)
            model.fit(x[fit], y[fit])
            prediction = model.predict(x[valid])
            mae = float(mean_absolute_error(y[valid], prediction))
            values.append(mae)
            log(
                f"W tune {label} alpha={alpha:g} inner={inner_fold} MAE={mae:.6f}"
            )
        rows.append(
            {
                "alpha": alpha,
                "mean_inner_mae": float(np.mean(values)),
                "sd_inner_mae": float(np.std(values, ddof=1)),
            }
        )
    curve = pd.DataFrame(rows)
    selected = float(curve.sort_values(["mean_inner_mae", "alpha"]).iloc[0]["alpha"])
    return selected, curve.sort_values("alpha")


def metrics(
    y_mapped: np.ndarray,
    y_native: np.ndarray,
    prediction_mapped: np.ndarray,
) -> Dict[str, float]:
    mapped = np.clip(prediction_mapped, 1.0, 5.0)
    native = np.clip(1.0 + 9.0 * (mapped - 1.0) / 4.0, 1.0, 10.0)
    return {
        "mae_1to5": float(mean_absolute_error(y_mapped, mapped)),
        "rmse_1to5": float(math.sqrt(mean_squared_error(y_mapped, mapped))),
        "mae_native": float(mean_absolute_error(y_native, native)),
        "rmse_native": float(math.sqrt(mean_squared_error(y_native, native))),
    }


def aggregate(folds: pd.DataFrame) -> pd.DataFrame:
    return (
        folds.groupby("model", as_index=False)
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
            alpha_geomean=("alpha", lambda x: 10 ** np.log10(x).mean()),
        )
        .sort_values("mae_1to5_mean")
    )


def paired_hotel(
    predictions: pd.DataFrame, first: str, second: str, n_resamples: int = 10_000
) -> Dict[str, Any]:
    left = predictions[predictions["model"].eq(first)]
    right = predictions[predictions["model"].eq(second)]
    joined = left.merge(
        right,
        on=["row_id", "offering_id", "y_true"],
        suffixes=("_first", "_second"),
        validate="one_to_one",
    )
    joined["first_error"] = np.abs(joined["y_true"] - joined["prediction_first"])
    joined["second_error"] = np.abs(joined["y_true"] - joined["prediction_second"])
    hotel = joined.groupby("offering_id")[["first_error", "second_error"]].mean()
    difference = (hotel["first_error"] - hotel["second_error"]).to_numpy()
    try:
        statistic, p_value = wilcoxon(difference)
    except ValueError:
        statistic, p_value = 0.0, 1.0
    rng = np.random.default_rng(SEED)
    samples = np.empty(n_resamples, dtype=float)
    for start in range(0, n_resamples, 500):
        size = min(500, n_resamples - start)
        draws = rng.integers(0, len(difference), size=(size, len(difference)))
        samples[start : start + size] = difference[draws].mean(axis=1)
    return {
        "first": first,
        "second": second,
        "difference": "first_minus_second_equal_hotel_MAE_on_1to5",
        "n_hotels": len(difference),
        "mean_difference": float(difference.mean()),
        "bootstrap_ci_95": [
            float(np.quantile(samples, 0.025)),
            float(np.quantile(samples, 0.975)),
        ],
        "wilcoxon_statistic": float(statistic),
        "wilcoxon_p": float(p_value),
    }


def main() -> None:
    args = parse_args()
    np.random.seed(SEED)
    dataset_path = resolve(args.dataset_file)
    sample_path = resolve(args.sample_file)
    qwen_path = resolve_qwen(args.qwen_path)
    alphas = [float(x) for x in args.alpha_grid.split(",") if x.strip()]
    phase3 = ROOT / "outputs" / "phase3"
    sample = load_sample(dataset_path, sample_path)
    row_ids = sample["row_id"].to_numpy(dtype=np.int64)
    features = load_features(qwen_path, row_ids)
    texts = sample["review_text"].fillna("").astype(str).to_numpy()
    y = sample["rating_overall_1to5"].to_numpy(dtype=float)
    y_native = sample["rating_overall_native"].to_numpy(dtype=float)
    groups = sample["offering_id"].to_numpy()
    outer_fold = sample["outer_fold"].to_numpy(dtype=np.int8)
    tuning_dir = phase3 / "w_alpha_curves"
    tuning_dir.mkdir(parents=True, exist_ok=True)
    fold_rows = []
    predictions = []

    for fold in tqdm(range(1, 6), desc="W outer folds"):
        train_pos = np.flatnonzero(outer_fold != fold)
        test_pos = np.flatnonzero(outer_fold == fold)
        prefix = cache_prefix(phase3, args.max_features, fold, sample_path)
        text_train, text_test = load_or_build_tfidf(
            texts, train_pos, test_pos, args.max_features, prefix, args.resume
        )
        scaled = {
            name: scale_dense(values, train_pos, test_pos)
            for name, values in features.items()
            if name != "ok"
        }
        variants = {
            "text_only": (text_train, text_test),
            "simple_fusion": (
                combine(text_train, scaled["simple"][0]),
                combine(text_test, scaled["simple"][1]),
            ),
            "interaction_fusion": (
                combine(text_train, scaled["interaction"][0]),
                combine(text_test, scaled["interaction"][1]),
            ),
        }
        for model_name, (x_train, x_test) in variants.items():
            curve_path = tuning_dir / f"{model_name}_fold{fold}.csv"
            if args.resume and curve_path.exists():
                curve = pd.read_csv(curve_path)
                alpha = float(
                    curve.sort_values(["mean_inner_mae", "alpha"]).iloc[0]["alpha"]
                )
            else:
                alpha, curve = tune_alpha(
                    x_train,
                    y[train_pos],
                    groups[train_pos],
                    alphas,
                    f"{model_name} fold{fold}",
                )
                curve.to_csv(curve_path, index=False)
            model = Ridge(alpha=alpha)
            model.fit(x_train, y[train_pos])
            prediction = model.predict(x_test)
            score = metrics(y[test_pos], y_native[test_pos], prediction)
            fold_rows.append(
                {
                    "model": model_name,
                    "fold": fold,
                    "alpha": alpha,
                    "n_test": len(test_pos),
                    **score,
                }
            )
            predictions.append(
                pd.DataFrame(
                    {
                        "model": model_name,
                        "fold": fold,
                        "row_id": row_ids[test_pos],
                        "offering_id": groups[test_pos],
                        "y_true": y[test_pos],
                        "prediction": np.clip(prediction, 1.0, 5.0),
                    }
                )
            )
            log(
                f"W fold={fold} {model_name} alpha={alpha:g} "
                f"MAE1-5={score['mae_1to5']:.6f} native={score['mae_native']:.6f}"
            )
        del text_train, text_test, scaled, variants
        gc.collect()

    fold_frame = pd.DataFrame(fold_rows)
    prediction_frame = pd.concat(predictions, ignore_index=True)
    summary = aggregate(fold_frame)
    comparisons = [
        paired_hotel(prediction_frame, "text_only", "simple_fusion"),
        paired_hotel(prediction_frame, "text_only", "interaction_fusion"),
        paired_hotel(prediction_frame, "simple_fusion", "interaction_fusion"),
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
    variants_close = bool(
        abs(
            indexed.loc["simple_fusion", "mae_1to5_mean"]
            - indexed.loc["interaction_fusion", "mae_1to5_mean"]
        )
        < 0.003
    )
    verdict = simple_beats and interaction_beats and variants_close
    fold_frame.to_csv(phase3 / "w_external_validation_folds.csv", index=False)
    summary.to_csv(phase3 / "w_external_validation_summary.csv", index=False)
    prediction_frame.to_parquet(
        phase3 / "w_external_validation_predictions.parquet", index=False
    )
    payload = {
        "experiment": "W2_W3",
        "sample_n": len(sample),
        "hotels": int(sample["offering_id"].nunique()),
        "sample_path": str(sample_path.relative_to(ROOT)),
        "qwen_path": str(qwen_path.relative_to(ROOT)),
        "seed": SEED,
        "outer_split": "GroupKFold(5), group=hotel address hash",
        "inner_cv": "GroupKFold(3), group=hotel address hash",
        "alpha_grid": alphas,
        "rating_mapping": "1 + 4 * (native - 1) / 9",
        "summary": summary.to_dict(orient="records"),
        "folds": fold_frame.to_dict(orient="records"),
        "paired_hotel_tests": comparisons,
        "criteria": {
            "simple_fusion_beats_text": simple_beats,
            "interaction_fusion_beats_text": interaction_beats,
            "simple_interaction_abs_gap_below_0.003": variants_close,
        },
        "qualitative_ordering_replicates": verdict,
    }
    (phase3 / "w_external_validation_results.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    lines = [
        "# Phase 3 W2/W3 — Booking External Validation",
        "",
        f"- hotel-intact sample: {len(sample):,} reviews / {sample['offering_id'].nunique():,} hotels",
        "- outer split: GroupKFold(5) by hotel-address hash",
        "- inner selection: GroupKFold(3) by hotel-address hash",
        "- rating mapping: `1 + 4 * (native - 1) / 9`",
        "- all predictions clipped to the applicable scale",
        "",
        "## Results on 1–5 equivalent and native 1–10 scale",
        "",
        summary.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Paired equal-hotel tests (1–5 equivalent MAE)",
        "",
        pd.DataFrame(comparisons).to_markdown(index=False, floatfmt=".6g"),
        "",
        "## W3 verdict",
        "",
        f"**Qualitative findings replicate: {verdict}.**",
        f"Fusion beats text: simple={simple_beats}, interaction={interaction_beats}; simple/interaction gap <0.003: {variants_close}.",
        "",
    ]
    (phase3 / "experiment_w_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    log(f"saved: {phase3 / 'experiment_w_report.md'}")


if __name__ == "__main__":
    main()
