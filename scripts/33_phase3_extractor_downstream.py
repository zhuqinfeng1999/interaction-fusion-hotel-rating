#!/usr/bin/env python3
"""Phase 3 L5/L6: decoding-matched extractor downstream sensitivity."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

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
ALPHA_GRID = [0.1, 1.0, 10.0, 100.0, 1000.0]


def log(message: str) -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen-profile", default="qwen25_7b")
    parser.add_argument("--llama-profile", default="llama31_8b")
    parser.add_argument("--qwen-path", type=Path, default=None)
    parser.add_argument("--llama-path", type=Path, default=None)
    parser.add_argument(
        "--sample-file",
        type=Path,
        default=Path("outputs/phase3/l1_sample_n200000_seed42.npz"),
    )
    parser.add_argument("--max-features", type=int, default=200_000)
    parser.add_argument("--alpha-grid", default="0.1,1,10,100,1000")
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def resolve_unique(path: Path | None, pattern: str) -> Path:
    if path is not None:
        resolved = resolve_path(path)
        if not resolved.exists():
            raise FileNotFoundError(resolved)
        return resolved
    matches = sorted((ROOT / "outputs" / "phase3").glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one {pattern}, found {matches}")
    return matches[0]


def load_sample(output_dir: Path, sample_path: Path) -> pd.DataFrame:
    arrays = np.load(sample_path)
    row_ids = arrays["row_id"].astype(np.int64)
    outer_folds = arrays["outer_fold"].astype(np.int8)
    dataset = pd.read_parquet(
        output_dir / "dataset_clean.parquet",
        columns=["offering_id", "rating_overall", "review_text"],
    ).reset_index(drop=True)
    dataset["row_id"] = np.arange(len(dataset), dtype=np.int64)
    sample = dataset.iloc[row_ids].copy().reset_index(drop=True)
    sample["outer_fold"] = outer_folds
    if not np.array_equal(sample["row_id"].to_numpy(), row_ids):
        raise ValueError("L1 sample alignment failed")
    return sample


def load_extractor_features(path: Path, row_ids: np.ndarray) -> Dict[str, np.ndarray]:
    score_cols = [f"aspect_{aspect}_score" for aspect in ASPECTS]
    mention_cols = [f"aspect_{aspect}_mentioned" for aspect in ASPECTS]
    frame = pd.read_parquet(
        path,
        columns=["row_id", "ok", "traveler_segment", *score_cols, *mention_cols],
    ).sort_values("row_id")
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
    segment_text = frame["traveler_segment"].fillna("unknown").astype(str).str.lower()
    segment_text = segment_text.where(segment_text.isin(SEGMENTS), "unknown")
    segments = (
        pd.get_dummies(segment_text)
        .reindex(columns=SEGMENTS, fill_value=False)
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
        "interaction": np.hstack(
            [scores, mentions, segments, p_s, p_b]
        ).astype(np.float32),
        "ok": ok.to_numpy(dtype=bool),
    }


def tfidf_cache_prefix(
    phase3: Path, max_features: int, fold: int, sample_path: Path
) -> Path:
    digest = hashlib.sha256(sample_path.read_bytes()).hexdigest()[:12]
    return (
        phase3
        / "l5_tfidf_cache"
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
    if resume and all(path.exists() for path in (train_path, test_path, vectorizer_path)):
        log(f"L5 TF-IDF cache hit: {prefix.name}")
        return sp.load_npz(train_path).tocsr(), sp.load_npz(test_path).tocsr()
    prefix.parent.mkdir(parents=True, exist_ok=True)
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=3,
        max_features=max_features,
        dtype=np.float32,
    )
    started = time.perf_counter()
    train = vectorizer.fit_transform(texts[train_pos]).tocsr()
    test = vectorizer.transform(texts[test_pos]).tocsr()
    sp.save_npz(train_path, train)
    sp.save_npz(test_path, test)
    joblib.dump(vectorizer, vectorizer_path)
    log(
        f"L5 TF-IDF built fold={prefix.name} train={train.shape} test={test.shape} "
        f"elapsed={time.perf_counter() - started:.1f}s"
    )
    return train, test


def scale_dense(
    dense: np.ndarray, train_pos: np.ndarray, test_pos: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    return (
        scaler.fit_transform(dense[train_pos]).astype(np.float32),
        scaler.transform(dense[test_pos]).astype(np.float32),
    )


def hstack(text: sp.csr_matrix, dense: np.ndarray | None) -> sp.csr_matrix:
    if dense is None:
        return text
    return sp.hstack([text, sp.csr_matrix(dense)], format="csr")


def tune_alpha(
    x_train: sp.csr_matrix,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    alphas: Sequence[float],
    label: str,
) -> Tuple[float, pd.DataFrame]:
    rows = []
    splits = list(GroupKFold(n_splits=3).split(x_train, y_train, groups_train))
    for alpha in alphas:
        fold_mae = []
        for inner_fold, (fit_pos, valid_pos) in enumerate(splits, start=1):
            model = Ridge(alpha=float(alpha), random_state=SEED)
            model.fit(x_train[fit_pos], y_train[fit_pos])
            prediction = model.predict(x_train[valid_pos])
            mae = float(mean_absolute_error(y_train[valid_pos], prediction))
            fold_mae.append(mae)
            log(
                f"L5 tune {label} alpha={alpha:g} inner={inner_fold} MAE={mae:.6f}"
            )
        rows.append(
            {
                "alpha": float(alpha),
                "mean_inner_mae": float(np.mean(fold_mae)),
                "sd_inner_mae": float(np.std(fold_mae, ddof=1)),
            }
        )
    curve = pd.DataFrame(rows).sort_values(["mean_inner_mae", "alpha"])
    return float(curve.iloc[0]["alpha"]), curve.sort_values("alpha")


def eval_metrics(y: np.ndarray, prediction: np.ndarray) -> Dict[str, float]:
    clipped = np.clip(prediction, 1.0, 5.0)
    return {
        "mae": float(mean_absolute_error(y, clipped)),
        "rmse": float(math.sqrt(mean_squared_error(y, clipped))),
        "raw_mae": float(mean_absolute_error(y, prediction)),
        "raw_rmse": float(math.sqrt(mean_squared_error(y, prediction))),
    }


def fit_predict(
    x_train: sp.csr_matrix,
    y_train: np.ndarray,
    x_test: sp.csr_matrix,
    alpha: float,
) -> np.ndarray:
    model = Ridge(alpha=alpha, random_state=SEED)
    model.fit(x_train, y_train)
    return model.predict(x_test)


def paired_hotel_test(
    predictions: pd.DataFrame,
    first: str,
    second: str,
    n_bootstrap: int = 10_000,
) -> Dict[str, Any]:
    first_rows = predictions[predictions["model"] == first]
    second_rows = predictions[predictions["model"] == second]
    joined = first_rows.merge(
        second_rows,
        on=["row_id", "offering_id", "y_true"],
        suffixes=("_first", "_second"),
        validate="one_to_one",
    )
    joined["abs_first"] = np.abs(joined["y_true"] - joined["y_pred_first"])
    joined["abs_second"] = np.abs(joined["y_true"] - joined["y_pred_second"])
    hotels = (
        joined.groupby("offering_id")[["abs_first", "abs_second"]]
        .mean()
        .dropna()
    )
    difference = hotels["abs_first"].to_numpy() - hotels["abs_second"].to_numpy()
    try:
        statistic, p_value = wilcoxon(difference)
    except ValueError:
        statistic, p_value = 0.0, 1.0
    rng = np.random.default_rng(SEED)
    bootstrap = np.empty(n_bootstrap, dtype=np.float64)
    for start in range(0, n_bootstrap, 500):
        size = min(500, n_bootstrap - start)
        draws = rng.integers(0, len(difference), size=(size, len(difference)))
        bootstrap[start : start + size] = difference[draws].mean(axis=1)
    return {
        "first": first,
        "second": second,
        "difference": "first_mae_minus_second_mae_equal_hotel_weight",
        "n_hotels": len(difference),
        "mean_difference": float(difference.mean()),
        "median_difference": float(np.median(difference)),
        "bootstrap_ci_95": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "wilcoxon_statistic": float(statistic),
        "wilcoxon_p": float(p_value),
    }


def aggregate(fold_results: pd.DataFrame) -> pd.DataFrame:
    return (
        fold_results.groupby("model", as_index=False)
        .agg(
            n=("n_test", "sum"),
            mae_mean=("mae", "mean"),
            mae_sd=("mae", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_sd=("rmse", "std"),
            alpha_geomean=("alpha", lambda values: 10 ** np.log10(values).mean()),
        )
        .sort_values("mae_mean")
    )


def full_qwen_reference(output_dir: Path) -> pd.DataFrame:
    mapping = {
        "text_only": "phase2_r0_text_shared_ridge",
        "qwen_simple": "phase2_r0_simple_shared_ridge",
        "qwen_interaction": "phase2_r0_interaction_shared_ridge",
    }
    rows = []
    for label, model_id in mapping.items():
        payload = json.loads(
            (output_dir / "metrics" / f"{model_id}.json").read_text(encoding="utf-8")
        )
        rows.append(
            {
                "model": label,
                "corpus": "full_877808_legacy_qwen_features_corrected_R0",
                "mae_mean": payload["aggregate"]["clipped"]["mean"]["mae"],
                "rmse_mean": payload["aggregate"]["clipped"]["mean"]["rmse"],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    alphas = [float(item) for item in args.alpha_grid.split(",") if item.strip()]
    output_dir = ROOT / "outputs"
    phase3 = output_dir / "phase3"
    sample_path = resolve_path(args.sample_file)
    qwen_path = resolve_unique(
        args.qwen_path, f"aspects_guided_{args.qwen_profile}_l1_n*.parquet"
    )
    llama_path = resolve_unique(
        args.llama_path, f"aspects_guided_{args.llama_profile}_l1_n*.parquet"
    )
    sample = load_sample(output_dir, sample_path)
    row_ids = sample["row_id"].to_numpy(dtype=np.int64)
    qwen = load_extractor_features(qwen_path, row_ids)
    llama = load_extractor_features(llama_path, row_ids)
    feature_sets = {
        "qwen_simple": qwen["z"],
        "qwen_interaction": qwen["interaction"],
        "llama_simple": llama["z"],
        "llama_interaction": llama["interaction"],
        "ensemble_simple": np.hstack([qwen["z"], llama["z"]]).astype(np.float32),
        "ensemble_interaction": np.hstack(
            [qwen["interaction"], llama["interaction"]]
        ).astype(np.float32),
    }
    texts = sample["review_text"].fillna("").astype(str).to_numpy()
    y = sample["rating_overall"].to_numpy(dtype=np.float64)
    groups = sample["offering_id"].to_numpy()
    outer_folds = sample["outer_fold"].to_numpy(dtype=np.int8)
    fold_rows = []
    prediction_frames = []
    tuning_dir = phase3 / "l5_alpha_curves"
    tuning_dir.mkdir(parents=True, exist_ok=True)

    for fold in tqdm(range(1, 6), desc="L5 outer folds"):
        test_pos = np.flatnonzero(outer_folds == fold)
        train_pos = np.flatnonzero(outer_folds != fold)
        prefix = tfidf_cache_prefix(phase3, args.max_features, fold, sample_path)
        text_train, text_test = load_or_build_tfidf(
            texts, train_pos, test_pos, args.max_features, prefix, args.resume
        )
        y_train, y_test = y[train_pos], y[test_pos]
        groups_train = groups[train_pos]

        text_curve_path = tuning_dir / f"text_only_fold{fold}.csv"
        if args.resume and text_curve_path.exists():
            text_curve = pd.read_csv(text_curve_path)
            text_alpha = float(
                text_curve.sort_values(["mean_inner_mae", "alpha"]).iloc[0]["alpha"]
            )
        else:
            text_alpha, text_curve = tune_alpha(
                text_train, y_train, groups_train, alphas, f"text f{fold}"
            )
            text_curve.to_csv(text_curve_path, index=False)
        text_pred = fit_predict(text_train, y_train, text_test, text_alpha)
        text_metrics = eval_metrics(y_test, text_pred)
        fold_rows.append(
            {"model": "text_only", "fold": fold, "alpha": text_alpha, "n_test": len(test_pos), **text_metrics}
        )
        prediction_frames.append(
            pd.DataFrame(
                {
                    "model": "text_only",
                    "fold": fold,
                    "row_id": row_ids[test_pos],
                    "offering_id": groups[test_pos],
                    "y_true": y_test,
                    "y_pred": np.clip(text_pred, 1.0, 5.0),
                }
            )
        )

        # Select simple/interaction alpha once with Qwen and freeze it for
        # Llama and ensemble. This is a decoding-matched comparison and avoids
        # granting any extractor a hyperparameter-search advantage.
        family_alphas: Dict[str, float] = {}
        scaled: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for family in ("simple", "interaction"):
            qwen_name = f"qwen_{family}"
            for name in [
                qwen_name,
                f"llama_{family}",
                f"ensemble_{family}",
            ]:
                scaled[name] = scale_dense(feature_sets[name], train_pos, test_pos)
            qwen_train = hstack(text_train, scaled[qwen_name][0])
            curve_path = tuning_dir / f"{qwen_name}_fold{fold}.csv"
            if args.resume and curve_path.exists():
                curve = pd.read_csv(curve_path)
                alpha = float(
                    curve.sort_values(["mean_inner_mae", "alpha"]).iloc[0]["alpha"]
                )
            else:
                alpha, curve = tune_alpha(
                    qwen_train,
                    y_train,
                    groups_train,
                    alphas,
                    f"{qwen_name} f{fold}",
                )
                curve.to_csv(curve_path, index=False)
            family_alphas[family] = alpha

        for name, dense in scaled.items():
            family = "interaction" if name.endswith("interaction") else "simple"
            alpha = family_alphas[family]
            x_train = hstack(text_train, dense[0])
            x_test = hstack(text_test, dense[1])
            prediction = fit_predict(x_train, y_train, x_test, alpha)
            model_metrics = eval_metrics(y_test, prediction)
            fold_rows.append(
                {"model": name, "fold": fold, "alpha": alpha, "n_test": len(test_pos), **model_metrics}
            )
            prediction_frames.append(
                pd.DataFrame(
                    {
                        "model": name,
                        "fold": fold,
                        "row_id": row_ids[test_pos],
                        "offering_id": groups[test_pos],
                        "y_true": y_test,
                        "y_pred": np.clip(prediction, 1.0, 5.0),
                    }
                )
            )
            log(
                f"L5 fold={fold} model={name} alpha={alpha:g} "
                f"MAE={model_metrics['mae']:.6f} RMSE={model_metrics['rmse']:.6f}"
            )

    fold_results = pd.DataFrame(fold_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    summary = aggregate(fold_results)
    fold_results.to_csv(phase3 / "l5_downstream_folds.csv", index=False)
    summary.to_csv(phase3 / "l5_downstream_summary.csv", index=False)
    predictions.to_parquet(phase3 / "l5_downstream_predictions.parquet", index=False)

    comparisons = []
    for extractor in ("qwen", "llama", "ensemble"):
        comparisons.append(paired_hotel_test(predictions, "text_only", f"{extractor}_simple"))
        comparisons.append(paired_hotel_test(predictions, "text_only", f"{extractor}_interaction"))
        comparisons.append(
            paired_hotel_test(
                predictions, f"{extractor}_simple", f"{extractor}_interaction"
            )
        )
    l6 = summary[summary["model"].isin(["text_only", "qwen_simple", "qwen_interaction"])][
        ["model", "mae_mean", "rmse_mean"]
    ].copy()
    l6["corpus"] = f"guided_qwen_hotel_intact_n{len(sample)}"
    l6 = pd.concat([l6, full_qwen_reference(output_dir)], ignore_index=True)
    l6.to_csv(phase3 / "l6_qwen_subsample_vs_full.csv", index=False)

    simple_beats_text = all(
        float(summary.set_index("model").loc[f"{extractor}_simple", "mae_mean"])
        < float(summary.set_index("model").loc["text_only", "mae_mean"])
        for extractor in ("qwen", "llama", "ensemble")
    )
    interaction_beats_text = all(
        float(summary.set_index("model").loc[f"{extractor}_interaction", "mae_mean"])
        < float(summary.set_index("model").loc["text_only", "mae_mean"])
        for extractor in ("qwen", "llama", "ensemble")
    )
    close = all(
        abs(
            float(summary.set_index("model").loc[f"{extractor}_simple", "mae_mean"])
            - float(summary.set_index("model").loc[f"{extractor}_interaction", "mae_mean"])
        )
        < 0.003
        for extractor in ("qwen", "llama", "ensemble")
    )
    qualitative_survives = simple_beats_text and interaction_beats_text and close
    payload = {
        "experiment": "L5_L6",
        "sample_n": len(sample),
        "sample_path": str(sample_path.relative_to(ROOT)),
        "qwen_path": str(qwen_path.relative_to(ROOT)),
        "llama_path": str(llama_path.relative_to(ROOT)),
        "alpha_grid": alphas,
        "inner_cv": "GroupKFold(3), group=offering_id",
        "regularization": "shared Ridge penalty; structured blocks standardized on outer training fold",
        "extractor_fairness": "simple/interaction alpha selected with Qwen and frozen for Llama/ensemble",
        "summary": summary.to_dict(orient="records"),
        "folds": fold_results.to_dict(orient="records"),
        "paired_hotel_tests": comparisons,
        "qualitative_ordering_survives": qualitative_survives,
        "criteria": {
            "all_simple_better_than_text": simple_beats_text,
            "all_interaction_better_than_text": interaction_beats_text,
            "all_simple_interaction_abs_gap_below_0.003": close,
        },
    }
    (phase3 / "l5_downstream_results.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    comparison_frame = pd.DataFrame(comparisons)
    lines = [
        "# Phase 3 L5/L6 — Extractor Downstream Sensitivity",
        "",
        f"- hotel-intact sample n: {len(sample):,}",
        "- outer split: canonical GroupKFold assignment by offering_id",
        f"- alpha grid: {alphas}",
        "- fairness: Qwen selects one alpha per fold/model family; the same alpha is frozen for Llama and ensemble",
        "",
        "## L5 downstream results",
        "",
        summary.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Paired equal-hotel tests",
        "",
        comparison_frame.to_markdown(index=False, floatfmt=".6g"),
        "",
        "## L5 verdict",
        "",
        f"**Qualitative ordering survives extractor change: {qualitative_survives}.**",
        f"Fusion beats text for both variants/extractors: simple={simple_beats_text}, interaction={interaction_beats_text}; all simple-vs-interaction gaps <0.003: {close}.",
        "",
        "## L6 Qwen guided subsample versus full corpus",
        "",
        l6.to_markdown(index=False, floatfmt=".6f"),
        "",
    ]
    report = phase3 / "experiment_l5_l6_downstream.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    log(f"saved: {report}")


if __name__ == "__main__":
    main()
