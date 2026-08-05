#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.aspect_extractor_base import ASPECTS  # noqa: E402

SEED = 42
N_SPLITS = 5


def log(message: str) -> None:
    print(message)


def current_rss_mb() -> Optional[float]:
    if psutil is None:
        return None
    try:
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        return None


class TimingRecorder:
    def __init__(self) -> None:
        self.records: Dict[str, List[float]] = {}

    def record(self, stage: str, elapsed: float) -> None:
        self.records.setdefault(stage, []).append(elapsed)

    def report_lines(self) -> List[str]:
        lines = [
            "# Timing Summary",
            "",
            "| stage | count | avg_sec | total_sec |",
            "| --- | --- | --- | --- |",
        ]
        for stage, values in sorted(self.records.items()):
            total = sum(values)
            avg = total / len(values) if values else 0.0
            lines.append(f"| {stage} | {len(values)} | {avg:.3f} | {total:.3f} |")
        lines.append("")
        return lines


def log_stage(
    fold: int,
    stage: str,
    elapsed: float,
    obj: Optional[Any] = None,
    timing: Optional[TimingRecorder] = None,
) -> None:
    shape = "-"
    nnz = "-"
    try:
        import scipy.sparse as sp  # type: ignore

        if obj is not None:
            shape = str(obj.shape)
            if sp.issparse(obj):
                nnz = str(obj.nnz)
    except Exception:
        if obj is not None and hasattr(obj, "shape"):
            shape = str(obj.shape)
    rss = current_rss_mb()
    rss_str = f"{rss:.1f}" if rss is not None else "na"
    log(
        f"fold {fold} stage={stage} elapsed={elapsed:.2f}s shape={shape} nnz={nnz} rss_mb={rss_str}"
    )
    if timing is not None:
        timing.record(stage, elapsed)


def read_parquet_columns(path: Path, columns: List[str]) -> Optional[pd.DataFrame]:
    try:
        return pd.read_parquet(path, columns=columns)
    except Exception:
        return None


def read_dataset(path: Path, with_text: bool, with_ratings: bool) -> Tuple[pd.DataFrame, str]:
    text_candidates = ["review_text", "text"]
    base_cols = ["offering_id", "rating_overall"]
    rating_cols = [
        "rating_service",
        "rating_cleanliness",
        "rating_location",
        "rating_rooms",
        "rating_value",
        "rating_sleep_quality",
    ]
    if with_ratings:
        base_cols += rating_cols
    for text_col in (text_candidates if with_text else [None]):
        cols = ["row_id"] + base_cols + ([text_col] if text_col else [])
        df = read_parquet_columns(path, [c for c in cols if c])
        if df is not None:
            if "row_id" not in df.columns:
                df["row_id"] = df.index.astype(int)
                log("row_id missing in dataset; using index as row_id.")
            return df, (text_col or "")
        cols = base_cols + ([text_col] if text_col else [])
        df = read_parquet_columns(path, [c for c in cols if c])
        if df is not None:
            df["row_id"] = df.index.astype(int)
            log("row_id missing in dataset; using index as row_id.")
            return df, (text_col or "")
    raise ValueError("failed to read dataset with required columns.")


def read_aspects(path: Path) -> pd.DataFrame:
    aspect_score_cols = [f"aspect_{a}_score" for a in ASPECTS]
    aspect_mentioned_cols = [f"aspect_{a}_mentioned" for a in ASPECTS]
    base_cols = ["row_id", "ok", "traveler_segment"]
    cols = base_cols + aspect_score_cols + aspect_mentioned_cols
    df = read_parquet_columns(path, cols)
    if df is not None:
        return df
    cols = ["row_index", "ok", "traveler_segment"] + aspect_score_cols + aspect_mentioned_cols
    df = read_parquet_columns(path, cols)
    if df is None:
        raise ValueError("failed to read aspects parquet with expected columns.")
    df = df.rename(columns={"row_index": "row_id"})
    return df


def load_splits(output_dir: Path) -> List[Dict[str, np.ndarray]]:
    splits = []
    for fold in range(1, N_SPLITS + 1):
        split_path = output_dir / f"splits_fold{fold}.npz"
        if not split_path.exists():
            raise FileNotFoundError(f"missing split file: {split_path}")
        data = np.load(split_path)
        split_hash = hashlib.sha1(split_path.read_bytes()).hexdigest()
        splits.append(
            {
                "train_idx": data["train_idx"],
                "test_idx": data["test_idx"],
                "hash": split_hash,
                "path": str(split_path),
            }
        )
    return splits


def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
    raw_mae = float(mean_absolute_error(y_true, y_pred))
    raw_rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    y_clip = np.clip(y_pred, 1, 5)
    clip_mae = float(mean_absolute_error(y_true, y_clip))
    clip_rmse = float(np.sqrt(mean_squared_error(y_true, y_clip)))
    return {
        "raw": {"mae": raw_mae, "rmse": raw_rmse},
        "clipped": {"mae": clip_mae, "rmse": clip_rmse},
    }


def summarize_metrics(folds: List[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, float]]]:
    raw_mae = np.array([f["raw"]["mae"] for f in folds], dtype=float)
    raw_rmse = np.array([f["raw"]["rmse"] for f in folds], dtype=float)
    clip_mae = np.array([f["clipped"]["mae"] for f in folds], dtype=float)
    clip_rmse = np.array([f["clipped"]["rmse"] for f in folds], dtype=float)
    return {
        "raw": {
            "mean": {"mae": float(raw_mae.mean()), "rmse": float(raw_rmse.mean())},
            "std": {"mae": float(raw_mae.std(ddof=1)), "rmse": float(raw_rmse.std(ddof=1))},
        },
        "clipped": {
            "mean": {"mae": float(clip_mae.mean()), "rmse": float(clip_rmse.mean())},
            "std": {"mae": float(clip_mae.std(ddof=1)), "rmse": float(clip_rmse.std(ddof=1))},
        },
    }


def cache_key_prefix(
    profile: str,
    max_features: int,
    min_df: int,
    ngram_range: Tuple[int, int],
    split_hash: str,
    fold: int,
    tag: str,
) -> str:
    ngram_tag = f"{ngram_range[0]}{ngram_range[1]}"
    return f"{profile}_mf{max_features}_ng{ngram_tag}_md{min_df}_{tag}_fold{fold}_{split_hash[:8]}"


def load_or_build_tfidf(
    text_train: np.ndarray,
    text_test: np.ndarray,
    max_features: int,
    cache_dir: Optional[Path],
    cache_prefix: str,
    fold: int,
    timing: TimingRecorder,
    stage_prefix: str,
) -> Tuple[Any, Any]:
    import joblib  # type: ignore
    import scipy.sparse as sp  # type: ignore

    vec_path = None
    xtr_path = None
    xte_path = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        vec_path = cache_dir / f"{cache_prefix}_vectorizer.joblib"
        xtr_path = cache_dir / f"{cache_prefix}_Xtr.npz"
        xte_path = cache_dir / f"{cache_prefix}_Xte.npz"
        if vec_path.exists() and xtr_path.exists() and xte_path.exists():
            start = time.perf_counter()
            vectorizer = joblib.load(vec_path)
            x_train = sp.load_npz(xtr_path)
            x_test = sp.load_npz(xte_path)
            elapsed = time.perf_counter() - start
            log(f"fold {fold} stage={stage_prefix}.tfidf cache_hit")
            log_stage(
                fold,
                f"{stage_prefix}.tfidf_cache_load",
                elapsed,
                x_train,
                timing,
            )
            return x_train, x_test

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=3,
        max_features=max_features,
        dtype=np.float32,
    )
    start = time.perf_counter()
    x_train = vectorizer.fit_transform(text_train)
    elapsed = time.perf_counter() - start
    log_stage(fold, f"{stage_prefix}.tfidf_fit_transform", elapsed, x_train, timing)

    start = time.perf_counter()
    x_test = vectorizer.transform(text_test)
    elapsed = time.perf_counter() - start
    log_stage(fold, f"{stage_prefix}.tfidf_transform", elapsed, x_test, timing)

    if vec_path is not None and xtr_path is not None and xte_path is not None:
        joblib.dump(vectorizer, vec_path)
        sp.save_npz(xtr_path, x_train)
        sp.save_npz(xte_path, x_test)
        log(f"fold {fold} stage={stage_prefix}.tfidf_cache_write path={vec_path}")
    return x_train, x_test


def build_llm_components(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[str], List[str]]:
    score_cols = [f"aspect_{a}_score" for a in ASPECTS]
    mention_cols = [f"aspect_{a}_mentioned" for a in ASPECTS]
    segment_col = "traveler_segment"
    for col in score_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int).clip(-2, 2)
    for col in mention_cols:
        df[col] = df[col].fillna(False).astype(bool)
    df[segment_col] = df[segment_col].fillna("unknown").astype(str).str.lower()
    allowed_segments = {"business", "couple", "family", "solo", "friends", "unknown"}
    df[segment_col] = df[segment_col].where(df[segment_col].isin(allowed_segments), "unknown")

    seg_dummies = pd.get_dummies(df[segment_col], prefix="segment")
    x_scores = df[score_cols].to_numpy(dtype=float)
    x_mentions = df[mention_cols].to_numpy(dtype=float)
    x_segments = seg_dummies.to_numpy(dtype=float)
    segment_cols = seg_dummies.columns.tolist()
    return x_scores, x_mentions, x_segments, score_cols, mention_cols, segment_cols


def build_feature_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, List[str], List[str]]:
    x_scores, x_mentions, x_segments, score_cols, mention_cols, segment_cols = build_llm_components(df)
    feature_names = score_cols + mention_cols + segment_cols
    x = np.hstack([x_scores, x_mentions, x_segments])
    return x, feature_names, segment_cols


def select_alpha(
    x_train: np.ndarray,
    y_train: np.ndarray,
    groups: np.ndarray,
    alphas: List[float],
) -> float:
    if len(alphas) == 1:
        return alphas[0]
    inner_cv = GroupKFold(n_splits=3)
    best_alpha = alphas[0]
    best_mae = float("inf")
    for alpha in alphas:
        maes = []
        for inner_train, inner_val in inner_cv.split(x_train, y_train, groups=groups):
            scaler = StandardScaler()
            x_tr = scaler.fit_transform(x_train[inner_train])
            x_val = scaler.transform(x_train[inner_val])
            model = Ridge(alpha=alpha, random_state=SEED)
            model.fit(x_tr, y_train[inner_train])
            preds = model.predict(x_val)
            maes.append(mean_absolute_error(y_train[inner_val], preds))
        mae = float(np.mean(maes)) if maes else float("inf")
        if mae < best_mae:
            best_mae = mae
            best_alpha = alpha
    return best_alpha


def run_llm_aspects(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    splits: List[Dict[str, np.ndarray]],
    alphas: List[float],
    timing: TimingRecorder,
) -> Dict[str, Any]:
    from tqdm import tqdm  # type: ignore

    folds: List[Dict[str, Any]] = []
    best_alphas: List[float] = []
    for fold_idx, split in enumerate(
        tqdm(splits, total=len(splits), desc="LLM aspects folds"), start=1
    ):
        train_idx = split["train_idx"]
        test_idx = split["test_idx"]
        x_train = x[train_idx]
        y_train = y[train_idx]
        x_test = x[test_idx]
        y_test = y[test_idx]
        group_train = groups[train_idx]

        alpha = select_alpha(x_train, y_train, group_train, alphas)
        best_alphas.append(alpha)
        scaler = StandardScaler()
        start = time.perf_counter()
        x_train_scaled = scaler.fit_transform(x_train)
        elapsed = time.perf_counter() - start
        log_stage(fold_idx, "llm_aspects.scale_fit", elapsed, x_train_scaled, timing)
        start = time.perf_counter()
        x_test_scaled = scaler.transform(x_test)
        elapsed = time.perf_counter() - start
        log_stage(fold_idx, "llm_aspects.scale_transform", elapsed, x_test_scaled, timing)
        model = Ridge(alpha=alpha, random_state=SEED)
        start = time.perf_counter()
        model.fit(x_train_scaled, y_train)
        elapsed = time.perf_counter() - start
        log_stage(fold_idx, "llm_aspects.model_fit", elapsed, x_train_scaled, timing)
        start = time.perf_counter()
        preds = model.predict(x_test_scaled)
        elapsed = time.perf_counter() - start
        log_stage(fold_idx, "llm_aspects.model_predict", elapsed, x_test_scaled, timing)
        start = time.perf_counter()
        metrics = eval_metrics(y_test, preds)
        elapsed = time.perf_counter() - start
        log_stage(fold_idx, "llm_aspects.metrics", elapsed, None, timing)
        folds.append(
            {
                "fold": fold_idx,
                "best_alpha": alpha,
                **metrics,
            }
        )
        log(
            f"LLM aspects fold {fold_idx}: alpha={alpha} MAE={metrics['clipped']['mae']:.4f} "
            f"RMSE={metrics['clipped']['rmse']:.4f}"
        )
    summary = summarize_metrics(folds)
    return {"folds": folds, "summary": summary, "best_alphas": best_alphas}


def run_text_plus_aspects(
    texts: np.ndarray,
    x_num: np.ndarray,
    y: np.ndarray,
    splits: List[Dict[str, np.ndarray]],
    max_features: int,
    alpha: float,
    cache_dir: Optional[Path],
    profile: str,
    timing: TimingRecorder,
) -> Dict[str, Any]:
    import scipy.sparse as sp
    from tqdm import tqdm  # type: ignore

    folds: List[Dict[str, Any]] = []
    for fold_idx, split in enumerate(
        tqdm(splits, total=len(splits), desc="Text+LLM folds"), start=1
    ):
        train_idx = split["train_idx"]
        test_idx = split["test_idx"]
        text_train = texts[train_idx]
        text_test = texts[test_idx]
        x_num_train = x_num[train_idx]
        x_num_test = x_num[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        cache_prefix = cache_key_prefix(
            profile,
            max_features,
            3,
            (1, 2),
            split["hash"],
            fold_idx,
            "full",
        )
        x_text_train, x_text_test = load_or_build_tfidf(
            text_train,
            text_test,
            max_features,
            cache_dir,
            cache_prefix,
            fold_idx,
            timing,
            "text_llm",
        )
        scaler = StandardScaler()
        start = time.perf_counter()
        x_num_train_scaled = scaler.fit_transform(x_num_train)
        elapsed = time.perf_counter() - start
        log_stage(fold_idx, "text_llm.scale_fit", elapsed, x_num_train_scaled, timing)
        start = time.perf_counter()
        x_num_test_scaled = scaler.transform(x_num_test)
        elapsed = time.perf_counter() - start
        log_stage(fold_idx, "text_llm.scale_transform", elapsed, x_num_test_scaled, timing)
        x_train = sp.hstack([x_text_train, sp.csr_matrix(x_num_train_scaled)], format="csr")
        x_test = sp.hstack([x_text_test, sp.csr_matrix(x_num_test_scaled)], format="csr")

        model = Ridge(alpha=alpha, random_state=SEED)
        start = time.perf_counter()
        model.fit(x_train, y_train)
        elapsed = time.perf_counter() - start
        log_stage(fold_idx, "text_llm.model_fit", elapsed, x_train, timing)
        start = time.perf_counter()
        preds = model.predict(x_test)
        elapsed = time.perf_counter() - start
        log_stage(fold_idx, "text_llm.model_predict", elapsed, x_test, timing)
        start = time.perf_counter()
        metrics = eval_metrics(y_test, preds)
        elapsed = time.perf_counter() - start
        log_stage(fold_idx, "text_llm.metrics", elapsed, None, timing)
        folds.append({"fold": fold_idx, **metrics})
        log(
            f"Text+LLM fold {fold_idx}: MAE={metrics['clipped']['mae']:.4f} "
            f"RMSE={metrics['clipped']['rmse']:.4f}"
        )
    summary = summarize_metrics(folds)
    return {"folds": folds, "summary": summary, "alpha": alpha}


def compute_importance(
    x: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    best_alpha: float,
) -> Dict[str, Any]:
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    model = Ridge(alpha=best_alpha, random_state=SEED)
    model.fit(x_scaled, y)
    coef = model.coef_
    importances = [
        {"feature": name, "coef": float(weight), "abs_coef": float(abs(weight))}
        for name, weight in zip(feature_names, coef)
    ]
    importances_sorted = sorted(importances, key=lambda x: x["abs_coef"], reverse=True)
    return {"coefficients": importances_sorted}


def write_aspect_importance(
    importances: Dict[str, Any],
    output_path: Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    coeffs = importances["coefficients"]
    aspect_feats = [c for c in coeffs if c["feature"].startswith("aspect_")]
    segment_feats = [c for c in coeffs if c["feature"].startswith("segment_")]
    lines = [
        "# Aspect Importance (LLM-aspects Ridge)",
        "",
        "## Aspect-related Features (abs coef)",
        "",
        "| feature | coef | abs_coef |",
        "| --- | --- | --- |",
    ]
    for row in aspect_feats[:20]:
        lines.append(f"| {row['feature']} | {row['coef']:.4f} | {row['abs_coef']:.4f} |")
    lines.extend(
        [
            "",
            "## Segment Features (abs coef)",
            "",
            "| feature | coef | abs_coef |",
            "| --- | --- | --- |",
        ]
    )
    for row in segment_feats[:20]:
        lines.append(f"| {row['feature']} | {row['coef']:.4f} | {row['abs_coef']:.4f} |")
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")
    log(f"saved: {output_path}")
    return aspect_feats, segment_feats


def plot_aspect_importance(
    aspect_feats: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    top = aspect_feats[:12]
    labels = [row["feature"] for row in top]
    values = [row["abs_coef"] for row in top]
    plt.figure(figsize=(10, 5))
    plt.barh(labels[::-1], values[::-1], color="#2f6f6f")
    plt.title("LLM-aspects Importance (abs coef)")
    plt.xlabel("abs(coef)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    log(f"saved: {output_path}")


def write_segment_preference(
    df: pd.DataFrame,
    output_path: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    score_cols = [f"aspect_{a}_score" for a in ASPECTS]
    mention_cols = [f"aspect_{a}_mentioned" for a in ASPECTS]
    segment_col = "traveler_segment"
    stats_score = df.groupby(segment_col)[score_cols].mean()
    stats_mention = df.groupby(segment_col)[mention_cols].mean()

    lines = [
        "# Segment Preference Summary",
        "",
        "## Aspect Score Means",
        "",
    ]
    lines.append(stats_score.to_markdown())
    lines.extend(["", "## Aspect Mentioned Rates", ""])
    lines.append(stats_mention.to_markdown())

    ranges = (stats_score.max() - stats_score.min()).sort_values(ascending=False)
    lines.extend(["", "## Largest Differences (max-min)", ""])
    for aspect, value in ranges.head(3).items():
        lines.append(f"- {aspect}: {value:.4f}")
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")
    log(f"saved: {output_path}")
    return stats_score, stats_mention


def plot_segment_heatmap(
    stats_score: pd.DataFrame,
    output_path: Path,
) -> None:
    order = ["business", "couple", "family", "solo", "friends", "unknown"]
    segments = [s for s in order if s in stats_score.index] + [
        s for s in stats_score.index if s not in order
    ]
    data = stats_score.loc[segments]
    plt.figure(figsize=(9, 5))
    plt.imshow(data.values, aspect="auto", cmap="coolwarm")
    plt.colorbar(label="mean score")
    plt.yticks(range(len(segments)), segments)
    plt.xticks(range(len(data.columns)), data.columns, rotation=45, ha="right")
    plt.title("Segment vs Aspect Score Means")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    log(f"saved: {output_path}")


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 2:
        return float("nan")
    denom = np.sum((y_true - y_true.mean()) ** 2)
    if denom == 0:
        return float("nan")
    return float(1.0 - np.sum((y_true - y_pred) ** 2) / denom)


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    x_std = np.std(x)
    y_std = np.std(y)
    if x_std == 0 or y_std == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    rx = pd.Series(x).rank(method="average").to_numpy()
    ry = pd.Series(y).rank(method="average").to_numpy()
    return pearson_corr(rx, ry)


def evaluate_extraction_quality(
    df: pd.DataFrame,
    output_path: Path,
) -> pd.DataFrame:
    rows = []
    for aspect in ASPECTS:
        score_col = f"aspect_{aspect}_score"
        rating_col = f"rating_{aspect}"
        if rating_col not in df.columns or score_col not in df.columns:
            continue
        valid = df[rating_col].notna()
        if not valid.any():
            continue
        y_true = df.loc[valid, rating_col].astype(float).to_numpy()
        scores = df.loc[valid, score_col].astype(float).to_numpy()
        preds = np.clip(3.0 + scores, 1.0, 5.0)
        mae = float(mean_absolute_error(y_true, preds))
        rmse = float(np.sqrt(mean_squared_error(y_true, preds)))
        pearson = pearson_corr(y_true, preds)
        spearman = spearman_corr(y_true, preds)
        rows.append(
            {
                "aspect": aspect,
                "n": int(valid.sum()),
                "mae": mae,
                "rmse": rmse,
                "pearson_r": pearson,
                "spearman_rho": spearman,
            }
        )
    df_eval = pd.DataFrame(rows)
    if not df_eval.empty:
        avg = {
            "aspect": "mean",
            "n": int(df_eval["n"].sum()),
            "mae": float(df_eval["mae"].mean()),
            "rmse": float(df_eval["rmse"].mean()),
            "pearson_r": float(df_eval["pearson_r"].mean()),
            "spearman_rho": float(df_eval["spearman_rho"].mean()),
        }
        df_eval = pd.concat([df_eval, pd.DataFrame([avg])], ignore_index=True)

    lines = [
        "# Aspect Extraction Quality (LLM vs Ratings)",
        "",
        "| aspect | n | mae | rmse | pearson_r | spearman_rho |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in df_eval.iterrows():
        lines.append(
            f"| {row['aspect']} | {int(row['n'])} | {row['mae']:.4f} | {row['rmse']:.4f} | "
            f"{row['pearson_r']:.4f} | {row['spearman_rho']:.4f} |"
        )
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")
    log(f"saved: {output_path}")
    return df_eval


def extract_evidence_strings(response_json: str) -> Dict[str, List[str]]:
    result: Dict[str, List[str]] = {}
    if not response_json:
        return result
    try:
        obj = json.loads(response_json)
    except Exception:
        return result
    aspects = obj.get("aspects", {}) if isinstance(obj, dict) else {}
    for aspect in ASPECTS:
        data = aspects.get(aspect, {}) if isinstance(aspects, dict) else {}
        evidence = data.get("evidence", [])
        if isinstance(evidence, list):
            items = [str(e).strip() for e in evidence if str(e).strip()]
        else:
            items = [str(evidence).strip()] if str(evidence).strip() else []
        if items:
            result[aspect] = items[:2]
    return result


def run_alignment_check(
    df_join: pd.DataFrame,
    output_path: Path,
    sample_n: int = 2000,
) -> Dict[str, float]:
    ok_df = df_join[df_join["ok"] == 1].copy()
    if ok_df.empty:
        log("alignment check skipped: no ok==1 rows.")
        return {}
    rng = np.random.default_rng(SEED)
    sample_n = min(sample_n, len(ok_df))
    sample = ok_df.sample(n=sample_n, random_state=SEED)
    total = 0
    hits = 0
    aspect_hits = {aspect: 0 for aspect in ASPECTS}
    aspect_total = {aspect: 0 for aspect in ASPECTS}

    for _, row in sample.iterrows():
        text = str(row.get("review_text", "") or row.get("text", "")).lower()
        evidence_map = extract_evidence_strings(str(row.get("response_json", "")))
        for aspect, evidences in evidence_map.items():
            for ev in evidences:
                total += 1
                aspect_total[aspect] += 1
                if ev.lower() in text:
                    hits += 1
                    aspect_hits[aspect] += 1

    overall_rate = hits / total if total else 0.0
    lines = [
        "# Join Alignment Check (Evidence vs Review Text)",
        "",
        f"- sample_n: {sample_n}",
        f"- total_evidence: {total}",
        f"- hit_rate: {overall_rate:.4f}",
        "",
        "| aspect | hit_rate | hits | total |",
        "| --- | --- | --- | --- |",
    ]
    for aspect in ASPECTS:
        denom = aspect_total[aspect]
        rate = (aspect_hits[aspect] / denom) if denom else 0.0
        lines.append(
            f"| {aspect} | {rate:.4f} | {aspect_hits[aspect]} | {denom} |"
        )
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")
    log(f"saved: {output_path}")

    if overall_rate < 0.6:
        log("WARNING: evidence hit_rate < 0.60; possible row_id misalignment.")
    return {
        "overall": overall_rate,
        **{f"{aspect}": (aspect_hits[aspect] / aspect_total[aspect] if aspect_total[aspect] else 0.0)
           for aspect in ASPECTS},
    }


def build_interactions(
    scores: np.ndarray,
    mentions: np.ndarray,
    segments: np.ndarray,
) -> np.ndarray:
    if segments.size == 0:
        return np.empty((scores.shape[0], 0), dtype=np.float32)
    parts = []
    for idx in range(segments.shape[1]):
        seg = segments[:, idx : idx + 1]
        parts.append(scores * seg)
        parts.append(mentions * seg)
    return np.hstack(parts).astype(np.float32)


def select_alpha_sparse(
    x_train,
    y_train: np.ndarray,
    groups: np.ndarray,
    alphas: List[float],
) -> float:
    if len(alphas) == 1:
        return alphas[0]
    inner_cv = GroupKFold(n_splits=3)
    best_alpha = alphas[0]
    best_mae = float("inf")
    for alpha in alphas:
        maes = []
        for inner_train, inner_val in inner_cv.split(x_train, y_train, groups=groups):
            model = Ridge(alpha=alpha, random_state=SEED)
            model.fit(x_train[inner_train], y_train[inner_train])
            preds = model.predict(x_train[inner_val])
            maes.append(mean_absolute_error(y_train[inner_val], preds))
        mae = float(np.mean(maes)) if maes else float("inf")
        if mae < best_mae:
            best_mae = mae
            best_alpha = alpha
    return best_alpha


def run_residual_on_llm(
    texts: np.ndarray,
    x_llm: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    splits: List[Dict[str, np.ndarray]],
    max_features: int,
    alpha_grid: List[float],
    cache_dir: Optional[Path],
    profile: str,
    timing: TimingRecorder,
) -> Dict[str, Any]:
    from tqdm import tqdm  # type: ignore

    folds: List[Dict[str, Any]] = []
    r2_train_list: List[float] = []
    r2_test_list: List[float] = []
    for fold_idx, split in enumerate(
        tqdm(splits, total=len(splits), desc="Residual+LLM folds"), start=1
    ):
        train_idx = split["train_idx"]
        test_idx = split["test_idx"]
        text_train = texts[train_idx]
        text_test = texts[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        groups_train = groups[train_idx]

        cache_prefix = cache_key_prefix(
            profile,
            max_features,
            3,
            (1, 2),
            split["hash"],
            fold_idx,
            "full",
        )
        x_text_train, x_text_test = load_or_build_tfidf(
            text_train,
            text_test,
            max_features,
            cache_dir,
            cache_prefix,
            fold_idx,
            timing,
            "residual",
        )
        text_model = Ridge(alpha=1.0, random_state=SEED)
        start = time.perf_counter()
        text_model.fit(x_text_train, y_train)
        elapsed = time.perf_counter() - start
        log_stage(fold_idx, "residual.text_fit", elapsed, x_text_train, timing)
        start = time.perf_counter()
        y_pred_text_train = text_model.predict(x_text_train)
        elapsed = time.perf_counter() - start
        log_stage(fold_idx, "residual.text_predict_train", elapsed, x_text_train, timing)
        start = time.perf_counter()
        y_pred_text_test = text_model.predict(x_text_test)
        elapsed = time.perf_counter() - start
        log_stage(fold_idx, "residual.text_predict_test", elapsed, x_text_test, timing)

        r_train = y_train - y_pred_text_train
        r_test = y_test - y_pred_text_test
        alpha = select_alpha(x_llm[train_idx], r_train, groups_train, alpha_grid)
        scaler = StandardScaler()
        start = time.perf_counter()
        x_train_scaled = scaler.fit_transform(x_llm[train_idx])
        elapsed = time.perf_counter() - start
        log_stage(fold_idx, "residual.llm_scale_fit", elapsed, x_train_scaled, timing)
        start = time.perf_counter()
        x_test_scaled = scaler.transform(x_llm[test_idx])
        elapsed = time.perf_counter() - start
        log_stage(fold_idx, "residual.llm_scale_transform", elapsed, x_test_scaled, timing)
        llm_model = Ridge(alpha=alpha, random_state=SEED)
        start = time.perf_counter()
        llm_model.fit(x_train_scaled, r_train)
        elapsed = time.perf_counter() - start
        log_stage(fold_idx, "residual.llm_fit", elapsed, x_train_scaled, timing)
        start = time.perf_counter()
        r_pred_train = llm_model.predict(x_train_scaled)
        elapsed = time.perf_counter() - start
        log_stage(fold_idx, "residual.llm_predict_train", elapsed, x_train_scaled, timing)
        start = time.perf_counter()
        r_pred_test = llm_model.predict(x_test_scaled)
        elapsed = time.perf_counter() - start
        log_stage(fold_idx, "residual.llm_predict_test", elapsed, x_test_scaled, timing)
        r2_train = r2_score(r_train, r_pred_train)
        r2_test = r2_score(r_test, r_pred_test)
        r2_train_list.append(r2_train)
        r2_test_list.append(r2_test)

        y_pred = y_pred_text_test + r_pred_test
        start = time.perf_counter()
        metrics = eval_metrics(y_test, y_pred)
        elapsed = time.perf_counter() - start
        log_stage(fold_idx, "residual.metrics", elapsed, None, timing)
        folds.append(
            {
                "fold": fold_idx,
                "best_alpha": alpha,
                "r2_train": r2_train,
                "r2_test": r2_test,
                **metrics,
            }
        )
        log(
            f"Residual+LLM fold {fold_idx}: alpha={alpha} MAE={metrics['clipped']['mae']:.4f} "
            f"RMSE={metrics['clipped']['rmse']:.4f} r2_test={r2_test:.4f}"
        )
    summary = summarize_metrics(folds)
    return {
        "folds": folds,
        "summary": summary,
        "r2_train_mean": float(np.nanmean(r2_train_list)) if r2_train_list else float("nan"),
        "r2_test_mean": float(np.nanmean(r2_test_list)) if r2_test_list else float("nan"),
    }


def run_text_llm_interactions(
    texts: np.ndarray,
    scores: np.ndarray,
    mentions: np.ndarray,
    segments: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    splits: List[Dict[str, np.ndarray]],
    max_features: int,
    alpha_grid: List[float],
    cache_dir: Optional[Path],
    profile: str,
    timing: TimingRecorder,
) -> Dict[str, Any]:
    import scipy.sparse as sp
    from tqdm import tqdm  # type: ignore

    folds: List[Dict[str, Any]] = []
    best_alphas: List[float] = []
    for fold_idx, split in enumerate(
        tqdm(splits, total=len(splits), desc="Text+LLM+Inter folds"), start=1
    ):
        train_idx = split["train_idx"]
        test_idx = split["test_idx"]
        text_train = texts[train_idx]
        text_test = texts[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]
        groups_train = groups[train_idx]

        cache_prefix = cache_key_prefix(
            profile,
            max_features,
            3,
            (1, 2),
            split["hash"],
            fold_idx,
            "full",
        )
        x_text_train, x_text_test = load_or_build_tfidf(
            text_train,
            text_test,
            max_features,
            cache_dir,
            cache_prefix,
            fold_idx,
            timing,
            "interaction",
        )

        base_train = np.hstack([scores[train_idx], mentions[train_idx], segments[train_idx]])
        base_test = np.hstack([scores[test_idx], mentions[test_idx], segments[test_idx]])
        start = time.perf_counter()
        inter_train = build_interactions(scores[train_idx], mentions[train_idx], segments[train_idx])
        elapsed = time.perf_counter() - start
        log_stage(fold_idx, "interaction.build_interactions_train", elapsed, inter_train, timing)
        start = time.perf_counter()
        inter_test = build_interactions(scores[test_idx], mentions[test_idx], segments[test_idx])
        elapsed = time.perf_counter() - start
        log_stage(fold_idx, "interaction.build_interactions_test", elapsed, inter_test, timing)
        dense_train = np.hstack([base_train, inter_train]).astype(np.float32)
        dense_test = np.hstack([base_test, inter_test]).astype(np.float32)

        scaler = StandardScaler()
        start = time.perf_counter()
        dense_train_scaled = scaler.fit_transform(dense_train)
        elapsed = time.perf_counter() - start
        log_stage(fold_idx, "interaction.scale_fit", elapsed, dense_train_scaled, timing)
        start = time.perf_counter()
        dense_test_scaled = scaler.transform(dense_test)
        elapsed = time.perf_counter() - start
        log_stage(fold_idx, "interaction.scale_transform", elapsed, dense_test_scaled, timing)

        x_train = sp.hstack([x_text_train, sp.csr_matrix(dense_train_scaled)], format="csr")
        x_test = sp.hstack([x_text_test, sp.csr_matrix(dense_test_scaled)], format="csr")

        alpha = select_alpha_sparse(x_train, y_train, groups_train, alpha_grid)
        best_alphas.append(alpha)
        model = Ridge(alpha=alpha, random_state=SEED)
        start = time.perf_counter()
        model.fit(x_train, y_train)
        elapsed = time.perf_counter() - start
        log_stage(fold_idx, "interaction.model_fit", elapsed, x_train, timing)
        start = time.perf_counter()
        preds = model.predict(x_test)
        elapsed = time.perf_counter() - start
        log_stage(fold_idx, "interaction.model_predict", elapsed, x_test, timing)
        start = time.perf_counter()
        metrics = eval_metrics(y_test, preds)
        elapsed = time.perf_counter() - start
        log_stage(fold_idx, "interaction.metrics", elapsed, None, timing)
        folds.append({"fold": fold_idx, "best_alpha": alpha, **metrics})
        log(
            f"Text+LLM+Interaction fold {fold_idx}: alpha={alpha} MAE={metrics['clipped']['mae']:.4f} "
            f"RMSE={metrics['clipped']['rmse']:.4f}"
        )
    summary = summarize_metrics(folds)
    return {"folds": folds, "summary": summary, "best_alphas": best_alphas}


def run_subset_eval(
    texts: np.ndarray,
    x_llm: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    mask: np.ndarray,
    splits: List[Dict[str, np.ndarray]],
    max_features: int,
    alpha: float,
    cache_dir: Optional[Path],
    profile: str,
    timing: TimingRecorder,
) -> Dict[str, Any]:
    import scipy.sparse as sp
    from tqdm import tqdm  # type: ignore

    text_folds = []
    mix_folds = []
    skipped = 0
    for fold_idx, split in enumerate(
        tqdm(splits, total=len(splits), desc="Subset folds"), start=1
    ):
        train_idx = split["train_idx"]
        test_idx = split["test_idx"]
        train_mask = mask[train_idx]
        test_mask = mask[test_idx]
        if not test_mask.any() or not train_mask.any():
            skipped += 1
            continue
        train_ids = train_idx[train_mask]
        test_ids = test_idx[test_mask]
        text_train = texts[train_ids]
        text_test = texts[test_ids]
        y_train = y[train_ids]
        y_test = y[test_ids]

        cache_prefix = cache_key_prefix(
            profile,
            max_features,
            3,
            (1, 2),
            split["hash"],
            fold_idx,
            "subset",
        )
        x_text_train, x_text_test = load_or_build_tfidf(
            text_train,
            text_test,
            max_features,
            cache_dir,
            cache_prefix,
            fold_idx,
            timing,
            "subset",
        )
        text_model = Ridge(alpha=1.0, random_state=SEED)
        start = time.perf_counter()
        text_model.fit(x_text_train, y_train)
        elapsed = time.perf_counter() - start
        log_stage(fold_idx, "subset.text_fit", elapsed, x_text_train, timing)
        start = time.perf_counter()
        text_preds = text_model.predict(x_text_test)
        elapsed = time.perf_counter() - start
        log_stage(fold_idx, "subset.text_predict", elapsed, x_text_test, timing)
        start = time.perf_counter()
        text_metrics = eval_metrics(y_test, text_preds)
        elapsed = time.perf_counter() - start
        log_stage(fold_idx, "subset.text_metrics", elapsed, None, timing)
        text_folds.append({"fold": fold_idx, **text_metrics})
        log(
            f"Subset Text-only fold {fold_idx}: MAE={text_metrics['clipped']['mae']:.4f} "
            f"RMSE={text_metrics['clipped']['rmse']:.4f}"
        )

        scaler = StandardScaler()
        start = time.perf_counter()
        x_train_scaled = scaler.fit_transform(x_llm[train_ids])
        elapsed = time.perf_counter() - start
        log_stage(fold_idx, "subset.llm_scale_fit", elapsed, x_train_scaled, timing)
        start = time.perf_counter()
        x_test_scaled = scaler.transform(x_llm[test_ids])
        elapsed = time.perf_counter() - start
        log_stage(fold_idx, "subset.llm_scale_transform", elapsed, x_test_scaled, timing)
        x_train = sp.hstack([x_text_train, sp.csr_matrix(x_train_scaled)], format="csr")
        x_test = sp.hstack([x_text_test, sp.csr_matrix(x_test_scaled)], format="csr")
        model = Ridge(alpha=alpha, random_state=SEED)
        start = time.perf_counter()
        model.fit(x_train, y_train)
        elapsed = time.perf_counter() - start
        log_stage(fold_idx, "subset.mix_fit", elapsed, x_train, timing)
        start = time.perf_counter()
        mix_preds = model.predict(x_test)
        elapsed = time.perf_counter() - start
        log_stage(fold_idx, "subset.mix_predict", elapsed, x_test, timing)
        start = time.perf_counter()
        mix_metrics = eval_metrics(y_test, mix_preds)
        elapsed = time.perf_counter() - start
        log_stage(fold_idx, "subset.mix_metrics", elapsed, None, timing)
        mix_folds.append({"fold": fold_idx, **mix_metrics})
        log(
            f"Subset Text+LLM fold {fold_idx}: MAE={mix_metrics['clipped']['mae']:.4f} "
            f"RMSE={mix_metrics['clipped']['rmse']:.4f}"
        )

    return {
        "text_only": {"folds": text_folds, "summary": summarize_metrics(text_folds)},
        "text_plus_llm": {"folds": mix_folds, "summary": summarize_metrics(mix_folds)},
        "skipped_folds": skipped,
    }

def load_baselines(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"missing baseline results: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_results_md(
    output_path: Path,
    baselines: Dict[str, Any],
    llm_results: Dict[str, Any],
    text_plus: Optional[Dict[str, Any]],
    unknown_share: float,
    residual_results: Optional[Dict[str, Any]],
    interaction_results: Optional[Dict[str, Any]],
    subset_results: Optional[Dict[str, Any]],
) -> None:
    text_mean = baselines["text_only"]["clipped"]["mean"]
    text_std = baselines["text_only"]["clipped"]["std"]
    upper_mean = baselines["upper_bound"]["clipped"]["mean"]
    upper_std = baselines["upper_bound"]["clipped"]["std"]
    llm_mean = llm_results["summary"]["clipped"]["mean"]
    llm_std = llm_results["summary"]["clipped"]["std"]

    lines = [
        "# LLM Aspects Results",
        "",
        "| Method | MAE (mean+/-std) | RMSE (mean+/-std) |",
        "| --- | --- | --- |",
        "| Text-only TF-IDF + Ridge | "
        f"{text_mean['mae']:.4f} +/- {text_std['mae']:.4f} | "
        f"{text_mean['rmse']:.4f} +/- {text_std['rmse']:.4f} |",
        "| Upper bound (true aspects) | "
        f"{upper_mean['mae']:.4f} +/- {upper_std['mae']:.4f} | "
        f"{upper_mean['rmse']:.4f} +/- {upper_std['rmse']:.4f} |",
        "| LLM aspects (scores+mentioned+segment) | "
        f"{llm_mean['mae']:.4f} +/- {llm_std['mae']:.4f} | "
        f"{llm_mean['rmse']:.4f} +/- {llm_std['rmse']:.4f} |",
    ]
    if text_plus:
        tp_mean = text_plus["summary"]["clipped"]["mean"]
        tp_std = text_plus["summary"]["clipped"]["std"]
        lines.append(
            "| Text + LLM aspects | "
            f"{tp_mean['mae']:.4f} +/- {tp_std['mae']:.4f} | "
            f"{tp_mean['rmse']:.4f} +/- {tp_std['rmse']:.4f} |"
        )
    llm_mae_delta = llm_mean["mae"] - text_mean["mae"]
    llm_upper_gap = llm_mean["mae"] - upper_mean["mae"]
    lines.extend(
        [
            "",
            "Notes:",
            "- Split: GroupKFold(5), grouped by offering_id (no hotel leakage).",
            "- Metrics: MAE and RMSE on rating_overall (1-5), clipped to [1,5].",
            "",
            "Key Findings:",
            f"- LLM aspects MAE vs text-only: {llm_mae_delta:+.4f}.",
            f"- Gap to upper bound (MAE): {llm_upper_gap:+.4f}.",
            "",
            "Why aspects-only may lag text-only:",
            "- Aspect extraction compresses review information; nuanced cues can be lost.",
            f"- traveler_segment unknown share is {unknown_share:.3f}, weakening segment signals.",
        ]
    )
    if text_plus:
        tp_mean = text_plus["summary"]["clipped"]["mean"]
        lines.append(
            f"- Text+LLM vs LLM-only (MAE): {tp_mean['mae'] - llm_mean['mae']:+.4f}."
        )
    lines.append("")

    if residual_results or interaction_results or subset_results:
        lines.extend(["## Incremental Value Tests", ""])
        if residual_results:
            res_mean = residual_results["summary"]["clipped"]["mean"]
            res_std = residual_results["summary"]["clipped"]["std"]
            text_mean = baselines["text_only"]["clipped"]["mean"]
            delta_mae = res_mean["mae"] - text_mean["mae"]
            delta_rmse = res_mean["rmse"] - text_mean["rmse"]
            lines.extend(
                [
                    "### Residual-on-LLM",
                    "",
                    "| MAE (mean+/-std) | RMSE (mean+/-std) | Delta MAE vs Text-only | Delta RMSE vs Text-only |",
                    "| --- | --- | --- | --- |",
                    f"| {res_mean['mae']:.4f} +/- {res_std['mae']:.4f} | "
                    f"{res_mean['rmse']:.4f} +/- {res_std['rmse']:.4f} | "
                    f"{delta_mae:+.4f} | {delta_rmse:+.4f} |",
                    f"- residual R2 train mean: {residual_results['r2_train_mean']:.4f}",
                    f"- residual R2 test mean: {residual_results['r2_test_mean']:.4f}",
                    "",
                ]
            )
        if interaction_results:
            inter_mean = interaction_results["summary"]["clipped"]["mean"]
            inter_std = interaction_results["summary"]["clipped"]["std"]
            alpha_med = float(np.median(interaction_results.get("best_alphas", [0.0])))
            lines.extend(
                [
                    "### Text + LLM + Interaction",
                    "",
                    "| MAE (mean+/-std) | RMSE (mean+/-std) | alpha (median) |",
                    "| --- | --- | --- |",
                    f"| {inter_mean['mae']:.4f} +/- {inter_std['mae']:.4f} | "
                    f"{inter_mean['rmse']:.4f} +/- {inter_std['rmse']:.4f} | {alpha_med:.4f} |",
                    "",
                ]
            )
        if subset_results:
            text_subset = subset_results["text_only"]["summary"]["clipped"]["mean"]
            text_subset_std = subset_results["text_only"]["summary"]["clipped"]["std"]
            mix_subset = subset_results["text_plus_llm"]["summary"]["clipped"]["mean"]
            mix_subset_std = subset_results["text_plus_llm"]["summary"]["clipped"]["std"]
            skipped = subset_results.get("skipped_folds", 0)
            lines.extend(
                [
                    "### Subset Sanity (segment != unknown, ok==1)",
                    "",
                    "| Method | MAE (mean+/-std) | RMSE (mean+/-std) | skipped_folds |",
                    "| --- | --- | --- | --- |",
                    f"| Text-only | {text_subset['mae']:.4f} +/- {text_subset_std['mae']:.4f} | "
                    f"{text_subset['rmse']:.4f} +/- {text_subset_std['rmse']:.4f} | {skipped} |",
                    f"| Text+LLM | {mix_subset['mae']:.4f} +/- {mix_subset_std['mae']:.4f} | "
                    f"{mix_subset['rmse']:.4f} +/- {mix_subset_std['rmse']:.4f} | {skipped} |",
                    "",
                ]
            )
        if residual_results:
            if residual_results["r2_test_mean"] < 0.01:
                lines.append("- Residual R2 is low; incremental signal from LLM aspects appears weak.")
            lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")
    log(f"saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM aspects analysis pipeline.")
    parser.add_argument("--profile", default="qwen25_7b")
    parser.add_argument("--with-text", action="store_true")
    parser.add_argument("--max_features", type=int, default=200000)
    parser.add_argument("--out_prefix", type=str, default=None)
    parser.add_argument("--cache_tfidf_dir", type=str, default="outputs/tfidf_cache")
    args = parser.parse_args()

    output_dir = Path(__file__).resolve().parents[1] / "outputs"
    log(
        "Note: this script is CPU/IO bound; low GPU utilization is expected. "
        "vLLM is only used in scripts/06_extract_aspects_llm.py."
    )
    dataset_path = output_dir / "dataset_clean.parquet"
    aspects_path = output_dir / f"aspects_llm_{args.profile}.parquet"
    splits = load_splits(output_dir)

    timing = TimingRecorder()
    cache_dir = None
    if args.cache_tfidf_dir:
        cache_dir = Path(args.cache_tfidf_dir)
        if not cache_dir.is_absolute():
            cache_dir = (Path(__file__).resolve().parents[1] / cache_dir).resolve()

    df_dataset, text_col = read_dataset(dataset_path, True, with_ratings=True)
    df_aspects = read_aspects(aspects_path)

    if "row_id" not in df_dataset.columns:
        df_dataset["row_id"] = df_dataset.index.astype(int)
    if "row_id" not in df_aspects.columns:
        raise ValueError("aspects file missing row_id.")
    df_dataset["row_id"] = pd.to_numeric(df_dataset["row_id"], errors="coerce")
    df_aspects["row_id"] = pd.to_numeric(df_aspects["row_id"], errors="coerce")
    if df_dataset["row_id"].isna().any() or df_aspects["row_id"].isna().any():
        raise ValueError("row_id contains NaN after coercion.")
    df_dataset["row_id"] = df_dataset["row_id"].astype(int)
    df_aspects["row_id"] = df_aspects["row_id"].astype(int)
    if df_aspects["row_id"].duplicated().any():
        dup = int(df_aspects["row_id"].duplicated().sum())
        raise ValueError(f"duplicate row_id in aspects: {dup}")

    df = df_dataset.merge(df_aspects, on="row_id", how="left", sort=False)
    if len(df) != 877808:
        raise ValueError(f"join row count {len(df)} != 877808")
    if df["row_id"].duplicated().any():
        dup = int(df["row_id"].duplicated().sum())
        raise ValueError(f"duplicate row_id after join: {dup}")

    ok_mask = df["ok"] == 1
    failed_n = int((~ok_mask).sum())
    unknown_share = float((df["traveler_segment"].fillna("unknown") == "unknown").mean())
    log(f"join rows={len(df)} ok_rate={ok_mask.mean():.4f} failed={failed_n}")
    log(f"unknown traveler_segment share={unknown_share:.4f}")

    score_cols = [f"aspect_{a}_score" for a in ASPECTS]
    mention_cols = [f"aspect_{a}_mentioned" for a in ASPECTS]
    df.loc[~ok_mask, score_cols] = 0
    df.loc[~ok_mask, mention_cols] = False
    df.loc[~ok_mask, "traveler_segment"] = "unknown"

    x_scores, x_mentions, x_segments, score_cols, mention_cols, segment_cols = build_llm_components(df)
    x = np.hstack([x_scores, x_mentions, x_segments])
    feature_names = score_cols + mention_cols + segment_cols
    y = df["rating_overall"].astype(float).to_numpy()
    groups = df["offering_id"].to_numpy()

    alphas = [0.1, 1.0, 10.0]
    llm_results = run_llm_aspects(x, y, groups, splits, alphas, timing)

    quality_path = output_dir / f"aspect_extraction_quality_{args.profile}.md"
    start = time.perf_counter()
    quality_df = evaluate_extraction_quality(df, quality_path)
    timing.record("write.extraction_quality", time.perf_counter() - start)
    if not quality_df.empty:
        aspect_sorted = quality_df[quality_df["aspect"] != "mean"].sort_values("mae")
        if not aspect_sorted.empty:
            top_aspect = aspect_sorted.iloc[0]
            bottom_aspect = aspect_sorted.iloc[-1]
            log(
                "extraction quality MAE best={best} ({best_mae:.4f}) worst={worst} ({worst_mae:.4f})".format(
                    best=top_aspect["aspect"],
                    best_mae=top_aspect["mae"],
                    worst=bottom_aspect["aspect"],
                    worst_mae=bottom_aspect["mae"],
                )
            )

    align_path = output_dir / f"join_alignment_check_{args.profile}.md"
    response_cols = ["row_id", "ok", "response_json"]
    df_resp = read_parquet_columns(aspects_path, response_cols)
    if df_resp is not None:
        df_resp["row_id"] = pd.to_numeric(df_resp["row_id"], errors="coerce")
        if df_resp["row_id"].isna().any():
            raise ValueError("response row_id contains NaN")
        df_resp["row_id"] = df_resp["row_id"].astype(int)
        df_text = read_parquet_columns(dataset_path, ["review_text"])
        if df_text is None:
            df_text = read_parquet_columns(dataset_path, ["text"])
        if df_text is not None:
            df_text = df_text.copy()
            df_text["row_id"] = df_text.index.astype(int)
            df_align = df_resp.merge(df_text, on="row_id", how="left", sort=False)
            start = time.perf_counter()
            run_alignment_check(df_align, align_path, sample_n=2000)
            timing.record("write.alignment_check", time.perf_counter() - start)
        else:
            log("alignment check skipped: review_text/text missing.")
    else:
        log("alignment check skipped: response_json missing.")

    best_alpha = float(np.median(llm_results["best_alphas"]))
    importance = compute_importance(x, y, feature_names, best_alpha)
    start = time.perf_counter()
    aspect_feats, segment_feats = write_aspect_importance(
        importance,
        output_dir / f"aspect_importance_{args.profile}.md",
    )
    timing.record("write.aspect_importance", time.perf_counter() - start)
    start = time.perf_counter()
    plot_aspect_importance(
        aspect_feats,
        output_dir / f"fig_aspect_importance_{args.profile}.png",
    )
    timing.record("write.fig_aspect_importance", time.perf_counter() - start)

    start = time.perf_counter()
    stats_score, _stats_mention = write_segment_preference(
        df,
        output_dir / f"segment_preference_{args.profile}.md",
    )
    timing.record("write.segment_preference", time.perf_counter() - start)
    start = time.perf_counter()
    plot_segment_heatmap(
        stats_score,
        output_dir / f"fig_segment_aspect_heatmap_{args.profile}.png",
    )
    timing.record("write.fig_segment_heatmap", time.perf_counter() - start)

    overall_scores = df[score_cols].mean()
    diff_rows = []
    for segment, row in stats_score.iterrows():
        diffs = (row - overall_scores).abs()
        for aspect, value in diffs.items():
            diff_rows.append((segment, aspect, float(value)))
    diff_rows_sorted = sorted(diff_rows, key=lambda x: x[2], reverse=True)[:3]

    top_aspects = [
        (row["feature"], row["abs_coef"])
        for row in aspect_feats
        if row["feature"].endswith("_score")
    ][:3]
    log(f"top3 aspect importance: {top_aspects}")
    log(f"top3 segment-aspect deviations: {diff_rows_sorted}")

    baselines = load_baselines(output_dir / "results_baselines.json")
    text_plus = None
    if args.with_text:
        if not text_col:
            raise ValueError("text column missing for --with-text")
        texts = df[text_col].fillna("").astype(str).to_numpy()
        text_plus = run_text_plus_aspects(
            texts=texts,
            x_num=x,
            y=y,
            splits=splits,
            max_features=args.max_features,
            alpha=0.1,
            cache_dir=cache_dir,
            profile=args.profile,
            timing=timing,
        )

    residual_results = None
    interaction_results = None
    subset_results = None
    if text_col:
        texts = df[text_col].fillna("").astype(str).to_numpy()
        residual_results = run_residual_on_llm(
            texts=texts,
            x_llm=x,
            y=y,
            groups=groups,
            splits=splits,
            max_features=args.max_features,
            alpha_grid=[0.1, 1.0, 10.0],
            cache_dir=cache_dir,
            profile=args.profile,
            timing=timing,
        )
        interaction_results = run_text_llm_interactions(
            texts=texts,
            scores=x_scores,
            mentions=x_mentions,
            segments=x_segments,
            y=y,
            groups=groups,
            splits=splits,
            max_features=args.max_features,
            alpha_grid=[0.01, 0.1, 1.0, 10.0],
            cache_dir=cache_dir,
            profile=args.profile,
            timing=timing,
        )
        subset_mask = (df["traveler_segment"] != "unknown") & (df["ok"] == 1)
        subset_results = run_subset_eval(
            texts=texts,
            x_llm=x,
            y=y,
            groups=groups,
            mask=subset_mask.to_numpy(),
            splits=splits,
            max_features=args.max_features,
            alpha=0.1,
            cache_dir=cache_dir,
            profile=args.profile,
            timing=timing,
        )

    out_prefix = args.out_prefix or f"results_llm_aspects_{args.profile}"
    result_json = {
        "meta": {
            "profile": args.profile,
            "seed": SEED,
            "n_splits": N_SPLITS,
            "features": {
                "aspect_scores": score_cols,
                "aspect_mentioned": mention_cols,
                "segment_cols": segment_cols,
            },
        },
        "llm_aspects": llm_results,
        "baselines": {
            "text_only": baselines["text_only"],
            "upper_bound": baselines["upper_bound"],
        },
    }
    if text_plus:
        result_json["text_plus_llm"] = text_plus
    if residual_results:
        result_json["residual_on_llm"] = residual_results
    if interaction_results:
        result_json["text_llm_interactions"] = interaction_results
    if subset_results:
        result_json["subset_eval"] = subset_results

    json_path = output_dir / f"{out_prefix}.json"
    start = time.perf_counter()
    json_path.write_text(json.dumps(result_json, indent=2), encoding="utf-8")
    timing.record("write.results_json", time.perf_counter() - start)
    log(f"saved: {json_path}")

    md_path = output_dir / f"{out_prefix}.md"
    start = time.perf_counter()
    write_results_md(
        md_path,
        baselines,
        llm_results,
        text_plus,
        unknown_share,
        residual_results,
        interaction_results,
        subset_results,
    )
    timing.record("write.results_md", time.perf_counter() - start)

    timing_path = output_dir / f"run_profile_{args.profile}_timing.md"
    timing_path.write_text("\n".join(timing.report_lines()), encoding="utf-8")
    log(f"saved: {timing_path}")


if __name__ == "__main__":
    main()
