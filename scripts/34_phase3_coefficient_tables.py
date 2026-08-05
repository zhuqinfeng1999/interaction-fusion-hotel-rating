#!/usr/bin/env python3
"""Phase 3 Experiment X: interpretable ridge coefficient supplements.

The aspects-only table is a conventional low-dimensional ridge fit.  For the
text-plus-interaction model, estimates are taken from the actual joint sparse
ridge fit, while uncertainty is computed for the structured block conditional
on the fitted text component.  The hotel bootstrap is a one-step cluster-pairs
bootstrap based on the cluster score contributions; this avoids pretending that
10,000 full 200k-dimensional TF-IDF refits are computationally feasible.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.aspect_extractor_base import ASPECTS  # noqa: E402
from src.phase2_sparse_ridge import (  # noqa: E402
    CenteredBlockOperator,
    fit_operator,
    predict_blocks,
)


SEED = 42
N_SPLITS = 5
SEGMENTS = ["business", "couple", "family", "friends", "solo", "unknown"]
INTERACTION_MODEL_ID = "phase2_r0_interaction_shared_ridge"


def log(message: str) -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="qwen25_7b")
    parser.add_argument("--max-features", type=int, default=200_000)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--max-iterations", type=int, default=2_000)
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def load_splits(output_dir: Path) -> Dict[int, Dict[str, np.ndarray]]:
    output: Dict[int, Dict[str, np.ndarray]] = {}
    for fold in range(1, N_SPLITS + 1):
        arrays = np.load(output_dir / f"splits_fold{fold}.npz")
        output[fold] = {
            "train_idx": arrays["train_idx"].astype(np.int64),
            "test_idx": arrays["test_idx"].astype(np.int64),
        }
    return output


def tfidf_paths(
    output_dir: Path, profile: str, max_features: int, fold: int
) -> Tuple[Path, Path]:
    pattern = f"{profile}_mf{max_features}_ng12_md3_full_fold{fold}_*_Xtr.npz"
    matches = sorted((output_dir / "tfidf_cache").glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"fold {fold}: expected one TF-IDF cache, found {matches}"
        )
    test = Path(str(matches[0]).replace("_Xtr.npz", "_Xte.npz"))
    if not test.exists():
        raise FileNotFoundError(test)
    return matches[0], test


def feature_names() -> Dict[str, List[str]]:
    z = [f"score_{aspect}" for aspect in ASPECTS]
    z += [f"mentioned_{aspect}" for aspect in ASPECTS]
    z += [f"segment_{segment}" for segment in SEGMENTS]
    score_interactions = [
        f"segment_{segment}:score_{aspect}"
        for segment in SEGMENTS
        for aspect in ASPECTS
    ]
    mention_interactions = [
        f"segment_{segment}:mentioned_{aspect}"
        for segment in SEGMENTS
        for aspect in ASPECTS
    ]
    return {"z": z, "p_s": score_interactions, "p_b": mention_interactions}


def load_raw_blocks(output_dir: Path, profile: str) -> Dict[str, np.ndarray]:
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
    if not np.array_equal(frame["row_id"].to_numpy(), np.arange(len(frame))):
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
    one_hot = (
        pd.get_dummies(segment)
        .reindex(columns=SEGMENTS, fill_value=False)
        .to_numpy(dtype=np.float32)
    )
    return {
        "z": np.hstack([scores, mentions, one_hot]).astype(np.float32),
        "p_s": np.hstack(
            [scores * one_hot[:, j : j + 1] for j in range(len(SEGMENTS))]
        ).astype(np.float32),
        "p_b": np.hstack(
            [mentions * one_hot[:, j : j + 1] for j in range(len(SEGMENTS))]
        ).astype(np.float32),
    }


def prepare_fold_blocks(
    raw: Mapping[str, np.ndarray], train_idx: np.ndarray, test_idx: np.ndarray
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    train: Dict[str, np.ndarray] = {}
    test: Dict[str, np.ndarray] = {}
    for name, values in raw.items():
        scaler = StandardScaler()
        train[name] = scaler.fit_transform(values[train_idx]).astype(np.float32)
        test[name] = scaler.transform(values[test_idx]).astype(np.float32)
    return train, test


def aspects_alpha(output_dir: Path, profile: str, fold: int) -> float:
    path = output_dir / f"results_llm_aspects_{profile}.json"
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("llm_aspects", {}).get("folds", [])
        selected = [row for row in rows if int(row["fold"]) == fold]
        if selected:
            return float(selected[0]["best_alpha"])
    return 0.1


def interaction_alpha(output_dir: Path, fold: int) -> float:
    path = (
        output_dir
        / "predictions"
        / INTERACTION_MODEL_ID
        / f"fold{fold}.meta.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    return float(payload["selected_hyperparameters"]["alpha"])


def conditional_inference(
    structured: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    estimates: np.ndarray,
    alpha: float,
    names: Sequence[str],
    n_resamples: int,
    seed: int,
) -> pd.DataFrame:
    """Sandwich SE and one-step hotel-pairs bootstrap for a small block."""
    z = np.asarray(structured, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    residual = y_true - y_pred
    n, width = z.shape
    if len(names) != width or len(estimates) != width + 1:
        raise ValueError("coefficient/design width mismatch")

    sums = z.sum(axis=0, dtype=np.float64)
    gram = np.empty((width + 1, width + 1), dtype=np.float64)
    gram[0, 0] = n
    gram[0, 1:] = sums
    gram[1:, 0] = sums
    gram[1:, 1:] = z.T @ z
    penalty = np.eye(width + 1, dtype=np.float64) * float(alpha)
    penalty[0, 0] = 0.0
    bread = np.linalg.pinv(gram + penalty)

    unique_groups, codes = np.unique(groups, return_inverse=True)
    scores = np.zeros((len(unique_groups), width + 1), dtype=np.float64)
    np.add.at(scores[:, 0], codes, residual)
    chunk = 100_000
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        contribution = z[start:stop] * residual[start:stop, None]
        np.add.at(scores[:, 1:], codes[start:stop], contribution)
    meat = scores.T @ scores
    covariance = bread @ meat @ bread
    parameters = width + 1
    clusters = len(unique_groups)
    if clusters > 1 and n > parameters:
        covariance *= (clusters / (clusters - 1)) * (
            (n - 1) / (n - parameters)
        )
    standard_error = np.sqrt(np.clip(np.diag(covariance), 0.0, None))

    rng = np.random.default_rng(seed)
    lower_samples = np.full((n_resamples, parameters), np.nan, dtype=np.float64)
    cursor = 0
    probability = np.full(clusters, 1.0 / clusters, dtype=np.float64)
    while cursor < n_resamples:
        batch = min(128, n_resamples - cursor)
        counts = rng.multinomial(clusters, probability, size=batch)
        perturbation = (counts - 1) @ scores
        lower_samples[cursor : cursor + batch] = (
            estimates[None, :] + perturbation @ bread.T
        )
        cursor += batch
    ci_low, ci_high = np.quantile(lower_samples, [0.025, 0.975], axis=0)
    z_value = np.divide(
        estimates,
        standard_error,
        out=np.full_like(estimates, np.nan),
        where=standard_error > 0,
    )
    p_value = 2 * stats.norm.sf(np.abs(z_value))
    return pd.DataFrame(
        {
            "feature": ["intercept", *names],
            "estimate": estimates,
            "cluster_robust_se": standard_error,
            "z_statistic": z_value,
            "p_value_normal": p_value,
            "bootstrap_ci_low": ci_low,
            "bootstrap_ci_high": ci_high,
            "n_clusters": clusters,
            "bootstrap_resamples": n_resamples,
            "bootstrap_seed": seed,
        }
    )


def goodness_of_fit(
    model: str,
    fold: int,
    y: np.ndarray,
    prediction: np.ndarray,
    n_features: int,
    alpha: float,
    seconds: float,
) -> Dict[str, Any]:
    n = len(y)
    residual = y - prediction
    r2 = float(r2_score(y, prediction))
    denominator = n - n_features - 1
    adjusted = (
        float(1 - (1 - r2) * (n - 1) / denominator)
        if denominator > 0
        else float("nan")
    )
    rse = (
        float(math.sqrt(np.dot(residual, residual) / denominator))
        if denominator > 0
        else float("nan")
    )
    return {
        "model": model,
        "fold": fold,
        "n_train": n,
        "n_features_including_text": n_features,
        "alpha": alpha,
        "r_squared": r2,
        "adjusted_r_squared": adjusted,
        "residual_standard_error": rse,
        "fit_seconds": seconds,
    }


def fit_aspects_only(
    fold: int,
    z_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    alpha: float,
    names: Sequence[str],
    n_resamples: int,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    started = time.perf_counter()
    model = Ridge(alpha=alpha)
    model.fit(z_train, y_train)
    prediction = model.predict(z_train)
    elapsed = time.perf_counter() - started
    estimates = np.concatenate([[float(model.intercept_)], model.coef_])
    table = conditional_inference(
        z_train,
        y_train,
        prediction,
        groups_train,
        estimates,
        alpha,
        names,
        n_resamples,
        SEED + fold,
    )
    table.insert(0, "block", ["intercept", *(["z"] * len(names))])
    table.insert(0, "fold", fold)
    table.insert(0, "model", "aspects_only_ridge")
    table["alpha"] = alpha
    table["inference_scope"] = "complete low-dimensional ridge model"
    gof = goodness_of_fit(
        "aspects_only_ridge", fold, y_train, prediction, len(names), alpha, elapsed
    )
    return table, gof


def fit_interaction_block(
    fold: int,
    text_train: sp.csr_matrix,
    blocks_train: Mapping[str, np.ndarray],
    y_train: np.ndarray,
    groups_train: np.ndarray,
    alpha: float,
    names_by_block: Mapping[str, Sequence[str]],
    n_resamples: int,
    tolerance: float,
    max_iterations: int,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    selected = {name: blocks_train[name] for name in ("z", "p_s", "p_b")}
    operator = CenteredBlockOperator(
        text_train, selected, {name: 1.0 for name in selected}
    )
    fit = fit_operator(
        operator,
        y_train,
        alpha,
        tolerance=tolerance,
        max_iterations=max_iterations,
        x0=None,
    )
    prediction = predict_blocks(text_train, selected, fit)
    ordered_blocks = ["z", "p_s", "p_b"]
    structured = np.hstack([selected[name] for name in ordered_blocks])
    names = [item for name in ordered_blocks for item in names_by_block[name]]
    estimates = np.concatenate(
        [[fit.intercept], *[fit.coefficients[name] for name in ordered_blocks]]
    )
    table = conditional_inference(
        structured,
        y_train,
        prediction,
        groups_train,
        estimates,
        alpha,
        names,
        n_resamples,
        SEED + 100 + fold,
    )
    block_column = ["intercept"]
    for name in ordered_blocks:
        block_column.extend([name] * len(names_by_block[name]))
    table.insert(0, "block", block_column)
    table.insert(0, "fold", fold)
    table.insert(0, "model", "interaction_fusion_structured_block")
    table["alpha"] = alpha
    table["inference_scope"] = (
        "joint ridge estimates; uncertainty conditional on fitted TF-IDF component"
    )
    gof = goodness_of_fit(
        "interaction_fusion",
        fold,
        y_train,
        prediction,
        text_train.shape[1] + structured.shape[1],
        alpha,
        fit.fit_seconds,
    )
    gof["lsqr_iterations"] = fit.iterations
    gof["lsqr_stop_code"] = fit.stop_code
    return table, gof


def write_markdown(
    coefficients: pd.DataFrame,
    gof: pd.DataFrame,
    interactions: pd.DataFrame,
    output_path: Path,
) -> None:
    lines = [
        "# S1 Coefficient Tables",
        "",
        "- All structured inputs are standardized within each outer training fold.",
        "- Standard errors use a hotel-clustered sandwich covariance.",
        "- CIs use 10,000 one-step cluster-pairs bootstrap resamples of hotels.",
        "- Interaction-fusion estimates come from the joint TF-IDF + structured ridge model; its structured-block uncertainty is conditional on the fitted text component.",
        "- Penalized-regression z tests and p-values are descriptive inferential diagnostics, not unpenalized OLS tests.",
        "",
        "## Goodness of Fit",
        "",
        gof.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Complete Fold-Level Coefficients",
        "",
    ]
    display = coefficients.copy()
    numeric = [
        "estimate",
        "cluster_robust_se",
        "z_statistic",
        "p_value_normal",
        "bootstrap_ci_low",
        "bootstrap_ci_high",
    ]
    for model in display["model"].unique():
        for fold in sorted(display.loc[display["model"].eq(model), "fold"].unique()):
            subset = display.loc[
                display["model"].eq(model) & display["fold"].eq(fold),
                ["block", "feature", *numeric],
            ]
            lines.extend(
                [
                    f"### {model}, fold {fold}",
                    "",
                    subset.to_markdown(index=False, floatfmt=".6g"),
                    "",
                ]
            )
    lines.extend(
        [
            "## Segment-by-Aspect Interaction Matrix",
            "",
            interactions.to_markdown(index=False, floatfmt=".6g"),
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    np.random.seed(SEED)
    output_dir = ROOT / "outputs"
    phase3 = output_dir / "phase3"
    cache_dir = phase3 / "x_coefficients"
    cache_dir.mkdir(parents=True, exist_ok=True)
    dataset = pd.read_parquet(
        output_dir / "dataset_clean.parquet",
        columns=["rating_overall", "offering_id"],
    )
    if len(dataset) != 877_808:
        raise ValueError(f"canonical row count failed: {len(dataset)}")
    y = dataset["rating_overall"].to_numpy(dtype=np.float64)
    groups = dataset["offering_id"].to_numpy()
    splits = load_splits(output_dir)
    raw = load_raw_blocks(output_dir, args.profile)
    names = feature_names()
    coefficient_parts: List[pd.DataFrame] = []
    gof_rows: List[Dict[str, Any]] = []

    for fold in tqdm(range(1, N_SPLITS + 1), desc="Experiment X folds"):
        train_idx = splits[fold]["train_idx"]
        test_idx = splits[fold]["test_idx"]
        blocks_train, _ = prepare_fold_blocks(raw, train_idx, test_idx)
        y_train = y[train_idx]
        groups_train = groups[train_idx]

        aspect_path = cache_dir / f"aspects_only_fold{fold}.csv"
        aspect_gof_path = cache_dir / f"aspects_only_fold{fold}.gof.json"
        if args.resume and aspect_path.exists() and aspect_gof_path.exists():
            log(f"X aspects-only fold {fold}: cache hit")
            aspect_table = pd.read_csv(aspect_path)
            aspect_gof = json.loads(aspect_gof_path.read_text(encoding="utf-8"))
        else:
            alpha = aspects_alpha(output_dir, args.profile, fold)
            aspect_table, aspect_gof = fit_aspects_only(
                fold,
                blocks_train["z"],
                y_train,
                groups_train,
                alpha,
                names["z"],
                args.bootstrap_resamples,
            )
            aspect_table.to_csv(aspect_path, index=False)
            aspect_gof_path.write_text(
                json.dumps(aspect_gof, indent=2), encoding="utf-8"
            )
        coefficient_parts.append(aspect_table)
        gof_rows.append(aspect_gof)

        interaction_path = cache_dir / f"interaction_fold{fold}.csv"
        interaction_gof_path = cache_dir / f"interaction_fold{fold}.gof.json"
        if args.resume and interaction_path.exists() and interaction_gof_path.exists():
            log(f"X interaction fold {fold}: cache hit")
            interaction_table = pd.read_csv(interaction_path)
            interaction_gof = json.loads(
                interaction_gof_path.read_text(encoding="utf-8")
            )
        else:
            train_path, _ = tfidf_paths(
                output_dir, args.profile, args.max_features, fold
            )
            text_train = sp.load_npz(train_path).tocsr()
            alpha = interaction_alpha(output_dir, fold)
            log(
                f"X interaction fold {fold}: fit n={len(train_idx):,}, "
                f"text={text_train.shape}, alpha={alpha:g}"
            )
            interaction_table, interaction_gof = fit_interaction_block(
                fold,
                text_train,
                blocks_train,
                y_train,
                groups_train,
                alpha,
                names,
                args.bootstrap_resamples,
                args.tolerance,
                args.max_iterations,
            )
            interaction_table.to_csv(interaction_path, index=False)
            interaction_gof_path.write_text(
                json.dumps(interaction_gof, indent=2), encoding="utf-8"
            )
            del text_train
        coefficient_parts.append(interaction_table)
        gof_rows.append(interaction_gof)
        del blocks_train, y_train, groups_train
        gc.collect()

    coefficients = pd.concat(coefficient_parts, ignore_index=True)
    gof = pd.DataFrame(gof_rows)
    interactions = coefficients.loc[
        coefficients["model"].eq("interaction_fusion_structured_block")
        & coefficients["block"].isin(["p_s", "p_b"])
    ].copy()
    split_names = interactions["feature"].str.extract(
        r"segment_([^:]+):(score|mentioned)_(.+)"
    )
    split_names.columns = ["segment", "interaction_type", "aspect"]
    interactions = pd.concat(
        [
            interactions[
                [
                    "fold",
                    "estimate",
                    "cluster_robust_se",
                    "bootstrap_ci_low",
                    "bootstrap_ci_high",
                    "p_value_normal",
                ]
            ].reset_index(drop=True),
            split_names.reset_index(drop=True),
        ],
        axis=1,
    )

    csv_path = phase3 / "S1_coefficient_tables.csv"
    md_path = phase3 / "S1_coefficient_tables.md"
    gof_path = phase3 / "S1_goodness_of_fit.csv"
    interaction_path = phase3 / "S1_segment_aspect_interactions.csv"
    coefficients.to_csv(csv_path, index=False)
    gof.to_csv(gof_path, index=False)
    interactions.to_csv(interaction_path, index=False)
    write_markdown(coefficients, gof, interactions, md_path)
    metadata = {
        "experiment": "X",
        "seed": SEED,
        "bootstrap_resamples": args.bootstrap_resamples,
        "bootstrap_unit": "offering_id (hotel)",
        "bootstrap_method": "one-step cluster-pairs bootstrap from cluster scores",
        "interaction_inference_scope": "conditional on fitted TF-IDF component",
        "standardization": "within each outer training fold",
        "coefficient_rows": len(coefficients),
        "paths": {
            "coefficients": str(csv_path.relative_to(ROOT)),
            "markdown": str(md_path.relative_to(ROOT)),
            "goodness_of_fit": str(gof_path.relative_to(ROOT)),
            "interaction_matrix": str(interaction_path.relative_to(ROOT)),
        },
    }
    (phase3 / "S1_coefficient_tables.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    log(f"saved {csv_path}")
    log(f"saved {md_path}")


if __name__ == "__main__":
    main()
