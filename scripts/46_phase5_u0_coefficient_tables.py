#!/usr/bin/env python3
"""Phase 5 P5.3: complete coefficient supplements under U0.

The aspects-only coefficients reproduce the U0 dense-ridge estimator.  The
interaction structured-block estimates are read from a fresh fit of the exact
joint U0 TF-IDF + z + score-interaction + mention-interaction model.  As in the
accepted Phase 3 supplement, uncertainty for the latter is conditional on the
fitted high-dimensional text component.  Hotel-cluster sandwich SEs and 10,000
one-step cluster-pairs bootstrap resamples are reported for every fold.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.aspect_extractor_base import ASPECTS  # noqa: E402
from src.phase2_ridge import fit_dense_ridge, predict_dense  # noqa: E402
from src.phase2_sparse_ridge import (  # noqa: E402
    CenteredBlockOperator,
    fit_operator,
    predict_blocks,
)


SEED = 42
N_ROWS = 877_808
SEGMENTS = ["business", "couple", "family", "solo", "friends", "unknown"]
ASPECT_MODEL_ID = "phase4_u0_llm_aspects_only"
INTERACTION_MODEL_ID = "phase4_u0_interaction_fusion"


def load_phase3_inference_module() -> Any:
    path = ROOT / "scripts" / "34_phase3_coefficient_tables.py"
    spec = importlib.util.spec_from_file_location("phase3_coefficient_inference", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


INFERENCE = load_phase3_inference_module()


def log(message: str) -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="qwen25_7b")
    parser.add_argument("--max-features", type=int, default=200_000)
    parser.add_argument("--folds", default="1,2,3,4,5")
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--max-iterations", type=int, default=2_000)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def parse_folds(value: str) -> List[int]:
    folds = sorted({int(item) for item in value.split(",") if item.strip()})
    if not folds or any(fold < 1 or fold > 5 for fold in folds):
        raise ValueError("--folds must contain values in 1..5")
    return folds


def load_splits(output_dir: Path) -> Dict[int, Dict[str, np.ndarray]]:
    result = {}
    for fold in range(1, 6):
        arrays = np.load(output_dir / f"splits_fold{fold}.npz")
        result[fold] = {
            "train_idx": arrays["train_idx"].astype(np.int64),
            "test_idx": arrays["test_idx"].astype(np.int64),
        }
    return result


def tfidf_train_path(output_dir: Path, profile: str, max_features: int, fold: int) -> Path:
    matches = sorted(
        (output_dir / "tfidf_cache").glob(
            f"{profile}_mf{max_features}_ng12_md3_full_fold{fold}_*_Xtr.npz"
        )
    )
    if len(matches) != 1:
        raise FileNotFoundError(f"fold {fold}: expected one TF-IDF cache, got {matches}")
    return matches[0]


def feature_names() -> Dict[str, List[str]]:
    z = [f"score_{aspect}" for aspect in ASPECTS]
    z += [f"mentioned_{aspect}" for aspect in ASPECTS]
    z += [f"segment_{segment}" for segment in SEGMENTS]
    p_s = [
        f"segment_{segment}:score_{aspect}"
        for segment in SEGMENTS
        for aspect in ASPECTS
    ]
    p_b = [
        f"segment_{segment}:mentioned_{aspect}"
        for segment in SEGMENTS
        for aspect in ASPECTS
    ]
    return {"z": z, "p_s": p_s, "p_b": p_b}


def load_raw_blocks(output_dir: Path, profile: str) -> Dict[str, np.ndarray]:
    score_cols = [f"aspect_{aspect}_score" for aspect in ASPECTS]
    mention_cols = [f"aspect_{aspect}_mentioned" for aspect in ASPECTS]
    frame = pd.read_parquet(
        output_dir / f"aspects_llm_{profile}.parquet",
        columns=["row_id", "ok", "traveler_segment", *score_cols, *mention_cols],
    ).sort_values("row_id")
    if len(frame) != N_ROWS or not np.array_equal(frame["row_id"], np.arange(N_ROWS)):
        raise ValueError("LLM row_id alignment failed")
    failed = ~frame["ok"].eq(1)
    frame.loc[failed, score_cols] = 0
    frame.loc[failed, mention_cols] = False
    frame.loc[failed, "traveler_segment"] = "unknown"
    scores = frame[score_cols].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(dtype=np.float32)
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
            [scores * one_hot[:, index : index + 1] for index in range(len(SEGMENTS))]
        ).astype(np.float32),
        "p_b": np.hstack(
            [mentions * one_hot[:, index : index + 1] for index in range(len(SEGMENTS))]
        ).astype(np.float32),
    }


def prepare_fold_blocks(
    raw: Mapping[str, np.ndarray], train_idx: np.ndarray, test_idx: np.ndarray
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    train, test = {}, {}
    for name, matrix in raw.items():
        scaler = StandardScaler()
        train[name] = scaler.fit_transform(matrix[train_idx]).astype(np.float32)
        test[name] = scaler.transform(matrix[test_idx]).astype(np.float32)
    return train, test


def stable_conditional_inference(
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
    """Numerically stable hotel-cluster sandwich and one-step pairs bootstrap.

    Computing ``bread @ (scores.T @ scores) @ bread`` can produce small
    negative diagonal values under the intentionally redundant full segment
    one-hot representation.  Forming per-cluster influence vectors first and
    then using ``influence.T @ influence`` is algebraically equivalent, PSD by
    construction, and also avoids the observed 0/0 z statistics.
    """
    z = np.asarray(structured, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    estimates = np.asarray(estimates, dtype=np.float64)
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
    cluster_scores = np.zeros((len(unique_groups), width + 1), dtype=np.float64)
    np.add.at(cluster_scores[:, 0], codes, residual)
    for start in range(0, n, 100_000):
        stop = min(start + 100_000, n)
        contribution = z[start:stop] * residual[start:stop, None]
        np.add.at(cluster_scores[:, 1:], codes[start:stop], contribution)

    # Stable PSD sandwich representation.
    influence = cluster_scores @ bread.T
    covariance = influence.T @ influence
    parameters = width + 1
    clusters = len(unique_groups)
    if clusters > 1 and n > parameters:
        covariance *= (clusters / (clusters - 1)) * (
            (n - 1) / (n - parameters)
        )
    standard_error = np.sqrt(np.maximum(np.diag(covariance), 0.0))

    rng = np.random.default_rng(seed)
    samples = np.empty((n_resamples, parameters), dtype=np.float64)
    cursor = 0
    probability = np.full(clusters, 1.0 / clusters, dtype=np.float64)
    while cursor < n_resamples:
        batch = min(128, n_resamples - cursor)
        counts = rng.multinomial(clusters, probability, size=batch)
        samples[cursor : cursor + batch] = (
            estimates[None, :] + (counts - 1) @ influence
        )
        cursor += batch
    ci_low, ci_high = np.quantile(samples, [0.025, 0.975], axis=0)
    z_value = np.divide(
        estimates,
        standard_error,
        out=np.zeros_like(estimates),
        where=standard_error > 0,
    )
    nonzero_with_zero_se = (standard_error == 0) & (estimates != 0)
    z_value[nonzero_with_zero_se] = np.sign(estimates[nonzero_with_zero_se]) * np.inf
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
            "sandwich_computation": "cluster_influence.T @ cluster_influence (PSD)",
        }
    )


def selected_alpha(output_dir: Path, model_id: str, fold: int) -> float:
    payload = json.loads((output_dir / "metrics" / f"{model_id}.json").read_text())
    rows = [row for row in payload["folds"] if int(row["fold"]) == fold]
    if len(rows) != 1:
        raise ValueError(f"missing/duplicate fold metric: {model_id} fold {fold}")
    return float(rows[0]["selected_hyperparameters"]["alpha"])


def fit_aspects_fold(
    fold: int,
    z_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    alpha: float,
    names: Sequence[str],
    n_resamples: int,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    coefficients, intercept, fit_seconds = fit_dense_ridge(z_train, y_train, alpha)
    prediction, predict_seconds = predict_dense(z_train, coefficients, intercept)
    estimates = np.concatenate([[intercept], coefficients])
    table = stable_conditional_inference(
        z_train,
        y_train,
        prediction,
        groups_train,
        estimates,
        alpha,
        names,
        n_resamples,
        SEED,
    )
    table.insert(0, "block", ["intercept", *(["z"] * len(names))])
    table.insert(0, "fold", fold)
    table.insert(0, "model", ASPECT_MODEL_ID)
    table["alpha"] = alpha
    table["inference_scope"] = "complete U0 low-dimensional dense ridge model"
    gof = INFERENCE.goodness_of_fit(
        ASPECT_MODEL_ID,
        fold,
        y_train,
        prediction,
        len(names),
        alpha,
        fit_seconds,
    )
    gof["predict_seconds"] = predict_seconds
    return table, gof


def fit_interaction_fold(
    fold: int,
    text_train: sp.csr_matrix,
    blocks_train: Mapping[str, np.ndarray],
    y_train: np.ndarray,
    groups_train: np.ndarray,
    alpha: float,
    names: Mapping[str, Sequence[str]],
    n_resamples: int,
    tolerance: float,
    max_iterations: int,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    operator = CenteredBlockOperator(
        text_train, blocks_train, {name: 1.0 for name in blocks_train}
    )
    fit = fit_operator(
        operator,
        y_train,
        alpha,
        tolerance=tolerance,
        max_iterations=max_iterations,
        x0=None,
    )
    prediction = predict_blocks(text_train, blocks_train, fit)
    order = ["z", "p_s", "p_b"]
    structured = np.hstack([blocks_train[name] for name in order])
    all_names = [feature for block in order for feature in names[block]]
    estimates = np.concatenate(
        [[fit.intercept], *[fit.coefficients[name] for name in order]]
    )
    table = stable_conditional_inference(
        structured,
        y_train,
        prediction,
        groups_train,
        estimates,
        alpha,
        all_names,
        n_resamples,
        SEED,
    )
    block_column = ["intercept"]
    for name in order:
        block_column.extend([name] * len(names[name]))
    table.insert(0, "block", block_column)
    table.insert(0, "fold", fold)
    table.insert(0, "model", INTERACTION_MODEL_ID)
    table["alpha"] = alpha
    table["inference_scope"] = (
        "joint U0 ridge estimates; uncertainty conditional on fitted TF-IDF component"
    )
    gof = INFERENCE.goodness_of_fit(
        INTERACTION_MODEL_ID,
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


def sign_pattern_table(coefficients: pd.DataFrame) -> pd.DataFrame:
    target_features = ["score_service", "score_rooms"] + [
        f"mentioned_{aspect}" for aspect in ASPECTS
    ]
    selected = coefficients[
        coefficients["block"].eq("z") & coefficients["feature"].isin(target_features)
    ]
    prior_path = ROOT / "outputs" / "phase3" / "S1_coefficient_tables.csv"
    prior = pd.read_csv(prior_path) if prior_path.exists() else pd.DataFrame()
    prior_model = {
        ASPECT_MODEL_ID: "aspects_only_ridge",
        INTERACTION_MODEL_ID: "interaction_fusion_structured_block",
    }
    rows = []
    for (model, feature), group in selected.groupby(["model", "feature"], sort=False):
        estimate = float(group["estimate"].mean())
        expected = "positive" if feature in {"score_service", "score_rooms"} else "negative"
        observed = "positive" if estimate > 0 else "negative" if estimate < 0 else "zero"
        prior_selected = prior[
            prior.get("model", pd.Series(dtype=str)).eq(prior_model[model])
            & prior.get("block", pd.Series(dtype=str)).eq("z")
            & prior.get("feature", pd.Series(dtype=str)).eq(feature)
        ]
        prior_estimate = (
            float(prior_selected["estimate"].mean())
            if len(prior_selected)
            else float("nan")
        )
        prior_sign = (
            "positive"
            if prior_estimate > 0
            else "negative"
            if prior_estimate < 0
            else "unknown"
        )
        rows.append(
            {
                "model": model,
                "feature": feature,
                "expected_published_sign": expected,
                "u0_fold_mean_estimate": estimate,
                "u0_sign": observed,
                "preserved": observed == expected,
                "phase3_fold_mean_estimate": prior_estimate,
                "phase3_sign": prior_sign,
                "sign_changed_from_phase3": observed != prior_sign,
                "positive_folds": int((group["estimate"] > 0).sum()),
                "negative_folds": int((group["estimate"] < 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def write_markdown(
    coefficients: pd.DataFrame,
    gof: pd.DataFrame,
    interactions: pd.DataFrame,
    sign_pattern: pd.DataFrame,
    path: Path,
) -> None:
    reversals = sign_pattern[~sign_pattern["preserved"]]
    lines = [
        "# S1 Complete U0 Coefficient Tables",
        "",
        "- Models exactly match the fold-specific U0 representations and selected alpha values.",
        "- All structured columns are standardized using outer-training-fold statistics.",
        "- Standard errors use a hotel-clustered sandwich covariance (`offering_id`).",
        "- The sandwich is evaluated in the numerically stable PSD form `cluster_influence.T @ cluster_influence`.",
        "- Bootstrap CIs use 10,000 one-step cluster-pairs hotel resamples with seed 42.",
        "- For interaction fusion, structured-block uncertainty is conditional on the fitted 200k-dimensional TF-IDF component.",
        "- Penalized-model z statistics and p values are descriptive diagnostics.",
        "",
        "## Published Table 3 sign-pattern audit",
        "",
        sign_pattern.to_markdown(index=False, floatfmt=".6g"),
        "",
    ]
    if len(reversals):
        lines.extend(
            [
                "**PROMINENT SIGN REVERSAL/EXCEPTION:** the following U0 fold-mean signs do not match the stated published sign pattern:",
                "",
                reversals[["model", "feature", "expected_published_sign", "u0_fold_mean_estimate", "u0_sign"]].to_markdown(index=False, floatfmt=".6g"),
                "",
            ]
        )
    else:
        lines.extend(["All requested signs are preserved under U0.", ""])
    lines.extend(["## Per-fold goodness of fit", "", gof.to_markdown(index=False, floatfmt=".6g"), "", "## Every fold-level coefficient", ""])
    numeric = [
        "estimate",
        "cluster_robust_se",
        "z_statistic",
        "p_value_normal",
        "bootstrap_ci_low",
        "bootstrap_ci_high",
    ]
    for model in coefficients["model"].unique():
        for fold in sorted(coefficients.loc[coefficients["model"].eq(model), "fold"].unique()):
            subset = coefficients[
                coefficients["model"].eq(model) & coefficients["fold"].eq(fold)
            ][["block", "feature", *numeric]]
            lines.extend([f"### {model}, fold {fold}", "", subset.to_markdown(index=False, floatfmt=".6g"), ""])
    lines.extend(
        [
            "## Segment-by-aspect interaction coefficient matrices",
            "",
            "Cells are `estimate [bootstrap 95% CI]`. The complete numeric long-form matrix is in `S1_segment_aspect_interactions.csv`.",
            "",
        ]
    )
    matrix = interactions.copy()
    matrix["cell"] = matrix.apply(
        lambda row: (
            f"{row['estimate']:.4f} "
            f"[{row['bootstrap_ci_low']:.4f}, {row['bootstrap_ci_high']:.4f}]"
        ),
        axis=1,
    )
    for fold in sorted(matrix["fold"].unique()):
        for interaction_type in ("score", "mentioned"):
            selected = matrix[
                matrix["fold"].eq(fold)
                & matrix["interaction_type"].eq(interaction_type)
            ]
            wide = selected.pivot(index="segment", columns="aspect", values="cell")
            wide = wide.reindex(index=SEGMENTS, columns=list(ASPECTS))
            lines.extend(
                [
                    f"### Fold {fold}: segment x {interaction_type}",
                    "",
                    wide.to_markdown(),
                    "",
                ]
            )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    np.random.seed(SEED)
    output_dir = ROOT / "outputs"
    phase5 = output_dir / "phase5"
    cache = phase5 / "coefficient_cache"
    phase5.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    dataset = pd.read_parquet(
        output_dir / "dataset_clean.parquet", columns=["rating_overall", "offering_id"]
    )
    if len(dataset) != N_ROWS:
        raise ValueError(f"canonical row count mismatch: {len(dataset)}")
    y = dataset["rating_overall"].to_numpy(dtype=np.float64)
    groups = dataset["offering_id"].to_numpy()
    splits = load_splits(output_dir)
    raw = load_raw_blocks(output_dir, args.profile)
    names = feature_names()
    coefficient_parts: List[pd.DataFrame] = []
    gof_rows: List[Dict[str, Any]] = []

    requested_folds = parse_folds(args.folds)
    for fold in tqdm(requested_folds, desc="P5 U0 coefficient folds"):
        train_idx = splits[fold]["train_idx"]
        test_idx = splits[fold]["test_idx"]
        blocks_train, _ = prepare_fold_blocks(raw, train_idx, test_idx)
        y_train = y[train_idx]
        groups_train = groups[train_idx]

        aspect_path = cache / f"u0_aspects_fold{fold}.csv"
        aspect_gof_path = cache / f"u0_aspects_fold{fold}.gof.json"
        if args.resume and aspect_path.exists() and aspect_gof_path.exists():
            aspect_table = pd.read_csv(aspect_path)
            aspect_gof = json.loads(aspect_gof_path.read_text())
            log(f"P5 coefficients aspects fold {fold}: cache hit")
        else:
            alpha = selected_alpha(output_dir, ASPECT_MODEL_ID, fold)
            aspect_table, aspect_gof = fit_aspects_fold(
                fold,
                blocks_train["z"],
                y_train,
                groups_train,
                alpha,
                names["z"],
                args.bootstrap_resamples,
            )
            aspect_table.to_csv(aspect_path, index=False)
            aspect_gof_path.write_text(json.dumps(aspect_gof, indent=2))
            log(f"P5 coefficients aspects fold {fold}: alpha={alpha:g} complete")
        coefficient_parts.append(aspect_table)
        gof_rows.append(aspect_gof)

        interaction_path = cache / f"u0_interaction_fold{fold}.csv"
        interaction_gof_path = cache / f"u0_interaction_fold{fold}.gof.json"
        if args.resume and interaction_path.exists() and interaction_gof_path.exists():
            interaction_table = pd.read_csv(interaction_path)
            interaction_gof = json.loads(interaction_gof_path.read_text())
            log(f"P5 coefficients interaction fold {fold}: cache hit")
        else:
            alpha = selected_alpha(output_dir, INTERACTION_MODEL_ID, fold)
            text_train = sp.load_npz(
                tfidf_train_path(output_dir, args.profile, args.max_features, fold)
            ).tocsr()
            started = time.perf_counter()
            interaction_table, interaction_gof = fit_interaction_fold(
                fold,
                text_train,
                {name: blocks_train[name] for name in ("z", "p_s", "p_b")},
                y_train,
                groups_train,
                alpha,
                names,
                args.bootstrap_resamples,
                args.tolerance,
                args.max_iterations,
            )
            interaction_table.to_csv(interaction_path, index=False)
            interaction_gof_path.write_text(json.dumps(interaction_gof, indent=2))
            log(
                f"P5 coefficients interaction fold {fold}: alpha={alpha:g}, "
                f"elapsed={time.perf_counter() - started:.1f}s complete"
            )
            del text_train
        coefficient_parts.append(interaction_table)
        gof_rows.append(interaction_gof)
        del blocks_train, y_train, groups_train
        gc.collect()

    if set(requested_folds) != set(range(1, 6)):
        log(
            f"P5.3 partial cache run complete for folds={requested_folds}; "
            "final tables deferred until an all-fold --resume run"
        )
        return

    coefficients = pd.concat(coefficient_parts, ignore_index=True)
    coefficients["p_value"] = coefficients["p_value_normal"]
    gof = pd.DataFrame(gof_rows)
    interactions = coefficients[
        coefficients["model"].eq(INTERACTION_MODEL_ID)
        & coefficients["block"].isin(["p_s", "p_b"])
    ].copy()
    parsed = interactions["feature"].str.extract(r"segment_([^:]+):(score|mentioned)_(.+)")
    parsed.columns = ["segment", "interaction_type", "aspect"]
    interactions = pd.concat(
        [
            interactions[
                [
                    "fold",
                    "estimate",
                    "cluster_robust_se",
                    "bootstrap_ci_low",
                    "bootstrap_ci_high",
                    "z_statistic",
                    "p_value",
                    "p_value_normal",
                ]
            ].reset_index(drop=True),
            parsed.reset_index(drop=True),
        ],
        axis=1,
    )
    signs = sign_pattern_table(coefficients)
    coefficients.to_csv(phase5 / "S1_coefficient_tables.csv", index=False)
    gof.to_csv(phase5 / "S1_goodness_of_fit.csv", index=False)
    interactions.to_csv(phase5 / "S1_segment_aspect_interactions.csv", index=False)
    signs.to_csv(phase5 / "S1_sign_pattern_audit.csv", index=False)
    write_markdown(
        coefficients,
        gof,
        interactions,
        signs,
        phase5 / "S1_coefficient_tables.md",
    )
    metadata = {
        "protocol": "Phase 4 U0",
        "models": [ASPECT_MODEL_ID, INTERACTION_MODEL_ID],
        "bootstrap_resamples": args.bootstrap_resamples,
        "bootstrap_seed": SEED,
        "bootstrap_unit": "offering_id",
        "bootstrap_method": "one-step cluster-pairs bootstrap from hotel score contributions",
        "sandwich_computation": "cluster_influence.T @ cluster_influence (PSD)",
        "coefficient_rows": len(coefficients),
        "interaction_rows": len(interactions),
        "goodness_of_fit": gof.to_dict(orient="records"),
        "sign_pattern": signs.to_dict(orient="records"),
    }
    (phase5 / "S1_coefficient_tables.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    log(f"P5.3 finalized: coefficients={len(coefficients)}, interactions={len(interactions)}")


if __name__ == "__main__":
    main()
