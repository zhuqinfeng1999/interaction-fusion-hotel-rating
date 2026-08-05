#!/usr/bin/env python3
"""Phase 3 Experiment G: alpha-response robustness and sample-size scaling."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import scipy.sparse as sp
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold
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
MODELS = ["text_only", "simple_fusion", "interaction_fusion"]
ALPHAS = np.logspace(-4, 6, 41, dtype=np.float64)
R0_MODEL_IDS = {
    "text_only": "phase2_r0_text_shared_ridge",
    "simple_fusion": "phase2_r0_simple_shared_ridge",
    "interaction_fusion": "phase2_r0_interaction_shared_ridge",
}


def log(message: str) -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=["benchmark", "g1", "g3", "g4", "finalize", "all"],
        default="all",
    )
    parser.add_argument("--profile", default="qwen25_7b")
    parser.add_argument("--folds", default="1,2,3,4,5")
    parser.add_argument("--max-features", type=int, default=200_000)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--max-iterations", type=int, default=2_000)
    parser.add_argument("--model-jobs", type=int, default=3)
    parser.add_argument("--inner-jobs", type=int, default=3)
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def parse_folds(raw: str) -> List[int]:
    folds = sorted({int(item.strip()) for item in raw.split(",") if item.strip()})
    if not folds or any(fold < 1 or fold > N_SPLITS for fold in folds):
        raise ValueError("--folds must contain values in 1..5")
    return folds


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_splits(output_dir: Path) -> Dict[int, Dict[str, Any]]:
    splits: Dict[int, Dict[str, Any]] = {}
    for fold in range(1, N_SPLITS + 1):
        path = output_dir / f"splits_fold{fold}.npz"
        arrays = np.load(path)
        splits[fold] = {
            "fold": fold,
            "train_idx": arrays["train_idx"].astype(np.int64),
            "test_idx": arrays["test_idx"].astype(np.int64),
            "sha256": sha256_file(path),
        }
    return splits


def tfidf_paths(
    output_dir: Path, profile: str, max_features: int, fold: int
) -> Tuple[Path, Path]:
    pattern = f"{profile}_mf{max_features}_ng12_md3_full_fold{fold}_*_Xtr.npz"
    matches = sorted((output_dir / "tfidf_cache").glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"fold {fold}: expected one TF-IDF training cache, got {matches}"
        )
    test_path = Path(str(matches[0]).replace("_Xtr.npz", "_Xte.npz"))
    if not test_path.exists():
        raise FileNotFoundError(test_path)
    return matches[0], test_path


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
        "p_s": p_s,
        "p_b": p_b,
    }


def block_names(model: str) -> List[str]:
    if model == "text_only":
        return []
    if model == "simple_fusion":
        return ["z"]
    if model == "interaction_fusion":
        return ["z", "p_s", "p_b"]
    raise ValueError(model)


def prepare_blocks(
    raw: Mapping[str, np.ndarray],
    fit_global_idx: np.ndarray,
    eval_global_idx: np.ndarray,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    fit: Dict[str, np.ndarray] = {}
    evaluate: Dict[str, np.ndarray] = {}
    for name, matrix in raw.items():
        scaler = StandardScaler()
        fit[name] = scaler.fit_transform(matrix[fit_global_idx]).astype(np.float32)
        evaluate[name] = scaler.transform(matrix[eval_global_idx]).astype(np.float32)
    return fit, evaluate


def model_blocks(
    model: str, blocks: Mapping[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    return {name: blocks[name] for name in block_names(model)}


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False)
    temp.replace(path)


def metrics(y: np.ndarray, prediction: np.ndarray) -> Tuple[float, float]:
    clipped = np.clip(prediction, 1.0, 5.0)
    return (
        float(mean_absolute_error(y, clipped)),
        float(math.sqrt(mean_squared_error(y, clipped))),
    )


def curve_complete(path: Path, alphas: Sequence[float] = ALPHAS) -> bool:
    if not path.exists():
        return False
    frame = pd.read_csv(path)
    if "alpha" not in frame:
        return False
    found = frame["alpha"].to_numpy(dtype=float)
    return all(np.any(np.isclose(found, alpha, rtol=1e-10)) for alpha in alphas)


def existing_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return pd.read_csv(path).to_dict(orient="records")


def run_outer_curve(
    model: str,
    fold: int,
    text_train: sp.csr_matrix,
    text_test: sp.csr_matrix,
    blocks_train_all: Mapping[str, np.ndarray],
    blocks_test_all: Mapping[str, np.ndarray],
    y_train: np.ndarray,
    y_test: np.ndarray,
    path: Path,
    tolerance: float,
    max_iterations: int,
    alphas: Sequence[float] = ALPHAS,
    extra: Optional[Mapping[str, Any]] = None,
    resume: bool = True,
) -> pd.DataFrame:
    if resume and curve_complete(path, alphas):
        log(f"G curve cache hit: {model} fold={fold} path={path.name}")
        return pd.read_csv(path)
    records = existing_records(path) if resume else []
    completed = np.asarray([row["alpha"] for row in records], dtype=float)
    blocks_train = model_blocks(model, blocks_train_all)
    blocks_test = model_blocks(model, blocks_test_all)
    lambdas = {name: 1.0 for name in blocks_train}
    operator = CenteredBlockOperator(text_train, blocks_train, lambdas)
    rss_mb = psutil.Process().memory_info().rss / (1024 * 1024)
    log(
        f"G curve start: {model} fold={fold} train={text_train.shape} "
        f"test={text_test.shape} blocks={list(blocks_train)} RSS={rss_mb:.0f}MB"
    )
    progress = tqdm(
        list(np.asarray(alphas, dtype=float)[::-1]),
        desc=f"G {model} f{fold}",
        leave=True,
    )
    for alpha in progress:
        if completed.size and np.any(np.isclose(completed, alpha, rtol=1e-10)):
            progress.set_postfix(cache="hit")
            continue
        fit = fit_operator(
            operator,
            y_train,
            float(alpha),
            tolerance=tolerance,
            max_iterations=max_iterations,
            # SciPy LSQR's damped problem must be solved independently at
            # every alpha. Passing the prior alpha's solution through ``x0``
            # changes the effective damped target and produced non-reproducible
            # oracle curves in a preflight check.
            x0=None,
        )
        train_prediction = operator.uncentered_prediction(fit.theta, float(y_train.mean()))
        test_prediction = predict_blocks(text_test, blocks_test, fit)
        train_mae, train_rmse = metrics(y_train, train_prediction)
        test_mae, test_rmse = metrics(y_test, test_prediction)
        raw_test_mae = float(mean_absolute_error(y_test, test_prediction))
        row: Dict[str, Any] = {
            "model": model,
            "fold": int(fold),
            "alpha": float(alpha),
            "test_mae": test_mae,
            "test_rmse": test_rmse,
            "train_mae": train_mae,
            "train_rmse": train_rmse,
            "test_mae_raw": raw_test_mae,
            "iterations": int(fit.iterations),
            "stop_code": int(fit.stop_code),
            "fit_seconds": float(fit.fit_seconds),
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "tolerance": float(tolerance),
            "max_iterations": int(max_iterations),
        }
        if extra:
            row.update(extra)
        records.append(row)
        atomic_csv(pd.DataFrame(records).sort_values("alpha"), path)
        progress.set_postfix(
            alpha=f"{alpha:.2g}", mae=f"{test_mae:.4f}", it=fit.iterations
        )
    frame = pd.DataFrame(records).sort_values("alpha").reset_index(drop=True)
    atomic_csv(frame, path)
    return frame


def load_core(
    output_dir: Path, profile: str
) -> Tuple[pd.DataFrame, Dict[int, Dict[str, Any]], Dict[str, np.ndarray]]:
    dataset = pd.read_parquet(
        output_dir / "dataset_clean.parquet",
        columns=["rating_overall", "offering_id"],
    )
    if len(dataset) != 877_808:
        raise ValueError(f"canonical row count failed: {len(dataset)}")
    splits = load_splits(output_dir)
    raw = load_raw_blocks(output_dir, profile)
    if len(raw["z"]) != len(dataset):
        raise ValueError("structured feature length mismatch")
    return dataset, splits, raw


def run_g1(args: argparse.Namespace, folds: Sequence[int]) -> None:
    output_dir = ROOT / "outputs"
    phase3 = output_dir / "phase3"
    curve_dir = phase3 / "g1_curves"
    dataset, splits, raw = load_core(output_dir, args.profile)
    y = dataset["rating_overall"].to_numpy(dtype=np.float64)
    for fold in folds:
        split = splits[fold]
        train_idx, test_idx = split["train_idx"], split["test_idx"]
        train_path, test_path = tfidf_paths(
            output_dir, args.profile, args.max_features, fold
        )
        log(f"G1 fold {fold}: loading TF-IDF")
        text_train = sp.load_npz(train_path).tocsr()
        text_test = sp.load_npz(test_path).tocsr()
        blocks_train, blocks_test = prepare_blocks(raw, train_idx, test_idx)
        kwargs = dict(
            fold=fold,
            text_train=text_train,
            text_test=text_test,
            blocks_train_all=blocks_train,
            blocks_test_all=blocks_test,
            y_train=y[train_idx],
            y_test=y[test_idx],
            tolerance=args.tolerance,
            max_iterations=args.max_iterations,
            resume=args.resume,
            extra={
                "experiment": "G1_outer_test_oracle_curve",
                "oracle_diagnostic_not_model_selection": True,
                "split_sha256": split["sha256"],
            },
        )
        worker_count = max(1, min(args.model_jobs, len(MODELS)))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(
                    run_outer_curve,
                    model=model,
                    path=curve_dir / f"{model}_fold{fold}.csv",
                    **kwargs,
                ): model
                for model in MODELS
            }
            for future in as_completed(futures):
                model = futures[future]
                frame = future.result()
                log(f"G1 completed: {model} fold={fold} rows={len(frame)}")
        del text_train, text_test, blocks_train, blocks_test
        gc.collect()
    combine_g1(phase3)


def combine_g1(phase3: Path) -> pd.DataFrame:
    paths = [
        phase3 / "g1_curves" / f"{model}_fold{fold}.csv"
        for model in MODELS
        for fold in range(1, N_SPLITS + 1)
    ]
    missing = [str(path) for path in paths if not curve_complete(path)]
    if missing:
        log(f"G1 combine deferred; incomplete curves={len(missing)}")
        return pd.DataFrame()
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    required = ["model", "fold", "alpha", "test_mae", "test_rmse", "train_mae"]
    ordered = required + [column for column in frame if column not in required]
    frame = frame[ordered].sort_values(["model", "fold", "alpha"])
    atomic_csv(frame, phase3 / "alpha_response_curves.csv")
    (phase3 / "alpha_response_curves.json").write_text(
        json.dumps(frame.to_dict(orient="records"), indent=2), encoding="utf-8"
    )
    log(f"saved G1 combined curves: {phase3 / 'alpha_response_curves.csv'}")
    return frame


def run_inner_fold_curve(
    model: str,
    outer_fold: int,
    inner_fold: int,
    text_outer: sp.csr_matrix,
    blocks_outer_all: Mapping[str, np.ndarray],
    y_outer: np.ndarray,
    fit_pos: np.ndarray,
    valid_pos: np.ndarray,
    path: Path,
    tolerance: float,
    max_iterations: int,
    resume: bool,
) -> pd.DataFrame:
    if resume and curve_complete(path):
        log(f"G3 cache hit: {model} outer={outer_fold} inner={inner_fold}")
        return pd.read_csv(path)
    records = existing_records(path) if resume else []
    completed = np.asarray([row["alpha"] for row in records], dtype=float)
    text_fit = text_outer[fit_pos].tocsr()
    text_valid = text_outer[valid_pos].tocsr()
    selected_blocks = model_blocks(model, blocks_outer_all)
    blocks_fit = {name: matrix[fit_pos] for name, matrix in selected_blocks.items()}
    blocks_valid = {name: matrix[valid_pos] for name, matrix in selected_blocks.items()}
    operator = CenteredBlockOperator(
        text_fit, blocks_fit, {name: 1.0 for name in blocks_fit}
    )
    for alpha in ALPHAS[::-1]:
        if completed.size and np.any(np.isclose(completed, alpha, rtol=1e-10)):
            continue
        fit = fit_operator(
            operator,
            y_outer[fit_pos],
            float(alpha),
            tolerance=tolerance,
            max_iterations=max_iterations,
            x0=None,
        )
        prediction = predict_blocks(text_valid, blocks_valid, fit)
        mae_raw = float(mean_absolute_error(y_outer[valid_pos], prediction))
        mae_clipped, rmse_clipped = metrics(y_outer[valid_pos], prediction)
        records.append(
            {
                "model": model,
                "outer_fold": outer_fold,
                "inner_fold": inner_fold,
                "inner_cv_splits": 5,
                "alpha": float(alpha),
                "inner_mae": mae_raw,
                "inner_mae_clipped": mae_clipped,
                "inner_rmse_clipped": rmse_clipped,
                "n_fit": int(len(fit_pos)),
                "n_valid": int(len(valid_pos)),
                "iterations": int(fit.iterations),
                "stop_code": int(fit.stop_code),
                "fit_seconds": float(fit.fit_seconds),
            }
        )
        atomic_csv(pd.DataFrame(records).sort_values("alpha"), path)
        log(
            f"G3 {model} outer={outer_fold} inner={inner_fold} "
            f"alpha={alpha:g} MAE={mae_raw:.6f} iter={fit.iterations}"
        )
    return pd.DataFrame(records).sort_values("alpha")


def run_g3(args: argparse.Namespace, folds: Sequence[int]) -> None:
    output_dir = ROOT / "outputs"
    phase3 = output_dir / "phase3"
    detail_dir = phase3 / "g3_inner_cv5"
    dataset, splits, raw = load_core(output_dir, args.profile)
    y = dataset["rating_overall"].to_numpy(dtype=np.float64)
    groups = dataset["offering_id"].to_numpy()
    for outer_fold in folds:
        split = splits[outer_fold]
        train_idx, test_idx = split["train_idx"], split["test_idx"]
        train_path, _ = tfidf_paths(
            output_dir, args.profile, args.max_features, outer_fold
        )
        text_outer = sp.load_npz(train_path).tocsr()
        blocks_outer, _ = prepare_blocks(raw, train_idx, test_idx)
        y_outer = y[train_idx]
        groups_outer = groups[train_idx]
        inner_splits = list(
            GroupKFold(n_splits=5).split(text_outer, y_outer, groups=groups_outer)
        )
        for model in MODELS:
            combined_path = detail_dir / f"{model}_outerfold{outer_fold}.csv"
            if args.resume and combined_path.exists():
                combined = pd.read_csv(combined_path)
                if len(combined) == len(ALPHAS):
                    log(f"G3 combined cache hit: {model} outer={outer_fold}")
                    continue
            workers = max(1, min(args.inner_jobs, len(inner_splits)))
            frames: List[pd.DataFrame] = []
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(
                        run_inner_fold_curve,
                        model=model,
                        outer_fold=outer_fold,
                        inner_fold=inner_fold,
                        text_outer=text_outer,
                        blocks_outer_all=blocks_outer,
                        y_outer=y_outer,
                        fit_pos=fit_pos,
                        valid_pos=valid_pos,
                        path=detail_dir
                        / f"{model}_outerfold{outer_fold}_innerfold{inner_fold}.csv",
                        tolerance=args.tolerance,
                        max_iterations=args.max_iterations,
                        resume=args.resume,
                    )
                    for inner_fold, (fit_pos, valid_pos) in enumerate(
                        inner_splits, start=1
                    )
                ]
                for future in as_completed(futures):
                    frames.append(future.result())
            details = pd.concat(frames, ignore_index=True)
            aggregate = (
                details.groupby("alpha", as_index=False)
                .agg(
                    mean_inner_mae=("inner_mae", "mean"),
                    sd_inner_mae=("inner_mae", "std"),
                    mean_inner_mae_clipped=("inner_mae_clipped", "mean"),
                    fit_seconds_total=("fit_seconds", "sum"),
                    mean_iterations=("iterations", "mean"),
                    max_stop_code=("stop_code", "max"),
                )
                .sort_values("alpha")
            )
            aggregate.insert(0, "outer_fold", outer_fold)
            aggregate.insert(0, "model", model)
            best_alpha = float(
                aggregate.sort_values(["mean_inner_mae", "alpha"]).iloc[0]["alpha"]
            )
            aggregate["selected"] = np.isclose(aggregate["alpha"], best_alpha)
            atomic_csv(aggregate, combined_path)
            log(
                f"G3 completed: {model} outer={outer_fold} "
                f"GroupKFold(5) alpha={best_alpha:g}"
            )
        del text_outer, blocks_outer
        gc.collect()
    finalize_g3_if_complete(phase3)


def phase2_cv3_alphas(output_dir: Path) -> Dict[Tuple[str, int], float]:
    result: Dict[Tuple[str, int], float] = {}
    for model, model_id in R0_MODEL_IDS.items():
        payload = json.loads(
            (output_dir / "metrics" / f"{model_id}.json").read_text(encoding="utf-8")
        )
        for row in payload["folds"]:
            result[(model, int(row["fold"]))] = float(
                row["selected_hyperparameters"]["alpha"]
            )
    return result


def oracle_and_plateau(curve: pd.DataFrame) -> Dict[str, float]:
    ordered = curve.sort_values("alpha").reset_index(drop=True)
    oracle_index = int(ordered["test_mae"].to_numpy().argmin())
    oracle = ordered.iloc[oracle_index]
    threshold = float(oracle["test_mae"] + 0.002)
    within = ordered["test_mae"].to_numpy() <= threshold + 1e-12
    left = oracle_index
    right = oracle_index
    while left > 0 and within[left - 1]:
        left -= 1
    while right + 1 < len(ordered) and within[right + 1]:
        right += 1
    low = float(ordered.iloc[left]["alpha"])
    high = float(ordered.iloc[right]["alpha"])
    return {
        "oracle_alpha": float(oracle["alpha"]),
        "oracle_mae": float(oracle["test_mae"]),
        "oracle_rmse": float(oracle["test_rmse"]),
        "plateau_alpha_low": low,
        "plateau_alpha_high": high,
        "plateau_width_decades": float(math.log10(high) - math.log10(low)),
    }


def lookup_curve_mae(curve: pd.DataFrame, alpha: float) -> float:
    position = np.flatnonzero(np.isclose(curve["alpha"].to_numpy(), alpha, rtol=1e-9))
    if not len(position):
        raise ValueError(f"alpha={alpha:g} absent from G1 curve")
    return float(curve.iloc[int(position[0])]["test_mae"])


def finalize_g3_if_complete(phase3: Path) -> pd.DataFrame:
    curves = combine_g1(phase3)
    if curves.empty:
        return pd.DataFrame()
    detail_dir = phase3 / "g3_inner_cv5"
    paths = [
        detail_dir / f"{model}_outerfold{fold}.csv"
        for model in MODELS
        for fold in range(1, N_SPLITS + 1)
    ]
    if any(not path.exists() or len(pd.read_csv(path)) != len(ALPHAS) for path in paths):
        log("G3 finalize deferred; CV5 curves incomplete")
        return pd.DataFrame()
    cv3 = phase2_cv3_alphas(ROOT / "outputs")
    rows = []
    for model in MODELS:
        for fold in range(1, N_SPLITS + 1):
            outer = curves[(curves["model"] == model) & (curves["fold"] == fold)]
            oracle = oracle_and_plateau(outer)
            cv5_curve = pd.read_csv(detail_dir / f"{model}_outerfold{fold}.csv")
            cv5_alpha = float(
                cv5_curve.sort_values(["mean_inner_mae", "alpha"]).iloc[0]["alpha"]
            )
            cv3_alpha = cv3[(model, fold)]
            cv3_mae = lookup_curve_mae(outer, cv3_alpha)
            cv5_mae = lookup_curve_mae(outer, cv5_alpha)
            rows.append(
                {
                    "model": model,
                    "fold": fold,
                    **oracle,
                    "inner_cv3_selected_alpha": cv3_alpha,
                    "inner_cv3_outer_mae": cv3_mae,
                    "inner_cv3_regret": cv3_mae - oracle["oracle_mae"],
                    "inner_cv3_log10_distance_to_oracle": abs(
                        math.log10(cv3_alpha) - math.log10(oracle["oracle_alpha"])
                    ),
                    "inner_cv5_selected_alpha": cv5_alpha,
                    "inner_cv5_outer_mae": cv5_mae,
                    "inner_cv5_regret": cv5_mae - oracle["oracle_mae"],
                    "inner_cv5_log10_distance_to_oracle": abs(
                        math.log10(cv5_alpha) - math.log10(oracle["oracle_alpha"])
                    ),
                    "cv5_minus_cv3_outer_mae": cv5_mae - cv3_mae,
                }
            )
    frame = pd.DataFrame(rows)
    atomic_csv(frame, phase3 / "g2_g3_oracle_plateau_selection.csv")
    (phase3 / "g2_g3_oracle_plateau_selection.json").write_text(
        json.dumps(frame.to_dict(orient="records"), indent=2), encoding="utf-8"
    )
    return frame


def hotel_sample_positions(
    groups: np.ndarray, target_rows: int, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    unique, counts = np.unique(groups, return_counts=True)
    order = np.random.default_rng(seed).permutation(len(unique))
    selected: List[Any] = []
    rows = 0
    for position in order:
        selected.append(unique[position])
        rows += int(counts[position])
        if rows >= min(target_rows, len(groups)):
            break
    selected_array = np.asarray(selected)
    positions = np.flatnonzero(np.isin(groups, selected_array)).astype(np.int64)
    return positions, selected_array


def run_g4(args: argparse.Namespace) -> None:
    output_dir = ROOT / "outputs"
    phase3 = output_dir / "phase3"
    curve_dir = phase3 / "g4_sample_size_curves"
    sample_dir = phase3 / "g4_samples"
    dataset, splits, raw = load_core(output_dir, args.profile)
    y = dataset["rating_overall"].to_numpy(dtype=np.float64)
    groups = dataset["offering_id"].to_numpy()
    split = splits[1]
    train_idx, test_idx = split["train_idx"], split["test_idx"]
    train_path, test_path = tfidf_paths(output_dir, args.profile, args.max_features, 1)
    text_outer = sp.load_npz(train_path).tocsr()
    text_test = sp.load_npz(test_path).tocsr()
    targets = [100_000, 200_000, 400_000, 700_000]
    for target in targets:
        sample_path = sample_dir / f"fold1_target{target}_seed{SEED}.npz"
        if args.resume and sample_path.exists():
            cached = np.load(sample_path)
            positions = cached["train_positions"].astype(np.int64)
            selected_hotels = cached["selected_hotels"]
        else:
            positions, selected_hotels = hotel_sample_positions(
                groups[train_idx], target, SEED
            )
            sample_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                sample_path,
                train_positions=positions,
                selected_hotels=selected_hotels,
                target_rows=np.asarray([target]),
                actual_rows=np.asarray([len(positions)]),
                seed=np.asarray([SEED]),
            )
        sample_global = train_idx[positions]
        blocks_sample, blocks_test = prepare_blocks(raw, sample_global, test_idx)
        text_sample = text_outer[positions].tocsr()
        kwargs = dict(
            fold=1,
            text_train=text_sample,
            text_test=text_test,
            blocks_train_all=blocks_sample,
            blocks_test_all=blocks_test,
            y_train=y[sample_global],
            y_test=y[test_idx],
            tolerance=args.tolerance,
            max_iterations=args.max_iterations,
            resume=args.resume,
            extra={
                "experiment": "G4_sample_size_outer_oracle",
                "target_train_rows": target,
                "actual_train_rows": int(len(positions)),
                "sample_hotels": int(len(selected_hotels)),
                "sample_seed": SEED,
                "tfidf_note": "fold1 outer-training TF-IDF representation held fixed",
            },
        )
        workers = max(1, min(args.model_jobs, len(MODELS)))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    run_outer_curve,
                    model=model,
                    path=curve_dir / f"{model}_target{target}.csv",
                    **kwargs,
                ): model
                for model in MODELS
            }
            for future in as_completed(futures):
                frame = future.result()
                log(
                    f"G4 completed: model={futures[future]} target={target} "
                    f"rows={len(frame)}"
                )
        del text_sample, blocks_sample, blocks_test
        gc.collect()
    finalize_g4_if_complete(phase3)


def finalize_g4_if_complete(phase3: Path) -> pd.DataFrame:
    targets = [100_000, 200_000, 400_000, 700_000]
    paths = [
        phase3 / "g4_sample_size_curves" / f"{model}_target{target}.csv"
        for model in MODELS
        for target in targets
    ]
    if any(not curve_complete(path) for path in paths):
        log("G4 finalize deferred; curves incomplete")
        return pd.DataFrame()
    rows = []
    all_curves = []
    for model in MODELS:
        for target in targets:
            curve = pd.read_csv(
                phase3 / "g4_sample_size_curves" / f"{model}_target{target}.csv"
            )
            all_curves.append(curve)
            oracle = oracle_and_plateau(curve)
            rows.append(
                {
                    "model": model,
                    "target_train_rows": target,
                    "actual_train_rows": int(curve.iloc[0]["actual_train_rows"]),
                    "sample_hotels": int(curve.iloc[0]["sample_hotels"]),
                    **oracle,
                }
            )
    result = pd.DataFrame(rows)
    atomic_csv(result, phase3 / "g4_sample_size_scaling.csv")
    atomic_csv(
        pd.concat(all_curves, ignore_index=True),
        phase3 / "g4_sample_size_alpha_curves.csv",
    )
    (phase3 / "g4_sample_size_scaling.json").write_text(
        json.dumps(result.to_dict(orient="records"), indent=2), encoding="utf-8"
    )
    return result


def format_mean_sd(values: pd.Series) -> str:
    return f"{values.mean():.6f} +/- {values.std(ddof=1):.6f}"


def plot_g1(curves: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    for axis, model in zip(axes, MODELS):
        part = curves[curves["model"] == model]
        for fold, fold_frame in part.groupby("fold"):
            axis.plot(
                fold_frame["alpha"],
                fold_frame["test_mae"],
                alpha=0.35,
                linewidth=1,
                label=f"fold {fold}",
            )
        mean_curve = part.groupby("alpha", as_index=False)["test_mae"].mean()
        axis.plot(
            mean_curve["alpha"],
            mean_curve["test_mae"],
            color="black",
            linewidth=2.2,
            label="fold mean",
        )
        axis.set_xscale("log")
        axis.set_title(model.replace("_", " "))
        axis.set_xlabel("Ridge alpha")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Outer-test MAE (clipped)")
    axes[-1].legend(fontsize=8)
    fig.suptitle("Phase 3 G1: outer-test oracle alpha-response curves")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def finalize_g(args: argparse.Namespace) -> None:
    output_dir = ROOT / "outputs"
    phase3 = output_dir / "phase3"
    curves = combine_g1(phase3)
    g23 = finalize_g3_if_complete(phase3)
    g4 = finalize_g4_if_complete(phase3)
    if curves.empty or g23.empty or g4.empty:
        raise RuntimeError("G1/G3/G4 must all be complete before --stage finalize")

    summary_rows = []
    for model, part in g23.groupby("model", sort=False):
        summary_rows.append(
            {
                "model": model,
                "oracle_mae_mean": float(part["oracle_mae"].mean()),
                "oracle_mae_sd": float(part["oracle_mae"].std(ddof=1)),
                "selected_cv3_mae_mean": float(part["inner_cv3_outer_mae"].mean()),
                "selected_cv3_mae_sd": float(part["inner_cv3_outer_mae"].std(ddof=1)),
                "regret_cv3_mean": float(part["inner_cv3_regret"].mean()),
                "regret_cv3_sd": float(part["inner_cv3_regret"].std(ddof=1)),
                "plateau_width_decades_mean": float(
                    part["plateau_width_decades"].mean()
                ),
                "plateau_width_decades_sd": float(
                    part["plateau_width_decades"].std(ddof=1)
                ),
                "selected_cv3_alpha_geomean": float(
                    10 ** np.log10(part["inner_cv3_selected_alpha"]).mean()
                ),
                "selected_cv3_alpha_log10_sd": float(
                    np.log10(part["inner_cv3_selected_alpha"]).std(ddof=1)
                ),
                "oracle_alpha_geomean": float(
                    10 ** np.log10(part["oracle_alpha"]).mean()
                ),
                "oracle_alpha_log10_sd": float(
                    np.log10(part["oracle_alpha"]).std(ddof=1)
                ),
                "selected_cv5_mae_mean": float(part["inner_cv5_outer_mae"].mean()),
                "regret_cv5_mean": float(part["inner_cv5_regret"].mean()),
                "selected_cv5_alpha_geomean": float(
                    10 ** np.log10(part["inner_cv5_selected_alpha"]).mean()
                ),
            }
        )
    summary = pd.DataFrame(summary_rows)
    atomic_csv(summary, phase3 / "g5_summary.csv")
    (phase3 / "g5_summary.json").write_text(
        json.dumps(summary.to_dict(orient="records"), indent=2), encoding="utf-8"
    )
    plot_g1(curves, phase3 / "g1_alpha_response_curves.png")

    simple = summary.set_index("model").loc["simple_fusion"]
    interaction = summary.set_index("model").loc["interaction_fusion"]
    flatter = bool(
        interaction["plateau_width_decades_mean"]
        > simple["plateau_width_decades_mean"] + 0.5
    )
    smaller_regret = bool(
        interaction["regret_cv3_mean"] < simple["regret_cv3_mean"] - 0.0005
    )
    robustness_supported = flatter and smaller_regret
    cv5_distance_delta = (
        g23["inner_cv5_log10_distance_to_oracle"]
        - g23["inner_cv3_log10_distance_to_oracle"]
    )
    cv5_mae_delta = g23["cv5_minus_cv3_outer_mae"]

    lines = [
        "# Phase 3 Experiment G — Regularization Sensitivity",
        "",
        "## G0 blocking answer",
        "",
        "**The original simple-fusion alpha 0.1 was fixed, not selected from a grid; interior/boundary status is therefore not applicable. The original interaction alpha 10 was the upper boundary in all five folds.**",
        "",
        "Full source evidence: `outputs/phase3/experiment_g0_alpha_audit.md`.",
        "",
        "## G5 manuscript summary",
        "",
        "| model | oracle MAE | selected CV3 MAE | regret | plateau width (decades) | selected alpha (geomean) | oracle alpha (geomean) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['model']} | {row['oracle_mae_mean']:.6f} +/- {row['oracle_mae_sd']:.6f} | "
            f"{row['selected_cv3_mae_mean']:.6f} +/- {row['selected_cv3_mae_sd']:.6f} | "
            f"{row['regret_cv3_mean']:.6f} +/- {row['regret_cv3_sd']:.6f} | "
            f"{row['plateau_width_decades_mean']:.3f} +/- {row['plateau_width_decades_sd']:.3f} | "
            f"{row['selected_cv3_alpha_geomean']:.4g} | {row['oracle_alpha_geomean']:.4g} |"
        )
    verdict = (
        "SUPPORTED: interaction has both a materially wider plateau and smaller CV3 regret."
        if robustness_supported
        else "NOT SUPPORTED under the prespecified joint rule: interaction does not have both a >0.5-decade wider plateau and >0.0005 smaller regret than simple fusion."
    )
    lines.extend(
        [
            "",
            f"**Robustness verdict: {verdict}**",
            "",
            "The G1 curves are outer-test oracle diagnostics, not a model-selection procedure. MAE/RMSE and train MAE use predictions clipped to [1,5], matching the manuscript's primary metric.",
            "",
            "## G2 fold-level oracle, plateau, and regret",
            "",
            g23.to_markdown(index=False, floatfmt=".6f"),
            "",
            "## G3 GroupKFold(3) versus GroupKFold(5)",
            "",
            f"- Mean change in absolute log10 distance to the outer oracle (CV5 - CV3): {cv5_distance_delta.mean():+.4f} decades.",
            f"- Mean outer-MAE change (CV5 - CV3): {cv5_mae_delta.mean():+.6f}.",
            f"- CV5 moved closer to the oracle in {int((cv5_distance_delta < 0).sum())}/{len(cv5_distance_delta)} model-folds and improved outer MAE in {int((cv5_mae_delta < 0).sum())}/{len(cv5_mae_delta)}.",
            "",
            "## G4 fold-1 sample-size scaling",
            "",
            g4.to_markdown(index=False, floatfmt=".6f"),
            "",
            "G4 keeps the fold-1 outer-training TF-IDF representation fixed and changes only the hotel-preserved supervised fitting rows. Samples are nested under seed 42.",
            "",
            "## Artifacts",
            "",
            "- `outputs/phase3/alpha_response_curves.csv` and `.json`",
            "- `outputs/phase3/g2_g3_oracle_plateau_selection.csv` and `.json`",
            "- `outputs/phase3/g4_sample_size_scaling.csv` and `.json`",
            "- `outputs/phase3/g4_sample_size_alpha_curves.csv`",
            "- `outputs/phase3/g5_summary.csv` and `.json`",
            "- `outputs/phase3/g1_alpha_response_curves.png`",
            "",
        ]
    )
    report = phase3 / "experiment_g_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    log(f"saved: {report}")


def run_benchmark(args: argparse.Namespace) -> None:
    output_dir = ROOT / "outputs"
    phase3 = output_dir / "phase3"
    dataset, splits, raw = load_core(output_dir, args.profile)
    split = splits[1]
    train_idx, test_idx = split["train_idx"], split["test_idx"]
    y = dataset["rating_overall"].to_numpy(dtype=np.float64)
    train_path, test_path = tfidf_paths(output_dir, args.profile, args.max_features, 1)
    text_train = sp.load_npz(train_path).tocsr()
    text_test = sp.load_npz(test_path).tocsr()
    blocks_train, blocks_test = prepare_blocks(raw, train_idx, test_idx)
    # ``run_outer_curve`` reverses its input to follow a strong-to-weak
    # regularization warm-start path, so keep this diagnostic list ascending.
    benchmark_alphas = np.asarray([1e-4, 10.0, 1e6], dtype=float)
    rows = []
    for model in MODELS:
        path = phase3 / "benchmark" / f"{model}_fold1.csv"
        started = time.perf_counter()
        frame = run_outer_curve(
            model,
            1,
            text_train,
            text_test,
            blocks_train,
            blocks_test,
            y[train_idx],
            y[test_idx],
            path,
            args.tolerance,
            args.max_iterations,
            alphas=benchmark_alphas,
            extra={"experiment": "G_benchmark"},
            resume=args.resume,
        )
        wall = time.perf_counter() - started
        rows.append(
            {
                "model": model,
                "benchmark_fits": len(benchmark_alphas),
                "benchmark_wall_seconds": wall,
                "mean_fit_seconds": float(frame["fit_seconds"].mean()),
                "projected_g1_serial_hours": float(
                    frame["fit_seconds"].mean() * len(ALPHAS) * N_SPLITS / 3600
                ),
            }
        )
    result = pd.DataFrame(rows)
    result["projected_g1_total_serial_hours"] = result[
        "projected_g1_serial_hours"
    ].sum()
    atomic_csv(result, phase3 / "g_benchmark.csv")
    (phase3 / "g_benchmark.json").write_text(
        json.dumps(result.to_dict(orient="records"), indent=2), encoding="utf-8"
    )
    log(result.to_string(index=False))


def main() -> None:
    args = parse_args()
    folds = parse_folds(args.folds)
    phase3 = ROOT / "outputs" / "phase3"
    phase3.mkdir(parents=True, exist_ok=True)
    log(
        f"Experiment G stage={args.stage} folds={folds} alpha_grid=logspace(-4,6,41) "
        f"model_jobs={args.model_jobs} inner_jobs={args.inner_jobs}"
    )
    if args.stage == "benchmark":
        run_benchmark(args)
        return
    if args.stage in {"g1", "all"}:
        run_g1(args, folds)
    if args.stage in {"g3", "all"}:
        run_g3(args, folds)
    if args.stage in {"g4", "all"}:
        run_g4(args)
    if args.stage in {"finalize", "all"}:
        finalize_g(args)


if __name__ == "__main__":
    main()
