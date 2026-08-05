#!/usr/bin/env python3
"""Phase 5 E2: controlled traveler-segment masking under the U0 protocol.

The population is restricted to rows whose extracted traveler segment was
originally known.  Segment labels are then reproducibly replaced by unknown at
rho in {0,.2,...,1}.  The fold-specific U0 A8 alpha (10 in all five folds) is
inherited without looking at outer-test outcomes.  Rho=0 is the immutable A8
identity reference; all other conditions are refit with the same U0 estimator.
"""

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
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import mean_absolute_error, mean_squared_error
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
CONTROL_SEEDS = [42, 43, 44]
RHO_VALUES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
SEGMENTS = ["business", "couple", "family", "friends", "solo", "unknown"]
N_ROWS = 877_808
SPLIT_SHA256 = "dfe3a434fb00c894218309c332627dc0131a2db3cc2cd241727fc712a4803942"
A8_MODEL_ID = "phase4_u0_a8_known_segment_only"


def log(message: str) -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="qwen25_7b")
    parser.add_argument("--max-features", type=int, default=200_000)
    parser.add_argument("--folds", default="1,2,3,4,5")
    parser.add_argument("--jobs", type=int, default=2, help="Concurrent outer-fold workers.")
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--max-iterations", type=int, default=2_000)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--finalize-only", action="store_true")
    return parser.parse_args()


def parse_folds(value: str) -> List[int]:
    folds = sorted({int(item) for item in value.split(",") if item.strip()})
    if not folds or any(fold < 1 or fold > 5 for fold in folds):
        raise ValueError("--folds must contain values in 1..5")
    return folds


def load_splits(output_dir: Path) -> Dict[int, Dict[str, np.ndarray]]:
    output = {}
    assignment = np.full(N_ROWS, -1, dtype=np.int8)
    for fold in range(1, 6):
        arrays = np.load(output_dir / f"splits_fold{fold}.npz")
        train_idx = arrays["train_idx"].astype(np.int64)
        test_idx = arrays["test_idx"].astype(np.int64)
        assignment[test_idx] = fold
        output[fold] = {"train_idx": train_idx, "test_idx": test_idx}
    digest = hashlib.sha256(assignment.tobytes()).hexdigest()
    if digest != SPLIT_SHA256:
        raise ValueError(f"canonical split hash mismatch: {digest}")
    return output


def tfidf_paths(output_dir: Path, profile: str, max_features: int, fold: int) -> Tuple[Path, Path]:
    matches = sorted(
        (output_dir / "tfidf_cache").glob(
            f"{profile}_mf{max_features}_ng12_md3_full_fold{fold}_*_Xtr.npz"
        )
    )
    if len(matches) != 1:
        raise FileNotFoundError(f"fold {fold}: expected one TF-IDF cache, got {matches}")
    test = Path(str(matches[0]).replace("_Xtr.npz", "_Xte.npz"))
    if not test.exists():
        raise FileNotFoundError(test)
    return matches[0], test


def load_components(output_dir: Path, profile: str) -> Dict[str, np.ndarray]:
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
    segment_text = frame["traveler_segment"].fillna("unknown").astype(str).str.lower()
    segment_text = segment_text.where(segment_text.isin(SEGMENTS), "unknown")
    segments = (
        pd.get_dummies(segment_text)
        .reindex(columns=SEGMENTS, fill_value=False)
        .to_numpy(dtype=np.float32)
    )
    return {
        "scores": scores,
        "mentions": mentions,
        "segments": segments,
        "known": segment_text.ne("unknown").to_numpy(dtype=bool),
    }


def build_blocks(scores: np.ndarray, mentions: np.ndarray, segments: np.ndarray) -> Dict[str, np.ndarray]:
    return {
        "z": np.hstack([scores, mentions, segments]).astype(np.float32),
        "p_s": np.hstack(
            [scores * segments[:, index : index + 1] for index in range(len(SEGMENTS))]
        ).astype(np.float32),
        "p_b": np.hstack(
            [mentions * segments[:, index : index + 1] for index in range(len(SEGMENTS))]
        ).astype(np.float32),
    }


def prepare_blocks(
    raw: Mapping[str, np.ndarray], train_idx: np.ndarray, test_idx: np.ndarray
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    train, test = {}, {}
    for name, matrix in raw.items():
        scaler = StandardScaler()
        train[name] = scaler.fit_transform(matrix[train_idx]).astype(np.float32)
        test[name] = scaler.transform(matrix[test_idx]).astype(np.float32)
    return train, test


def condition_path(cache_dir: Path, fold: int, rho: float, seed: int) -> Path:
    return cache_dir / f"rho{int(round(rho * 100)):03d}_seed{seed}_fold{fold}.json"


def selected_a8_alpha(output_dir: Path, fold: int) -> float:
    payload = json.loads(
        (output_dir / "predictions" / A8_MODEL_ID / f"fold{fold}.meta.json").read_text()
    )
    return float(payload["selected_hyperparameters"]["alpha"])


def reference_rho0(output_dir: Path, fold: int, seed: int) -> Dict[str, Any]:
    meta = json.loads(
        (output_dir / "predictions" / A8_MODEL_ID / f"fold{fold}.meta.json").read_text()
    )
    return {
        "fold": fold,
        "rho": 0.0,
        "seed": seed,
        "n": int(meta["n_test"]),
        "mae": float(meta["clipped"]["mae"]),
        "rmse": float(meta["clipped"]["rmse"]),
        "alpha": float(meta["selected_hyperparameters"]["alpha"]),
        "iterations": int(meta.get("extra", {}).get("iterations", -1)),
        "stop_code": int(meta.get("extra", {}).get("stop_code", -1)),
        "fit_seconds": 0.0,
        "source": "identity copy of phase4_u0_a8_known_segment_only",
        "split_sha256": SPLIT_SHA256,
    }


def fit_fold(
    args: argparse.Namespace,
    fold: int,
    split: Mapping[str, np.ndarray],
    output_dir: Path,
    cache_dir: Path,
    components: Mapping[str, np.ndarray],
    y: np.ndarray,
) -> List[Dict[str, Any]]:
    train_idx = split["train_idx"]
    test_idx = split["test_idx"]
    known = components["known"]
    train_keep = known[train_idx]
    test_keep = known[test_idx]
    known_train_idx = train_idx[train_keep]
    known_test_idx = test_idx[test_keep]
    train_path, test_path = tfidf_paths(output_dir, args.profile, args.max_features, fold)
    log(f"E2 fold {fold}: loading TF-IDF; known train={len(known_train_idx):,}, test={len(known_test_idx):,}")
    text_train = sp.load_npz(train_path).tocsr()[train_keep].tocsr()
    text_test = sp.load_npz(test_path).tocsr()[test_keep].tocsr()
    alpha = selected_a8_alpha(output_dir, fold)
    records: List[Dict[str, Any]] = []
    progress = tqdm(
        [(rho, seed) for rho in RHO_VALUES for seed in CONTROL_SEEDS],
        desc=f"P5 E2 fold {fold}",
        position=fold - 1,
        leave=True,
    )
    for rho, seed in progress:
        path = condition_path(cache_dir, fold, rho, seed)
        if args.resume and path.exists():
            row = json.loads(path.read_text())
            records.append(row)
            progress.set_postfix(rho=rho, seed=seed, cache="hit", mae=f"{row['mae']:.4f}")
            continue
        if np.isclose(rho, 0.0):
            row = reference_rho0(output_dir, fold, seed)
        else:
            rng = np.random.default_rng(seed + int(round(rho * 1000)))
            segments = components["segments"].copy()
            known_positions = np.flatnonzero(known)
            n_mask = int(round(rho * len(known_positions)))
            if n_mask:
                selected = rng.choice(known_positions, size=n_mask, replace=False)
                segments[selected] = 0.0
                segments[selected, SEGMENTS.index("unknown")] = 1.0
            raw = build_blocks(components["scores"], components["mentions"], segments)
            blocks_train, blocks_test = prepare_blocks(raw, known_train_idx, known_test_idx)
            operator = CenteredBlockOperator(
                text_train, blocks_train, {name: 1.0 for name in blocks_train}
            )
            fit = fit_operator(
                operator,
                y[known_train_idx],
                alpha,
                tolerance=args.tolerance,
                max_iterations=args.max_iterations,
                x0=None,
            )
            prediction = np.clip(predict_blocks(text_test, blocks_test, fit), 1.0, 5.0)
            truth = y[known_test_idx]
            row = {
                "fold": fold,
                "rho": rho,
                "seed": seed,
                "n": int(len(truth)),
                "mae": float(mean_absolute_error(truth, prediction)),
                "rmse": float(math.sqrt(mean_squared_error(truth, prediction))),
                "alpha": alpha,
                "iterations": int(fit.iterations),
                "stop_code": int(fit.stop_code),
                "fit_seconds": float(fit.fit_seconds),
                "source": "U0 refit on originally-known subset after deterministic masking",
                "split_sha256": SPLIT_SHA256,
            }
            del segments, raw, blocks_train, blocks_test, operator, fit, prediction
            gc.collect()
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(row, indent=2), encoding="utf-8")
        temporary.replace(path)
        records.append(row)
        progress.set_postfix(rho=rho, seed=seed, mae=f"{row['mae']:.4f}")
    del text_train, text_test
    gc.collect()
    log(f"E2 fold {fold}: complete")
    return records


def finalize(cache_dir: Path, phase5: Path) -> None:
    expected = [
        condition_path(cache_dir, fold, rho, seed)
        for fold in range(1, 6)
        for rho in RHO_VALUES
        for seed in CONTROL_SEEDS
    ]
    missing = [path for path in expected if not path.exists()]
    if missing:
        raise RuntimeError(f"E2 incomplete: {len(missing)} condition-fold files missing")
    fold = pd.DataFrame([json.loads(path.read_text()) for path in expected]).sort_values(
        ["rho", "seed", "fold"]
    )
    seed_level = (
        fold.groupby(["rho", "seed"], as_index=False)
        .agg(
            n=("n", "sum"),
            mae=("mae", "mean"),
            rmse=("rmse", "mean"),
            mean_fit_seconds=("fit_seconds", "mean"),
        )
    )
    summary = (
        seed_level.groupby("rho", as_index=False)
        .agg(
            n=("n", "first"),
            mae_mean=("mae", "mean"),
            mae_sd=("mae", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_sd=("rmse", "std"),
        )
    )
    fold.to_csv(phase5 / "E2_masking_fold_results.csv", index=False)
    seed_level.to_csv(phase5 / "E2_masking_seed_results.csv", index=False)
    summary.to_csv(phase5 / "E2_masking_summary.csv", index=False)
    payload = {
        "protocol": "U0 A8 known-segment subset; inherited fold-specific alpha; deterministic global masking",
        "seeds": CONTROL_SEEDS,
        "rho_values": RHO_VALUES,
        "split_sha256": SPLIT_SHA256,
        "fold_results": fold.to_dict(orient="records"),
        "seed_results": seed_level.to_dict(orient="records"),
        "summary": summary.to_dict(orient="records"),
    }
    (phase5 / "E2_masking_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# E2 Controlled Segment Missingness under U0",
        "",
        "- Population: rows with an originally known extracted traveler segment.",
        "- Masking: an exact global fraction is deterministically relabeled `unknown` for each seed.",
        "- Estimator: Phase 4 U0 interaction fusion, with the U0 A8 fold-specific alpha inherited (alpha=10 in all folds).",
        "- Rho=0: identity copy of the immutable U0 A8 outer predictions.",
        "- Summary SD is across three seed-level five-fold metric means.",
        "",
        summary.to_markdown(index=False, floatfmt=".6f"),
        "",
    ]
    (phase5 / "E2_masking_u0.md").write_text("\n".join(lines), encoding="utf-8")
    log(f"E2 finalized: {len(fold)} fold-condition rows, {len(seed_level)} seed rows")


def main() -> None:
    args = parse_args()
    np.random.seed(SEED)
    output_dir = ROOT / "outputs"
    phase5 = output_dir / "phase5"
    cache_dir = phase5 / "e2_cache"
    phase5.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    folds = parse_folds(args.folds)
    if not args.finalize_only:
        dataset = pd.read_parquet(
            output_dir / "dataset_clean.parquet", columns=["rating_overall"]
        )
        y = dataset["rating_overall"].to_numpy(dtype=np.float64)
        splits = load_splits(output_dir)
        components = load_components(output_dir, args.profile)
        log(
            f"P5 E2 start: jobs={args.jobs}, known={components['known'].sum():,}, "
            f"conditions/fold={len(RHO_VALUES) * len(CONTROL_SEEDS)}"
        )
        with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
            futures = {
                pool.submit(
                    fit_fold, args, fold, splits[fold], output_dir, cache_dir, components, y
                ): fold
                for fold in folds
            }
            for future in as_completed(futures):
                future.result()
    if set(folds) == set(range(1, 6)):
        finalize(cache_dir, phase5)
    else:
        log("E2 partial fold run complete; finalization deferred until all folds exist")


if __name__ == "__main__":
    main()
