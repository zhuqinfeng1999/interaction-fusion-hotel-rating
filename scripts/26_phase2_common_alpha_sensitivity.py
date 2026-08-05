#!/usr/bin/env python3
"""Fit the missing simple-fusion arm at the text-selected common alpha."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil
import scipy.sparse as sp
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation_harness import (  # noqa: E402
    EvaluationHarness,
    PeakRSSMonitor,
    bootstrap_mean_difference,
    paired_wilcoxon,
    write_json,
)
from src.phase2_sparse_ridge import (  # noqa: E402
    CenteredBlockOperator,
    fit_operator,
    predict_blocks,
)


MODEL_ID = "phase2_claim_simple_at_text_alpha"
TEXT_MODEL = "phase2_r0_text_shared_ridge"
INTERACTION_MODEL = "phase2_r2a_interaction_at_text_alpha"
SEED = 42


def load_pa_module() -> Any:
    path = ROOT / "scripts" / "21_phase2_controls_ablations.py"
    spec = importlib.util.spec_from_file_location("phase2_pa_common_alpha", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PA = load_pa_module()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="qwen25_7b")
    parser.add_argument("--max-features", type=int, default=200_000)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--max-iterations", type=int, default=2_000)
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def text_alpha(output_dir: Path, fold: int) -> float:
    payload = json.loads(
        (
            output_dir
            / "predictions"
            / TEXT_MODEL
            / f"fold{fold}.meta.json"
        ).read_text(encoding="utf-8")
    )
    return float(payload["selected_hyperparameters"]["alpha"])


def fit_fold(
    args: argparse.Namespace,
    fold: int,
    output_dir: Path,
    harness: EvaluationHarness,
    splits: dict[int, dict[str, Any]],
    components: dict[str, np.ndarray],
    y: np.ndarray,
    groups: np.ndarray,
    row_ids: np.ndarray,
) -> None:
    if args.resume and harness.has_fold(MODEL_ID, fold):
        print(f"{MODEL_ID} fold {fold}: cache hit", flush=True)
        return
    split = splits[fold]
    train_idx = split["train_idx"]
    test_idx = split["test_idx"]
    train_path, test_path = PA.tfidf_paths(
        output_dir, args.profile, args.max_features, fold
    )
    text_train = sp.load_npz(train_path).tocsr()
    text_test = sp.load_npz(test_path).tocsr()
    raw = PA.true_blocks(components)
    block_train, block_test = PA.prepare_blocks(raw, train_idx, test_idx)
    simple_train = {"z": block_train["z"]}
    simple_test = {"z": block_test["z"]}
    alpha = text_alpha(output_dir, fold)

    with PeakRSSMonitor() as memory:
        operator = CenteredBlockOperator(text_train, simple_train, {"z": 1.0})
        fit = fit_operator(
            operator,
            y[train_idx],
            alpha,
            tolerance=args.tolerance,
            max_iterations=args.max_iterations,
        )
        predict_started = time.perf_counter()
        prediction = predict_blocks(text_test, simple_test, fit)
        predict_seconds = time.perf_counter() - predict_started
    artifact = harness.write_fold(
        model_id=MODEL_ID,
        fold=fold,
        row_id=row_ids[test_idx],
        offering_id=groups[test_idx],
        y_true=y[test_idx],
        y_pred_raw=prediction,
        fit_seconds=fit.fit_seconds,
        predict_seconds=predict_seconds,
        peak_memory_mb=max(
            memory.peak_mb,
            psutil.Process().memory_info().rss / (1024 * 1024),
        ),
        selected_hyperparameters={
            "alpha": alpha,
            "lambdas": {"z": 1.0},
            "alpha_source": f"{TEXT_MODEL} fold {fold} inner-CV selection",
            "selection_note": (
                "No candidate model used the outer test fold to choose alpha; "
                "the text-selected alpha is imposed on all three feature sets."
            ),
            "solver": "scipy.sparse.linalg.lsqr centered LinearOperator",
            "tolerance": args.tolerance,
            "max_iterations": args.max_iterations,
        },
        extra={
            "phase": 2,
            "experiment": "common_alpha_claim_sensitivity",
            "variant": "simple_fusion_at_text_selected_alpha",
            "n_features": int(operator.shape[1]),
            "outer_split_sha256": split["sha256"],
            "lsqr_iterations": fit.iterations,
            "lsqr_stop_code": fit.stop_code,
        },
    )
    print(
        f"{MODEL_ID} fold {fold}: alpha={alpha:.6g}, "
        f"MAE={artifact.record['clipped']['mae']:.6f}, "
        f"RMSE={artifact.record['clipped']['rmse']:.6f}, "
        f"fit={fit.fit_seconds:.1f}s",
        flush=True,
    )
    del text_train, text_test, raw, block_train, block_test
    gc.collect()


def summary_row(output_dir: Path, model_id: str) -> dict[str, Any]:
    payload = json.loads(
        (output_dir / "metrics" / f"{model_id}.json").read_text(encoding="utf-8")
    )
    clipped = payload["aggregate"]["clipped"]
    return {
        "model_id": model_id,
        "n": int(payload["n_predictions"]),
        "mae_mean": float(clipped["mean"]["mae"]),
        "mae_std": float(clipped["std"]["mae"]),
        "rmse_mean": float(clipped["mean"]["rmse"]),
        "rmse_std": float(clipped["std"]["rmse"]),
    }


def main() -> None:
    args = parse_args()
    np.random.seed(SEED)
    output_dir = ROOT / "outputs"
    phase2 = output_dir / "phase2"
    harness = EvaluationHarness(output_dir)
    splits = PA.load_splits(output_dir)
    components = PA.load_components(output_dir, args.profile)
    dataset = pd.read_parquet(
        output_dir / "dataset_clean.parquet",
        columns=["rating_overall", "offering_id"],
    )
    y = dataset["rating_overall"].to_numpy(dtype=float)
    groups = dataset["offering_id"].to_numpy()
    row_ids = np.arange(len(dataset), dtype=np.int64)

    for fold in tqdm(range(1, 6), desc="Common-alpha simple fusion"):
        fit_fold(
            args,
            fold,
            output_dir,
            harness,
            splits,
            components,
            y,
            groups,
            row_ids,
        )
    harness.finalize_model(
        MODEL_ID,
        {
            "phase": 2,
            "experiment": "common_alpha_claim_sensitivity",
            "features": "x + z",
            "alpha_protocol": "fold-specific text-only inner-CV alpha imposed unchanged",
            "seed": SEED,
        },
    )

    model_ids = [TEXT_MODEL, MODEL_ID, INTERACTION_MODEL]
    summary = pd.DataFrame([summary_row(output_dir, model) for model in model_ids])
    simple_hotel = output_dir / "per_hotel" / f"{MODEL_ID}.parquet"
    interaction_hotel = output_dir / "per_hotel" / f"{INTERACTION_MODEL}.parquet"
    paired = paired_wilcoxon(simple_hotel, interaction_hotel)
    bootstrap = bootstrap_mean_difference(simple_hotel, interaction_hotel)
    simple_mae = float(summary.set_index("model_id").loc[MODEL_ID, "mae_mean"])
    interaction_mae = float(
        summary.set_index("model_id").loc[INTERACTION_MODEL, "mae_mean"]
    )
    result = {
        "protocol": (
            "All three models use each fold's text-only inner-CV-selected alpha; "
            "no outer-test-based model selection."
        ),
        "models": summary.to_dict(orient="records"),
        "interaction_gain_over_simple_mae": simple_mae - interaction_mae,
        "paired_hotel_simple_minus_interaction": paired,
        "bootstrap_hotel_simple_minus_interaction": bootstrap,
        "interpretation_rule": (
            "Positive simple-minus-interaction values favor interaction fusion."
        ),
    }
    summary.to_csv(phase2 / "experiment_common_alpha_model_table.csv", index=False)
    write_json(phase2 / "experiment_common_alpha_sensitivity.json", result)
    lines = [
        "# Common-alpha Interaction Sensitivity",
        "",
        result["protocol"],
        "",
        summary.to_markdown(index=False, floatfmt=".6f"),
        "",
        f"- Interaction MAE gain over simple: {result['interaction_gain_over_simple_mae']:.6f}",
        f"- Paired hotel simple-minus-interaction: {paired['mean_difference']:.6f}; Wilcoxon p={paired['p_value']:.6g}",
        f"- Bootstrap 95% CI: {bootstrap['ci_95']}",
        "",
        "This is a pre-specified regularization sensitivity contrast. It does not replace the independently tuned R0 or block-wise R1 primary analyses.",
        "",
    ]
    (phase2 / "experiment_common_alpha_sensitivity.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
