#!/usr/bin/env python3
"""Second-round revision: two new sparse variants under the unchanged U0 protocol.

TA  text + true platform aspect sub-ratings (median imputation + missingness
    indicators, identical encoding to the U0 true-aspect reference), fused with
    the same TF-IDF block. Not deployable.
NS  full interaction design (A4) with every sleep-quality column removed
    (score, mention, and the 12 segment interactions).

Both reuse the functions of scripts/41_phase4_u0_models.py unchanged: inner
GroupKFold(5) on the outer training fold, 41-point log grid 1e-4..1e6 with
two-decade boundary extension, LSQR atol=btol=1e-4, iter_lim=2000.
All artefacts go to outputs/r2; nothing under outputs/ is written.
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
MR = ROOT / "outputs" / "r2"
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("u0mod", ROOT / "scripts" / "41_phase4_u0_models.py")
u0 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(u0)

MODEL_IDS = {
    "TA": "mr_u0_text_true_aspects_reference",
    "NS": "mr_u0_a9_interaction_without_sleep_quality",
}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=sorted(MODEL_IDS), required=True)
    p.add_argument("--folds", default="1,2,3,4,5")
    p.add_argument("--profile", default="qwen25_7b")
    p.add_argument("--max-features", type=int, default=200_000)
    p.add_argument("--inner-jobs", type=int, default=5)
    p.add_argument("--tolerance", type=float, default=1e-4)
    p.add_argument("--max-iterations", type=int, default=2_000)
    p.add_argument("--max-extensions", type=int, default=4)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()
    folds = u0.parse_folds(args.folds)
    model_id = MODEL_IDS[args.variant]

    out = MR
    phase4 = out / "phase4"
    harness = u0.EvaluationHarness(out)
    dataset = u0.load_dataset(ROOT / "outputs")
    splits = u0.load_splits(ROOT / "outputs")
    y_all = dataset["rating_overall"].to_numpy(dtype=np.float64)
    groups_all = dataset["offering_id"].to_numpy()
    aspects = list(u0.ASPECTS)
    k = len(aspects)
    sleep = aspects.index("sleep_quality")
    rating_cols = [f"rating_{a}" for a in aspects]

    raw = None
    if args.variant == "NS":
        full = u0.load_components(ROOT / "outputs", args.profile)
        n_seg = len(u0.SEGMENTS)
        keep_z = [i for i in range(full["z"].shape[1]) if i not in (sleep, k + sleep)]
        keep_p = [i for i in range(n_seg * k) if i % k != sleep]
        raw = {
            "z": full["z"][:, keep_z],
            "p_s": full["p_s"][:, keep_p],
            "p_b": full["p_b"][:, keep_p],
        }
        u0.log(f"NS widths z={raw['z'].shape[1]} p_s={raw['p_s'].shape[1]} p_b={raw['p_b'].shape[1]}")

    for fold in folds:
        if args.resume and harness.has_fold(model_id, fold):
            u0.log(f"{model_id} fold={fold}: cache hit")
            continue
        train_idx = splits[fold]["train_idx"]
        test_idx = splits[fold]["test_idx"]
        tr_path, te_path = u0.tfidf_paths(ROOT / "outputs", args.profile, args.max_features, fold)
        t0 = time.perf_counter()
        text_train = sp.load_npz(tr_path).tocsr()
        text_test = sp.load_npz(te_path).tocsr()
        load_seconds = time.perf_counter() - t0
        if args.variant == "TA":
            imputer = SimpleImputer(strategy="median", add_indicator=True)
            a_tr = imputer.fit_transform(dataset.iloc[train_idx][rating_cols].to_numpy(dtype=np.float64))
            a_te = imputer.transform(dataset.iloc[test_idx][rating_cols].to_numpy(dtype=np.float64))
            sc = StandardScaler()
            blocks_train = {"true": sc.fit_transform(a_tr).astype(np.float32)}
            blocks_test = {"true": sc.transform(a_te).astype(np.float32)}
        else:
            blocks_train, blocks_test = u0.prepare_blocks(raw, train_idx, test_idx)
        alpha, boundary, extensions, tuning_seconds = u0.sparse_tune_cv5(
            args.variant, fold, text_train, blocks_train, y_all[train_idx], groups_all[train_idx], phase4, args,
        )
        u0.write_sparse_fold(
            harness, model_id, fold, dataset, train_idx, test_idx, text_train, text_test,
            blocks_train, blocks_test, alpha, boundary, tuning_seconds, args,
            {
                "experiment": "r2",
                "variant": args.variant,
                "boundary_extensions": extensions,
                "deployable": args.variant != "TA",
                "tfidf_load_seconds": load_seconds,
            },
        )
        del text_train, text_test, blocks_train, blocks_test
        gc.collect()

    if set(folds) == set(range(1, 6)):
        harness.finalize_model(model_id, {
            "protocol": "Phase 4 U0 unified GroupKFold(5)",
            "alpha_grid": u0.ALPHAS.tolist(),
            "boundary_extension": "two decades until interior",
            "split_assignment_sha256": u0.SPLIT_SHA256,
            "seed": u0.SEED,
            "model_id": model_id,
        })


if __name__ == "__main__":
    main()
