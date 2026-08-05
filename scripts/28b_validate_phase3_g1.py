#!/usr/bin/env python3
"""Validate G1 completeness and audit agreement at Phase 2 selected alphas.

The Phase 2 estimator uses an iterative LSQR solve with a 1e-4 stopping
tolerance.  Repeated or concurrently scheduled solves can therefore stop at
slightly different iterations.  We preserve those differences in the audit
instead of incorrectly claiming bitwise/exact metric reproduction.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
PHASE3 = OUTPUTS / "phase3"
MODELS = {
    "text_only": "phase2_r0_text_shared_ridge",
    "simple_fusion": "phase2_r0_simple_shared_ridge",
    "interaction_fusion": "phase2_r0_interaction_shared_ridge",
}
EXACT_TOLERANCE = 1e-10
# This is an audit guardrail, not a scientific equivalence margin.  The actual
# fold-level deltas and their maxima are always emitted below.
NUMERICAL_DRIFT_GUARDRAIL = 3e-3


def main() -> None:
    parts = []
    checks = []
    for model, model_id in MODELS.items():
        reference = json.loads(
            (OUTPUTS / "metrics" / f"{model_id}.json").read_text(encoding="utf-8")
        )
        reference_by_fold = {int(row["fold"]): row for row in reference["folds"]}
        for fold in range(1, 6):
            path = PHASE3 / "g1_curves" / f"{model}_fold{fold}.csv"
            if not path.exists():
                raise FileNotFoundError(path)
            curve = pd.read_csv(path)
            if len(curve) != 41 or curve["alpha"].nunique() != 41:
                raise ValueError(f"incomplete G1 curve: {path}, rows={len(curve)}")
            expected = reference_by_fold[fold]
            alpha = float(expected["selected_hyperparameters"]["alpha"])
            match = curve[np.isclose(curve["alpha"], alpha, rtol=1e-10)]
            if len(match) != 1:
                raise ValueError(f"selected alpha {alpha:g} absent/duplicated: {path}")
            actual = match.iloc[0]
            mae_delta = float(actual["test_mae"] - expected["clipped"]["mae"])
            rmse_delta = float(actual["test_rmse"] - expected["clipped"]["rmse"])
            if (
                abs(mae_delta) > NUMERICAL_DRIFT_GUARDRAIL
                or abs(rmse_delta) > NUMERICAL_DRIFT_GUARDRAIL
            ):
                raise ValueError(
                    f"G1 drift exceeds guardrail for {model} fold {fold}: "
                    f"MAE delta={mae_delta}, "
                    f"RMSE delta={rmse_delta}"
                )
            checks.append(
                {
                    "model": model,
                    "fold": fold,
                    "selected_alpha": alpha,
                    "g1_test_mae": float(actual["test_mae"]),
                    "phase2_test_mae": float(expected["clipped"]["mae"]),
                    "mae_delta": mae_delta,
                    "g1_test_rmse": float(actual["test_rmse"]),
                    "phase2_test_rmse": float(expected["clipped"]["rmse"]),
                    "rmse_delta": rmse_delta,
                }
            )
            parts.append(curve)
    combined = pd.concat(parts, ignore_index=True).sort_values(
        ["model", "fold", "alpha"]
    )
    if len(combined) != 615 or combined.duplicated(["model", "fold", "alpha"]).any():
        raise ValueError("combined G1 must contain 615 unique model-fold-alpha rows")
    required = ["model", "fold", "alpha", "test_mae", "test_rmse", "train_mae"]
    combined = combined[required + [c for c in combined if c not in required]]
    combined.to_csv(PHASE3 / "alpha_response_curves.csv", index=False)
    (PHASE3 / "alpha_response_curves.json").write_text(
        json.dumps(combined.to_dict(orient="records"), indent=2), encoding="utf-8"
    )
    audit = pd.DataFrame(checks)
    audit.to_csv(PHASE3 / "g1_selected_alpha_reproduction_audit.csv", index=False)
    max_abs_mae_delta = float(audit["mae_delta"].abs().max())
    max_abs_rmse_delta = float(audit["rmse_delta"].abs().max())
    exact = bool(
        max_abs_mae_delta <= EXACT_TOLERANCE
        and max_abs_rmse_delta <= EXACT_TOLERANCE
    )
    payload = {
        "status": "PASS_EXACT" if exact else "PASS_WITH_NUMERICAL_DRIFT",
        "rows": len(combined),
        "unique_model_fold_alpha": int(
            combined[["model", "fold", "alpha"]].drop_duplicates().shape[0]
        ),
        "exact_tolerance": EXACT_TOLERANCE,
        "numerical_drift_guardrail": NUMERICAL_DRIFT_GUARDRAIL,
        "max_abs_mae_delta": max_abs_mae_delta,
        "max_abs_rmse_delta": max_abs_rmse_delta,
        "interpretation": (
            "Curves are complete. Non-exact agreement reflects the Phase 2 "
            "iterative LSQR tolerance/scheduling and is retained as an anomaly."
        ),
        "checks": checks,
    }
    (PHASE3 / "g1_selected_alpha_reproduction_audit.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(
        f"G1 validation {payload['status']}: 615 unique rows; "
        f"max |MAE delta|={max_abs_mae_delta:.6g}, "
        f"max |RMSE delta|={max_abs_rmse_delta:.6g}",
        flush=True,
    )


if __name__ == "__main__":
    main()
