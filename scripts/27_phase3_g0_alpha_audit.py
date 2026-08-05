#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"
PHASE3_DIR = OUTPUT_DIR / "phase3"


MODEL_SPECS: List[Dict[str, Any]] = [
    {
        "model_id": "text_only_tfidf_ridge",
        "display_name": "text_only_tfidf_ridge",
        "selection": "fixed",
        "fixed_alpha": 1.0,
        "alpha_grid": None,
        "alpha_field": "alpha",
        "source_evidence": [
            "scripts/03_run_baselines.py:88-101",
            "scripts/08_phase0_reproduce.py:657-687",
        ],
        "source_note": "Ridge(alpha=1.0) is fit directly; no alpha search is called.",
    },
    {
        "model_id": "llm_aspects_only",
        "display_name": "llm_aspects_only",
        "selection": "inner_group_cv",
        "fixed_alpha": None,
        "alpha_grid": [0.1, 1.0, 10.0],
        "alpha_field": "alpha",
        "source_evidence": [
            "scripts/07_run_llm_aspects_analysis.py:302-327",
            "scripts/07_run_llm_aspects_analysis.py:1327-1328",
            "scripts/08_phase0_reproduce.py:487-504",
        ],
        "source_note": "GroupKFold(3) selects from [0.1, 1, 10].",
    },
    {
        "model_id": "text_llm_simple_fusion",
        "display_name": "text_llm_simple_fusion",
        "selection": "fixed",
        "fixed_alpha": 0.1,
        "alpha_grid": None,
        "alpha_field": "alpha",
        "source_evidence": [
            "scripts/07_run_llm_aspects_analysis.py:392-468",
            "scripts/07_run_llm_aspects_analysis.py:1421-1431",
            "scripts/08_phase0_reproduce.py:695-733",
        ],
        "source_note": "The caller passes alpha=0.1 and Ridge fits it directly; no grid exists.",
    },
    {
        "model_id": "residual_on_llm",
        "display_name": "residual_on_llm",
        "selection": "inner_group_cv",
        "fixed_alpha": None,
        "alpha_grid": [0.1, 1.0, 10.0],
        "alpha_field": "residual_alpha",
        "source_evidence": [
            "scripts/07_run_llm_aspects_analysis.py:797-912",
            "scripts/07_run_llm_aspects_analysis.py:1438-1449",
            "scripts/08_phase0_reproduce.py:742-798",
        ],
        "source_note": "Text stage fixes alpha=1.0; GroupKFold(3) selects residual alpha from [0.1, 1, 10].",
    },
    {
        "model_id": "upper_bound_true_aspects",
        "display_name": "upper_bound_true_aspects",
        "selection": "inner_group_cv",
        "fixed_alpha": None,
        "alpha_grid": [0.01, 0.1, 1.0, 10.0, 100.0],
        "alpha_field": "alpha",
        "source_evidence": [
            "scripts/03_run_baselines.py:117-160",
            "scripts/08_phase0_reproduce.py:487-488",
            "scripts/08_phase0_reproduce.py:542-592",
        ],
        "source_note": "GroupKFold(3) selects Ridge alpha from [0.01, 0.1, 1, 10, 100].",
    },
    {
        "model_id": "interaction_fusion",
        "display_name": "interaction_fusion",
        "selection": "inner_group_cv",
        "fixed_alpha": None,
        "alpha_grid": [0.01, 0.1, 1.0, 10.0],
        "alpha_field": "alpha",
        "source_evidence": [
            "scripts/07_run_llm_aspects_analysis.py:772-794",
            "scripts/07_run_llm_aspects_analysis.py:915-1012",
            "scripts/07_run_llm_aspects_analysis.py:1450-1463",
            "scripts/08_phase0_reproduce.py:819-918",
        ],
        "source_note": "GroupKFold(3) selects from [0.01, 0.1, 1, 10].",
    },
]


def load_fold_hyperparameters(model_id: str) -> List[Dict[str, Any]]:
    path = OUTPUT_DIR / "metrics" / f"{model_id}.json"
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    folds = payload.get("folds", [])
    if len(folds) != 5:
        raise ValueError(f"{path}: expected 5 folds, found {len(folds)}")
    return folds


def selected_alpha(record: Dict[str, Any], field: str) -> float:
    params = record.get("selected_hyperparameters", {})
    if field not in params:
        raise KeyError(f"fold {record.get('fold')}: missing {field} in {params}")
    return float(params[field])


def boundary_status(alpha: float, grid: Optional[List[float]], selection: str) -> str:
    if selection == "fixed" or grid is None:
        return "not_applicable_fixed"
    if alpha == float(grid[0]):
        return "lower_boundary"
    if alpha == float(grid[-1]):
        return "upper_boundary"
    if float(grid[0]) < alpha < float(grid[-1]):
        return "interior"
    return "outside_declared_grid"


def audit_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for spec in MODEL_SPECS:
        folds = load_fold_hyperparameters(spec["model_id"])
        for record in folds:
            alpha = selected_alpha(record, spec["alpha_field"])
            expected = spec["fixed_alpha"]
            if expected is not None and alpha != float(expected):
                raise ValueError(
                    f"{spec['model_id']} fold {record['fold']}: {alpha} != fixed {expected}"
                )
            rows.append(
                {
                    "model_id": spec["model_id"],
                    "fold": int(record["fold"]),
                    "selection_method": spec["selection"],
                    "alpha_grid": spec["alpha_grid"],
                    "selected_alpha": alpha,
                    "boundary_status": boundary_status(
                        alpha, spec["alpha_grid"], spec["selection"]
                    ),
                    "source_evidence": spec["source_evidence"],
                    "source_note": spec["source_note"],
                }
            )
    return rows


def write_markdown(rows: List[Dict[str, Any]], path: Path) -> None:
    frame = pd.DataFrame(rows)
    summary_rows = []
    for spec in MODEL_SPECS:
        part = frame[frame["model_id"] == spec["model_id"]]
        selections = sorted(part["selected_alpha"].unique().tolist())
        statuses = sorted(part["boundary_status"].unique().tolist())
        grid_text = "no grid (fixed)" if spec["alpha_grid"] is None else json.dumps(spec["alpha_grid"])
        summary_rows.append(
            {
                "model": spec["display_name"],
                "exact_alpha_grid": grid_text,
                "selected_alpha_folds_1_to_5": ", ".join(
                    f"{value:g}" for value in part.sort_values("fold")["selected_alpha"]
                ),
                "boundary_status": ", ".join(statuses),
                "source": "; ".join(spec["source_evidence"]),
            }
        )
    summary = pd.DataFrame(summary_rows)
    lines = [
        "# Phase 3 G0 — Original Alpha-Grid Audit",
        "",
        "## Blocking answer",
        "",
        "**The original simple-fusion alpha of 0.1 was not interior to a grid and was not at a grid boundary: no grid search existed for that model. It was a fixed, hard-coded alpha.**",
        "",
        "By contrast, interaction fusion selected `alpha=10` in all five folds, which is the **upper boundary** of its original `[0.01, 0.1, 1, 10]` grid. This makes the original interaction search boundary-limited and motivates the expanded oracle curves in G1.",
        "",
        "## Exact audit",
        "",
        "| model | exact alpha grid | fold 1–5 selected alpha | status | source evidence |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['model']} | {row['exact_alpha_grid']} | "
            f"{row['selected_alpha_folds_1_to_5']} | {row['boundary_status']} | "
            f"{row['source']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation rules",
            "",
            "- `not_applicable_fixed` is intentionally distinct from a boundary selection: a fixed constant is not a one-point model-selection grid.",
            "- `lower_boundary` and `upper_boundary` mean the inner-CV optimum was constrained by the smallest or largest tested candidate.",
            "- The residual model has two stages: text Ridge is fixed at 1.0, while the reported residual alpha is selected from `[0.1, 1, 10]`.",
            "- Fold-level values were read from the canonical `outputs/metrics/{model_id}.json` artifacts, not reconstructed from rounded manuscript tables.",
            "",
            "Machine-readable companions: `experiment_g0_alpha_audit.csv` and `experiment_g0_alpha_audit.json`.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    PHASE3_DIR.mkdir(parents=True, exist_ok=True)
    rows = audit_rows()
    csv_rows = []
    for row in rows:
        converted = dict(row)
        converted["alpha_grid"] = (
            json.dumps(row["alpha_grid"]) if row["alpha_grid"] is not None else ""
        )
        converted["source_evidence"] = "; ".join(row["source_evidence"])
        csv_rows.append(converted)

    csv_path = PHASE3_DIR / "experiment_g0_alpha_audit.csv"
    json_path = PHASE3_DIR / "experiment_g0_alpha_audit.json"
    md_path = PHASE3_DIR / "experiment_g0_alpha_audit.md"
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    write_markdown(rows, md_path)

    print("BLOCKING G0 ANSWER:")
    print("  simple fusion alpha=0.1 was FIXED; there was no alpha grid, so boundary status is N/A.")
    print("  interaction alpha=10 was the UPPER GRID BOUNDARY in all five folds.")
    print(f"saved: {csv_path}")
    print(f"saved: {json_path}")
    print(f"saved: {md_path}")


if __name__ == "__main__":
    main()
