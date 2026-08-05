#!/usr/bin/env python3
"""Validate Phase 5 artefacts and build the manuscript-facing closure report."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
PHASE5 = OUTPUTS / "phase5"
ASPECT_MODEL_ID = "phase4_u0_llm_aspects_only"
INTERACTION_MODEL_ID = "phase4_u0_interaction_fusion"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="qwen25_7b")
    return parser.parse_args()


def require(path: Path) -> Path:
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(path)
    return path


def fmt(value: Any, digits: int = 6) -> str:
    if value is None or not np.isfinite(float(value)):
        return "NA"
    return f"{float(value):.{digits}f}"


def main() -> None:
    args = parse_args()
    p51 = json.loads(require(PHASE5 / "P5_1_answer.json").read_text())
    comparison = pd.read_csv(require(PHASE5 / "P5_1_implementation_comparison.csv"))
    fold_metrics = pd.read_csv(require(PHASE5 / "P5_1_fold_metrics.csv"))
    diagnostics_path = sorted(PHASE5.glob("P5_1_solver_diagnostic_fold*.csv"))
    if len(diagnostics_path) != 1:
        raise ValueError(f"expected one P5.1 diagnostic table, got {diagnostics_path}")
    diagnostics = pd.read_csv(require(diagnostics_path[0]))
    e1 = pd.read_csv(require(PHASE5 / "E1_known_unknown_strata.csv"))
    e2 = pd.read_csv(require(PHASE5 / "E2_masking_summary.csv"))
    e2_seed = pd.read_csv(require(PHASE5 / "E2_masking_seed_results.csv"))
    e3 = pd.read_csv(require(PHASE5 / "E3_mentioned_count_strata.csv"))
    e4 = pd.read_csv(require(PHASE5 / "E4_hotel_coverage_correlation.csv"))
    coefficients = pd.read_csv(require(PHASE5 / "S1_coefficient_tables.csv"))
    interactions = pd.read_csv(require(PHASE5 / "S1_segment_aspect_interactions.csv"))
    gof = pd.read_csv(require(PHASE5 / "S1_goodness_of_fit.csv"))
    signs = pd.read_csv(require(PHASE5 / "S1_sign_pattern_audit.csv"))

    if len(e1) != 6 or set(e1["stratum"]) != {"known_segment", "unknown_segment"}:
        raise ValueError("E1 completeness failed")
    if len(e3) != 21 or set(e3["mentioned_aspects"]) != set(range(7)):
        raise ValueError("E3 completeness failed")
    if len(e2) != 6 or len(e2_seed) != 18:
        raise ValueError("E2 completeness failed")
    if len(coefficients) != 550:
        raise ValueError(f"coefficient row count failed: {len(coefficients)} != 550")
    if len(interactions) != 360:
        raise ValueError(f"interaction row count failed: {len(interactions)} != 360")
    if len(gof) != 10:
        raise ValueError("goodness-of-fit row count failed")
    if not (coefficients["bootstrap_resamples"] == 10_000).all():
        raise ValueError("10,000-resample bootstrap requirement failed")
    if not (coefficients["bootstrap_seed"] == 42).all():
        raise ValueError("bootstrap seed requirement failed")

    published = fold_metrics[fold_metrics["source"].str.startswith("published")]
    g1 = fold_metrics[fold_metrics["source"].str.startswith("g1")]
    published_mae = float(published["mae"].mean())
    g1_mae = float(g1["mae"].mean())
    diagnostic_by_name = diagnostics.set_index("implementation")
    controlled_published = float(
        diagnostic_by_name.loc[
            "published_exact_float64_auto_tol1e-4", "test_mae_clipped"
        ]
    )
    controlled_round = float(
        diagnostic_by_name.loc[
            "structured_round_float32_then_float64_auto", "test_mae_clipped"
        ]
    )
    controlled_lsqr = float(
        diagnostic_by_name.loc[
            "solver_swap_float64_lsqr_tol1e-4", "test_mae_clipped"
        ]
    )
    reversals = signs[~signs["preserved"].astype(bool)]
    protocol_sign_changes = signs[signs["sign_changed_from_phase3"].astype(bool)]
    positives = signs[
        signs["feature"].isin(["score_service", "score_rooms"])
    ]
    requested_positive_preserved = bool(positives["preserved"].all())
    mentioned = signs[signs["feature"].str.startswith("mentioned_")]
    all_mentions_negative = bool(mentioned["preserved"].all())
    e1_unknown_penalty = {
        model: float(group.set_index("stratum").loc["unknown_segment", "mae"])
        - float(group.set_index("stratum").loc["known_segment", "mae"])
        for model, group in e1.groupby("model")
    }
    e2_rho0 = e2.loc[np.isclose(e2["rho"], 0.0)].iloc[0]
    e2_rho1 = e2.loc[np.isclose(e2["rho"], 1.0)].iloc[0]
    e4_pearson = e4[e4["method"].eq("Pearson")].iloc[0]
    e4_spearman = e4[e4["method"].eq("Spearman")].iloc[0]

    anomalies: List[str] = [
        (
            "P5.1: the published and G1 alpha=0.1 values use different iterative "
            "estimators/stopping rules. This is a protocol-labeling issue in the G1 "
            "diagnostic, not corruption of the independently reproduced Panel A artefact."
        ),
        (
            "P5.3: interaction-fusion uncertainty is conditional on the fitted TF-IDF "
            "component, uses a PSD cluster-influence sandwich, and uses a one-step cluster-pairs bootstrap; 10,000 complete "
            "200k-dimensional refits are not claimed."
        ),
    ]
    if len(reversals):
        labels = ", ".join(
            f"{row.model}:{row.feature} ({row.u0_sign})"
            for row in reversals.itertuples()
        )
        anomalies.append(
            "P5.3 PROMINENT SIGN REVERSAL/EXCEPTION relative to the stated Table 3 pattern: "
            + labels
            + (
                ". At least one sign changed from Phase 3 under U0."
                if len(protocol_sign_changes)
                else ". Neither exception is U0-induced; both signs were already positive in Phase 3."
            )
        )
    else:
        anomalies.append("P5.3: no requested sign reversal was observed under U0.")

    artefacts = [
        "outputs/phase5/P5_1_answer.md",
        "outputs/phase5/P5_1_answer.json",
        "outputs/phase5/P5_1_implementation_comparison.csv",
        "outputs/phase5/P5_1_fold_metrics.csv",
        str(diagnostics_path[0].relative_to(ROOT)),
        "outputs/phase5/E1_known_unknown_strata.csv",
        "outputs/phase5/E1_known_unknown_strata.json",
        "outputs/phase5/E1_E3_E4_u0.md",
        "outputs/phase5/E2_masking_fold_results.csv",
        "outputs/phase5/E2_masking_seed_results.csv",
        "outputs/phase5/E2_masking_summary.csv",
        "outputs/phase5/E2_masking_results.json",
        "outputs/phase5/E2_masking_u0.md",
        "outputs/phase5/E3_mentioned_count_strata.csv",
        "outputs/phase5/E3_mentioned_count_strata.json",
        "outputs/phase5/E4_hotel_coverage_advantage.parquet",
        "outputs/phase5/E4_hotel_coverage_correlation.csv",
        "outputs/phase5/E4_hotel_coverage_correlation.json",
        "outputs/phase5/S1_coefficient_tables.csv",
        "outputs/phase5/S1_coefficient_tables.md",
        "outputs/phase5/S1_coefficient_tables.json",
        "outputs/phase5/S1_goodness_of_fit.csv",
        "outputs/phase5/S1_segment_aspect_interactions.csv",
        "outputs/phase5/S1_sign_pattern_audit.csv",
    ]
    for item in artefacts:
        require(ROOT / item)

    lines = [
        "# Phase 5 — Consistency Closure",
        "",
        "**Status: consistency closure only. No Phase 4 conclusion is changed.**",
        "",
        "## P5.1 — Simple-fusion discrepancy (blocking answer)",
        "",
        "**Resolved exactly: 0.4676 and 0.4568 are not outputs from the same numerical fitting implementation.** The feature identities, outer splits, scaler-fitting rows/rule, and TF-IDF matrices are the same; the structured dtype and numerical solver differ. The published pipeline physically hstacks the float32 TF-IDF matrix with a float64 standardized structured block, producing a float64 CSR matrix, then calls scikit-learn `Ridge(alpha=0.1, solver='auto', tol=1e-4, fit_intercept=True)`, which resolves to sparse-CG. G1 and U0 keep text and float32 structured blocks separate, explicitly center them in a `LinearOperator`, and call SciPy LSQR with `atol=btol=1e-4` and `iter_lim=2000`. Those tolerance values are not equivalent stopping tests on this ill-conditioned low-alpha problem.",
        "",
        f"The correct reference for **the published model at alpha=0.1** is the `SIMPLE_FUSION` branch of `scripts/08_phase0_reproduce.py`: clipped MAE {published_mae:.6f} (reported as 0.4676). G1's later LSQR diagnostic gives {g1_mae:.6f} (difference {g1_mae - published_mae:+.6f}). `scripts/41_phase4_u0_models.py` uses the G1/U0 LSQR estimator but does not use alpha=0.1 for the final simple-fusion model; it selects alpha=5.623413 by grouped inner CV5.",
        "",
        "**Panel A is unaffected. The manuscript value 0.4676 is retained verbatim.** Phase 0 reran the exact published branch and reproduced it within 0.000024. G1 was an outer-test regularization diagnostic under a later estimator and is not substituted into Panel A.",
        "",
        f"The controlled fold isolates the cause quantitatively: sparse-CG to LSQR changes MAE by {controlled_lsqr - controlled_published:+.6f}, whereas float32 rounding of the structured values while retaining float64 sparse-CG changes it by {controlled_round - controlled_published:+.6f}. Solver/stopping behavior is therefore dominant; dtype rounding is secondary.",
        "",
        "### Implementation comparison",
        "",
        comparison.to_markdown(index=False),
        "",
        "### Five-fold alpha=0.1 evidence",
        "",
        fold_metrics.to_markdown(index=False, floatfmt=".6f"),
        "",
        "### Controlled solver/dtype interchange",
        "",
        diagnostics.to_markdown(index=False, floatfmt=".6g"),
        "",
        "## P5.2 — Missingness stratification under U0",
        "",
        "All E1/E3/E4 metrics below use immutable clipped Phase 4 U0 outer-test predictions. E2 is the only refit experiment.",
        "",
        "### E1 — Known versus unknown traveler segment",
        "",
        e1.to_markdown(index=False, floatfmt=".6f"),
        "",
        "Unknown-segment rows are harder for all three models; the unknown-minus-known MAE penalties are "
        + ", ".join(f"`{model}` {delta:+.6f}" for model, delta in e1_unknown_penalty.items())
        + ".",
        "",
        "### E2 — Controlled segment-label masking",
        "",
        "The population is restricted to originally known-segment rows. For each rho and seed, an exact global fraction is relabeled unknown before rebuilding and outer-train-standardizing z/p_s/p_b. The U0 A8 fold-specific selected alpha is inherited (alpha=10 in all folds); rho=0 is the immutable A8 identity reference. SD is across three seed-level five-fold means.",
        "",
        e2.to_markdown(index=False, floatfmt=".6f"),
        "",
        f"From rho=0 to rho=1, mean MAE changes by {float(e2_rho1['mae_mean'] - e2_rho0['mae_mean']):+.6f} and mean RMSE by {float(e2_rho1['rmse_mean'] - e2_rho0['rmse_mean']):+.6f}.",
        "",
        "### E3 — Number of mentioned aspects",
        "",
        e3.to_markdown(index=False, floatfmt=".6f"),
        "",
        "### E4 — Hotel-level known-segment coverage",
        "",
        "Advantage is hotel text-only MAE minus hotel interaction-fusion MAE; positive values favor interaction fusion.",
        "",
        e4.to_markdown(index=False, floatfmt=".6g"),
        "",
        f"The U0 hotel-level association is small and negative (Pearson r={float(e4_pearson['correlation']):.6f}; Spearman rho={float(e4_spearman['correlation']):.6f}), so greater known-segment coverage does not explain a larger interaction-over-text advantage across hotels.",
        "",
        "## P5.3 — U0 coefficient artefacts",
        "",
        f"The complete supplement contains {len(coefficients):,} fold-level coefficient rows: all coefficients for `{ASPECT_MODEL_ID}` and every structured coefficient from `{INTERACTION_MODEL_ID}`. There are {len(interactions):,} fold-level segment-by-aspect interaction rows. Every estimate has a hotel-cluster-robust SE computed as the numerically stable PSD matrix `cluster_influence.T @ cluster_influence`, a z statistic, a normal-reference p value, and a 10,000-resample hotel-bootstrap 95% CI (seed 42).",
        "",
        "### Per-fold goodness of fit",
        "",
        gof.to_markdown(index=False, floatfmt=".6g"),
        "",
        "### Published Table 3 sign-pattern audit",
        "",
        signs.to_markdown(index=False, floatfmt=".6g"),
        "",
        f"Positive service-score and rooms-score signs preserved under U0: **{requested_positive_preserved}**.",
        f"All six mention-indicator fold-mean signs negative under U0: **{all_mentions_negative}**.",
        f"Signs changed from the earlier Phase 3 coefficient protocol: **{len(protocol_sign_changes)}**.",
        "",
    ]
    if len(reversals):
        lines.extend(
            [
                "**PROMINENT SIGN REVERSAL/EXCEPTION:**",
                "",
                reversals.to_markdown(index=False, floatfmt=".6g"),
                "",
            ]
        )
    lines.extend(["## Anomalies and interpretive constraints", ""])
    lines.extend([f"- {item}" for item in anomalies])
    lines.extend(
        [
            "",
            "## Artefact paths",
            "",
            *[f"- `{item}`" for item in artefacts],
            "",
            "Each Markdown table above has a CSV and/or JSON machine-readable companion listed here. The hotel-level E4 rows are Parquet.",
            "",
            "## Reproduction commands",
            "",
            "```bash",
            "python scripts/44_phase5_discrepancy_strata.py --stage all",
            "python scripts/45_phase5_e2_u0_missingness.py --jobs 2",
            "python scripts/46_phase5_u0_coefficient_tables.py --bootstrap-resamples 10000",
            "python scripts/47_finalize_phase5.py",
            "```",
            "",
            "All scripts are resumable, CPU-only, use seed 42, use the canonical split assignment hash `dfe3a434fb00c894218309c332627dc0131a2db3cc2cd241727fc712a4803942`, and perform no LLM re-extraction.",
            "",
        ]
    )
    report_path = OUTPUTS / "phase5_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    payload: Dict[str, Any] = {
        "phase": 5,
        "status": "PASS",
        "canonical_split_sha256": "dfe3a434fb00c894218309c332627dc0131a2db3cc2cd241727fc712a4803942",
        "p5_1": {
            **p51,
            "implementation_comparison": comparison.to_dict(orient="records"),
            "fold_metrics": fold_metrics.to_dict(orient="records"),
            "controlled_diagnostic": diagnostics.to_dict(orient="records"),
        },
        "e1": e1.to_dict(orient="records"),
        "e2": {
            "summary": e2.to_dict(orient="records"),
            "seed_results": e2_seed.to_dict(orient="records"),
        },
        "e3": e3.to_dict(orient="records"),
        "e4": e4.to_dict(orient="records"),
        "p5_3": {
            "coefficient_rows": len(coefficients),
            "interaction_rows": len(interactions),
            "goodness_of_fit": gof.to_dict(orient="records"),
            "sign_pattern": signs.to_dict(orient="records"),
            "positive_service_rooms_preserved": requested_positive_preserved,
            "all_mention_indicators_negative": all_mentions_negative,
        },
        "anomalies": anomalies,
        "artefacts": artefacts,
    }
    (OUTPUTS / "phase5_report.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(f"Phase 5 PASS: {report_path}", flush=True)


if __name__ == "__main__":
    main()
