#!/usr/bin/env python3
"""Consolidate all Phase 2 experiments into the final report and master table."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation_harness import write_json  # noqa: E402


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def infer_channel(model_id: str) -> str:
    if "qwen_direct" in model_id:
        return "direct LLM"
    if "roberta" in model_id or "bert_base" in model_id:
        return "transformer text"
    if "tfidf_svd" in model_id:
        return "TF-IDF SVD"
    if "tfidf" in model_id or model_id.startswith("phase2_"):
        return "sparse TF-IDF"
    if "llm_aspects_only" in model_id:
        return "LLM structured only"
    if "upper_bound" in model_id:
        return "true platform aspects"
    return "other"


def infer_structured(model_id: str) -> str:
    if any(token in model_id for token in ("zint", "interaction", "p1_", "p2_", "p3_", "p4_")):
        return "z + score/mention interactions"
    if "score_interactions_only" in model_id:
        return "z + score interactions"
    if "mention_interactions_only" in model_id:
        return "z + mention interactions"
    if "plus_z" in model_id or "simple" in model_id or "residual" in model_id:
        return "LLM main effects z"
    if "llm_aspects_only" in model_id:
        return "LLM main effects z"
    return "none"


def infer_protocol(model_id: str) -> str:
    if model_id.startswith("phase2_r"):
        return "approved reduced R0/R1 grouped tuning"
    if model_id.startswith("phase2_claim"):
        return "common text-selected alpha sensitivity"
    if model_id.startswith("phase2_p") or model_id.startswith("phase2_a") or model_id.startswith("phase2_e2"):
        return "fold-matched R1c penalty control"
    if "crossfit" in model_id:
        return "cross-fitted representation + corrected R0"
    return "accepted Phase 1 / original protocol"


def first_not_none(values: List[Any]) -> Any:
    return next((value for value in values if value is not None), None)


def build_master(output_dir: Path) -> pd.DataFrame:
    d4 = pd.read_csv(output_dir / "phase2" / "experiment_d4_all_metrics.csv")
    rows: List[Dict[str, Any]] = []
    for _, drow in d4.iterrows():
        model_id = str(drow["model_id"])
        metrics_path = output_dir / "metrics" / f"{model_id}.json"
        payload = read_json(metrics_path) if metrics_path.exists() else {}
        folds = payload.get("folds", [])
        aggregate = payload.get("aggregate", {})
        clipped = aggregate.get("clipped", {})
        raw = aggregate.get("raw", {})
        hyperparameters = [fold.get("selected_hyperparameters", {}) for fold in folds]
        extras = [fold.get("extra", {}) for fold in folds]
        alphas = [item.get("alpha") for item in hyperparameters]
        lambdas = [item.get("lambdas") for item in hyperparameters]
        boundaries = [
            item.get("alpha_boundary_status") for item in hyperparameters
        ]
        n_features = first_not_none(
            [extra.get("n_features") for extra in extras]
        )
        fit_seconds = float(
            sum(float(fold.get("fit_seconds", 0.0)) for fold in folds)
        )
        gpu_hours = float(
            sum(float(extra.get("gpu_hours", 0.0)) for extra in extras)
        )
        rows.append(
            {
                "model_id": model_id,
                "channel": infer_channel(model_id),
                "structured_block": infer_structured(model_id),
                "protocol": infer_protocol(model_id),
                "n": int(drow["n"]),
                "n_features": n_features,
                "mae_clipped_fold_mean": clipped.get("mean", {}).get("mae", drow["clipped_mae"]),
                "mae_clipped_fold_std": clipped.get("std", {}).get("mae"),
                "rmse_clipped_fold_mean": clipped.get("mean", {}).get("rmse", drow["clipped_rmse"]),
                "rmse_clipped_fold_std": clipped.get("std", {}).get("rmse"),
                "mae_raw_fold_mean": raw.get("mean", {}).get("mae", drow["raw_mae"]),
                "rmse_raw_fold_mean": raw.get("mean", {}).get("rmse", drow["raw_rmse"]),
                "mae_rounded_pooled": float(drow["rounded_mae"]),
                "rmse_rounded_pooled": float(drow["rounded_rmse"]),
                "exact_accuracy_pooled": float(drow["exact_accuracy"]),
                "selected_alpha_by_fold": json.dumps(alphas),
                "selected_lambdas_by_fold": json.dumps(lambdas),
                "boundary_by_fold": json.dumps(boundaries),
                "fit_seconds_total": fit_seconds,
                "gpu_hours": gpu_hours,
                "metrics_path": str(metrics_path.relative_to(ROOT)) if metrics_path.exists() else None,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["mae_clipped_fold_mean", "model_id"], na_position="last"
    )


def table_or_note(path: Path, columns: List[str] | None = None, max_rows: int = 30) -> str:
    if not path.exists():
        return f"Pending/missing: `{path.relative_to(ROOT)}`"
    frame = pd.read_csv(path)
    if columns:
        frame = frame[[column for column in columns if column in frame.columns]]
    if len(frame) > max_rows:
        frame = frame.head(max_rows)
    return frame.to_markdown(index=False, floatfmt=".6f")


def main() -> None:
    output_dir = ROOT / "outputs"
    phase2 = output_dir / "phase2"
    required = [
        "experiment_s_forensics.json",
        "experiment_c_enforcement.json",
        "experiment_x_results.json",
        "experiment_r_results.json",
        "experiment_p_results.json",
        "experiment_a_results.json",
        "experiment_e_results.json",
        "experiment_d_results.json",
        "experiment_t_results.json",
        "experiment_claim_protocol_sensitivity.json",
    ]
    missing = [name for name in required if not (phase2 / name).exists()]
    if missing:
        raise FileNotFoundError(f"Phase 2 finalization missing: {missing}")
    report_paths = {
        code: phase2 / f"experiment_{code}_report.md"
        for code in ("s", "c", "x", "r", "p", "a", "e", "d", "t")
    }
    missing_reports = [
        path.name for path in report_paths.values() if not path.exists()
    ]
    if missing_reports:
        raise FileNotFoundError(
            f"Phase 2 finalization missing detailed reports: {missing_reports}"
        )

    s = read_json(phase2 / "experiment_s_forensics.json")
    c = read_json(phase2 / "experiment_c_enforcement.json")
    x = read_json(phase2 / "experiment_x_results.json")
    r = read_json(phase2 / "experiment_r_results.json")
    p = read_json(phase2 / "experiment_p_results.json")
    a = read_json(phase2 / "experiment_a_results.json")
    e = read_json(phase2 / "experiment_e_results.json")
    d = read_json(phase2 / "experiment_d_results.json")
    t = read_json(phase2 / "experiment_t_results.json")
    claim = read_json(phase2 / "experiment_claim_protocol_sensitivity.json")

    master = build_master(output_dir)
    master.to_csv(phase2 / "master_model_table.csv", index=False)
    master_records = (
        master.astype(object).where(pd.notna(master), None).to_dict(orient="records")
    )
    write_json(
        phase2 / "master_model_table.json",
        {"models": master_records},
    )

    x_increment = pd.DataFrame(x["interaction_comparisons"])
    x_summary = (
        x_increment.groupby(["source", "pooling"], as_index=False)
        .agg(
            gain_mean=("interaction_gain_z_minus_zint", "mean"),
            gain_std=("interaction_gain_z_minus_zint", "std"),
            gain_median=("interaction_gain_z_minus_zint", "median"),
            gain_min=("interaction_gain_z_minus_zint", "min"),
            gain_max=("interaction_gain_z_minus_zint", "max"),
        )
    )
    r_models = pd.DataFrame(r["models"])
    p_summary = pd.DataFrame(p["summary"])
    a_models = pd.DataFrame(a["models"])
    e2 = pd.DataFrame(e["e2_summary"])
    t1 = pd.read_csv(phase2 / "experiment_t1_lr_sweep.csv")
    claim_comparisons = pd.read_csv(
        phase2 / "experiment_claim_paired_comparisons.csv"
    )
    crossfit_mean_row = x_summary.loc[
        x_summary["source"].eq("crossfit")
        & x_summary["pooling"].eq("mean")
    ]
    if len(crossfit_mean_row) != 1:
        raise ValueError(
            "expected exactly one crossfit/mean row in Experiment X summary"
        )
    crossfit_mean_gain = float(crossfit_mean_row.iloc[0]["gain_mean"])
    crossfit_median_gain = float(crossfit_mean_row.iloc[0]["gain_median"])

    guided = c["guided_decoding"]
    guided_summary = guided.get("summary", {})
    throughput = guided.get("throughput", {})
    r_decision = r["decision_rule"]
    anomalies = [
        {
            "id": "scale_mismatch",
            "finding": "Published Table 2 and Table 4 used different score scales.",
            "evidence": f"Table 2: {s['table2_scale']}; Table 4: {s['table4_scale']}",
        },
        {
            "id": "legacy_json_claim",
            "finding": "Legacy extraction was prompt/repair/retry based, not true guided decoding.",
            "evidence": f"Legacy terminal failures: {c['legacy_audit']['n_non_ok']}; guided sample output: {guided.get('output_path')}",
        },
        {
            "id": "crossfit_gate_instability",
            "finding": "The X gate was triggered by fold 1, but the interaction increment did not generalize consistently.",
            "evidence": (
                f"Fold-1 mean gain={x['pilot']['interaction_gain_z_minus_zint']:.6f}; "
                f"five-fold mean={crossfit_mean_gain:.6f}; "
                f"median={crossfit_median_gain:.6f}."
            ),
        },
        {
            "id": "interaction_refuted",
            "finding": "Under fair reduced R1 block tuning, interaction fusion is worse than simple fusion.",
            "evidence": (
                f"simple-interaction MAE={r_decision['simple_minus_interaction_mae']:.6f}; "
                f"hotel bootstrap CI={r_decision['bootstrap']['ci_95']}; "
                f"Wilcoxon p={r_decision['wilcoxon']['p_value']:.3e}."
            ),
        },
        {
            "id": "negative_control_denominator",
            "finding": "P fractions use a negative signed true gain and cannot be interpreted as recovered positive benefit.",
            "evidence": f"R1b-R1c signed gain={p['true_signed_gain_simple_minus_interaction']:.6f}.",
        },
    ]

    key_ids = [
        "qwen_direct_zero_n100000",
        "roberta_base_regression",
        "roberta_base_phase2_lr5em05_regression",
        "roberta_base_phase2_maxlen512_5em05_regression",
        "bert_base_uncased_regression",
        "text_only_tfidf_ridge",
        "phase2_r0_text_shared_ridge",
        "phase2_r1b_simple_block_ridge",
        "phase2_r1c_interaction_block_ridge",
        "phase2_r2a_interaction_at_text_alpha",
        "phase2_claim_simple_at_text_alpha",
        "llm_aspects_only",
        "upper_bound_true_aspects",
    ]
    key_master = master[master["model_id"].isin(key_ids)][
        [
            "model_id",
            "channel",
            "structured_block",
            "n",
            "mae_clipped_fold_mean",
            "rmse_clipped_fold_mean",
            "mae_rounded_pooled",
            "rmse_rounded_pooled",
            "exact_accuracy_pooled",
            "gpu_hours",
        ]
    ]

    commands = [
        "python scripts/17_phase2_schema_forensics.py",
        "python scripts/18_phase2_schema_enforcement.py --sample-size 50000 --resume",
        "python scripts/19_phase2_crossfit_representations.py --folds 1,2,3,4,5 --poolings mean,cls --resume",
        "python scripts/20_phase2_regularization_forensics.py --mode all --folds 1,2,3,4,5 --inner-jobs 3 --resume",
        "python scripts/21_phase2_controls_ablations.py --stage all --folds 1,2,3,4,5 --resume",
        "python scripts/22_phase2_missingness_error_analysis.py --stage all --folds 1,2,3,4,5 --resume",
        "bash scripts/23b_phase2_t_orchestrator.sh",
        "python scripts/24_finalize_phase2.py",
    ]

    lines = [
        "# Phase 2 — Forensics, Controls, and Cross-fitted Representations",
        "",
        "## Anomalies and narrative-changing results (read first)",
        "",
    ]
    for item in anomalies:
        lines.extend([f"- **{item['finding']}** {item['evidence']}"])
    lines.extend(
        [
            "",
            "## Manuscript claim preservation and protocol sensitivity",
            "",
            "**The current title-level claim remains supported.** On the same canonical 877,808 reviews and saved hotel-grouped splits, interaction fusion significantly improves over text-only under the published protocol, corrected shared-penalty R0, and corrected block-wise R1.",
            "",
            "The stronger statement that interaction fusion is universally the best fusion variant is protocol-sensitive and is not supported by corrected R0/R1. Both findings are retained below; neither result replaces the other.",
            "",
            claim_comparisons[
                [
                    "comparison_id",
                    "protocol",
                    "mae_gain_baseline_minus_candidate",
                    "hotel_bootstrap_ci_low",
                    "hotel_bootstrap_ci_high",
                    "wilcoxon_p",
                ]
            ].to_markdown(index=False, floatfmt=".6f"),
            "",
            f"Recommended wording: {claim['conclusions']['recommended_manuscript_claim']}",
            "",
            f"Common-alpha interpretation: {claim['conclusions']['common_alpha_assessment']}",
            "",
            "## Master result table (selected models)",
            "",
            key_master.to_markdown(index=False, floatfmt=".6f"),
            "",
            f"The complete {len(master)}-model table, including per-fold alpha/lambda and boundary status, is in `outputs/phase2/master_model_table.csv` and `.json`.",
            "",
            "## Experiment S — schema and scale forensics",
            "",
            f"- Cached aspect scores actually use {s['actual_emitted_range']}; the extractor was instructed to emit 1–5: {s['extractor_ever_instructed_1_to_5']}.",
            f"- **Tables used the same scale: {s['tables_same_scale']}.** Table 2 used `{s['table2_scale']}`; Table 4 used `{s['table4_scale']}`.",
            f"- The two feature parameterizations have identical spans: {s['column_span_check']['column_spans_identical']}; ridge differs because the transform is not orthonormal (condition number {s['column_span_check']['transform_condition_number']:.4f}).",
            "- Full source-line evidence and both recomputed tables: `outputs/phase2/experiment_s_report.md`.",
            "",
            "## Experiment C — enforcement audit",
            "",
            f"- Legacy cache: n={c['legacy_audit']['n_cached']:,}, non-ok={c['legacy_audit']['n_non_ok']:,}, OK rate={c['legacy_audit']['ok_rate']:.4%}.",
            f"- Guided 50k output: `{guided.get('output_path')}`; full re-extraction performed: {guided.get('full_reextraction_performed')}.",
            f"- Guided summary: `{json.dumps(guided_summary)}`.",
            f"- Throughput/cost: `{json.dumps(throughput)}`.",
            "- Full agreement and failure taxonomy: `outputs/phase2/experiment_c_report.md`.",
            "",
            "## Experiment X — cross-fitted representations",
            "",
            f"- Fold-1 pilot gain: {x['pilot']['interaction_gain_z_minus_zint']:.6f}; decision: **{x['pilot']['decision']}**.",
            x_summary.to_markdown(index=False, floatfmt=".6f"),
            "",
            "Full train/test R² gaps and coefficient norms: `outputs/phase2/experiment_x_fold_results.csv`.",
            "",
            "## Experiment R — regularization forensics",
            "",
            r_models.to_markdown(index=False, floatfmt=".6f"),
            "",
            f"**Decision rule:** {r_decision['category']}; simple minus interaction={r_decision['simple_minus_interaction_mae']:.6f}. The interaction block does not clear 0.003 and is significantly worse.",
            "",
            "The user-approved reduced protocol screened all 57 alpha candidates on a deterministic hotel-preserved 200k sample, refined locally on full training data, and used two lambda coordinate passes plus local Cartesian refinement. Curves retain stage, iterations, stop code, and boundary status.",
            "",
            "## Experiment P — permutation and dimension controls",
            "",
            p_summary.to_markdown(index=False, floatfmt=".6f"),
            "",
            "Because the true signed interaction gain is negative, `fraction_of_true_signed_gain` is reported only as an algebraic diagnostic.",
            "",
            "## Experiment A — ablations",
            "",
            a_models.to_markdown(index=False, floatfmt=".6f"),
            "",
            "A6 is an identity check: the enforced schema stores score=0 whenever an aspect is unmentioned, so an unmasked latent score is not recoverable.",
            "",
            "## Experiment E — missingness and coverage",
            "",
            "### E2 dose response",
            "",
            e2.to_markdown(index=False, floatfmt=".6f"),
            "",
            f"E4 hotel-level correlation: `{json.dumps(e['e4'])}`.",
            "",
            "Full E1/E3 strata: `outputs/phase2/experiment_e1_segment_strata.csv` and `experiment_e3_mentioned_count.csv`; curve: `experiment_e2_missingness_curve.png`.",
            "",
            "## Experiment D — error distribution and metric behaviour",
            "",
            f"- Models evaluated: {d['n_models']}; skipped: {len(d['skipped'])}.",
            f"- D5: `{json.dumps(d['d5'])}`.",
            "- Full rating-level, tail, confusion, clipped, rounded and exact-accuracy tables: `outputs/phase2/experiment_d1_rating_breakdown.csv` through `experiment_d4_all_metrics.csv`.",
            "",
            "## Experiment T — transformer tuning defensibility",
            "",
            t1.to_markdown(index=False, floatfmt=".6f"),
            "",
            f"- Selected learning rate: {t['t1']['selected_learning_rate']:.1e}; five-fold rerun required: {t['t1']['requires_five_fold_rerun']}.",
            f"- T2 max-length comparison: `{json.dumps(t['t2'])}`.",
            "",
            "## Reproducibility, commands, and artifacts",
            "",
            "- Seed: 42 (control seeds are explicitly recorded).",
            "- Canonical saved outer splits are reused; `offering_id` grouping prevents hotel leakage.",
            "- Every fold prediction is stored under `outputs/predictions/{model_id}/fold{k}.parquet`; aggregate metrics and hotel-level errors are under `outputs/metrics/` and `outputs/per_hotel/`.",
            "- Caches are versioned or model/fold/seed tagged; all long scripts support resume.",
            "",
            "Exact commands:",
            "",
            "```bash",
            *commands,
            "```",
            "",
            "Experiment-specific reports: `outputs/phase2/experiment_{s,c,x,r,p,a,e,d,t}_report.md`.",
            "Claim-preservation audit: `outputs/phase2/experiment_claim_protocol_sensitivity.md` and `.json`.",
            "",
        ]
    )
    lines.extend(
        [
            "## Detailed experiment reports",
            "",
            "The following appendices are embedded from the machine-generated experiment reports so that this final report is self-contained.",
            "",
        ]
    )
    report_names = {
        "s": "S — Schema and scale forensics",
        "c": "C — Schema enforcement audit",
        "x": "X — Cross-fitted representations",
        "r": "R — Regularization forensics",
        "p": "P — Permutation and dimension controls",
        "a": "A — Fair-tuning ablations",
        "e": "E — Missingness and segment coverage",
        "d": "D — Error distribution and metric behaviour",
        "t": "T — Transformer tuning defensibility",
    }
    for code, title in report_names.items():
        detail_lines = report_paths[code].read_text(encoding="utf-8").splitlines()
        if detail_lines and detail_lines[0].startswith("#"):
            detail_lines = detail_lines[1:]
        lines.extend([f"### Experiment {title}", "", *detail_lines, ""])
    claim_detail = (
        phase2 / "experiment_claim_protocol_sensitivity.md"
    ).read_text(encoding="utf-8").splitlines()
    if claim_detail and claim_detail[0].startswith("#"):
        claim_detail = claim_detail[1:]
    lines.extend(
        [
            "### Manuscript claim preservation and protocol sensitivity",
            "",
            *claim_detail,
            "",
        ]
    )

    report_path = output_dir / "phase2_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    handoff_ids = [
        "text_only_tfidf_ridge",
        "text_llm_simple_fusion",
        "interaction_fusion",
        "phase2_r0_text_shared_ridge",
        "phase2_r0_simple_shared_ridge",
        "phase2_r0_interaction_shared_ridge",
        "phase2_r1b_simple_block_ridge",
        "phase2_r1c_interaction_block_ridge",
        "phase2_claim_simple_at_text_alpha",
        "phase2_r2a_interaction_at_text_alpha",
        "roberta_base_regression",
        "roberta_base_phase2_lr5em05_regression",
    ]
    handoff_table = master.loc[
        master["model_id"].isin(handoff_ids),
        [
            "model_id",
            "protocol",
            "n",
            "mae_clipped_fold_mean",
            "mae_clipped_fold_std",
            "rmse_clipped_fold_mean",
            "rmse_clipped_fold_std",
        ],
    ]
    handoff_lines = [
        "# Claude Handoff — Phase 2 Results",
        "",
        "## Paper and defensible central claim",
        "",
        "Paper title: *Interaction fusion of text and LLM-extracted aspects improves hotel rating prediction at scale: A verifiable study on TripAdvisor*.",
        "",
        f"**Recommended claim:** {claim['conclusions']['recommended_manuscript_claim']}",
        "",
        f"**Title assessment:** {claim['conclusions']['title_assessment']}",
        "",
        "Do not replace this with the stronger statement that interaction fusion is universally better than every simple-fusion specification; the corrected experiments do not support that statement.",
        "",
        "## Original and corrected results together",
        "",
        handoff_table.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Protocol-sensitive simple-versus-interaction result",
        "",
        claim_comparisons[
            [
                "comparison_id",
                "mae_gain_baseline_minus_candidate",
                "hotel_bootstrap_ci_low",
                "hotel_bootstrap_ci_high",
                "wilcoxon_p",
            ]
        ].to_markdown(index=False, floatfmt=".6f"),
        "",
        f"- {claim['conclusions']['common_alpha_assessment']}",
        "- The dataset and outer split did not change: n=877,808 and split assignment SHA-256 `dfe3a434fb00c894218309c332627dc0131a2db3cc2cd241727fc712a4803942`.",
        "- Interaction versus text-only remains significantly favorable under the legacy, corrected shared-penalty, and corrected block-wise protocols.",
        "",
        "## Other narrative-changing Phase 2 findings",
        "",
        *[f"- **{item['finding']}** {item['evidence']}" for item in anomalies],
        f"- Tuning selected RoBERTa lr={t['t1']['selected_learning_rate']:.1e}; the rerun materially improves the original RoBERTa result.",
        f"- Segment-missingness E4 correlation is near zero (Pearson r={e['e4']['pearson_r']:.4f}, p={e['e4']['pearson_p']:.4f}); the E2 curve is not monotonically degrading.",
        "",
        "## Authoritative artifacts",
        "",
        "- Full self-contained report: `outputs/phase2_report.md`",
        "- Every-model table: `outputs/phase2/master_model_table.csv` and `.json`",
        "- Claim audit: `outputs/phase2/experiment_claim_protocol_sensitivity.md` and `.json`",
        "- Common-alpha audit: `outputs/phase2/experiment_common_alpha_sensitivity.md` and `.json`",
        "- Experiment reports: `outputs/phase2/experiment_{s,c,x,r,p,a,e,d,t}_report.md`",
        "- Predictions, aggregate metrics, and hotel-level errors: `outputs/predictions/`, `outputs/metrics/`, and `outputs/per_hotel/`",
        "",
    ]
    handoff_path = output_dir / "phase2_claude_handoff.md"
    handoff_path.write_text("\n".join(handoff_lines), encoding="utf-8")
    summary = {
        "report_path": str(report_path.relative_to(ROOT)),
        "claude_handoff_path": str(handoff_path.relative_to(ROOT)),
        "master_table_csv": "outputs/phase2/master_model_table.csv",
        "master_table_json": "outputs/phase2/master_model_table.json",
        "n_master_models": int(len(master)),
        "anomalies": anomalies,
        "r_decision_rule": r_decision,
        "claim_protocol_sensitivity": claim["conclusions"],
        "commands": commands,
    }
    write_json(phase2 / "phase2_summary.json", summary)
    print(f"saved: {report_path}")
    print(f"saved: {handoff_path}")
    print(f"master models: {len(master)}")


if __name__ == "__main__":
    main()
