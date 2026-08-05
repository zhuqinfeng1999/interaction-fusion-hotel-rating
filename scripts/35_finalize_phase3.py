#!/usr/bin/env python3
"""Assemble Phase 3 outputs into the requested audit report and manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
PHASE3 = OUTPUTS / "phase3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Write an explicitly labelled checkpoint report when a required arm is missing.",
    )
    return parser.parse_args()


def read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def mean_sd(value: pd.Series, digits: int = 6) -> str:
    return f"{value.mean():.{digits}f} +/- {value.std(ddof=1):.{digits}f}"


def legacy_corrected_table() -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    sources = [
        ("published/legacy", "text_only", OUTPUTS / "results_baselines.json", "text_only"),
        ("published/legacy", "simple_fusion", OUTPUTS / "results_llm_aspects_qwen25_7b.json", "text_plus_llm"),
        ("published/legacy", "interaction_fusion", OUTPUTS / "results_llm_aspects_qwen25_7b.json", "text_llm_interactions"),
    ]
    for protocol, model, path, key in sources:
        payload = read_json(path)
        if payload is None or key not in payload:
            continue
        node = payload[key]
        if path.name == "results_baselines.json":
            mean = node.get("clipped", {}).get("mean", node.get("mean", {}))
            std = node.get("clipped", {}).get("std", node.get("std", {}))
        else:
            summary = node.get("summary", {}).get("clipped", {})
            mean, std = summary.get("mean", {}), summary.get("std", {})
        rows.append(
            {
                "protocol": protocol,
                "model": model,
                "mae_mean": mean.get("mae"),
                "mae_sd": std.get("mae"),
                "rmse_mean": mean.get("rmse"),
                "rmse_sd": std.get("rmse"),
            }
        )
    corrected = {
        "text_only": "phase2_r0_text_shared_ridge",
        "simple_fusion": "phase2_r0_simple_shared_ridge",
        "interaction_fusion": "phase2_r0_interaction_shared_ridge",
        "interaction_block_penalty": "phase2_r1c_interaction_block_ridge",
        "simple_common_text_alpha": "phase2_claim_simple_at_text_alpha",
        "interaction_common_text_alpha": "phase2_r2a_interaction_at_text_alpha",
    }
    for model, model_id in corrected.items():
        payload = read_json(OUTPUTS / "metrics" / f"{model_id}.json")
        if payload is None:
            continue
        clipped = payload["aggregate"]["clipped"]
        rows.append(
            {
                "protocol": "Phase 2 corrected/sensitivity",
                "model": model,
                "mae_mean": clipped["mean"]["mae"],
                "mae_sd": clipped["std"]["mae"],
                "rmse_mean": clipped["mean"]["rmse"],
                "rmse_sd": clipped["std"]["rmse"],
            }
        )
    return pd.DataFrame(rows)


def v_verdict(payload: Optional[Dict[str, Any]]) -> str:
    if payload is None:
        return "PENDING"
    agreement = pd.DataFrame(payload.get("agreement", []))
    if agreement.empty:
        return "PENDING"
    original = agreement[agreement["condition"].eq("original")]
    full = original[original["field"].eq("full_record")]
    exact = (
        float(full["all_three_exact_rate_all_ok"].mean())
        if len(full)
        else float("nan")
    )
    return (
        "SUPPORTED as an exact rerun-determinism claim"
        if math.isclose(exact, 1.0, abs_tol=1e-12)
        else "NOT SUPPORTED as an exact claim; cache-persisted reproducibility is supported"
    )


def runtime_gpu_hours() -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for path in sorted(
        list(PHASE3.glob("l_guided_runtime_*.json"))
        + list(PHASE3.glob("w_guided_runtime_*.json"))
    ):
        payload = read_json(path) or {}
        wall = payload.get("wall_seconds", payload.get("elapsed_seconds"))
        endpoints = payload.get("n_endpoints") or len(
            payload.get("endpoint_stats", [])
        ) or len(payload.get("endpoints", [])) or 1
        gpu_hours = payload.get("gpu_hours")
        rows.append(
            {
                "stage": path.stem,
                "wall_hours": float(wall) / 3600 if wall is not None else None,
                "gpu_count": endpoints,
                "gpu_hours": (
                    float(gpu_hours)
                    if gpu_hours is not None
                    else float(wall) * endpoints / 3600
                    if wall is not None
                    else None
                ),
                "source": rel(path),
            }
        )
    v_stats = sorted(PHASE3.glob("v_qwen25_7b_*_n20000_seed*.stats.json"))
    for path in v_stats:
        payload = read_json(path) or {}
        wall = payload.get("wall_seconds")
        endpoints = payload.get("n_endpoints", len(payload.get("endpoint_stats", [])) or 2)
        rows.append(
            {
                "stage": path.stem,
                "wall_hours": float(wall) / 3600 if wall is not None else None,
                "gpu_count": endpoints,
                "gpu_hours": float(wall) * endpoints / 3600 if wall is not None else None,
                "source": rel(path),
            }
        )
    return pd.DataFrame(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_manifest(exclude: Iterable[Path]) -> pd.DataFrame:
    excluded = {path.resolve() for path in exclude}
    rows = []
    for path in sorted(PHASE3.rglob("*")):
        if not path.is_file() or path.resolve() in excluded:
            continue
        rows.append(
            {
                "path": rel(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "modified_utc": datetime.fromtimestamp(
                    path.stat().st_mtime, tz=timezone.utc
                ).isoformat(),
            }
        )
    return pd.DataFrame(rows)


def required_status() -> Dict[str, bool]:
    return {
        "G": (PHASE3 / "experiment_g_report.md").exists(),
        "L4": (PHASE3 / "l4_extractor_agreement.json").exists(),
        "L5_L6": (PHASE3 / "l5_downstream_results.json").exists(),
        "V": (PHASE3 / "v_determinism_results.json").exists()
        and any(PHASE3.glob("v_qwen25_7b_original_n20000_seed*.parquet")),
        "X": (PHASE3 / "S1_coefficient_tables.csv").exists(),
    }


def main() -> None:
    args = parse_args()
    status = required_status()
    missing = [name for name, complete in status.items() if not complete]
    if missing and not args.allow_incomplete:
        raise RuntimeError(
            f"required Phase 3 sections incomplete: {missing}; use --allow-incomplete only for a labelled checkpoint"
        )

    g0 = read_json(PHASE3 / "experiment_g0_alpha_audit.json") or {}
    g1_reproduction = read_json(
        PHASE3 / "g1_selected_alpha_reproduction_audit.json"
    ) or {}
    g5 = (
        pd.read_csv(PHASE3 / "g5_summary.csv")
        if (PHASE3 / "g5_summary.csv").exists()
        else pd.DataFrame()
    )
    g23 = (
        pd.read_csv(PHASE3 / "g2_g3_oracle_plateau_selection.csv")
        if (PHASE3 / "g2_g3_oracle_plateau_selection.csv").exists()
        else pd.DataFrame()
    )
    g4 = (
        pd.read_csv(PHASE3 / "g4_sample_size_scaling.csv")
        if (PHASE3 / "g4_sample_size_scaling.csv").exists()
        else pd.DataFrame()
    )
    l4 = read_json(PHASE3 / "l4_extractor_agreement.json")
    l5 = read_json(PHASE3 / "l5_downstream_results.json")
    v = read_json(PHASE3 / "v_determinism_results.json")
    w = read_json(PHASE3 / "w_external_validation_results.json")
    x_meta = read_json(PHASE3 / "S1_coefficient_tables.json")
    context = legacy_corrected_table()
    runtime = runtime_gpu_hours()

    g_robustness = "PENDING"
    if not g5.empty:
        indexed = g5.set_index("model")
        simple = indexed.loc["simple_fusion"]
        interaction = indexed.loc["interaction_fusion"]
        supported = (
            interaction["plateau_width_decades_mean"]
            > simple["plateau_width_decades_mean"] + 0.5
            and interaction["regret_cv3_mean"]
            < simple["regret_cv3_mean"] - 0.0005
        )
        g_robustness = "SUPPORTED" if supported else "NOT SUPPORTED"

    l_verdict = (
        str(l5.get("qualitative_ordering_survives")) if l5 else "PENDING"
    )
    w_verdict = (
        str(w.get("qualitative_ordering_replicates")) if w else "NOT RUN"
    )
    report_state = "FINAL" if not missing else "INCOMPLETE CHECKPOINT"
    lines = [
        f"# Phase 3 Report — {report_state}",
        "",
        f"- generated UTC: {datetime.now(timezone.utc).isoformat()}",
        "- canonical split SHA-256: `dfe3a434fb00c894218309c332627dc0131a2db3cc2cd241727fc712a4803942`",
        f"- completed sections: {status}",
        "",
        "## G0 blocking answer (read first)",
        "",
        "**The original simple-fusion alpha 0.1 was fixed; it was not selected from any grid. Its interior/boundary status is therefore not applicable. The original interaction-fusion alpha 10 was selected at the upper boundary of `[0.01, 0.1, 1, 10]` in all five folds.**",
        "",
        "Source-line audit: `outputs/phase3/experiment_g0_alpha_audit.md`.",
        "",
        "## Anomalies and claim constraints",
        "",
        f"- G robustness contrast (prespecified rule): **{g_robustness}**.",
        f"- Cross-extractor qualitative ordering: **{l_verdict}**.",
        f"- Exact extractor determinism: **{v_verdict(v)}**.",
        f"- External Booking replication: **{w_verdict}**.",
        (
            "- G1 selected-alpha reproduction audit: "
            f"**{g1_reproduction.get('status', 'PENDING')}**; "
            f"max absolute MAE drift={g1_reproduction.get('max_abs_mae_delta', float('nan')):.6g}, "
            f"max absolute RMSE drift={g1_reproduction.get('max_abs_rmse_delta', float('nan')):.6g}. "
            "This numerical LSQR/scheduling drift is retained rather than described as exact reproduction."
        ),
        "- The original/published interaction result and the corrected/sensitivity results are retained side by side. No protocol-sensitive result is silently substituted for another.",
        "",
        "## Original and corrected model results retained together",
        "",
        context.to_markdown(index=False, floatfmt=".6f") if not context.empty else "PENDING",
        "",
        "## Experiment G — regularization response",
        "",
        "G1 curves are outer-test oracle diagnostics, not a permissible model-selection procedure.",
        "",
        g5.to_markdown(index=False, floatfmt=".6f") if not g5.empty else "PENDING",
        "",
        f"**G5 verdict: {g_robustness}.**",
        "",
        "### G2/G3 inner-versus-outer alpha",
        "",
        g23.to_markdown(index=False, floatfmt=".6f") if not g23.empty else "PENDING",
        "",
        "### G4 sample-size scaling (fold 1)",
        "",
        g4.to_markdown(index=False, floatfmt=".6f") if not g4.empty else "PENDING",
        "",
        "## Experiment L — extractor sensitivity",
        "",
    ]
    if l4:
        lines.extend(
            [
                f"- Qwen OK rate: {l4['qwen_ok_rate']:.4%}",
                f"- Llama OK rate: {l4['llama_ok_rate']:.4%}",
                f"- segment kappa: {l4['segment_kappa']:.6f}",
                "",
                "### L4 score agreement",
                "",
                pd.DataFrame(l4["scores"]).to_markdown(index=False, floatfmt=".6f"),
                "",
                "### L4 mention agreement",
                "",
                pd.DataFrame(l4["mentions"]).to_markdown(index=False, floatfmt=".6f"),
                "",
                "Full 6x6 segment matrix and evidence results: `outputs/phase3/experiment_l4_agreement.md`.",
                "",
            ]
        )
    else:
        lines.extend(["PENDING: decoding-matched Llama output is unavailable.", ""])
    if l5:
        lines.extend(
            [
                "### L5 downstream sensitivity",
                "",
                pd.DataFrame(l5["summary"]).to_markdown(index=False, floatfmt=".6f"),
                "",
                f"**Qualitative ordering survives extractor change: {l5['qualitative_ordering_survives']}.**",
                "",
                "L6 and paired hotel tests: `outputs/phase3/experiment_l5_l6_downstream.md`.",
                "",
            ]
        )
    lines.extend(
        [
            "## Experiment V — extractor determinism",
            "",
            f"**V3 verdict: {v_verdict(v)}.**",
            "",
            pd.DataFrame(v.get("agreement", [])).to_markdown(index=False, floatfmt=".6f")
            if v
            else "PENDING",
            "",
            "## Experiment W — external corpus",
            "",
            (
                f"**W3 replication verdict: {w_verdict}.** See `outputs/phase3/experiment_w_report.md`."
                if w
                else "Not run. Source/licence/feasibility must be reported separately; this is not evidence of external replication."
            ),
            "",
            "## Experiment X — coefficient supplements",
            "",
            (
                "Complete coefficient, cluster-robust uncertainty, bootstrap CI, interaction matrix, and goodness-of-fit files are present."
                if x_meta
                else "PENDING"
            ),
            "",
            "- `outputs/phase3/S1_coefficient_tables.csv` and `.md`",
            "- `outputs/phase3/S1_segment_aspect_interactions.csv`",
            "- `outputs/phase3/S1_goodness_of_fit.csv`",
            "",
            "## Compute cost",
            "",
            runtime.to_markdown(index=False, floatfmt=".4f") if not runtime.empty else "PENDING",
            "",
            "## Exact commands and seeds",
            "",
            "```bash",
            "python scripts/27_phase3_g0_alpha_audit.py",
            "python scripts/28_phase3_regularization_curves.py --stage g1 --folds 1,2,3,4,5 --model-jobs 3 --no-resume",
            "python scripts/28_phase3_regularization_curves.py --stage g3 --folds 1,2,3,4,5 --inner-jobs 3 --resume",
            "python scripts/28_phase3_regularization_curves.py --stage g4 --model-jobs 3 --resume",
            "python scripts/28_phase3_regularization_curves.py --stage finalize",
            "python scripts/29_phase3_l_sample.py --sample-size 200000 --seed 42",
            "python scripts/30_phase3_determinism.py --profile qwen25_7b --sample-size 20000 --seeds 42,43,44 --concurrency 16 --resume --endpoints http://localhost:8000/v1,http://localhost:8001/v1",
            "python scripts/31_phase3_guided_extract.py --extractor qwen --profile qwen25_7b --concurrency 16 --resume --endpoints http://localhost:8000/v1,http://localhost:8001/v1",
            "python scripts/31_phase3_guided_extract.py --extractor llama --profile llama31_8b --model /path/to/Llama-3.1-8B-Instruct --concurrency 16 --resume --endpoints http://localhost:8000/v1,http://localhost:8001/v1",
            "python scripts/32_phase3_extractor_agreement.py",
            "python scripts/33_phase3_extractor_downstream.py",
            "python scripts/34_phase3_coefficient_tables.py --bootstrap-resamples 10000 --resume",
            "python scripts/37_phase3_booking_prepare.py --sample-size 200000 --seed 42 --resume",
            "python scripts/31_phase3_guided_extract.py --extractor qwen --profile qwen25_7b --sample-file outputs/phase3/w_sample_n200000_seed42.npz --dataset-file outputs/phase3/booking_dataset_clean.parquet --cache-root outputs/phase3/w_guided_cache --output-tag booking_qwen_guided_n200035 --runtime-prefix w_guided_runtime --experiment W2 --endpoints http://localhost:8000/v1,http://localhost:8001/v1 --concurrency 16 --resume",
            "python scripts/38_phase3_booking_analyze.py --qwen-path outputs/phase3/aspects_guided_booking_qwen_guided_n200035.parquet --max-features 200000 --resume",
            "python scripts/35_finalize_phase3.py",
            "```",
            "",
            "All sampling and bootstrap procedures use seed 42 unless the repeated-extraction seed is explicitly listed as 42/43/44.",
            "",
            "## Artifact index",
            "",
            "The complete size/hash inventory is `outputs/phase3/phase3_artifact_manifest.csv`. Machine-readable report metadata are in `outputs/phase3_report.json`.",
            "",
        ]
    )
    report_path = OUTPUTS / "phase3_report.md"
    json_path = OUTPUTS / "phase3_report.json"
    manifest_path = PHASE3 / "phase3_artifact_manifest.csv"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "report_state": report_state,
        "completion": status,
        "missing": missing,
        "split_sha256": "dfe3a434fb00c894218309c332627dc0131a2db3cc2cd241727fc712a4803942",
        "g0_blocking_answer": {
            "simple_fusion_alpha": 0.1,
            "simple_fusion_selection": "fixed; no grid; boundary not applicable",
            "interaction_alpha": 10.0,
            "interaction_grid": [0.01, 0.1, 1.0, 10.0],
            "interaction_boundary": "upper in all five folds",
        },
        "verdicts": {
            "g_robustness": g_robustness,
            "l_extractor_ordering": l_verdict,
            "v_determinism": v_verdict(v),
            "w_external_replication": w_verdict,
        },
        "legacy_and_corrected_results": context.to_dict(orient="records"),
        "g5": g5.to_dict(orient="records"),
        "g2_g3": g23.to_dict(orient="records"),
        "g4": g4.to_dict(orient="records"),
        "l4": l4,
        "l5_l6": l5,
        "v": v,
        "w": w,
        "x": x_meta,
        "compute_cost": runtime.to_dict(orient="records"),
        "g0_source_audit": g0,
        "g1_selected_alpha_reproduction_audit": g1_reproduction,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    manifest = artifact_manifest([manifest_path])
    manifest.to_csv(manifest_path, index=False)
    print(f"saved: {report_path}", flush=True)
    print(f"saved: {json_path}", flush=True)
    print(f"saved: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
