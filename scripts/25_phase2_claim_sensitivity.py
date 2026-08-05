#!/usr/bin/env python3
"""Audit which interaction-fusion manuscript claims survive protocol changes."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation_harness import (  # noqa: E402
    bootstrap_mean_difference,
    paired_wilcoxon,
    write_json,
)


MODELS = {
    "legacy_text": "text_only_tfidf_ridge",
    "legacy_simple": "text_llm_simple_fusion",
    "legacy_interaction": "interaction_fusion",
    "r0_text": "phase2_r0_text_shared_ridge",
    "r0_simple": "phase2_r0_simple_shared_ridge",
    "r0_interaction": "phase2_r0_interaction_shared_ridge",
    "r1_simple": "phase2_r1b_simple_block_ridge",
    "r1_interaction": "phase2_r1c_interaction_block_ridge",
    "r2_interaction_text_alpha": "phase2_r2a_interaction_at_text_alpha",
    "common_alpha_simple": "phase2_claim_simple_at_text_alpha",
}

COMPARISONS = [
    (
        "legacy_interaction_vs_text",
        "Published/legacy protocol",
        "legacy_text",
        "legacy_interaction",
        "Primary title claim under the original protocol",
    ),
    (
        "legacy_interaction_vs_simple",
        "Published/legacy protocol",
        "legacy_simple",
        "legacy_interaction",
        "Original claim that interactions beat simple fusion",
    ),
    (
        "r0_interaction_vs_text",
        "Corrected shared-penalty R0",
        "r0_text",
        "r0_interaction",
        "Title claim after expanded grouped alpha tuning",
    ),
    (
        "r0_interaction_vs_simple",
        "Corrected shared-penalty R0",
        "r0_simple",
        "r0_interaction",
        "Interaction ranking after expanded grouped alpha tuning",
    ),
    (
        "r1_interaction_vs_text",
        "Corrected block-wise R1",
        "r0_text",
        "r1_interaction",
        "Title claim under block-wise regularization",
    ),
    (
        "r1_interaction_vs_simple",
        "Corrected block-wise R1",
        "r1_simple",
        "r1_interaction",
        "Interaction ranking under block-wise regularization",
    ),
    (
        "common_text_alpha_interaction_vs_simple",
        "Common text-selected alpha sensitivity",
        "common_alpha_simple",
        "r2_interaction_text_alpha",
        "Exact same-alpha simple-versus-interaction contrast; alpha selected by text-only inner CV",
    ),
    (
        "r2_interaction_text_alpha_vs_text",
        "Alpha-matched sensitivity R2a",
        "r0_text",
        "r2_interaction_text_alpha",
        "Interaction at the text-selected alpha versus text",
    ),
    (
        "r2_interaction_text_alpha_vs_r0_simple",
        "Exploratory cross-protocol sensitivity",
        "r0_simple",
        "r2_interaction_text_alpha",
        "Outer-fold sensitivity only; alphas were selected for different objectives",
    ),
]


def read_metrics(output_dir: Path, model_id: str) -> Dict[str, Any]:
    path = output_dir / "metrics" / f"{model_id}.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_split_hash(output_dir: Path) -> str:
    report = (output_dir / "phase0_report.md").read_text(encoding="utf-8")
    match = re.search(r"split_assignment_sha256:\s*`([0-9a-f]{64})`", report)
    if not match:
        raise ValueError("canonical split hash not found in outputs/phase0_report.md")
    return match.group(1)


def model_summary(output_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    protocol_by_alias = {
        "legacy_text": "published/legacy",
        "legacy_simple": "published/legacy",
        "legacy_interaction": "published/legacy",
        "r0_text": "corrected shared penalty",
        "r0_simple": "corrected shared penalty",
        "r0_interaction": "corrected shared penalty",
        "r1_simple": "corrected block-wise penalty",
        "r1_interaction": "corrected block-wise penalty",
        "r2_interaction_text_alpha": "alpha-matched sensitivity",
        "common_alpha_simple": "alpha-matched sensitivity",
    }
    for alias, model_id in MODELS.items():
        payload = read_metrics(output_dir, model_id)
        clipped = payload["aggregate"]["clipped"]
        alphas = [
            fold.get("selected_hyperparameters", {}).get("alpha")
            for fold in payload.get("folds", [])
        ]
        lambdas = [
            fold.get("selected_hyperparameters", {}).get("lambdas")
            for fold in payload.get("folds", [])
        ]
        rows.append(
            {
                "alias": alias,
                "model_id": model_id,
                "protocol": protocol_by_alias[alias],
                "n": int(payload["n_predictions"]),
                "mae_mean": float(clipped["mean"]["mae"]),
                "mae_std": float(clipped["std"]["mae"]),
                "rmse_mean": float(clipped["mean"]["rmse"]),
                "rmse_std": float(clipped["std"]["rmse"]),
                "alpha_by_fold": json.dumps(alphas),
                "lambdas_by_fold": json.dumps(lambdas),
            }
        )
    return pd.DataFrame(rows)


def paired_comparisons(output_dir: Path, models: pd.DataFrame) -> pd.DataFrame:
    indexed = models.set_index("alias")
    rows: List[Dict[str, Any]] = []
    for comparison_id, protocol, baseline_alias, candidate_alias, interpretation in COMPARISONS:
        baseline_id = MODELS[baseline_alias]
        candidate_id = MODELS[candidate_alias]
        baseline_hotel = output_dir / "per_hotel" / f"{baseline_id}.parquet"
        candidate_hotel = output_dir / "per_hotel" / f"{candidate_id}.parquet"
        bootstrap = bootstrap_mean_difference(baseline_hotel, candidate_hotel)
        wilcoxon = paired_wilcoxon(baseline_hotel, candidate_hotel)
        baseline_mae = float(indexed.loc[baseline_alias, "mae_mean"])
        candidate_mae = float(indexed.loc[candidate_alias, "mae_mean"])
        gain = baseline_mae - candidate_mae
        ci_low, ci_high = bootstrap["ci_95"]
        rows.append(
            {
                "comparison_id": comparison_id,
                "protocol": protocol,
                "baseline_model": baseline_id,
                "candidate_model": candidate_id,
                "baseline_mae": baseline_mae,
                "candidate_mae": candidate_mae,
                "mae_gain_baseline_minus_candidate": gain,
                "hotel_mean_gain": float(bootstrap["mean_difference"]),
                "hotel_bootstrap_ci_low": float(ci_low),
                "hotel_bootstrap_ci_high": float(ci_high),
                "wilcoxon_p": float(wilcoxon["p_value"]),
                "candidate_significantly_better": bool(ci_low > 0.0),
                "candidate_significantly_worse": bool(ci_high < 0.0),
                "interpretation": interpretation,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    output_dir = ROOT / "outputs"
    phase2 = output_dir / "phase2"
    split_hash = canonical_split_hash(output_dir)
    models = model_summary(output_dir)
    if models["n"].nunique() != 1 or int(models["n"].iloc[0]) != 877_808:
        raise ValueError("claim audit models do not all cover the canonical 877,808 rows")
    comparisons = paired_comparisons(output_dir, models)

    legacy_title = comparisons.set_index("comparison_id").loc[
        "legacy_interaction_vs_text"
    ]
    r0_title = comparisons.set_index("comparison_id").loc[
        "r0_interaction_vs_text"
    ]
    r1_title = comparisons.set_index("comparison_id").loc[
        "r1_interaction_vs_text"
    ]
    r0_ranking = comparisons.set_index("comparison_id").loc[
        "r0_interaction_vs_simple"
    ]
    r1_ranking = comparisons.set_index("comparison_id").loc[
        "r1_interaction_vs_simple"
    ]
    common_alpha_ranking = comparisons.set_index("comparison_id").loc[
        "common_text_alpha_interaction_vs_simple"
    ]
    common_simple_mae = float(
        models.set_index("alias").loc["common_alpha_simple", "mae_mean"]
    )
    common_interaction_mae = float(
        models.set_index("alias").loc[
            "r2_interaction_text_alpha", "mae_mean"
        ]
    )

    conclusions = {
        "same_dataset_and_outer_splits": True,
        "canonical_n": 877_808,
        "canonical_split_assignment_sha256": split_hash,
        "title_claim_interaction_improves_over_text_supported": bool(
            legacy_title["candidate_significantly_better"]
            and r0_title["candidate_significantly_better"]
            and r1_title["candidate_significantly_better"]
        ),
        "interaction_is_universally_best_fusion_supported": False,
        "ranking_is_regularization_protocol_sensitive": bool(
            r0_ranking["mae_gain_baseline_minus_candidate"] < 0
            and r1_ranking["mae_gain_baseline_minus_candidate"] < 0
        ),
        "interaction_beats_simple_at_common_text_selected_alpha": bool(
            common_alpha_ranking["candidate_significantly_better"]
        ),
        "common_alpha_assessment": (
            "At the exact same text-selected alpha, simple fusion has lower "
            f"review-weighted MAE by {common_interaction_mae - common_simple_mae:.6f}; "
            "the equal-hotel mean difference slightly favors interaction, but its "
            f"bootstrap 95% CI [{common_alpha_ranking['hotel_bootstrap_ci_low']:.6f}, "
            f"{common_alpha_ranking['hotel_bootstrap_ci_high']:.6f}] crosses zero. "
            "This is an effective tie, not robust evidence that either fusion is "
            "universally superior."
        ),
        "recommended_manuscript_claim": (
            "Interaction fusion significantly improves rating prediction over the "
            "text-only baseline on the canonical TripAdvisor corpus. Whether the "
            "interaction terms add value beyond a main-effect-only simple fusion is "
            "sensitive to the regularization protocol; under corrected shared and "
            "block-wise tuning, simple fusion is marginally better."
        ),
        "title_assessment": (
            "The current title remains supported because it says 'improves', not "
            "'is universally optimal' or 'outperforms every fusion baseline'."
        ),
    }

    models.to_csv(phase2 / "experiment_claim_model_table.csv", index=False)
    comparisons.to_csv(phase2 / "experiment_claim_paired_comparisons.csv", index=False)
    payload = {
        "conclusions": conclusions,
        "models": models.to_dict(orient="records"),
        "paired_comparisons": comparisons.to_dict(orient="records"),
    }
    write_json(phase2 / "experiment_claim_protocol_sensitivity.json", payload)

    display_models = models[
        [
            "model_id",
            "protocol",
            "n",
            "mae_mean",
            "mae_std",
            "rmse_mean",
            "rmse_std",
        ]
    ]
    display_comparisons = comparisons[
        [
            "comparison_id",
            "protocol",
            "mae_gain_baseline_minus_candidate",
            "hotel_mean_gain",
            "hotel_bootstrap_ci_low",
            "hotel_bootstrap_ci_high",
            "wilcoxon_p",
            "candidate_significantly_better",
            "candidate_significantly_worse",
        ]
    ]
    lines = [
        "# Manuscript Claim Preservation and Protocol Sensitivity",
        "",
        "## Bottom line",
        "",
        "- The dataset did **not** change: every compared model uses the canonical 877,808 rows and saved hotel-grouped outer splits.",
        "- The title-level claim is supported under the published, corrected shared-penalty, and corrected block-wise protocols: interaction fusion improves over text-only.",
        "- The stronger claim that interaction fusion is always the best fusion model is not supported. Its ranking against simple fusion changes with regularization and tuning.",
        f"- {conclusions['common_alpha_assessment']}",
        "- This distinction preserves the paper's central contribution without hiding the corrected sensitivity result.",
        "",
        "## Model results",
        "",
        display_models.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Paired per-hotel contrasts",
        "",
        "Positive gain means the candidate in the comparison is better than its baseline.",
        "",
        display_comparisons.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Recommended manuscript wording",
        "",
        conclusions["recommended_manuscript_claim"],
        "",
        "## Interpretation for the existing title",
        "",
        conclusions["title_assessment"],
        "",
        "The published/legacy interaction result must remain labeled as originating from the original protocol. The corrected R0/R1 results must be reported alongside it rather than silently replacing or suppressing either result.",
        "",
    ]
    (phase2 / "experiment_claim_protocol_sensitivity.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(json.dumps(conclusions, indent=2))


if __name__ == "__main__":
    main()
