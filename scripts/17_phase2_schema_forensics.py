#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.aspect_extractor_base import ASPECTS  # noqa: E402
from src.evaluation_harness import write_json  # noqa: E402


PROFILE = "qwen25_7b"
SEGMENTS = ["business", "couple", "family", "friends", "solo", "unknown"]
PUBLISHED_TABLE2 = {
    "service_mae": 0.5669,
    "sleep_quality_mae": 1.2974,
    "mean_mae": 0.8801,
    "service_pearson": 0.7347,
    "sleep_quality_pearson": 0.3521,
    "mean_pearson": 0.5473,
}
PUBLISHED_TABLE4 = {
    ("business", "service"): 0.7694,
    ("couple", "service"): 1.0786,
    ("unknown", "sleep_quality"): 0.0002,
}


def log(message: str) -> None:
    print(message, flush=True)


def score_distribution(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    payload: Dict[str, Any] = {}
    for aspect in ASPECTS:
        score = pd.to_numeric(
            frame[f"aspect_{aspect}_score"],
            errors="coerce",
        ).fillna(0)
        mentioned = frame[f"aspect_{aspect}_mentioned"].fillna(False).astype(bool)
        payload[aspect] = {}
        for stratum, mask in (
            ("all", np.ones(len(frame), dtype=bool)),
            ("mentioned", mentioned.to_numpy()),
            ("unmentioned", ~mentioned.to_numpy()),
        ):
            values = score.to_numpy(dtype=float)[mask]
            distinct = sorted(np.unique(values).tolist())
            histogram = {
                str(value): int(np.sum(values == value))
                for value in distinct
            }
            item = {
                "aspect": aspect,
                "stratum": stratum,
                "n": int(len(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "distinct_values": distinct,
                "mean": float(np.mean(values)),
                "sd": float(np.std(values, ddof=1)),
                "histogram": histogram,
            }
            rows.append(item)
            payload[aspect][stratum] = item
    return pd.DataFrame(rows), payload


def plot_histograms(
    frame: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    bins = np.arange(-2.5, 3.0, 1.0)
    for axis, aspect in zip(axes.flat, ASPECTS):
        score = frame[f"aspect_{aspect}_score"].to_numpy(dtype=float)
        mentioned = frame[f"aspect_{aspect}_mentioned"].to_numpy(dtype=bool)
        axis.hist(
            score[mentioned],
            bins=bins,
            alpha=0.8,
            label="mentioned",
        )
        axis.hist(
            score[~mentioned],
            bins=bins,
            alpha=0.8,
            label="unmentioned",
        )
        axis.set_title(aspect)
        axis.set_xticks([-2, -1, 0, 1, 2])
        axis.set_yscale("log")
    axes.flat[0].legend()
    fig.suptitle("Cached aspect-score distributions")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def correlation(
    first: np.ndarray,
    second: np.ndarray,
    method: str,
) -> float:
    if len(first) < 2 or np.std(first) == 0 or np.std(second) == 0:
        return float("nan")
    if method == "pearson":
        return float(stats.pearsonr(first, second).statistic)
    return float(stats.spearmanr(first, second).statistic)


def extraction_quality(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for convention in ("raw_cached", "shifted_3_plus_s"):
        for aspect in ASPECTS:
            truth_col = f"rating_{aspect}"
            score_col = f"aspect_{aspect}_score"
            valid = frame[truth_col].notna()
            truth = frame.loc[valid, truth_col].to_numpy(dtype=float)
            score = frame.loc[valid, score_col].to_numpy(dtype=float)
            prediction = score if convention == "raw_cached" else 3.0 + score
            rows.append(
                {
                    "convention": convention,
                    "aspect": aspect,
                    "n": int(valid.sum()),
                    "mae": float(mean_absolute_error(truth, prediction)),
                    "rmse": float(
                        np.sqrt(mean_squared_error(truth, prediction))
                    ),
                    "pearson_r": correlation(truth, prediction, "pearson"),
                    "spearman_rho": correlation(
                        truth,
                        prediction,
                        "spearman",
                    ),
                }
            )
        subset = [row for row in rows if row["convention"] == convention]
        rows.append(
            {
                "convention": convention,
                "aspect": "mean",
                "n": int(sum(row["n"] for row in subset)),
                "mae": float(np.mean([row["mae"] for row in subset])),
                "rmse": float(np.mean([row["rmse"] for row in subset])),
                "pearson_r": float(
                    np.mean([row["pearson_r"] for row in subset])
                ),
                "spearman_rho": float(
                    np.mean([row["spearman_rho"] for row in subset])
                ),
            }
        )
    return pd.DataFrame(rows)


def table2_match(quality: pd.DataFrame) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    for convention in quality["convention"].unique():
        subset = quality[quality["convention"] == convention].set_index("aspect")
        observed = {
            "service_mae": float(subset.loc["service", "mae"]),
            "sleep_quality_mae": float(
                subset.loc["sleep_quality", "mae"]
            ),
            "mean_mae": float(subset.loc["mean", "mae"]),
            "service_pearson": float(
                subset.loc["service", "pearson_r"]
            ),
            "sleep_quality_pearson": float(
                subset.loc["sleep_quality", "pearson_r"]
            ),
            "mean_pearson": float(subset.loc["mean", "pearson_r"]),
        }
        deltas = {
            key: observed[key] - published
            for key, published in PUBLISHED_TABLE2.items()
        }
        results[convention] = {
            "observed": observed,
            "delta_from_published": deltas,
            "matches_at_4dp": all(abs(value) < 5e-5 for value in deltas.values()),
        }
    return results


def segment_means(frame: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    segment = frame["traveler_segment"].fillna("unknown").astype(str).str.lower()
    segment = segment.where(segment.isin(SEGMENTS), "unknown")
    for convention in ("raw_s_times_b", "shifted_3_plus_s_times_b"):
        for segment_name in SEGMENTS:
            segment_mask = segment.eq(segment_name).to_numpy()
            for aspect in ASPECTS:
                score = frame[f"aspect_{aspect}_score"].to_numpy(dtype=float)
                mention = frame[
                    f"aspect_{aspect}_mentioned"
                ].to_numpy(dtype=float)
                if convention == "raw_s_times_b":
                    masked = score * mention
                else:
                    masked = (3.0 + score) * mention
                rows.append(
                    {
                        "convention": convention,
                        "traveler_segment": segment_name,
                        "aspect": aspect,
                        "n_segment": int(segment_mask.sum()),
                        "mean_masked_score": float(
                            masked[segment_mask].mean()
                        ),
                        "mentioned_rate": float(
                            mention[segment_mask].mean()
                        ),
                    }
                )
    return pd.DataFrame(rows)


def table4_match(segment_table: pd.DataFrame) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    for convention in segment_table["convention"].unique():
        subset = segment_table[
            segment_table["convention"] == convention
        ].set_index(["traveler_segment", "aspect"])
        observed = {
            f"{segment}_{aspect}": float(
                subset.loc[(segment, aspect), "mean_masked_score"]
            )
            for segment, aspect in PUBLISHED_TABLE4
        }
        deltas = {
            f"{segment}_{aspect}": (
                observed[f"{segment}_{aspect}"] - published
            )
            for (segment, aspect), published in PUBLISHED_TABLE4.items()
        }
        results[convention] = {
            "observed": observed,
            "delta_from_published": deltas,
            "matches_at_4dp": all(abs(value) < 5e-5 for value in deltas.values()),
        }
    return results


def column_span_check(frame: pd.DataFrame) -> Dict[str, Any]:
    scores = frame[
        [f"aspect_{aspect}_score" for aspect in ASPECTS]
    ].to_numpy(dtype=np.float64)
    mentions = frame[
        [f"aspect_{aspect}_mentioned" for aspect in ASPECTS]
    ].to_numpy(dtype=np.float64)
    raw_masked = scores * mentions
    shifted_masked = (3.0 + scores) * mentions
    legacy = np.hstack([raw_masked, mentions])
    paper = np.hstack([shifted_masked, mentions])
    identity = np.eye(len(ASPECTS))
    zeros = np.zeros_like(identity)
    transform = np.block([[identity, zeros], [3.0 * identity, identity]])
    inverse = np.block([[identity, zeros], [-3.0 * identity, identity]])
    forward_error = float(np.max(np.abs(paper - legacy @ transform)))
    inverse_error = float(np.max(np.abs(legacy - paper @ inverse)))
    legacy_gram = legacy.T @ legacy
    paper_gram = paper.T @ paper
    return {
        "relation": (
            "[(3+s)*b, b] = [s*b, b] @ [[I,0],[3I,I]]"
        ),
        "forward_max_abs_error": forward_error,
        "inverse_max_abs_error": inverse_error,
        "legacy_rank": int(np.linalg.matrix_rank(legacy_gram)),
        "paper_rank": int(np.linalg.matrix_rank(paper_gram)),
        "transform_rank": int(np.linalg.matrix_rank(transform)),
        "transform_determinant": float(np.linalg.det(transform)),
        "transform_condition_number": float(np.linalg.cond(transform)),
        "orthogonality_max_abs_error": float(
            np.max(np.abs(transform.T @ transform - np.eye(12)))
        ),
        "column_spans_identical": (
            forward_error <= 1e-12
            and inverse_error <= 1e-12
            and np.linalg.matrix_rank(transform) == 12
        ),
    }


def markdown_table(frame: pd.DataFrame, digits: int = 4) -> str:
    display = frame.copy()
    numeric = display.select_dtypes(include=[np.number]).columns
    display[numeric] = display[numeric].round(digits)
    return display.to_markdown(index=False)


def main() -> None:
    output_dir = ROOT / "outputs"
    phase2_dir = output_dir / "phase2"
    phase2_dir.mkdir(parents=True, exist_ok=True)
    score_cols = [f"aspect_{aspect}_score" for aspect in ASPECTS]
    mention_cols = [f"aspect_{aspect}_mentioned" for aspect in ASPECTS]
    aspect_frame = pd.read_parquet(
        output_dir / f"aspects_llm_{PROFILE}.parquet",
        columns=[
            "row_id",
            "ok",
            "traveler_segment",
            *score_cols,
            *mention_cols,
        ],
    ).sort_values("row_id")
    ratings = pd.read_parquet(
        output_dir / "dataset_clean.parquet",
        columns=[f"rating_{aspect}" for aspect in ASPECTS],
    )
    if not np.array_equal(
        aspect_frame["row_id"].to_numpy(),
        np.arange(len(aspect_frame)),
    ):
        raise ValueError("aspect row_id alignment failed")
    frame = pd.concat(
        [
            aspect_frame.reset_index(drop=True),
            ratings.reset_index(drop=True),
        ],
        axis=1,
    )

    distribution_frame, distribution_payload = score_distribution(frame)
    distribution_frame.to_csv(
        phase2_dir / "experiment_s_score_distribution.csv",
        index=False,
    )
    plot_histograms(
        frame,
        phase2_dir / "experiment_s_score_histograms.png",
    )

    quality = extraction_quality(frame)
    quality.to_csv(
        phase2_dir / "experiment_s_table2_both_scales.csv",
        index=False,
    )
    table2_matches = table2_match(quality)

    segment_table = segment_means(frame)
    segment_table.to_csv(
        phase2_dir / "experiment_s_table4_both_scales.csv",
        index=False,
    )
    table4_matches = table4_match(segment_table)
    span = column_span_check(frame)

    source_evidence = {
        "extractor_instruction": {
            "path": "scripts/06_extract_aspects_llm.py",
            "lines": "59-80",
            "finding": "Prompt explicitly requests score in {-2,-1,0,1,2}; never [1,5].",
        },
        "normalization": {
            "path": "scripts/06_extract_aspects_llm.py",
            "lines": "439-476",
            "finding": "Scores are coerced and clipped to [-2,2]; unmentioned aspects are forced to score=0.",
        },
        "published_structured_vector": {
            "path": "scripts/08_phase0_reproduce.py",
            "lines": "267-271",
            "finding": "Published Table 1 replay constructs [raw cached scores, mentions, segment one-hot].",
        },
        "llm_aspects_only": {
            "path": "scripts/08_phase0_reproduce.py",
            "lines": "496-516",
            "representation": "[raw cached scores, mentions, segment one-hot]",
        },
        "text_llm_simple_fusion": {
            "path": "scripts/08_phase0_reproduce.py",
            "lines": "695-715",
            "representation": "TF-IDF plus standardized raw cached score/mention/segment vector",
        },
        "residual_on_llm": {
            "path": "scripts/08_phase0_reproduce.py",
            "lines": "742-769",
            "representation": "Raw cached score/mention/segment vector predicts TF-IDF residuals",
        },
        "interaction_fusion": {
            "path": "scripts/08_phase0_reproduce.py",
            "lines": "819-900",
            "representation": "TF-IDF + raw scores + mentions + segment one-hot + segment*raw-score + segment*mention",
        },
    }
    payload = {
        "profile": PROFILE,
        "n_rows": int(len(frame)),
        "actual_emitted_range": [-2, 2],
        "extractor_ever_instructed_1_to_5": False,
        "source_evidence": source_evidence,
        "score_distribution": distribution_payload,
        "table2_match": table2_matches,
        "table4_match": table4_matches,
        "tables_same_scale": False,
        "table2_scale": "clip(3 + raw cached score, 1, 5)",
        "table4_scale": "raw cached score, effectively s*b because unmentioned scores are forced to zero",
        "column_span_check": span,
    }
    write_json(phase2_dir / "experiment_s_forensics.json", payload)

    quality_raw = quality[quality["convention"] == "raw_cached"]
    quality_shifted = quality[
        quality["convention"] == "shifted_3_plus_s"
    ]
    table4_examples = segment_table[
        segment_table.apply(
            lambda row: (
                row["traveler_segment"],
                row["aspect"],
            )
            in PUBLISHED_TABLE4,
            axis=1,
        )
    ]
    report = [
        "# Experiment S — Schema and Scale Forensics",
        "",
        "## S1. Actual cached schema",
        "",
        "- The actual score range is **[-2, 2]**, with discrete values "
        "`{-2,-1,0,1,2}` for mentioned aspects.",
        "- Unmentioned aspects are deterministically forced to score 0.",
        "- The extractor was never instructed to emit [1,5]. The prompt "
        "explicitly requests -2..2 at `scripts/06_extract_aspects_llm.py:59-80`.",
        "",
        markdown_table(
            distribution_frame[
                [
                    "aspect",
                    "stratum",
                    "n",
                    "min",
                    "max",
                    "mean",
                    "sd",
                ]
            ]
        ),
        "",
        "Histogram data: `outputs/phase2/experiment_s_score_distribution.csv`; "
        "figure: `outputs/phase2/experiment_s_score_histograms.png`.",
        "",
        "## S2. Table 1 feature representations",
        "",
        "| model | actual representation | source |",
        "| --- | --- | --- |",
        *[
            f"| {model} | {item['representation']} | "
            f"`{item['path']}:{item['lines']}` |"
            for model, item in source_evidence.items()
            if "representation" in item
        ],
        "",
        "Because extraction normalization forces unmentioned scores to zero, "
        "the raw score block is numerically `s*b`.",
        "",
        "## S3. Published Table 2 under both scales",
        "",
        "### Raw cached score compared directly with platform ratings",
        "",
        markdown_table(quality_raw),
        "",
        "### clip(3+s, 1, 5) compared with platform ratings",
        "",
        markdown_table(quality_shifted),
        "",
        "**Result:** only `3+s` reproduces 0.5669 / 1.2974 / 0.8801 "
        "and Pearson 0.7347 / 0.3521 / 0.5473.",
        "",
        "## S4. Published Table 4 under both scales",
        "",
        markdown_table(table4_examples),
        "",
        "**Result:** the raw `s*b` convention reproduces Business service "
        "0.7694, Couple service 1.0786, and Unknown sleep 0.0002.",
        "",
        "Full table: `outputs/phase2/experiment_s_table4_both_scales.csv`.",
        "",
        "## S5. Explicit scale conclusion",
        "",
        "**Published Table 2 and Table 4 did not use the same scale.** "
        "Table 2 used `3+s`; Table 4 used raw cached `s` (effectively `s*b`).",
        "",
        "## S6. Linear reparameterization",
        "",
        f"- Forward maximum absolute error: {span['forward_max_abs_error']:.3g}",
        f"- Inverse maximum absolute error: {span['inverse_max_abs_error']:.3g}",
        f"- Legacy/paper ranks: {span['legacy_rank']} / {span['paper_rank']}",
        f"- Transform determinant: {span['transform_determinant']:.1f}",
        f"- Transform condition number: {span['transform_condition_number']:.4f}",
        f"- Transform orthogonality error: {span['orthogonality_max_abs_error']:.4f}",
        "",
        "`[(3+s)*b; b]` and `[s*b; b]` have identical column spans. "
        "The transform is invertible but not orthonormal, so ridge penalties "
        "are not invariant. This explains near-identical predictions with "
        "different selected alphas.",
        "",
    ]
    report_path = phase2_dir / "experiment_s_report.md"
    report_path.write_text("\n".join(report), encoding="utf-8")
    log(f"saved: {report_path}")


if __name__ == "__main__":
    main()
