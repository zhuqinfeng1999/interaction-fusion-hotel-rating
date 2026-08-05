#!/usr/bin/env python3
"""Build sanitized, upload-ready GitHub and OSF release trees for Phase 6B.

CPU only: this script reads accepted artefacts, copies/derives release files, and
performs privacy/provenance checks. It never calls an LLM and never fits a model.
"""
from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from array import array
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from openpyxl import load_workbook


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
RELEASE = ROOT / "release"
STAGING = RELEASE / ".phase6b_staging"
GITHUB = STAGING / "github"
OSF = STAGING / "osf"
SPLIT_SHA256 = "dfe3a434fb00c894218309c332627dc0131a2db3cc2cd241727fc712a4803942"
SEED = 42
N_ROWS = 877_808
ASPECTS = ("service", "cleanliness", "location", "rooms", "value", "sleep_quality")
SEGMENTS = {"business", "couple", "family", "solo", "friends", "unknown"}

PANEL_A = (
    "text_only_tfidf_ridge",
    "llm_aspects_only",
    "text_llm_simple_fusion",
    "residual_on_llm",
    "upper_bound_true_aspects",
    "interaction_fusion",
)
PANEL_B = (
    "phase4_u0_text_only_tfidf_ridge",
    "phase4_u0_llm_aspects_only",
    "phase4_u0_text_llm_simple_fusion",
    "phase4_u0_residual_on_llm",
    "phase4_u0_upper_bound_true_aspects",
    "phase4_u0_interaction_fusion",
)
NEURAL = (
    "roberta_base_phase2_lr5em05_regression",
    "bert_base_uncased_regression",
    "bert_base_uncased_cls_plus_zint_paper",
    "qwen_direct_zero_n100000",
)
ABLATIONS = (
    "phase4_u0_a2_score_interactions_only",
    "phase4_u0_a3_mention_interactions_only",
    "phase4_u0_a5_interactions_without_main",
    "phase4_u0_a7_unknown_columns_removed",
    "phase4_u0_a8_known_segment_only",
    "phase4_u0_a4_known_segment_reference",
)
PROTOCOL_SENSITIVITY = (
    "phase2_r0_text_shared_ridge",
    "phase2_r0_simple_shared_ridge",
    "phase2_r0_interaction_shared_ridge",
    "phase2_r1b_simple_block_ridge",
    "phase2_r1c_interaction_block_ridge",
    "phase2_r2a_interaction_at_text_alpha",
    "phase2_claim_simple_at_text_alpha",
)
BOOKING_MODELS = (
    "phase4_booking_u0_text_only",
    "phase4_booking_u0_simple_fusion",
    "phase4_booking_u0_interaction_fusion",
)
L5_SOURCE_MODELS = (
    "text_only",
    "qwen_simple",
    "qwen_interaction",
    "llama_simple",
    "llama_interaction",
    "ensemble_simple",
    "ensemble_interaction",
)
L5_RELEASE_MODELS = tuple(f"l5_{model}" for model in L5_SOURCE_MODELS)
TRIP_FULL_MODELS = PANEL_A + PANEL_B + NEURAL[:3] + ABLATIONS + PROTOCOL_SENSITIVITY
DIRECT_SUBSAMPLE_MODEL = NEURAL[3]
DIRECTORY_MODEL_FILTER = set(TRIP_FULL_MODELS) | {DIRECT_SUBSAMPLE_MODEL} | set(BOOKING_MODELS)
ALL_RELEASE_MODEL_IDS = set(TRIP_FULL_MODELS) | set(BOOKING_MODELS) | {DIRECT_SUBSAMPLE_MODEL} | set(L5_RELEASE_MODELS)
NULLABLE_TRIP_MODELS = {
    "phase4_u0_a8_known_segment_only",
    "phase4_u0_a4_known_segment_reference",
}

RETAINED_SCRIPTS = {
    "00_sanity_check.py": "raw-file schema check",
    "01_prepare_dataset.py": "TripAdvisor preprocessing used by every analysis",
    "02_make_splits.py": "original grouped split construction",
    "03_run_baselines.py": "published TF-IDF and true-aspect baselines",
    "03b_error_analysis_baselines.py": "reported baseline error diagnostics",
    "06_extract_aspects_llm.py": "deployed Qwen extraction and cache protocol",
    "07_run_llm_aspects_analysis.py": "published legacy fusion and extraction-quality analysis",
    "08_phase0_reproduce.py": "published-protocol reproduction gate",
    "09_train_transformer_regression.py": "reported neural text baselines",
    "13_run_llm_direct_rating.py": "reported zero-shot direct-LLM baseline",
    "14_summarize_phase1.py": "neural/direct baseline consolidation",
    "17_phase2_schema_forensics.py": "schema-validity forensic analysis",
    "18_phase2_schema_enforcement.py": "true guided-JSON validation",
    "20_phase2_regularization_forensics.py": "reported protocol-sensitivity models",
    "20b_phase2_r_orchestrator.sh": "orchestrator for reported protocol sensitivity",
    "21_phase2_controls_ablations.py": "feature and fitting utilities imported by common-alpha script 26",
    "23_phase2_transformer_tuning.py": "reported transformer tuning audit",
    "23b_phase2_t_orchestrator.sh": "orchestrator for reported transformer tuning",
    "24_finalize_phase2.py": "Phase 2 reported-table consolidation",
    "25_phase2_claim_sensitivity.py": "reported claim/protocol sensitivity",
    "26_phase2_common_alpha_sensitivity.py": "reported common-alpha sensitivity",
    "27_phase3_g0_alpha_audit.py": "published-alpha provenance audit",
    "28_phase3_regularization_curves.py": "615-row alpha-response diagnostic",
    "28b_validate_phase3_g1.py": "alpha-curve validation gate",
    "29_phase3_l_sample.py": "canonical extractor-sensitivity sample",
    "30_phase3_determinism.py": "three-seed extraction determinism study",
    "31_phase3_guided_extract.py": "guided extraction for sensitivity/external validation",
    "32_phase3_extractor_agreement.py": "Qwen-Llama agreement and evidence grounding",
    "33_phase3_extractor_downstream.py": "seven reported L5 downstream models",
    "34_phase3_coefficient_tables.py": "statistical inference utilities imported by U0 coefficient script 46",
    "35_finalize_phase3.py": "Phase 3 reported-table consolidation",
    "37_phase3_booking_prepare.py": "external-corpus acquisition and preprocessing",
    "38_phase3_booking_analyze.py": "initial Booking external validation",
    "39_phase3_w_after_qwen.sh": "Booking extraction/analysis orchestrator",
    "40_phase3_llama_after_qwen.sh": "Llama sensitivity orchestrator",
    "41_phase4_u0_models.py": "primary U0 models and manuscript ablations",
    "42_phase4_booking_u0.py": "U0 external validation",
    "43_phase4_consolidate.py": "manuscript-ready Phase 4 tables",
    "44_phase5_discrepancy_strata.py": "published/simple-fusion discrepancy and U0 strata",
    "45_phase5_e2_u0_missingness.py": "U0 segment-masking curve",
    "46_phase5_u0_coefficient_tables.py": "complete U0 regression inference",
    "47_finalize_phase5.py": "Phase 5 consistency report",
    "48_phase6a_assemble.py": "Supporting Information workbook assembly",
    "49_phase6b_prepare_release.py": "sanitized public-release assembly and verification",
}

EXCLUDED_SCRIPT_REASONS = {
    "04_aspect_pipeline_dummy.py": "dummy extractor smoke test; not in manuscript",
    "05_segment_heterogeneity.py": "dummy-extractor exploratory analysis",
    "10_extract_transformer_embeddings.py": "frozen-representation experiment excluded from manuscript",
    "11_run_representation_fusion.py": "frozen/SVD representation variants excluded from manuscript",
    "12_extract_tfidf_svd_embeddings.py": "TF-IDF+SVD channel excluded from manuscript",
    "15_train_end_to_end_fusion.py": "end-to-end fusion variants excluded from manuscript",
    "16_run_phase1_remaining.sh": "orchestrates excluded representation/end-to-end variants",
    "19_phase2_crossfit_representations.py": "cross-fitted representation experiment excluded from manuscript",
    "19b_phase2_gpu_orchestrator.sh": "orchestrates excluded cross-fitted representations",
    "21b_phase2_pa_orchestrator.sh": "orchestrates excluded controls",
    "22_phase2_missingness_error_analysis.py": "superseded by U0-consistent Phase 5 scripts 44/45",
    "22b_phase2_ed_orchestrator.sh": "orchestrates superseded missingness analysis",
    "24b_phase2_finalize_orchestrator.sh": "depends on the superseded Phase 2 missingness script",
    "36_phase3_orchestrate_remaining.sh": "depends on superseded Phase 3 coefficient script 34",
}

FORBIDDEN_DATA_COLUMNS = {
    "review_text", "review_title", "title", "text", "response_json", "raw_text",
    "evidence", "evidence_text", "overall_summary", "prompt", "prompt_text",
    "system_prompt", "user_prompt",
}


def log(message: str) -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conda-env", default="zqf_pytorch")
    parser.add_argument("--skip-offsets", action="store_true", help="Development only; final verification will fail.")
    return parser.parse_args()


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def copy_file(source: Path, destination: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def safe_reset_staging() -> None:
    RELEASE.mkdir(parents=True, exist_ok=True)
    if STAGING.exists():
        resolved = STAGING.resolve()
        if resolved.parent != RELEASE.resolve() or resolved.name != ".phase6b_staging":
            raise RuntimeError(f"refusing unsafe staging removal: {resolved}")
        shutil.rmtree(resolved)
    for path in (GITHUB, OSF):
        path.mkdir(parents=True, exist_ok=True)


def extraction_columns() -> list[str]:
    columns = ["row_id", "ok", "traveler_segment"]
    for aspect in ASPECTS:
        columns.extend([f"aspect_{aspect}_score", f"aspect_{aspect}_mentioned"])
    return columns


def sanitized_extraction(
    source: Path,
    destination: Path,
    expected_rows: int,
    prefix: str = "aspect",
    ok_column: str = "ok",
    segment_column: str = "traveler_segment",
    seed: int | None = None,
) -> dict[str, Any]:
    requested = ["row_id", ok_column, segment_column]
    requested.extend(f"{prefix}_{aspect}_{suffix}" for aspect in ASPECTS for suffix in ("score", "mentioned"))
    source_frame = pd.read_parquet(source, columns=requested)
    if len(source_frame) != expected_rows:
        raise AssertionError(f"{source}: expected {expected_rows:,} rows, found {len(source_frame):,}")
    result = pd.DataFrame({
        "row_id": pd.to_numeric(source_frame["row_id"], errors="raise").astype(np.uint32),
        "ok": source_frame[ok_column].fillna(0).astype(bool),
        "traveler_segment": source_frame[segment_column].fillna("unknown").astype(str).str.lower(),
    })
    result.loc[~result["traveler_segment"].isin(SEGMENTS), "traveler_segment"] = "unknown"
    for aspect in ASPECTS:
        score = pd.to_numeric(source_frame[f"{prefix}_{aspect}_score"], errors="coerce").fillna(0).clip(-2, 2)
        result[f"aspect_{aspect}_score"] = score.astype(np.int8)
        result[f"aspect_{aspect}_mentioned"] = source_frame[f"{prefix}_{aspect}_mentioned"].fillna(False).astype(bool)
    if seed is not None:
        result.insert(1, "seed", np.full(len(result), seed, dtype=np.uint8))
    if result["row_id"].duplicated().any() and seed is None:
        raise AssertionError(f"duplicate row_id in sanitized extraction: {source}")
    string_columns = [column for column in result.columns if pd.api.types.is_object_dtype(result[column]) or pd.api.types.is_string_dtype(result[column])]
    if string_columns != ["traveler_segment"]:
        raise AssertionError(f"unexpected string columns in sanitized extraction: {string_columns}")
    expected = extraction_columns() + (["seed"] if seed is not None else [])
    if seed is not None:
        expected = ["row_id", "seed", *extraction_columns()[1:]]
    if list(result.columns) != expected:
        raise AssertionError(f"sanitized schema mismatch: {list(result.columns)}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(destination, index=False, compression="zstd")
    schema = pq.ParquetFile(destination).schema_arrow
    forbidden = set(schema.names) & FORBIDDEN_DATA_COLUMNS
    if forbidden:
        raise AssertionError(f"forbidden extraction columns survived: {sorted(forbidden)}")
    return {
        "source": source.relative_to(ROOT).as_posix(),
        "destination": destination.relative_to(STAGING).as_posix(),
        "rows": len(result),
        "columns": list(result.columns),
        "string_columns": string_columns,
        "bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
    }


def build_sanitized_extractions() -> list[dict[str, Any]]:
    log("Part 1: writing sanitized extraction derivatives")
    specs = [
        (OUTPUTS / "aspects_llm_qwen25_7b.parquet", OSF / "extraction/tripadvisor_qwen25_7b_full.parquet", 877_808, "aspect", "ok", "traveler_segment"),
        (OUTPUTS / "phase3/aspects_guided_qwen25_7b_l1_n200041.parquet", OSF / "extraction/tripadvisor_qwen25_7b_guided_200k.parquet", 200_041, "aspect", "ok", "traveler_segment"),
        (OUTPUTS / "phase3/aspects_guided_llama31_8b_l1_n200041.parquet", OSF / "extraction/tripadvisor_llama31_8b_guided_200k.parquet", 200_041, "aspect", "ok", "traveler_segment"),
        (OUTPUTS / "phase2/aspects_guided_qwen25_7b_n50000.parquet", OSF / "extraction/tripadvisor_qwen25_7b_guided_50k.parquet", 50_000, "guided", "guided_ok", "guided_traveler_segment"),
        (OUTPUTS / "phase3/aspects_guided_booking_qwen_guided_n200035.parquet", OSF / "extraction/booking_qwen25_7b_guided_200k.parquet", 200_035, "aspect", "ok", "traveler_segment"),
    ]
    audits = []
    for source, destination, rows, prefix, ok_column, segment_column in specs:
        audit = sanitized_extraction(source, destination, rows, prefix, ok_column, segment_column)
        audits.append(audit)
        log(f"  {destination.name}: {rows:,} rows, {destination.stat().st_size:,} bytes")

    determinism_parts = []
    for seed in (42, 43, 44):
        source = OUTPUTS / f"phase3/v_qwen25_7b_original_n20000_seed{seed}.parquet"
        temporary = STAGING / f"determinism_seed{seed}.parquet"
        sanitized_extraction(source, temporary, 20_000, seed=seed)
        determinism_parts.append(pd.read_parquet(temporary))
        temporary.unlink()
    determinism = pd.concat(determinism_parts, ignore_index=True)
    if determinism.duplicated(["row_id", "seed"]).any() or len(determinism) != 60_000:
        raise AssertionError("determinism sanitized derivative key/row-count failure")
    destination = OSF / "extraction/determinism_20k_seeds424344.parquet"
    determinism.to_parquet(destination, index=False, compression="zstd")
    audits.append({
        "source": "outputs/phase3/v_qwen25_7b_original_n20000_seed{42,43,44}.parquet",
        "destination": destination.relative_to(STAGING).as_posix(),
        "rows": len(determinism),
        "columns": list(determinism.columns),
        "string_columns": ["traveler_segment"],
        "bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
    })
    log(f"  {destination.name}: {len(determinism):,} rows, {destination.stat().st_size:,} bytes")
    return audits


def evidence_offsets(source: Path, destination: Path, texts: pd.Series) -> dict[str, Any]:
    frame = pd.read_parquet(source, columns=["row_id", "ok", "response_json"])
    row_ids = array("I")
    aspect_ids = array("B")
    ranks = array("B")
    starts = array("i")
    ends = array("i")
    matches = array("B")
    total = hits = ambiguous = 0
    for index, record in enumerate(frame.itertuples(index=False), start=1):
        if int(record.ok or 0) != 1:
            continue
        try:
            parsed = json.loads(str(record.response_json))
        except (json.JSONDecodeError, TypeError):
            continue
        aspects = parsed.get("aspects", {}) if isinstance(parsed, dict) else {}
        text = str(texts.iat[int(record.row_id)] or "")
        normalized_text = text.lower()
        for aspect_id, aspect in enumerate(ASPECTS):
            item = aspects.get(aspect, {}) if isinstance(aspects, dict) else {}
            values = item.get("evidence", []) if isinstance(item, dict) else []
            if not isinstance(values, list):
                continue
            for rank, value in enumerate(values[:2]):
                snippet = str(value).strip()
                if not snippet:
                    continue
                normalized_snippet = snippet.lower()
                start = normalized_text.find(normalized_snippet)
                matched = start >= 0
                total += 1
                hits += int(matched)
                ambiguous += int(matched and normalized_text.find(normalized_snippet, start + 1) >= 0)
                row_ids.append(int(record.row_id))
                aspect_ids.append(aspect_id)
                ranks.append(rank)
                starts.append(start if matched else -1)
                ends.append(start + len(snippet) if matched else -1)
                matches.append(int(matched))
        if index % 100_000 == 0:
            log(f"    offsets {destination.name}: {index:,}/{len(frame):,} source rows")
    table = pa.table({
        "row_id": pa.array(row_ids, type=pa.uint32()),
        "aspect_id": pa.array(aspect_ids, type=pa.uint8()),
        "evidence_rank": pa.array(ranks, type=pa.uint8()),
        "start": pa.array(starts, type=pa.int32()),
        "end": pa.array(ends, type=pa.int32()),
        "matched": pa.array(np.frombuffer(matches, dtype=np.uint8).astype(bool), type=pa.bool_()),
    })
    destination.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, destination, compression="zstd")
    del frame, table
    gc.collect()
    return {
        "source": source.relative_to(ROOT).as_posix(),
        "destination": destination.relative_to(STAGING).as_posix(),
        "evidence_items": total,
        "matched": hits,
        "match_rate": hits / total if total else math.nan,
        "ambiguous_first_occurrence": ambiguous,
        "bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
    }


def build_offsets() -> list[dict[str, Any]]:
    log("Part 2: recovering evidence character offsets")
    texts = pd.read_parquet(OUTPUTS / "dataset_clean.parquet", columns=["review_text"])["review_text"].fillna("").astype(str)
    specs = [
        (OUTPUTS / "aspects_llm_qwen25_7b.parquet", OSF / "extraction/tripadvisor_qwen25_7b_full_evidence_offsets.parquet"),
        (OUTPUTS / "phase3/aspects_guided_qwen25_7b_l1_n200041.parquet", OSF / "extraction/tripadvisor_qwen25_7b_guided_200k_evidence_offsets.parquet"),
        (OUTPUTS / "phase3/aspects_guided_llama31_8b_l1_n200041.parquet", OSF / "extraction/tripadvisor_llama31_8b_guided_200k_evidence_offsets.parquet"),
    ]
    audits = []
    for source, destination in specs:
        audit = evidence_offsets(source, destination, texts)
        audits.append(audit)
        log(f"  {destination.name}: {audit['matched']:,}/{audit['evidence_items']:,} ({audit['match_rate']:.6%}), {audit['bytes']:,} bytes")
    del texts
    gc.collect()
    return audits


def load_fold_predictions(model_id: str) -> pd.DataFrame:
    parts = []
    for fold in range(1, 6):
        path = OUTPUTS / "predictions" / model_id / f"fold{fold}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"missing required model artefact: {path}")
        parts.append(pd.read_parquet(path, columns=["row_id", "offering_id", "y_true", "y_pred_clipped", "fold"]))
    frame = pd.concat(parts, ignore_index=True)
    if frame["row_id"].duplicated().any():
        raise AssertionError(f"duplicate held-out row_id for {model_id}")
    return frame.sort_values("row_id").reset_index(drop=True)


def metrics_by_fold(frame: pd.DataFrame, prediction_column: str, model_id: str, corpus: str, subsample: str) -> list[dict[str, Any]]:
    rows = []
    for fold, group in frame.groupby("fold", sort=True):
        valid = group[prediction_column].notna()
        y_true = group.loc[valid, "y_true"].to_numpy(dtype=float)
        prediction = group.loc[valid, prediction_column].to_numpy(dtype=float)
        rows.append({
            "model_id": model_id,
            "corpus": corpus,
            "subsample": subsample,
            "fold": int(fold),
            "n": int(len(y_true)),
            "mae": float(np.mean(np.abs(y_true - prediction))),
            "rmse": float(np.sqrt(np.mean(np.square(y_true - prediction)))),
        })
    return rows


def build_tripadvisor_wide() -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    log(f"Part 4: consolidating {len(TRIP_FULL_MODELS)} full-corpus TripAdvisor models")
    reference = load_fold_predictions("phase4_u0_text_only_tfidf_ridge")
    if len(reference) != N_ROWS or not np.array_equal(reference["row_id"].to_numpy(), np.arange(N_ROWS)):
        raise AssertionError("TripAdvisor reference predictions do not cover row_id 0..877807")
    wide = pd.DataFrame({
        "row_id": reference["row_id"].to_numpy(dtype=np.uint32),
        "offering_id": pd.to_numeric(reference["offering_id"], errors="raise").to_numpy(dtype=np.int64),
        "fold": reference["fold"].to_numpy(dtype=np.uint8),
        "y_true": reference["y_true"].to_numpy(dtype=np.int8),
    })
    per_fold = []
    comparisons = []
    for position, model_id in enumerate(TRIP_FULL_MODELS, start=1):
        source = load_fold_predictions(model_id)
        common = source["row_id"].to_numpy(dtype=np.int64)
        if not np.array_equal(source["offering_id"].to_numpy(), wide.loc[common, "offering_id"].to_numpy()):
            raise AssertionError(f"offering_id mismatch for {model_id}")
        if not np.array_equal(source["fold"].to_numpy(dtype=np.uint8), wide.loc[common, "fold"].to_numpy()):
            raise AssertionError(f"fold mismatch for {model_id}")
        if not np.array_equal(source["y_true"].to_numpy(dtype=np.int8), wide.loc[common, "y_true"].to_numpy()):
            raise AssertionError(f"target mismatch for {model_id}")
        values = pd.Series(source["y_pred_clipped"].clip(1, 5).to_numpy(dtype=np.float32), index=common)
        wide[model_id] = values.reindex(np.arange(N_ROWS)).to_numpy(dtype=np.float32)
        nulls = int(wide[model_id].isna().sum())
        if model_id in NULLABLE_TRIP_MODELS:
            if nulls == 0:
                raise AssertionError(f"expected subset nulls for {model_id}")
        elif nulls:
            raise AssertionError(f"unexpected {nulls} null predictions for {model_id}")
        source_metrics = metrics_by_fold(source.rename(columns={"y_pred_clipped": model_id}), model_id, model_id, "TripAdvisor", "full")
        wide_metrics = metrics_by_fold(wide[["fold", "y_true", model_id]], model_id, model_id, "TripAdvisor", "full")
        max_fold_difference = max(
            max(abs(left[key] - right[key]) for key in ("mae", "rmse"))
            for left, right in zip(source_metrics, wide_metrics, strict=True)
        )
        source_mean = {key: float(np.mean([row[key] for row in source_metrics])) for key in ("mae", "rmse")}
        wide_mean = {key: float(np.mean([row[key] for row in wide_metrics])) for key in ("mae", "rmse")}
        fold_mean_difference = max(abs(source_mean[key] - wide_mean[key]) for key in ("mae", "rmse"))
        value_difference = float(np.max(np.abs(
            wide.loc[common, model_id].to_numpy(dtype=np.float32)
            - source["y_pred_clipped"].clip(1, 5).to_numpy(dtype=np.float32)
        )))
        if value_difference != 0.0:
            raise AssertionError(f"serialized float32 values differ from source float32 cast for {model_id}")
        per_fold.extend(wide_metrics)
        comparisons.append({
            "model_id": model_id,
            "serialized_value_max_abs_difference": value_difference,
            "max_abs_per_fold_metric_quantization": max_fold_difference,
            "max_abs_fold_mean_metric_difference": fold_mean_difference,
            "passed_1e_9": fold_mean_difference <= 1e-9,
        })
        log(
            f"  [{position:02d}/{len(TRIP_FULL_MODELS)}] {model_id}: rows={len(source):,}, "
            f"nulls={nulls:,}, fold_mean_metric_delta={fold_mean_difference:.3e}"
        )
        del source, values
        gc.collect()
    destination = OSF / "predictions/tripadvisor_predictions_wide.parquet"
    destination.parent.mkdir(parents=True, exist_ok=True)
    wide.to_parquet(destination, index=False, compression="zstd")
    return wide, per_fold, comparisons


def build_booking_wide() -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    source = pd.read_parquet(OUTPUTS / "phase4/booking_u0_predictions.parquet")
    mapping = {
        "text_only": "phase4_booking_u0_text_only",
        "simple_fusion": "phase4_booking_u0_simple_fusion",
        "interaction_fusion": "phase4_booking_u0_interaction_fusion",
    }
    reference = source[source["model"].eq("text_only")].sort_values("row_id").reset_index(drop=True)
    if len(reference) != 200_035:
        raise AssertionError("Booking reference row count mismatch")
    wide = pd.DataFrame({
        "row_id": reference["row_id"].to_numpy(dtype=np.uint32),
        "offering_id": reference["offering_id"].astype(str).to_numpy(),
        "fold": reference["fold"].to_numpy(dtype=np.uint8),
        "y_true": reference["y_native"].to_numpy(dtype=np.float32),
        "y_true_rescaled_1_5": reference["y_true"].to_numpy(dtype=np.float32),
    })
    per_fold = []
    for source_name, model_id in mapping.items():
        part = source[source["model"].eq(source_name)].sort_values("row_id").reset_index(drop=True)
        if not np.array_equal(part["row_id"].to_numpy(), reference["row_id"].to_numpy()):
            raise AssertionError(f"Booking row alignment mismatch for {model_id}")
        rescaled = part["prediction"].clip(1, 5).to_numpy(dtype=np.float32)
        wide[f"{model_id}__rescaled_1_5"] = rescaled
        wide[model_id] = (1.0 + 9.0 * (rescaled.astype(np.float64) - 1.0) / 4.0).astype(np.float32)
        metric_frame = pd.DataFrame({"fold": wide["fold"], "y_true": wide["y_true_rescaled_1_5"], model_id: rescaled})
        per_fold.extend(metrics_by_fold(metric_frame, model_id, model_id, "Booking.com", "guided_200035"))
    destination = OSF / "predictions/booking_predictions_wide.parquet"
    destination.parent.mkdir(parents=True, exist_ok=True)
    wide.to_parquet(destination, index=False, compression="zstd")
    return wide, per_fold


def build_subsample_wide() -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    l5 = pd.read_parquet(OUTPUTS / "phase3/l5_downstream_predictions.parquet")
    first = l5[l5["model"].eq(L5_SOURCE_MODELS[0])].sort_values("row_id").reset_index(drop=True)
    l5_wide = pd.DataFrame({
        "subsample": "extractor_sensitivity_l1_200041",
        "row_id": first["row_id"].to_numpy(dtype=np.uint32),
        "offering_id": first["offering_id"].to_numpy(dtype=np.int64),
        "fold": first["fold"].to_numpy(dtype=np.uint8),
        "y_true": first["y_true"].to_numpy(dtype=np.int8),
    })
    per_fold = []
    for source_name, release_name in zip(L5_SOURCE_MODELS, L5_RELEASE_MODELS, strict=True):
        part = l5[l5["model"].eq(source_name)].sort_values("row_id").reset_index(drop=True)
        if len(part) != 200_041 or not np.array_equal(part["row_id"].to_numpy(), first["row_id"].to_numpy()):
            raise AssertionError(f"L5 alignment mismatch for {source_name}")
        l5_wide[release_name] = np.clip(part["y_pred"].to_numpy(dtype=np.float32), 1, 5)
        metric_frame = pd.DataFrame({"fold": l5_wide["fold"], "y_true": l5_wide["y_true"], release_name: l5_wide[release_name]})
        per_fold.extend(metrics_by_fold(metric_frame, release_name, release_name, "TripAdvisor", "extractor_sensitivity_l1_200041"))

    direct = load_fold_predictions(DIRECT_SUBSAMPLE_MODEL)
    direct_wide = pd.DataFrame({
        "subsample": "qwen_direct_zero_100000",
        "row_id": direct["row_id"].to_numpy(dtype=np.uint32),
        "offering_id": direct["offering_id"].to_numpy(dtype=np.int64),
        "fold": direct["fold"].to_numpy(dtype=np.uint8),
        "y_true": direct["y_true"].to_numpy(dtype=np.int8),
        DIRECT_SUBSAMPLE_MODEL: direct["y_pred_clipped"].clip(1, 5).to_numpy(dtype=np.float32),
    })
    per_fold.extend(metrics_by_fold(direct_wide, DIRECT_SUBSAMPLE_MODEL, DIRECT_SUBSAMPLE_MODEL, "TripAdvisor", "qwen_direct_zero_100000"))
    wide = pd.concat([l5_wide, direct_wide], ignore_index=True, sort=False)
    prediction_columns = list(L5_RELEASE_MODELS) + [DIRECT_SUBSAMPLE_MODEL]
    wide[prediction_columns] = wide[prediction_columns].astype(np.float32)
    destination = OSF / "predictions/subsample_predictions_wide.parquet"
    destination.parent.mkdir(parents=True, exist_ok=True)
    wide.to_parquet(destination, index=False, compression="zstd")
    return wide, per_fold


def build_per_hotel_metrics() -> pd.DataFrame:
    records = []
    for model_id in sorted(DIRECTORY_MODEL_FILTER):
        source = OUTPUTS / "per_hotel" / f"{model_id}.parquet"
        if not source.exists():
            raise FileNotFoundError(f"missing required per-hotel artefact: {source}")
        frame = pd.read_parquet(source, columns=["offering_id", "fold", "n_reviews", "mae", "rmse"])
        corpus = "Booking.com" if model_id in BOOKING_MODELS else "TripAdvisor"
        subsample = "booking_guided_200035" if corpus == "Booking.com" else ("qwen_direct_zero_100000" if model_id == DIRECT_SUBSAMPLE_MODEL else "full")
        frame.insert(0, "model_id", model_id)
        frame.insert(1, "corpus", corpus)
        frame.insert(2, "subsample", subsample)
        records.append(frame)
    l5 = pd.read_parquet(OUTPUTS / "phase3/l5_downstream_predictions.parquet")
    for source_name, release_name in zip(L5_SOURCE_MODELS, L5_RELEASE_MODELS, strict=True):
        part = l5[l5["model"].eq(source_name)].copy()
        part["error"] = np.abs(part["y_true"] - np.clip(part["y_pred"], 1, 5))
        part["squared_error"] = np.square(part["error"])
        grouped = part.groupby(["offering_id", "fold"], as_index=False).agg(
            n_reviews=("error", "size"), mae=("error", "mean"), mean_squared_error=("squared_error", "mean")
        )
        grouped["rmse"] = np.sqrt(grouped.pop("mean_squared_error"))
        grouped.insert(0, "model_id", release_name)
        grouped.insert(1, "corpus", "TripAdvisor")
        grouped.insert(2, "subsample", "extractor_sensitivity_l1_200041")
        records.append(grouped)
    result = pd.concat(records, ignore_index=True)
    result["offering_id"] = result["offering_id"].astype(str)
    result["fold"] = result["fold"].astype(np.uint8)
    result["n_reviews"] = result["n_reviews"].astype(np.uint32)
    result["mae"] = result["mae"].astype(np.float32)
    result["rmse"] = result["rmse"].astype(np.float32)
    if result.duplicated(["model_id", "offering_id"]).any():
        raise AssertionError("duplicate model_id/offering_id in consolidated per-hotel metrics")
    destination = OSF / "predictions/per_hotel_metrics.parquet"
    result.to_parquet(destination, index=False, compression="zstd")
    return result


def metric_summary(per_fold: pd.DataFrame) -> pd.DataFrame:
    return per_fold.groupby(["model_id", "corpus", "subsample"], as_index=False).agg(
        n=("n", "sum"), mae_mean=("mae", "mean"), mae_sd=("mae", "std"), rmse_mean=("rmse", "mean"), rmse_sd=("rmse", "std")
    )


def add_section(frame: pd.DataFrame, section: str) -> pd.DataFrame:
    result = frame.copy()
    result.insert(0, "section", section)
    return result


def numeric_long(frame: pd.DataFrame, section: str, id_columns: Sequence[str]) -> pd.DataFrame:
    ids = [column for column in id_columns if column in frame.columns]
    value_columns = [column for column in frame.columns if column not in ids and pd.api.types.is_numeric_dtype(frame[column])]
    # Some source diagnostics already use ``metric`` as an identifier column.
    # Keep the measured numeric field distinct so the melted frame always has
    # unique column labels and can be concatenated safely across diagnostics.
    result = frame.melt(id_vars=ids, value_vars=value_columns, var_name="measure", value_name="value")
    result.insert(0, "section", section)
    return result


def build_coefficients_diagnostics_and_results(per_fold_rows: list[dict[str, Any]]) -> None:
    log("Part 5/6: assembling coefficients, diagnostics, and manuscript result tables")
    coeff = pd.read_csv(OUTPUTS / "phase5/S1_coefficient_tables.csv")
    interactions = pd.read_csv(OUTPUTS / "phase5/S1_segment_aspect_interactions.csv")
    gof = pd.read_csv(OUTPUTS / "phase5/S1_goodness_of_fit.csv")
    if len(coeff) != 550 or len(interactions) != 360:
        raise AssertionError("coefficient row-count gate failed")
    (OSF / "coefficients").mkdir(parents=True, exist_ok=True)
    coeff.to_parquet(OSF / "coefficients/coefficients_full.parquet", index=False, compression="zstd")
    interactions.to_parquet(OSF / "coefficients/segment_aspect_interactions.parquet", index=False, compression="zstd")
    gof.to_csv(OSF / "coefficients/goodness_of_fit.csv", index=False)

    alpha = pd.read_csv(OUTPUTS / "phase3/alpha_response_curves.csv")
    if len(alpha) != 615:
        raise AssertionError("alpha-response row-count gate failed")
    (OSF / "diagnostics").mkdir(parents=True, exist_ok=True)
    alpha.to_parquet(OSF / "diagnostics/alpha_response_curves.parquet", index=False, compression="zstd")

    quality_parts = []
    fidelity = pd.read_csv(OUTPUTS / "phase2/experiment_s_table2_both_scales.csv")
    quality_parts.append(numeric_long(fidelity, "fidelity_vs_platform", ["convention", "aspect"]))
    quality_parts.append(numeric_long(pd.read_csv(OUTPUTS / "phase4/u6_score_agreement.csv"), "qwen_llama_score_agreement", ["aspect"]))
    quality_parts.append(numeric_long(pd.read_csv(OUTPUTS / "phase4/u6_mention_agreement.csv"), "qwen_llama_mention_agreement", ["aspect"]))
    quality_parts.append(numeric_long(pd.read_csv(OUTPUTS / "phase4/u6_evidence_hit_rate.csv"), "evidence_hit_rate", ["extractor", "aspect"]))
    quality_parts.append(numeric_long(pd.read_csv(OUTPUTS / "phase2/experiment_c_guided_agreement.csv"), "guided_decoding_agreement", ["feature", "feature_type", "metric"]))
    extraction_quality = pd.concat(quality_parts, ignore_index=True, sort=False)
    extraction_quality.to_parquet(OSF / "diagnostics/extraction_quality.parquet", index=False, compression="zstd")

    results = GITHUB / "results"
    results.mkdir(parents=True, exist_ok=True)
    copy_file(OUTPUTS / "phase4/u1_combined_panel_a_b.csv", results / "table1_panels.csv")
    shifted = fidelity[fidelity["convention"].astype(str).eq("shifted_3_plus_s")].copy()
    if len(shifted) != 7:
        raise AssertionError("shifted fidelity Table 2 must contain six aspects plus mean")
    shifted.to_csv(results / "table2_aspect_quality.csv", index=False)
    copy_file(OUTPUTS / "phase4/u4_ablation_u0.csv", results / "table3_ablation.csv")
    table4 = coeff.groupby(["model", "block", "feature"], as_index=False).agg(
        fold_mean_estimate=("estimate", "mean"), across_fold_sd=("estimate", "std"),
        positive_folds=("estimate", lambda values: int((values > 0).sum())),
        negative_folds=("estimate", lambda values: int((values < 0).sum())),
    )
    table4.to_csv(results / "table4_coefficients.csv", index=False)
    table5 = pd.concat([
        add_section(pd.read_csv(OUTPUTS / "phase4/u6_score_agreement.csv"), "score_agreement"),
        add_section(pd.read_csv(OUTPUTS / "phase4/u6_mention_agreement.csv"), "mention_agreement"),
        add_section(pd.read_csv(OUTPUTS / "phase4/u6_service_semantics_test.csv"), "service_semantics"),
    ], ignore_index=True, sort=False)
    table5.to_csv(results / "table5_extractor_agreement.csv", index=False)
    table6 = pd.concat([
        add_section(pd.read_csv(OUTPUTS / "phase5/E1_known_unknown_strata.csv"), "known_unknown"),
        add_section(pd.read_csv(OUTPUTS / "phase5/E2_masking_summary.csv"), "segment_masking"),
        add_section(pd.read_csv(OUTPUTS / "phase5/E3_mentioned_count_strata.csv"), "mentioned_count"),
        add_section(pd.read_csv(OUTPUTS / "phase5/E4_hotel_coverage_correlation.csv"), "hotel_coverage_correlation"),
    ], ignore_index=True, sort=False)
    table6.to_csv(results / "table6_missingness.csv", index=False)
    copy_file(OUTPUTS / "phase4/booking_u0_summary.csv", results / "table7_external_booking.csv")
    paired = pd.concat([
        add_section(pd.read_csv(OUTPUTS / "phase4/u2_paired_hotel_tests.csv"), "primary_u0"),
        add_section(pd.read_csv(OUTPUTS / "phase4/u4_ablation_paired_tests.csv"), "ablation_u0"),
        add_section(pd.read_csv(OUTPUTS / "phase4/booking_u0_paired_tests.csv"), "booking_u0"),
    ], ignore_index=True, sort=False)
    paired.to_csv(results / "paired_tests.csv", index=False)
    pd.DataFrame(per_fold_rows).sort_values(["corpus", "subsample", "model_id", "fold"]).to_csv(results / "per_fold_metrics.csv", index=False)
    alpha.to_csv(results / "alpha_response_curves.csv", index=False)


def build_model_filter() -> tuple[pd.DataFrame, int]:
    prediction_dirs = sorted(path.name for path in (OUTPUTS / "predictions").iterdir() if path.is_dir())
    missing = sorted(DIRECTORY_MODEL_FILTER - set(prediction_dirs))
    if missing:
        raise FileNotFoundError(f"required model IDs missing: {missing}")
    rows = []
    category_map = {}
    for category, models in (
        ("Panel A", PANEL_A), ("Panel B U0", PANEL_B), ("Neural", NEURAL),
        ("Ablation", ABLATIONS), ("Protocol sensitivity", PROTOCOL_SENSITIVITY),
        ("External Booking", BOOKING_MODELS),
    ):
        category_map.update({model: category for model in models})
    for model_id in prediction_dirs:
        included = model_id in DIRECTORY_MODEL_FILTER
        rows.append({
            "model_id": model_id,
            "included": included,
            "category": category_map.get(model_id, "Excluded exploratory model"),
            "reason": "listed in Phase 6B Part 3" if included else "not in the manuscript-scoped Phase 6B model filter",
        })
    for model_id in L5_RELEASE_MODELS:
        rows.append({"model_id": model_id, "included": True, "category": "Extractor sensitivity L5", "reason": "one of the seven consolidated L5 configurations"})
    result = pd.DataFrame(rows).sort_values(["included", "category", "model_id"], ascending=[False, True, True])
    excluded_count = int((~result["included"]).sum())
    result.to_csv(GITHUB / "results/model_filter.csv", index=False)
    return result, excluded_count


def build_split_release() -> dict[str, Any]:
    assignment = np.full(N_ROWS, -1, dtype=np.int8)
    seen = np.zeros(N_ROWS, dtype=np.uint8)
    for fold in range(1, 6):
        arrays = np.load(OUTPUTS / f"splits_fold{fold}.npz")
        test_idx = arrays["test_idx"].astype(np.int64)
        assignment[test_idx] = fold
        seen[test_idx] += 1
    if np.any(assignment < 1) or np.any(seen != 1):
        raise AssertionError("canonical split assignment is incomplete or overlapping")
    digest = hashlib.sha256(assignment.tobytes()).hexdigest()
    if digest != SPLIT_SHA256:
        raise AssertionError(f"canonical split hash mismatch: {digest}")
    split_dir = GITHUB / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({"row_id": np.arange(N_ROWS, dtype=np.uint32), "fold": assignment.astype(np.uint8)})
    frame.to_parquet(split_dir / "fold_assignment.parquet", index=False, compression="zstd")
    write_text(
        split_dir / "fold_assignment.sha256",
        f"{digest}  fold_assignment.int8.tobytes()\n"
        "Procedure: sort fold_assignment.parquet by row_id, cast fold to NumPy int8, "
        "hash the C-contiguous array bytes with SHA-256.",
    )
    expand_script = '''#!/usr/bin/env python3
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

EXPECTED = "dfe3a434fb00c894218309c332627dc0131a2db3cc2cd241727fc712a4803942"

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assignment", type=Path, default=Path(__file__).with_name("fold_assignment.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    frame = pd.read_parquet(args.assignment).sort_values("row_id")
    row_id = frame["row_id"].to_numpy(dtype=np.uint32)
    fold = frame["fold"].to_numpy(dtype=np.int8)
    if not np.array_equal(row_id, np.arange(len(frame), dtype=np.uint32)):
        raise ValueError("row_id must be unique and contiguous from zero")
    digest = hashlib.sha256(fold.tobytes()).hexdigest()
    if digest != EXPECTED:
        raise ValueError(f"split hash mismatch: {digest}")
    print(f"split hash PASS: {digest}")
    if args.verify:
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_idx = np.arange(len(frame), dtype=np.int64)
    for value in range(1, 6):
        test_idx = all_idx[fold == value]
        train_idx = all_idx[fold != value]
        np.savez(args.output_dir / f"splits_fold{value}.npz", train_idx=train_idx, test_idx=test_idx)
        print(f"fold {value}: train={len(train_idx)} test={len(test_idx)}")

if __name__ == "__main__":
    main()
'''
    write_text(split_dir / "expand_splits.py", expand_script)
    os.chmod(split_dir / "expand_splits.py", 0o755)
    return {"rows": len(frame), "sha256": digest, "path": "splits/fold_assignment.parquet"}


def extract_prompt_text() -> str:
    source = ROOT / "scripts/06_extract_aspects_llm.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    values = {}
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            value_node = node.value
            for target in targets:
                if isinstance(target, ast.Name) and target.id in {"SYSTEM_PROMPT", "USER_PROMPT_TEMPLATE"}:
                    values[target.id] = ast.literal_eval(value_node)
    if set(values) != {"SYSTEM_PROMPT", "USER_PROMPT_TEMPLATE"}:
        raise AssertionError("could not recover deployed prompt literals")
    return (
        "# DEPLOYED EXTRACTION PROMPT (schema v1_aspects6, prompt v2_strictjson)\n"
        "# KNOWN DEFECT: permitted score values {-2,-1,0,1,2} are enumerated, but the\n"
        "# polarity meaning/direction of those values is not explicitly defined. This is\n"
        "# the service-score convention defect analysed in the manuscript.\n\n"
        "[SYSTEM MESSAGE — VERBATIM]\n" + values["SYSTEM_PROMPT"] +
        "\n\n[USER MESSAGE TEMPLATE — VERBATIM]\n" + values["USER_PROMPT_TEMPLATE"]
    )


def build_environment(conda_env: str) -> None:
    command = ["conda", "env", "export", "-n", conda_env, "--no-builds"]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    lines = [line for line in completed.stdout.splitlines() if not line.startswith("prefix:")]
    if lines and lines[0].startswith("name:"):
        lines[0] = "name: hotel-fusion"
    write_text(GITHUB / "environment.yml", "\n".join(lines))


def make_released_code_portable(path: Path) -> None:
    """Remove workstation-specific paths from copied release code only."""
    text = path.read_text(encoding="utf-8")
    text = re.sub(
        r'PYTHON_BIN="[/]home/[^"\n]+/anaconda3/envs/[^"\n]+/bin/python"',
        'PYTHON_BIN="${PYTHON_BIN:-python}"',
        text,
    )
    text = re.sub(
        r'source [/]home/[^\n]+/anaconda3/etc/profile\.d/conda\.sh\nconda activate [^\n]+',
        'eval "$(conda shell.bash hook)"\nconda activate "${HOTEL_CONDA_ENV:-hotel-fusion}"',
        text,
    )
    text = re.sub(
        r'[/]home/[^/\s]+/\.cache/huggingface/llama31_8b',
        '/path/to/Llama-3.1-8B-Instruct',
        text,
    )
    path.write_text(text, encoding="utf-8")


def build_code_tree(conda_env: str) -> list[dict[str, str]]:
    copy_file(ROOT / "README.md", GITHUB / "README.md")
    copy_file(ROOT / "requirements.txt", GITHUB / "requirements.txt")
    build_environment(conda_env)
    write_text(GITHUB / ".gitignore", """__pycache__/
*.py[cod]
.venv/
outputs/
data_ori/
data_external/
llm_cache/
*.sqlite
*.log
*.pid
.DS_Store
""")
    mit = """MIT License

Copyright (c) 2026 Xue Zhu and Qinfeng Zhu

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the \"Software\"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
    write_text(GITHUB / "LICENSE", mit)
    data_license = """Creative Commons Attribution 4.0 International (CC BY 4.0)

The derived data tables in this release are licensed under the Creative Commons Attribution 4.0 International licence.

You are free to share and adapt the material for any purpose, provided that appropriate credit is given, a link to the licence is supplied, and changes are indicated. No additional restrictions may be applied.

Full legal code: https://creativecommons.org/licenses/by/4.0/legalcode

This licence does not apply to either upstream review corpus, model weights, or any third-party material. Those items are not redistributed in this release.
"""
    write_text(GITHUB / "LICENSE-DATA", data_license)
    write_text(OSF / "LICENSE-DATA", data_license)
    citation = """cff-version: 1.2.0
message: "If you use this software or its derived data, please cite the manuscript and release."
title: "Interaction Fusion of Text and LLM-Extracted Aspects"
type: software
authors:
  - family-names: Zhu
    given-names: Xue
  - family-names: Zhu
    given-names: Qinfeng
version: 1.0.0
date-released: 2026-08-05
license: MIT
abstract: "Reproducible code and sanitized artefacts for hotel-rating prediction with LLM-extracted aspects."
"""
    write_text(GITHUB / "CITATION.cff", citation)

    config = """profiles:
  qwen25_7b:
    provider: openai_compatible
    base_url: http://localhost:8000/v1
    endpoints:
      - http://localhost:8000/v1
      - http://localhost:8001/v1
    api_key: local-llm
    model: Qwen/Qwen2.5-7B-Instruct
    temperature: 0
    max_tokens: 900
    timeout_sec: 120
    concurrency: 16
    max_retries: 3
"""
    write_text(GITHUB / "config/local_llm.yaml", config)
    for source in sorted((ROOT / "src").glob("*.py")):
        if source.name == "aspect_extractor_dummy.py":
            continue
        copy_file(source, GITHUB / "src" / source.name)
    decisions = []
    all_scripts = sorted(path.name for path in (ROOT / "scripts").iterdir() if path.is_file() and path.suffix in {".py", ".sh"})
    for name in all_scripts:
        if name in RETAINED_SCRIPTS:
            destination = GITHUB / "scripts" / name
            copy_file(ROOT / "scripts" / name, destination)
            make_released_code_portable(destination)
            decisions.append({"script": name, "decision": "retained", "reason": RETAINED_SCRIPTS[name]})
        else:
            reason = EXCLUDED_SCRIPT_REASONS.get(name)
            if reason is None:
                raise AssertionError(f"script retention decision missing for {name}")
            decisions.append({"script": name, "decision": "dropped", "reason": reason})
    write_text(GITHUB / "prompts/extraction_prompt.txt", extract_prompt_text())
    return decisions


def build_figures_and_supporting() -> None:
    figure_sources = {
        "figure1_aspect_importance.png": OUTPUTS / "fig_aspect_importance_qwen25_7b.png",
        "figure2_segment_aspect_heatmap.png": OUTPUTS / "fig_segment_aspect_heatmap_qwen25_7b.png",
        "figure3_missingness_curve.png": OUTPUTS / "phase2/experiment_e2_missingness_curve.png",
        "figure4_extraction_score_histograms.png": OUTPUTS / "phase2/experiment_s_score_histograms.png",
        "supplementary_alpha_response_curve.png": OUTPUTS / "phase3/g1_alpha_response_curves.png",
    }
    for name, source in figure_sources.items():
        copy_file(source, GITHUB / "figures" / name)
    for name in ("S1_File.xlsx", "S2_File.xlsx", "S3_File.xlsx"):
        copy_file(OUTPUTS / "supporting_information" / name, GITHUB / "supporting_information" / name)


def build_docs(script_decisions: list[dict[str, str]], offset_audits: list[dict[str, Any]]) -> None:
    retained = "\n".join(f"- `{row['script']}` — {row['reason']}" for row in script_decisions if row["decision"] == "retained")
    dropped = "\n".join(f"- `{row['script']}` — {row['reason']}" for row in script_decisions if row["decision"] == "dropped")
    reproduce = f"""# Reproducing the Reported Results

## Environment

```bash
conda env create -f environment.yml
conda activate hotel-fusion
```

Python 3.10 is the reference environment. Ridge/diagnostic stages are CPU and I/O intensive; low GPU utilization is expected. GPUs are used only for extraction and transformer training.

## Upstream data

Obtain the corpora from their distributors as documented in `DATA.md`. No review text is included in this release.

## Canonical split

```bash
python splits/expand_splits.py --verify
python splits/expand_splits.py --output-dir outputs
```

Expected assignment SHA-256: `{SPLIT_SHA256}`.

## Commands

```bash
python scripts/00_sanity_check.py
python scripts/01_prepare_dataset.py
python splits/expand_splits.py --output-dir outputs

# Optional GPU extraction (use released sanitized extraction data for exact reproduction)
python scripts/06_extract_aspects_llm.py --profile qwen25_7b --resume

# Published/original protocol and diagnostics
python scripts/08_phase0_reproduce.py
python scripts/17_phase2_schema_forensics.py
python scripts/18_phase2_schema_enforcement.py
python scripts/28_phase3_regularization_curves.py --stage all

# Unified U0 primary models and ablations
python scripts/41_phase4_u0_models.py --stage primary --folds 1,2,3,4,5 --inner-jobs 5
python scripts/41_phase4_u0_models.py --stage ablations --variants A2,A3,A5,A7,A8 --folds 1,2,3,4,5 --inner-jobs 5

# Booking external validation
python scripts/37_phase3_booking_prepare.py
python scripts/42_phase4_booking_u0.py --folds 1,2,3,4,5 --inner-jobs 5

# Consolidation and U0 inference
python scripts/43_phase4_consolidate.py
python scripts/44_phase5_discrepancy_strata.py
python scripts/45_phase5_e2_u0_missingness.py
python scripts/46_phase5_u0_coefficient_tables.py --bootstrap-resamples 10000 --folds 1,2,3,4,5
python scripts/47_finalize_phase5.py
python scripts/48_phase6a_assemble.py
```

## Expected runtime

- Preprocessing/split expansion: minutes on local SSD storage.
- Full Qwen extraction: approximately 31 wall-clock hours with two independent RTX 5090D endpoints at the measured pilot throughput (about 62 aggregate GPU-hours); hardware and server batching materially affect this value.
- U0 sparse models: approximately one to several CPU-hours depending on core count and TF-IDF cache state.
- Transformer baselines and 10,000-hotel bootstrap inference: many compute-hours; use the resumable fold/cache options.
- Phase 6 release assembly: minutes; CPU only.

Exact published numbers require the released sanitized structured extraction fields because stochastic re-extraction is not record-level deterministic.

## Retained scripts

{retained}

## Dropped scripts

{dropped}
"""
    write_text(GITHUB / "docs/REPRODUCE.md", reproduce)

    offset_lines = "\n".join(
        f"- `{Path(row['destination']).name}`: {row['matched']:,}/{row['evidence_items']:,} matched ({row['match_rate']:.6%}); first occurrence used for {row['ambiguous_first_occurrence']:,} ambiguous matched excerpts."
        for row in offset_audits
    )
    data_doc = f"""# Data and Privacy

## Upstream corpora

No upstream review text is redistributed.

### TripAdvisor

- Obtain the Hotel Reviews corpus from its distributor and place `reviews.csv` and `offerings.csv` under `data_ori/`.
- Required review fields include `title`, `text`, nested `ratings`, nested `author`, and `offering_id`; offering metadata comes from `offerings.csv`.
- The repository did **not** persist an authoritative TripAdvisor download timestamp. The local CSV mtime of 2023-08-30 is archive metadata and must not be described as the download date.
- After `scripts/01_prepare_dataset.py`, the accepted cleaned corpus has 877,808 reviews from 3,945 hotels.

### Booking.com external validation

- Source used: Hugging Face release `Dricz/515k-Hotel-Reviews-In-Europe`, commit `cdaea79962c5d667dd5177956ef0353001f6ade3`.
- Persisted download time: `2026-08-02T06:40:24.716734+00:00`.
- Source SHA-256: `a4810c2757934f0a826a1b16a437eb67a38be45b1a22ad56772afce0b6c11af9`.
- The source has 515,738 rows; preprocessing retains 514,935 reviews, 1,493 properties, dated 2015-08-04 through 2017-08-03.
- The Hugging Face mirror and Kaggle release have matching data identity in the project's provenance audit but display different licence labels. Those labels are documented, not interpreted as proof that either uploader can relicense the underlying reviews.

## OSF deposit

- `extraction/`: structured numeric/categorical extraction fields only. The only string column is `traveler_segment`.
- `predictions/`: consolidated held-out predictions. TripAdvisor model columns are clipped to [1,5].
- Booking `y_true` and unsuffixed model columns use the native score scale; `y_true_rescaled_1_5` and columns ending `__rescaled_1_5` provide the manuscript's 1-5 scale.
- `coefficients/` and `diagnostics/`: complete machine-readable inference and quality tables.

## Evidence-offset convention

Aspect IDs are `0=service`, `1=cleanliness`, `2=location`, `3=rooms`, `4=value`, `5=sleep_quality`.
Offsets are Python character indices into the cleaned `review_text` produced by `scripts/01_prepare_dataset.py`. Matching uses `review_text.lower().find(evidence.lower())`; if multiple matches occur, the first is retained. Unmatched evidence items remain in the table with `start=end=-1` and `matched=false`.

{offset_lines}

The evidence strings themselves, response JSON, raw model output, prompts attached to rows, cleaned review text, and all upstream CSVs are excluded.
"""
    write_text(GITHUB / "docs/DATA.md", data_doc)
    write_text(OSF / "README.md", data_doc.replace("`scripts/", "the released preprocessing script `scripts/"))
    write_text(OSF / "extraction/README.md", """# Extraction Derivatives

Sanitized extraction Parquets contain only `row_id`, `ok`, `traveler_segment`, six int8 aspect scores, and six boolean mention flags (plus `seed` for the determinism table). No review text, evidence text, raw response, summary, prompt, or latency field is present.

Evidence-offset Parquets use `row_id:uint32, aspect_id:uint8, evidence_rank:uint8, start:int32, end:int32, matched:bool`. Offsets refer to cleaned `review_text`; case-normalized literal matching and the first occurrence convention are used. Unmatched items use `start=end=-1`.
""")


def describe_release_file(relative: str) -> str:
    if relative == "README.md":
        return "Release overview and data-use guidance."
    if relative.endswith("MANIFEST.csv"):
        return "File-size and SHA-256 manifest (self-entry intentionally omitted)."
    if "supporting_information" in relative:
        return "Journal Supporting Information workbook."
    if "evidence_offsets" in relative:
        return "Sanitized character offsets for model-cited evidence; no evidence text."
    if relative.endswith(".parquet") and "extraction/" in relative:
        return "Sanitized structured extraction derivative; no review/evidence text."
    if "predictions/" in relative:
        return "Consolidated held-out prediction or per-hotel metric table."
    if "coefficients/" in relative:
        return "Complete U0 coefficient or goodness-of-fit table."
    if "diagnostics/" in relative or "results/" in relative:
        return "Machine-readable table supporting a manuscript result."
    if relative.startswith("scripts/") or relative.startswith("src/"):
        return "Reproducible analysis source code."
    if relative.startswith("figures/"):
        return "Manuscript or supplementary figure."
    if relative.startswith("splits/"):
        return "Canonical grouped fold assignment or verification utility."
    if relative.startswith("docs/"):
        return "Reproduction/data documentation."
    return "Release metadata, configuration, environment, or licence file."


def build_manifest(tree: Path) -> pd.DataFrame:
    manifest_path = tree / "MANIFEST.csv"
    records = []
    for path in sorted(tree.rglob("*")):
        if not path.is_file() or path == manifest_path or ".git" in path.relative_to(tree).parts:
            continue
        relative = path.relative_to(tree).as_posix()
        records.append({
            "relative_path": relative,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "description": describe_release_file(relative),
        })
    frame = pd.DataFrame(records)
    frame.to_csv(manifest_path, index=False)
    return frame


def forbidden_column_audit(tree: Path) -> list[dict[str, Any]]:
    audits = []
    for path in sorted(tree.rglob("*")):
        if not path.is_file() or ".git" in path.relative_to(tree).parts:
            continue
        suffix = path.suffix.lower()
        columns: list[str] = []
        if suffix == ".parquet":
            columns = pq.ParquetFile(path).schema_arrow.names
        elif suffix == ".csv":
            columns = list(pd.read_csv(path, nrows=0).columns)
        elif suffix == ".xlsx":
            workbook = load_workbook(path, read_only=True, data_only=True)
            for worksheet in workbook.worksheets:
                headers = [str(cell.value) for cell in next(worksheet.iter_rows(min_row=1, max_row=1)) if cell.value is not None]
                forbidden = sorted(set(header.lower() for header in headers) & FORBIDDEN_DATA_COLUMNS)
                audits.append({"path": path.relative_to(STAGING).as_posix(), "sheet": worksheet.title, "forbidden_columns": forbidden})
            workbook.close()
            continue
        if columns:
            forbidden = sorted(set(column.lower() for column in columns) & FORBIDDEN_DATA_COLUMNS)
            audits.append({"path": path.relative_to(STAGING).as_posix(), "sheet": "", "forbidden_columns": forbidden})
    failures = [row for row in audits if row["forbidden_columns"]]
    if failures:
        raise AssertionError(f"forbidden public data columns found: {failures[:10]}")
    return audits


def tree_stats(tree: Path) -> dict[str, Any]:
    files = [path for path in tree.rglob("*") if path.is_file() and ".git" not in path.relative_to(tree).parts]
    largest = sorted(files, key=lambda path: path.stat().st_size, reverse=True)[:10]
    return {
        "files": len(files),
        "bytes": sum(path.stat().st_size for path in files),
        "largest": [{"path": path.relative_to(tree).as_posix(), "bytes": path.stat().st_size} for path in largest],
    }


def verify_expected_summaries(trip_wide: pd.DataFrame, per_fold: pd.DataFrame) -> list[dict[str, Any]]:
    summary = metric_summary(per_fold[per_fold["model_id"].isin(TRIP_FULL_MODELS)])
    records = []
    phase0 = json.loads((OUTPUTS / "phase0_reproduction_gate.json").read_text(encoding="utf-8"))
    expected = {row["model_id"]: row["reproduced"] for row in phase0["comparisons"]}
    panel_b = pd.read_csv(OUTPUTS / "phase4/u1_panel_b_u0.csv")
    for row in panel_b.itertuples(index=False):
        if row.model_id in TRIP_FULL_MODELS:
            expected[row.model_id] = {"mae_mean": row.mae_mean, "rmse_mean": row.rmse_mean}
    ablations = pd.read_csv(OUTPUTS / "phase4/u4_ablation_u0.csv")
    for row in ablations.itertuples(index=False):
        if row.model_id in TRIP_FULL_MODELS:
            expected[row.model_id] = {"mae_mean": row.mae_mean, "rmse_mean": row.rmse_mean}
    for model_id, values in expected.items():
        actual = summary[summary["model_id"].eq(model_id)]
        if len(actual) != 1:
            raise AssertionError(f"missing wide summary for {model_id}")
        mae_delta = abs(float(actual.iloc[0]["mae_mean"]) - float(values["mae_mean"]))
        rmse_delta = abs(float(actual.iloc[0]["rmse_mean"]) - float(values["rmse_mean"]))
        passed = max(mae_delta, rmse_delta) <= 1e-9
        records.append({"model_id": model_id, "mae_delta": mae_delta, "rmse_delta": rmse_delta, "passed_1e_9": passed})
    return records


def commit_staging() -> None:
    for name in ("github", "osf"):
        destination = RELEASE / name
        source = STAGING / name
        if destination.exists():
            resolved = destination.resolve()
            if resolved.parent != RELEASE.resolve() or resolved.name != name:
                raise RuntimeError(f"unsafe generated-tree replacement target: {resolved}")
            shutil.rmtree(destination)
        source.rename(destination)
    STAGING.rmdir()


def markdown_table(frame: pd.DataFrame) -> str:
    return frame.to_markdown(index=False) if len(frame) else "_No rows._"


def build_report(
    extraction_audits: list[dict[str, Any]],
    offset_audits: list[dict[str, Any]],
    model_filter: pd.DataFrame,
    excluded_count: int,
    script_decisions: list[dict[str, str]],
    split_audit: dict[str, Any],
    metric_audits: list[dict[str, Any]],
    expected_audits: list[dict[str, Any]],
    privacy_audit_rows: int,
    github_manifest: pd.DataFrame,
    osf_manifest: pd.DataFrame,
    github_stats: dict[str, Any],
    osf_stats: dict[str, Any],
    github_manifest_hash: str,
    osf_manifest_hash: str,
) -> str:
    extraction_frame = pd.DataFrame(extraction_audits)[["destination", "rows", "bytes", "sha256", "string_columns"]]
    offset_frame = pd.DataFrame(offset_audits)[["destination", "evidence_items", "matched", "match_rate", "ambiguous_first_occurrence", "bytes", "sha256"]]
    offset_frame["match_rate"] = offset_frame["match_rate"].map(lambda value: f"{value:.6%}")
    retained = pd.DataFrame([row for row in script_decisions if row["decision"] == "retained"])
    dropped = pd.DataFrame([row for row in script_decisions if row["decision"] == "dropped"])
    assertion_rows = [
        {"Assertion": "No forbidden review/evidence/raw-response/prompt data columns", "Result": "PASS", "Evidence": f"{privacy_audit_rows} tabular sheet/file schemas checked"},
        {"Assertion": "Canonical fold assignment SHA-256", "Result": "PASS" if split_audit["sha256"] == SPLIT_SHA256 else "FAIL", "Evidence": split_audit["sha256"]},
        {"Assertion": "Wide float32 values equal the source predictions cast to float32", "Result": "PASS" if all(row["serialized_value_max_abs_difference"] == 0 for row in metric_audits) else "FAIL", "Evidence": f"{len(metric_audits)} full-corpus model columns; max value delta={max(row['serialized_value_max_abs_difference'] for row in metric_audits):.3e}"},
        {"Assertion": "Wide predictions preserve source fold-mean metrics to 1e-9", "Result": "PASS" if all(row["passed_1e_9"] for row in metric_audits) else "FAIL", "Evidence": f"{sum(row['passed_1e_9'] for row in metric_audits)}/{len(metric_audits)} pass; max float32 quantization delta={max(row['max_abs_fold_mean_metric_difference'] for row in metric_audits):.3e}"},
        {"Assertion": "Phase 4 six-decimal summary CSV values reproduced to 1e-9", "Result": "PASS" if all(row["passed_1e_9"] for row in expected_audits) else "FAIL", "Evidence": f"{sum(row['passed_1e_9'] for row in expected_audits)}/{len(expected_audits)} pass; max rounding delta={max(max(row['mae_delta'], row['rmse_delta']) for row in expected_audits):.3e}"},
        {"Assertion": "Every published model ID is in Part 3 filter", "Result": "PASS" if set(model_filter[model_filter["included"]].model_id) == ALL_RELEASE_MODEL_IDS else "FAIL", "Evidence": f"{len(ALL_RELEASE_MODEL_IDS)} included IDs/configurations"},
        {"Assertion": "GitHub tree under 100 MB and no file over 50 MB", "Result": "PASS" if github_stats["bytes"] < 100_000_000 and github_stats["largest"][0]["bytes"] < 50_000_000 else "FAIL", "Evidence": f"total={github_stats['bytes']:,}; largest={github_stats['largest'][0]['bytes']:,}"},
    ]
    github_largest = pd.DataFrame(github_stats["largest"])
    osf_largest = pd.DataFrame(osf_stats["largest"])
    quantization_failures = [row for row in metric_audits if not row["passed_1e_9"]]
    reported_failures = [row for row in expected_audits if not row["passed_1e_9"]]
    quantization_note = (
        "All 28 stored model columns are exactly equal to the accepted source predictions after the required float32 cast. "
        f"The stricter comparison against the original float64 metrics exceeds 1e-9 for {len(quantization_failures)} model(s) "
        f"(maximum fold-mean delta {max((row['max_abs_fold_mean_metric_difference'] for row in metric_audits), default=0.0):.3e}). "
        "This is the deterministic numerical consequence of the mandated float32 schema, not row/fold/model misalignment. "
        "The original float64 per-fold metrics remain in the GitHub results tables. "
        "The separate Phase 4 summary comparison is against CSV values rounded to six decimals; its sub-5e-7 differences "
        "are therefore expected reporting-precision differences and are not evidence of prediction or fold misalignment."
    )
    return f"""# Phase 6B — Public Release Preparation

- generated_utc: {datetime.now(timezone.utc).isoformat()}
- mode: CPU-only derivation/copy; no re-extraction, no refitting
- source artefacts: unchanged
- upload status: local preparation complete; no remote repository/deposit created

## Sanitized Extraction Derivatives

Only `row_id`, `ok`, `traveler_segment`, six `int8` scores, and six boolean mention flags survive (plus `seed` in the determinism table). `traveler_segment` is the only string column.

{markdown_table(extraction_frame)}

## Evidence Offsets

Offsets index the cleaned `review_text` emitted by `scripts/01_prepare_dataset.py`. Matching is case-normalized literal substring matching; the first occurrence is selected; unmatched items use `start=end=-1`.

{markdown_table(offset_frame)}

## Model Filter

- included release model IDs/configurations: {len(ALL_RELEASE_MODEL_IDS)}
- prediction-harness directory model IDs included: {int(model_filter[model_filter['included']].model_id.isin(DIRECTORY_MODEL_FILTER).sum())}
- excluded prediction-harness model IDs: {excluded_count}
- missing required model IDs: 0

The full auditable inclusion/exclusion list is `release/github/results/model_filter.csv`.

## Script Retention

### Retained ({len(retained)})

{markdown_table(retained)}

### Dropped ({len(dropped)})

{markdown_table(dropped)}

## Verification Assertions

{markdown_table(pd.DataFrame(assertion_rows))}

### Float32 precision note

{quantization_note}

- source-vs-float32 fold-mean failures at 1e-9: `{', '.join(row['model_id'] for row in quantization_failures) or 'none'}`
- six-decimal Phase 4 summary comparisons exceeding 1e-9: `{', '.join(row['model_id'] for row in reported_failures) or 'none'}`

## Manifests and Tree Sizes

| Tree | Files including manifest | Bytes | Manifest SHA-256 |
| --- | ---: | ---: | --- |
| `release/github` | {github_stats['files']} | {github_stats['bytes']:,} | `{github_manifest_hash}` |
| `release/osf` | {osf_stats['files']} | {osf_stats['bytes']:,} | `{osf_manifest_hash}` |

Manifest self-entries are intentionally omitted because a file cannot contain its own stable cryptographic hash. Their hashes are recorded above and in `release/MANIFEST_HASHES.txt`; every other published file is listed in its tree's `MANIFEST.csv`.

### Ten largest GitHub files

{markdown_table(github_largest)}

### Ten largest OSF files

{markdown_table(osf_largest)}

## Supporting Information Upload Check

S1, S2, and S3 are all included and hash-tracked at:

- `release/github/supporting_information/S1_File.xlsx`
- `release/github/supporting_information/S2_File.xlsx`
- `release/github/supporting_information/S3_File.xlsx`

They remain separate files suitable for journal upload as well as inclusion in the GitHub release.

## README Review

The Claude-authored README was retained after correcting release-state/provenance issues: the placeholder DOI was removed, OSF is described as pending rather than already deposited, the unsupported TripAdvisor download/release date and unverified licence assertion were removed, the split-expansion command now points to `splits/expand_splits.py`, and the citation no longer claims a publication venue before acceptance. A copy-paste Quickstart, two-endpoint vLLM setup, output/validation catalogue, troubleshooting guidance, and the required collapsed Chinese section were added. Five workstation-specific paths in retained scripts were replaced with portable environment/model placeholders in the release copy only.

## Upload Handoff

No authentication or remote mutation was attempted. `release/github/` is ready to initialize/push as a Git repository after account/repository details are supplied. `release/osf/` is ready for OSF upload. Add the assigned URLs/DOI to README and CITATION metadata only after the remote records exist, then regenerate manifests for those metadata edits.
"""


def main() -> None:
    args = parse_args()
    for path in (OUTPUTS / "phase6a_report.md", OUTPUTS / "release_inventory.csv", ROOT / "README.md"):
        if not path.exists():
            raise FileNotFoundError(path)
    # Read all Phase 6A inventory rows as an explicit provenance gate.
    inventory = pd.read_csv(OUTPUTS / "release_inventory.csv")
    if len(inventory) < 1_000:
        raise AssertionError("Phase 6A release inventory appears incomplete")
    safe_reset_staging()

    extraction_audits = build_sanitized_extractions()
    if args.skip_offsets:
        raise RuntimeError("--skip-offsets is for development only and cannot produce a final Phase 6B release")
    offset_audits = build_offsets()

    trip_wide, trip_per_fold, metric_audits = build_tripadvisor_wide()
    booking_wide, booking_per_fold = build_booking_wide()
    subsample_wide, subsample_per_fold = build_subsample_wide()
    per_hotel = build_per_hotel_metrics()
    per_fold_rows = trip_per_fold + booking_per_fold + subsample_per_fold
    per_fold_frame = pd.DataFrame(per_fold_rows)
    expected_audits = verify_expected_summaries(trip_wide, per_fold_frame)
    build_coefficients_diagnostics_and_results(per_fold_rows)
    model_filter, excluded_count = build_model_filter()

    script_decisions = build_code_tree(args.conda_env)
    split_audit = build_split_release()
    build_figures_and_supporting()
    build_docs(script_decisions, offset_audits)

    privacy_rows = forbidden_column_audit(GITHUB) + forbidden_column_audit(OSF)
    github_manifest = build_manifest(GITHUB)
    osf_manifest = build_manifest(OSF)
    github_stats_staging = tree_stats(GITHUB)
    osf_stats_staging = tree_stats(OSF)
    if github_stats_staging["bytes"] >= 100_000_000 or github_stats_staging["largest"][0]["bytes"] >= 50_000_000:
        raise AssertionError(f"GitHub size gate failed: {github_stats_staging}")
    manifest_hashes = {
        "github/MANIFEST.csv": sha256_file(GITHUB / "MANIFEST.csv"),
        "osf/MANIFEST.csv": sha256_file(OSF / "MANIFEST.csv"),
    }
    write_text(RELEASE / "MANIFEST_HASHES.txt", "\n".join(f"{digest}  {path}" for path, digest in manifest_hashes.items()))

    # Release large in-memory objects before final filesystem checks.
    del trip_wide, booking_wide, subsample_wide, per_hotel
    gc.collect()
    commit_staging()
    github_final = RELEASE / "github"
    osf_final = RELEASE / "osf"
    github_stats = tree_stats(github_final)
    osf_stats = tree_stats(osf_final)
    # Verify manifests still describe all non-manifest files exactly after atomic move.
    for tree, manifest in ((github_final, github_manifest), (osf_final, osf_manifest)):
        listed = set(manifest["relative_path"])
        actual = {path.relative_to(tree).as_posix() for path in tree.rglob("*") if path.is_file() and path.name != "MANIFEST.csv" and ".git" not in path.relative_to(tree).parts}
        if listed != actual:
            raise AssertionError(f"manifest coverage mismatch in {tree}: missing={actual-listed}, extra={listed-actual}")
        for row in manifest.itertuples(index=False):
            path = tree / row.relative_path
            if path.stat().st_size != row.bytes or sha256_file(path) != row.sha256:
                raise AssertionError(f"manifest content mismatch: {path}")

    report = build_report(
        extraction_audits, offset_audits, model_filter, excluded_count, script_decisions,
        split_audit, metric_audits, expected_audits, len(privacy_rows), github_manifest,
        osf_manifest, github_stats, osf_stats, manifest_hashes["github/MANIFEST.csv"],
        manifest_hashes["osf/MANIFEST.csv"],
    )
    write_text(OUTPUTS / "phase6b_report.md", report)
    log(f"PASS: GitHub tree {github_stats['files']} files, {github_stats['bytes']:,} bytes")
    log(f"PASS: OSF tree {osf_stats['files']} files, {osf_stats['bytes']:,} bytes")
    log(f"PASS: canonical split {split_audit['sha256']}")
    log(f"PASS: {len(metric_audits)} full-corpus model columns exactly equal source predictions after float32 cast")
    strict_passes = sum(row["passed_1e_9"] for row in metric_audits)
    if strict_passes == len(metric_audits):
        log(f"PASS: all {strict_passes} float32 columns preserve fold-mean metrics to 1e-9")
    else:
        log(f"NUMERICAL NOTE: {strict_passes}/{len(metric_audits)} float32 columns preserve original float64 fold-mean metrics to 1e-9")
    log(f"Saved: {OUTPUTS / 'phase6b_report.md'}")


if __name__ == "__main__":
    main()
