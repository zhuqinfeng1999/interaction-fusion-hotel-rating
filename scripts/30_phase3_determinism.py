#!/usr/bin/env python3
"""Phase 3 Experiment V: repeated extraction determinism on 20k reviews."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
SEED = 42
RUN_SEEDS = (42, 43, 44)


def load_legacy_module() -> Any:
    path = ROOT / "scripts" / "06_extract_aspects_llm.py"
    spec = importlib.util.spec_from_file_location("phase3_legacy_extractor", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


LEGACY = load_legacy_module()
ASPECTS = tuple(LEGACY.ASPECTS)


def log(message: str) -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="qwen25_7b")
    parser.add_argument("--sample-size", type=int, default=20_000)
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument("--max-chars", type=int, default=4_000)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument(
        "--endpoints",
        default=None,
        help="Comma-separated endpoints; defaults to config/local_llm.yaml.",
    )
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Analyze completed run parquets without contacting vLLM.",
    )
    return parser.parse_args()


def parse_seeds(raw: str) -> List[int]:
    seeds = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if len(seeds) < 3:
        raise ValueError("Experiment V requires at least three RNG seeds")
    return seeds


def fold_assignment(output_dir: Path, n_rows: int) -> np.ndarray:
    assignment = np.full(n_rows, -1, dtype=np.int8)
    for fold in range(1, 6):
        arrays = np.load(output_dir / f"splits_fold{fold}.npz")
        assignment[arrays["test_idx"].astype(np.int64)] = fold
    if np.any(assignment < 1):
        raise ValueError("outer fold assignment incomplete")
    return assignment


def load_or_make_sample(
    output_dir: Path, sample_size: int, seed: int
) -> pd.DataFrame:
    phase3 = output_dir / "phase3"
    sample_path = phase3 / f"v_sample_n{sample_size}_seed{seed}.npz"
    columns = ["id", "title", "review_text", "rating_overall"]
    frame = pd.read_parquet(output_dir / "dataset_clean.parquet", columns=columns)
    frame = frame.reset_index(drop=True)
    frame["row_index"] = np.arange(len(frame), dtype=np.int64)
    folds = fold_assignment(output_dir, len(frame))
    if sample_path.exists():
        sampled = np.load(sample_path)["row_id"].astype(np.int64)
    else:
        strata = pd.Series(folds.astype(str)) + "_" + frame[
            "rating_overall"
        ].astype(int).astype(str)
        indices = np.arange(len(frame), dtype=np.int64)
        sampled, _ = train_test_split(
            indices,
            train_size=min(sample_size, len(frame)),
            random_state=seed,
            stratify=strata if sample_size >= strata.nunique() else None,
        )
        sampled = np.sort(np.asarray(sampled, dtype=np.int64))
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            sample_path,
            row_id=sampled,
            outer_fold=folds[sampled],
            rating_overall=frame.iloc[sampled]["rating_overall"].to_numpy(
                dtype=np.int8
            ),
            seed=np.asarray([seed]),
        )
    sample = frame.iloc[sampled].copy()
    if sample["row_index"].duplicated().any():
        raise ValueError("duplicate row IDs in V sample")
    return sample


def cache_path(
    output_dir: Path,
    profile: str,
    condition: str,
    sample_size: int,
    seed: int,
    endpoint_idx: int,
) -> Path:
    return (
        output_dir
        / "phase3"
        / "v_cache"
        / f"{profile}_{condition}_n{sample_size}_seed{seed}_ep{endpoint_idx}"
        / "cache.sqlite"
    )


def output_path(
    output_dir: Path, profile: str, condition: str, sample_size: int, seed: int
) -> Path:
    return (
        output_dir
        / "phase3"
        / f"v_{profile}_{condition}_n{sample_size}_seed{seed}.parquet"
    )


def build_infos(
    sample: pd.DataFrame,
    model: str,
    max_chars: int,
) -> List[Any]:
    return [
        LEGACY.build_row_info(
            row=row,
            row_id=int(row["row_index"]),
            model=model,
            title_col="title",
            text_col="review_text",
            max_chars=max_chars,
        )
        for _, row in sample.iterrows()
    ]


def cache_to_output(info: Any, cached: Dict[str, Any], profile: Any) -> Dict[str, Any]:
    response_json = cached.get("response_json") or ""
    parsed = LEGACY.safe_json_loads(response_json)
    normalized = LEGACY.normalize_response(parsed)
    return LEGACY.build_output_row(
        row_info=info,
        normalized=normalized,
        ok=int(cached.get("ok", 0)),
        error=cached.get("error"),
        prompt_tokens=cached.get("prompt_tokens"),
        completion_tokens=cached.get("completion_tokens"),
        elapsed_ms=cached.get("elapsed_ms"),
        model=profile.model,
        prompt_hash=info.prompt_hash,
        response_json=response_json or json.dumps(normalized, ensure_ascii=True),
    )


def result_to_output(info_map: Dict[int, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    info = info_map[int(result["row_id"])]
    return LEGACY.build_output_row(
        row_info=info,
        normalized=result["normalized"],
        ok=int(result["ok"]),
        error=result.get("error"),
        prompt_tokens=result.get("prompt_tokens"),
        completion_tokens=result.get("completion_tokens"),
        elapsed_ms=result.get("elapsed_ms"),
        model=result["model"],
        prompt_hash=result["prompt_hash"],
        response_json=result["response_json"],
    )


def run_endpoint(
    endpoint_idx: int,
    endpoint: str,
    infos: Sequence[Any],
    profile: Any,
    seed: int,
    condition: str,
    sample_size: int,
    output_dir: Path,
    resume: bool,
) -> Dict[str, Any]:
    path = cache_path(
        output_dir, profile.name, condition, sample_size, seed, endpoint_idx
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    LEGACY.init_cache(connection)
    keys = [info.cache_key for info in infos]
    cached = LEGACY.fetch_cache_map(connection, keys) if resume else {}
    hits = {
        key: value for key, value in cached.items() if int(value.get("ok", 0)) == 1
    }
    pending = [info for info in infos if info.cache_key not in hits]
    endpoint_profile = replace(profile, base_url=endpoint, endpoints=[endpoint])
    started = time.perf_counter()
    results, _ = LEGACY.run_requests_with_progress(
        rows=list(pending),
        profile=endpoint_profile,
        seed=seed,
        concurrency=profile.concurrency,
        fail_log_path=output_dir
        / "phase3"
        / f"v_fail_{profile.name}_{condition}_seed{seed}_ep{endpoint_idx}.jsonl",
        progress_label=f"V {condition} seed{seed} ep{endpoint_idx}",
        progress_position=endpoint_idx,
    )
    LEGACY.upsert_cache(connection, results)
    all_cached = LEGACY.fetch_cache_map(connection, keys)
    connection.close()
    wall = time.perf_counter() - started
    return {
        "endpoint_idx": endpoint_idx,
        "endpoint": endpoint,
        "cache_path": str(path.relative_to(ROOT)),
        "cache_hits": len(hits),
        "requested_n": len(pending),
        "processed_n": len(results),
        "ok_new_n": int(sum(int(row["ok"]) for row in results)),
        "wall_seconds": wall,
        "rows_per_second": len(results) / wall if wall else None,
        "cached_rows": all_cached,
    }


def run_condition_seed(
    sample: pd.DataFrame,
    base_profile: Any,
    condition: str,
    temperature: float,
    seed: int,
    output_dir: Path,
    max_chars: int,
    resume: bool,
) -> Dict[str, Any]:
    path = output_path(
        output_dir, base_profile.name, condition, len(sample), seed
    )
    stats_path = path.with_suffix(".stats.json")
    if resume and path.exists() and stats_path.exists():
        log(f"V cache hit: {path.name}")
        return json.loads(stats_path.read_text(encoding="utf-8"))
    profile = replace(base_profile, temperature=temperature)
    infos = build_infos(sample, profile.model, max_chars)
    info_map = {int(info.row_id): info for info in infos}
    endpoint_infos = [
        [info for info in infos if info.row_id % len(profile.endpoints) == endpoint_idx]
        for endpoint_idx in range(len(profile.endpoints))
    ]
    started = time.perf_counter()
    endpoint_stats: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(profile.endpoints)) as executor:
        futures = [
            executor.submit(
                run_endpoint,
                endpoint_idx,
                endpoint,
                endpoint_infos[endpoint_idx],
                profile,
                seed,
                condition,
                len(sample),
                output_dir,
                resume,
            )
            for endpoint_idx, endpoint in enumerate(profile.endpoints)
        ]
        for future in as_completed(futures):
            endpoint_stats.append(future.result())
    outputs: List[Dict[str, Any]] = []
    for endpoint_idx, infos_for_endpoint in enumerate(endpoint_infos):
        stat = next(row for row in endpoint_stats if row["endpoint_idx"] == endpoint_idx)
        for info in infos_for_endpoint:
            cached = stat["cached_rows"].get(info.cache_key)
            if cached is None:
                raise ValueError(f"missing V cache row {info.row_id}")
            outputs.append(cache_to_output(info, cached, profile))
        del stat["cached_rows"]
    output = pd.DataFrame(outputs).sort_values("row_id").reset_index(drop=True)
    if len(output) != len(sample) or output["row_id"].duplicated().any():
        raise ValueError("V output alignment failed")
    output.to_parquet(path, index=False)
    wall = time.perf_counter() - started
    stats = {
        "condition": condition,
        "temperature": temperature,
        "seed": seed,
        "n": len(output),
        "ok_n": int(output["ok"].eq(1).sum()),
        "ok_rate": float(output["ok"].eq(1).mean()),
        "wall_seconds": wall,
        "combined_rows_per_second": len(output) / wall if wall else None,
        "endpoint_stats": sorted(endpoint_stats, key=lambda row: row["endpoint_idx"]),
        "output_path": str(path.relative_to(ROOT)),
    }
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    log(
        f"V completed {condition} seed={seed}: ok={stats['ok_rate']:.4%} "
        f"wall={wall / 60:.1f}min rows/s={stats['combined_rows_per_second']:.3f}"
    )
    return stats


def canonical_json(value: Any) -> str:
    try:
        parsed = json.loads(str(value))
    except Exception:
        return str(value)
    return json.dumps(parsed, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def analyze_condition(
    output_dir: Path,
    profile: str,
    condition: str,
    temperature: float,
    sample_size: int,
    seeds: Sequence[int],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    frames = []
    stats = []
    columns = [
        "row_id",
        "ok",
        "response_json",
        "traveler_segment",
        *[f"aspect_{aspect}_score" for aspect in ASPECTS],
        *[f"aspect_{aspect}_mentioned" for aspect in ASPECTS],
    ]
    for seed in seeds:
        path = output_path(output_dir, profile, condition, sample_size, seed)
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_parquet(path, columns=columns).sort_values("row_id")
        frame = frame.set_index("row_id", verify_integrity=True)
        frames.append(frame)
        stats.append(
            json.loads(path.with_suffix(".stats.json").read_text(encoding="utf-8"))
        )
    common = frames[0].index
    if any(not common.equals(frame.index) for frame in frames[1:]):
        raise ValueError("V repeated-run row alignment failed")
    all_ok = np.logical_and.reduce([frame["ok"].eq(1).to_numpy() for frame in frames])
    records = [
        {
            "condition": condition,
            "temperature": temperature,
            "field": "full_record",
            "aspect": "all",
            "n_all": len(common),
            "n_all_ok": int(all_ok.sum()),
            "all_three_exact_rate_all": float(
                np.logical_and.reduce(
                    [
                        frames[0]["response_json"].map(canonical_json).to_numpy()
                        == frame["response_json"].map(canonical_json).to_numpy()
                        for frame in frames[1:]
                    ]
                ).mean()
            ),
            "all_three_exact_rate_all_ok": float(
                np.logical_and.reduce(
                    [
                        frames[0]["response_json"].map(canonical_json).to_numpy()
                        == frame["response_json"].map(canonical_json).to_numpy()
                        for frame in frames[1:]
                    ]
                )[all_ok].mean()
            )
            if all_ok.any()
            else None,
        }
    ]
    fields = [("segment", "all", "traveler_segment")]
    for aspect in ASPECTS:
        fields.append(("score", aspect, f"aspect_{aspect}_score"))
        fields.append(("mentioned", aspect, f"aspect_{aspect}_mentioned"))
    for field_type, aspect, column in fields:
        equal = np.logical_and.reduce(
            [frames[0][column].to_numpy() == frame[column].to_numpy() for frame in frames[1:]]
        )
        records.append(
            {
                "condition": condition,
                "temperature": temperature,
                "field": field_type,
                "aspect": aspect,
                "n_all": len(common),
                "n_all_ok": int(all_ok.sum()),
                "all_three_exact_rate_all": float(equal.mean()),
                "all_three_exact_rate_all_ok": float(equal[all_ok].mean())
                if all_ok.any()
                else None,
            }
        )
    summary = {
        "condition": condition,
        "temperature": temperature,
        "seeds": list(seeds),
        "n": len(common),
        "all_ok_n": int(all_ok.sum()),
        "run_stats": stats,
        "gpu_hours": float(
            sum(
                sum(float(endpoint["wall_seconds"]) for endpoint in run["endpoint_stats"])
                for run in stats
            )
            / 3600
        ),
    }
    return records, summary


def write_report(
    output_dir: Path,
    agreement: pd.DataFrame,
    summaries: Sequence[Dict[str, Any]],
    original_temperature: float,
) -> None:
    phase3 = output_dir / "phase3"
    agreement.to_csv(phase3 / "v_determinism_agreement.csv", index=False)
    payload = {
        "experiment": "V",
        "legacy_mechanism": "prompt + repair + retry (scripts/06_extract_aspects_llm.py)",
        "original_temperature_from_config": original_temperature,
        "conditions": list(summaries),
        "agreement": agreement.to_dict(orient="records"),
    }
    (phase3 / "v_determinism_results.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    full = agreement[agreement["field"] == "full_record"]
    score_mean = (
        agreement[agreement["field"] == "score"]
        .groupby("condition")["all_three_exact_rate_all_ok"]
        .mean()
    )
    mention_mean = (
        agreement[agreement["field"] == "mentioned"]
        .groupby("condition")["all_three_exact_rate_all_ok"]
        .mean()
    )
    original_condition = "original"
    exact = float(
        full[full["condition"] == original_condition][
            "all_three_exact_rate_all_ok"
        ].iloc[0]
    )
    supported = exact >= 0.999999
    verdict = (
        "SUPPORTED as written on this 20k sample: all successful normalized records exactly match across RNG seeds."
        if supported
        else "NOT SUPPORTED as an exact determinism claim: successful normalized records differ across RNG seeds. The accurate claim is reproducibility conditional on persisted cache entries, with empirical repeated-run agreement reported below."
    )
    lines = [
        "# Phase 3 Experiment V — Extractor Determinism",
        "",
        f"- original configured temperature: {original_temperature:g}",
        "- mechanism tested: legacy prompt/repair/retry extraction, not guided JSON",
        "- RNG seeds: " + ", ".join(map(str, summaries[0]["seeds"])),
        f"- total GPU-hours: {sum(item['gpu_hours'] for item in summaries):.3f}",
        "",
        "## Agreement",
        "",
        agreement.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## V3 verdict",
        "",
        f"**{verdict}**",
        "",
        f"Mean score exact agreement (original): {score_mean[original_condition]:.6f}; mean mention exact agreement: {mention_mean[original_condition]:.6f}.",
        "",
    ]
    if original_temperature == 0:
        lines.extend(
            [
                "V2 required no additional condition because the original configuration was already temperature=0/greedy. The three-seed original condition is therefore also the V2 greedy test.",
                "",
            ]
        )
    (phase3 / "experiment_v_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    output_dir = ROOT / "outputs"
    phase3 = output_dir / "phase3"
    phase3.mkdir(parents=True, exist_ok=True)
    seeds = parse_seeds(args.seeds)
    profile = LEGACY.load_profile(ROOT / "config" / "local_llm.yaml", args.profile)
    if args.endpoints:
        endpoints = [
            item.strip().rstrip("/")
            for item in args.endpoints.split(",")
            if item.strip()
        ]
        profile = replace(profile, endpoints=endpoints, base_url=endpoints[0])
    if args.concurrency:
        profile = replace(profile, concurrency=args.concurrency)
    sample = load_or_make_sample(output_dir, args.sample_size, SEED)
    conditions = [("original", float(profile.temperature))]
    if profile.temperature != 0:
        conditions.append(("greedy", 0.0))
    run_summaries = []
    if not args.analyze_only:
        for condition, temperature in conditions:
            for seed in seeds:
                run_summaries.append(
                    run_condition_seed(
                        sample,
                        profile,
                        condition,
                        temperature,
                        seed,
                        output_dir,
                        args.max_chars,
                        args.resume,
                    )
                )
    agreement_rows: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    for condition, temperature in conditions:
        records, summary = analyze_condition(
            output_dir, args.profile, condition, temperature, len(sample), seeds
        )
        agreement_rows.extend(records)
        summaries.append(summary)
    agreement = pd.DataFrame(agreement_rows)
    write_report(output_dir, agreement, summaries, float(profile.temperature))
    log(f"saved: {phase3 / 'experiment_v_report.md'}")


if __name__ == "__main__":
    main()
