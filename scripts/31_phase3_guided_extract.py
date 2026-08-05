#!/usr/bin/env python3
"""Phase 3 guided JSON extraction for L2/L3 and the external W sample."""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import sqlite3
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SEED = 42


def load_guided_module() -> Any:
    path = ROOT / "scripts" / "18_phase2_schema_enforcement.py"
    spec = importlib.util.spec_from_file_location("phase3_guided_extractor", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


GUIDED = load_guided_module()
ASPECTS = tuple(GUIDED.ASPECTS)


def log(message: str) -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extractor", choices=["qwen", "llama"], required=True)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--api-key", default="local-llm")
    parser.add_argument("--endpoints", default=None)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--max-chars", type=int, default=4_000)
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument("--timeout-sec", type=int, default=120)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--sample-file",
        type=Path,
        default=Path("outputs/phase3/l1_sample_n200000_seed42.npz"),
    )
    parser.add_argument(
        "--dataset-file",
        type=Path,
        default=Path("outputs/dataset_clean.parquet"),
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=None,
        help="Optional isolated cache root (required for external corpora).",
    )
    parser.add_argument("--output-tag", default=None)
    parser.add_argument("--runtime-prefix", default="l_guided_runtime")
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--stage-label", default=None)
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def make_profile(args: argparse.Namespace) -> Any:
    if args.extractor == "qwen":
        name = args.profile or "qwen25_7b"
        proxy_args = SimpleNamespace(
            endpoints=args.endpoints,
            max_tokens=args.max_tokens,
            concurrency=args.concurrency,
        )
        profile = GUIDED.load_profile(
            ROOT / "config" / "local_llm.yaml", name, proxy_args
        )
        return GUIDED.Profile(
            name=profile.name,
            endpoints=profile.endpoints,
            api_key=profile.api_key,
            model=profile.model,
            temperature=0.0,
            max_tokens=args.max_tokens,
            timeout_sec=args.timeout_sec,
            concurrency=args.concurrency,
            max_retries=args.max_retries,
        )
    name = args.profile or "llama31_8b"
    if not args.model:
        raise ValueError("--model is required for --extractor llama")
    if not args.endpoints:
        raise ValueError("--endpoints is required for --extractor llama")
    endpoints = tuple(
        item.strip().rstrip("/")
        for item in args.endpoints.split(",")
        if item.strip()
    )
    return GUIDED.Profile(
        name=name,
        endpoints=endpoints,
        api_key=args.api_key,
        model=args.model,
        temperature=0.0,
        max_tokens=args.max_tokens,
        timeout_sec=args.timeout_sec,
        concurrency=args.concurrency,
        max_retries=args.max_retries,
    )


def load_sample(dataset_path: Path, sample_path: Path) -> pd.DataFrame:
    if not sample_path.exists():
        raise FileNotFoundError(
            f"missing sample: {sample_path}"
        )
    row_ids = np.load(sample_path)["row_id"].astype(np.int64)
    dataset = pd.read_parquet(
        dataset_path, columns=["review_text"]
    ).reset_index(drop=True)
    dataset["row_id"] = np.arange(len(dataset), dtype=np.int64)
    sample = dataset.iloc[row_ids].copy()
    if not np.array_equal(sample["row_id"].to_numpy(), row_ids):
        raise ValueError("L sample row alignment failed")
    return sample


def endpoint_cache_path(
    output_dir: Path,
    extractor: str,
    profile_name: str,
    endpoint_idx: int,
    cache_root: Path | None = None,
) -> Path:
    if cache_root is not None:
        return cache_root / f"{profile_name}_ep{endpoint_idx}" / "cache.sqlite"
    if extractor == "qwen":
        # Deliberately reuse Phase 2 C's decoding-matched guided cache.
        return (
            output_dir
            / "phase2"
            / "guided_cache"
            / f"{profile_name}_ep{endpoint_idx}"
            / "cache.sqlite"
        )
    return (
        output_dir
        / "phase3"
        / "l_guided_cache"
        / f"{profile_name}_ep{endpoint_idx}"
        / "cache.sqlite"
    )


async def run_all_endpoints(
    sample: pd.DataFrame,
    profile: Any,
    extractor: str,
    output_dir: Path,
    resume: bool,
    seed: int,
    max_chars: int,
    cache_root: Path | None,
    stage_label: str,
) -> List[Dict[str, Any]]:
    shards = GUIDED.split_sample_by_endpoint(sample, profile.endpoints)
    tasks = []
    for endpoint_idx, (endpoint, shard) in enumerate(
        zip(profile.endpoints, shards)
    ):
        tasks.append(
            GUIDED.process_endpoint(
                endpoint_idx=endpoint_idx,
                endpoint=endpoint,
                shard=shard,
                profile=profile,
                cache_path=endpoint_cache_path(
                    output_dir,
                    extractor,
                    profile.name,
                    endpoint_idx,
                    cache_root,
                ),
                resume=resume,
                seed=seed,
                max_chars=max_chars,
                position=endpoint_idx,
                stage=stage_label,
            )
        )
    return list(await asyncio.gather(*tasks))


def read_selected_cache(
    output_dir: Path,
    extractor: str,
    profile: Any,
    row_ids: Sequence[int],
    cache_root: Path | None,
) -> pd.DataFrame:
    selected = set(int(value) for value in row_ids)
    frames = []
    for endpoint_idx in range(len(profile.endpoints)):
        path = endpoint_cache_path(
            output_dir, extractor, profile.name, endpoint_idx, cache_root
        )
        connection = GUIDED.init_guided_cache(path)
        frame = pd.read_sql_query(
            "SELECT * FROM cache WHERE schema_version=? AND prompt_version=? AND model=?",
            connection,
            params=(
                GUIDED.GUIDED_SCHEMA_VERSION,
                GUIDED.GUIDED_PROMPT_VERSION,
                profile.model,
            ),
        )
        connection.close()
        frames.append(frame[frame["row_id"].isin(selected)])
    result = pd.concat(frames, ignore_index=True)
    result = result.sort_values(["row_id", "created_at"]).drop_duplicates(
        "row_id", keep="last"
    )
    if len(result) != len(selected):
        raise ValueError(f"guided cache missing {len(selected) - len(result)} rows")
    return result.sort_values("row_id").reset_index(drop=True)


def output_frame(cache: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for record in cache.itertuples(index=False):
        try:
            parsed = json.loads(record.response_json) if record.response_json else {}
        except Exception:
            parsed = {}
        aspects = parsed.get("aspects", {}) if isinstance(parsed, dict) else {}
        row: Dict[str, Any] = {
            "row_id": int(record.row_id),
            "ok": int(record.ok),
            "error": record.error,
            "model": record.model,
            "prompt_hash": record.prompt_hash,
            "schema_version": record.schema_version,
            "prompt_version": record.prompt_version,
            "response_json": record.response_json,
            "raw_text": record.raw_text,
            "prompt_tokens": record.prompt_tokens,
            "completion_tokens": record.completion_tokens,
            "elapsed_ms": record.elapsed_ms,
            "attempt_count": record.attempt_count,
            "traveler_segment": parsed.get("traveler_segment", "unknown")
            if isinstance(parsed, dict)
            else "unknown",
            "overall_summary": parsed.get("overall_summary", "")
            if isinstance(parsed, dict)
            else "",
        }
        for aspect in ASPECTS:
            item = aspects.get(aspect, {}) if isinstance(aspects, dict) else {}
            row[f"aspect_{aspect}_score"] = int(item.get("score", 0) or 0)
            row[f"aspect_{aspect}_mentioned"] = bool(
                item.get("mentioned", False)
            )
        rows.append(row)
    output = pd.DataFrame(rows).sort_values("row_id").reset_index(drop=True)
    if output["row_id"].duplicated().any():
        raise ValueError("duplicate row_id in guided output")
    return output


def main() -> None:
    args = parse_args()
    output_dir = ROOT / "outputs"
    phase3 = output_dir / "phase3"
    phase3.mkdir(parents=True, exist_ok=True)
    sample_path = resolve_path(args.sample_file)
    dataset_path = resolve_path(args.dataset_file)
    cache_root = resolve_path(args.cache_root) if args.cache_root else None
    sample = load_sample(dataset_path, sample_path)
    profile = make_profile(args)
    tag = args.output_tag or f"{profile.name}_l1_n{len(sample)}"
    parquet_path = phase3 / f"aspects_guided_{tag}.parquet"
    stats_path = phase3 / f"{args.runtime_prefix}_{tag}.json"
    experiment = args.experiment or ("L2" if args.extractor == "llama" else "L3")
    stage_label = args.stage_label or f"L {args.extractor} guided"
    log(
        f"L guided extractor={args.extractor} model={profile.model} n={len(sample)} "
        f"endpoints={len(profile.endpoints)} concurrency/endpoint={profile.concurrency}"
    )
    started = time.perf_counter()
    endpoint_stats = asyncio.run(
        run_all_endpoints(
            sample,
            profile,
            args.extractor,
            output_dir,
            args.resume,
            args.seed,
            args.max_chars,
            cache_root,
            stage_label,
        )
    )
    cache = read_selected_cache(
        output_dir,
        args.extractor,
        profile,
        sample["row_id"].tolist(),
        cache_root,
    )
    output = output_frame(cache)
    output.to_parquet(parquet_path, index=False)
    wall = time.perf_counter() - started
    processed = sum(int(row["processed_n"]) for row in endpoint_stats)
    endpoint_gpu_seconds = sum(float(row["wall_seconds"]) for row in endpoint_stats)
    payload = {
        "experiment": experiment,
        "extractor": args.extractor,
        "profile": profile.name,
        "model": profile.model,
        "schema_version": GUIDED.GUIDED_SCHEMA_VERSION,
        "prompt_version": GUIDED.GUIDED_PROMPT_VERSION,
        "response_format": "json_schema strict=true",
        "sample_path": str(sample_path.relative_to(ROOT)),
        "dataset_path": str(dataset_path.relative_to(ROOT)),
        "cache_root": str(cache_root.relative_to(ROOT)) if cache_root else None,
        "sample_n": len(sample),
        "ok_n": int(output["ok"].eq(1).sum()),
        "ok_rate": float(output["ok"].eq(1).mean()),
        "processed_uncached_n": processed,
        "cache_hits": sum(int(row["cache_hits"]) for row in endpoint_stats),
        "wall_seconds": wall,
        "gpu_hours": endpoint_gpu_seconds / 3600,
        "combined_rows_per_second_uncached": processed / max(
            (float(row["wall_seconds"]) for row in endpoint_stats), default=1.0
        )
        if processed
        else None,
        "endpoint_stats": endpoint_stats,
        "output_path": str(parquet_path.relative_to(ROOT)),
    }
    stats_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log(
        f"L {args.extractor} complete: ok={payload['ok_rate']:.4%} "
        f"cache_hits={payload['cache_hits']} processed={processed} "
        f"wall={wall / 3600:.2f}h GPU-hours={payload['gpu_hours']:.2f}"
    )
    log(f"saved: {parquet_path}")
    log(f"saved: {stats_path}")


if __name__ == "__main__":
    main()
