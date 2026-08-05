#!/usr/bin/env python3
"""Phase 2 Experiment C: extraction enforcement audit and guided JSON trial."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import sqlite3
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import httpx
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.aspect_extractor_base import ASPECTS  # noqa: E402
from src.evaluation_harness import write_json  # noqa: E402


SEED = 42
PROFILE = "qwen25_7b"
GUIDED_SCHEMA_VERSION = "v1_aspects6"
GUIDED_PROMPT_VERSION = "v4_guided_json_schema_compact"
SEGMENTS = ("business", "couple", "family", "solo", "friends", "unknown")
SENTIMENTS = ("pos", "neg", "neu")
SCORES = (-2, -1, 0, 1, 2)

SYSTEM_PROMPT = (
    "Extract the requested information from the hotel review. The response is "
    "constrained by a JSON schema. Use only evidence copied from the review."
)
USER_PROMPT = """Extract six hotel-review aspects.
For an unmentioned aspect use mentioned=false, sentiment="neu", score=0,
and evidence=[]. Evidence must be copied from the review, contain at most two
strings per aspect, and each string must be at most 20 words.

Review:
{review_text}
"""

JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "aspects": {
            "type": "object",
            "properties": {
                aspect: {
                    "type": "object",
                    "properties": {
                        "mentioned": {"type": "boolean"},
                        "sentiment": {"type": "string", "enum": list(SENTIMENTS)},
                        "score": {"type": "integer", "enum": list(SCORES)},
                        "evidence": {
                            "type": "array",
                            "items": {"type": "string"},
                            "maxItems": 2,
                        },
                    },
                    "required": ["mentioned", "sentiment", "score", "evidence"],
                    "additionalProperties": False,
                }
                for aspect in ASPECTS
            },
            "required": list(ASPECTS),
            "additionalProperties": False,
        },
        "traveler_segment": {"type": "string", "enum": list(SEGMENTS)},
        "overall_summary": {"type": "string"},
    },
    "required": ["aspects", "traveler_segment", "overall_summary"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class Profile:
    name: str
    endpoints: Tuple[str, ...]
    api_key: str
    model: str
    temperature: float
    max_tokens: int
    timeout_sec: int
    concurrency: int
    max_retries: int


def log(message: str) -> None:
    tqdm.write(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default=PROFILE)
    parser.add_argument("--sample-size", type=int, default=50_000)
    parser.add_argument("--pilot", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument("--max-chars", type=int, default=4_000)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--endpoints",
        default=None,
        help="Comma-separated OpenAI-compatible endpoints; defaults to config.",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Write C1/C2 cache audit without contacting vLLM.",
    )
    return parser.parse_args()


def load_profile(path: Path, name: str, args: argparse.Namespace) -> Profile:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required for Phase 2 C") from exc
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg = payload["profiles"][name]
    configured = cfg.get("endpoints") or [cfg["base_url"]]
    endpoints = (
        tuple(item.strip().rstrip("/") for item in args.endpoints.split(",") if item.strip())
        if args.endpoints
        else tuple(str(item).rstrip("/") for item in configured)
    )
    if not endpoints:
        raise ValueError("at least one endpoint is required")
    return Profile(
        name=name,
        endpoints=endpoints,
        api_key=str(cfg["api_key"]),
        model=str(cfg["model"]),
        temperature=float(cfg.get("temperature", 0.0)),
        max_tokens=int(args.max_tokens or cfg.get("max_tokens", 900)),
        timeout_sec=int(cfg.get("timeout_sec", 120)),
        concurrency=int(args.concurrency or cfg.get("concurrency", 16)),
        max_retries=int(cfg.get("max_retries", 3)),
    )


def canonical_split_hash(output_dir: Path) -> str:
    digest = hashlib.sha256()
    for fold in range(1, 6):
        path = output_dir / f"splits_fold{fold}.npz"
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def outer_fold_assignment(output_dir: Path, n_rows: int) -> np.ndarray:
    assignment = np.full(n_rows, -1, dtype=np.int8)
    for fold in range(1, 6):
        data = np.load(output_dir / f"splits_fold{fold}.npz")
        test_idx = np.asarray(data["test_idx"], dtype=np.int64)
        if np.any(assignment[test_idx] != -1):
            raise ValueError(f"overlapping test indices in fold {fold}")
        assignment[test_idx] = fold
    if np.any(assignment == -1):
        raise ValueError("saved outer splits do not cover every row")
    return assignment


def cache_paths(output_dir: Path, profile: str) -> List[Path]:
    paths = sorted((output_dir / "llm_cache").glob(f"{profile}_ep*/cache.sqlite"))
    if not paths:
        legacy = output_dir / "llm_cache" / profile / "cache.sqlite"
        if legacy.exists():
            paths = [legacy]
    if not paths:
        raise FileNotFoundError(f"no extraction cache found for profile={profile}")
    return paths


def sqlite_columns(connection: sqlite3.Connection, table: str) -> List[str]:
    return [str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})")]


def audit_existing_cache(output_dir: Path, profile: str) -> Dict[str, Any]:
    frames: List[pd.DataFrame] = []
    cache_details: List[Dict[str, Any]] = []
    for path in cache_paths(output_dir, profile):
        with sqlite3.connect(path) as connection:
            columns = sqlite_columns(connection, "cache")
            selected = [
                column
                for column in (
                    "row_id",
                    "ok",
                    "error",
                    "schema_version",
                    "prompt_version",
                )
                if column in columns
            ]
            frame = pd.read_sql_query(
                f"SELECT {', '.join(selected)} FROM cache",
                connection,
            )
        frames.append(frame)
        cache_details.append(
            {
                "path": str(path.relative_to(ROOT)),
                "n_rows": int(len(frame)),
                "columns": columns,
                "retry_count_persisted": "attempt_count" in columns,
            }
        )
    cache = pd.concat(frames, ignore_index=True)
    if cache["row_id"].duplicated().any():
        raise ValueError("existing endpoint caches contain duplicate row_id values")
    failures = cache[cache["ok"].astype(int) != 1].copy()
    error_counts = failures["error"].fillna("").astype(str).value_counts().to_dict()

    taxonomy = {
        "json_parse": int(failures["error"].eq("json_parse_failed").sum()),
        "missing_fields": 0,
        "type_errors": 0,
        "range_violations": 0,
        "invalid_categorical_values": 0,
        "retry_exhaustion": int(len(failures)),
        "other_terminal_errors": int(
            (~failures["error"].eq("json_parse_failed")).sum()
        ),
    }
    retry_observability = {
        "attempt_count_persisted": False,
        "exact_distribution_identifiable": False,
        "known_attempt_3_rows": int(len(failures)),
        "known_required_at_least_one_retry_lower_bound": int(len(failures)),
        "successful_rows_attempt_distribution": None,
        "explanation": (
            "The legacy cache stores only the terminal response, not attempt_count. "
            "All terminal parse failures exhausted max_retries=3. Whether successful "
            "rows succeeded on attempt 1, 2, or 3 cannot be reconstructed."
        ),
    }
    taxonomy_frame = pd.DataFrame(
        [{"failure_class": key, "n": value} for key, value in taxonomy.items()]
    )
    taxonomy_frame.to_csv(
        output_dir / "phase2" / "experiment_c_failure_taxonomy.csv",
        index=False,
    )
    return {
        "n_cached": int(len(cache)),
        "n_ok": int(cache["ok"].astype(int).eq(1).sum()),
        "n_non_ok": int(len(failures)),
        "ok_rate": float(cache["ok"].astype(int).eq(1).mean()),
        "terminal_error_counts": {str(key): int(value) for key, value in error_counts.items()},
        "taxonomy": taxonomy,
        "retry_observability": retry_observability,
        "cache_details": cache_details,
        "source_evidence": {
            "prompt_constraints": "scripts/06_extract_aspects_llm.py:54-84",
            "repair_parser": "scripts/06_extract_aspects_llm.py:359-410",
            "normalization": "scripts/06_extract_aspects_llm.py:439-487",
            "retry_loop": "scripts/06_extract_aspects_llm.py:630-674",
            "terminal_failure": "scripts/06_extract_aspects_llm.py:699-719",
            "http_payload_without_response_format": "scripts/06_extract_aspects_llm.py:637-646",
        },
        "mechanism": (
            "Prompt-only schema instructions, best-effort JSON stripping/repair, "
            "normalization/default filling, and up to three HTTP generations. The "
            "legacy request payload contains neither response_format nor guided_json."
        ),
    }


def select_stratified_sample(
    output_dir: Path,
    sample_size: int,
    seed: int,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    dataset_path = output_dir / "dataset_clean.parquet"
    frame = pd.read_parquet(
        dataset_path,
        columns=["review_text", "rating_overall"],
    ).reset_index(drop=True)
    frame.insert(0, "row_id", np.arange(len(frame), dtype=np.int64))
    if not 0 < sample_size <= len(frame):
        raise ValueError(f"sample-size must be in [1,{len(frame)}]")
    folds = outer_fold_assignment(output_dir, len(frame))
    ratings = pd.to_numeric(frame["rating_overall"], errors="coerce").round().astype("Int64")
    strata = pd.Series(folds.astype(str), index=frame.index) + "_" + ratings.astype(str)
    indices = np.arange(len(frame), dtype=np.int64)
    if sample_size < len(frame):
        stratify_values = strata if sample_size >= strata.nunique() else None
        sampled, _ = train_test_split(
            indices,
            train_size=sample_size,
            random_state=seed,
            stratify=stratify_values,
        )
        sampled = np.sort(np.asarray(sampled, dtype=np.int64))
    else:
        sampled = indices
    sample_path = (
        output_dir
        / "phase2"
        / f"guided_sample_n{sample_size}_seed{seed}.npz"
    )
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        sample_path,
        row_id=sampled,
        outer_fold=folds[sampled],
        rating_overall=ratings.iloc[sampled].to_numpy(dtype=np.int8),
    )
    selected = frame.iloc[sampled].copy()
    metadata = {
        "sample_path": str(sample_path.relative_to(ROOT)),
        "sample_size": int(len(selected)),
        "seed": int(seed),
        "stratified_by": (
            ["saved_outer_fold", "rating_overall"]
            if sample_size >= strata.nunique()
            else ["none_for_tiny_smoke_test"]
        ),
        "split_hash": canonical_split_hash(output_dir),
        "dataset_total_n": int(len(frame)),
    }
    return selected, metadata


def init_guided_cache(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS cache (
            cache_key TEXT PRIMARY KEY,
            row_id INTEGER NOT NULL,
            model TEXT NOT NULL,
            prompt_hash TEXT NOT NULL,
            schema_version TEXT NOT NULL,
            prompt_version TEXT NOT NULL,
            endpoint_idx INTEGER NOT NULL,
            ok INTEGER NOT NULL,
            response_json TEXT,
            raw_text TEXT,
            error TEXT,
            prompt_tokens INTEGER,
            completion_tokens INTEGER,
            elapsed_ms INTEGER,
            attempt_count INTEGER NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_cache_row_version "
        "ON cache(row_id, schema_version, prompt_version)"
    )
    connection.commit()
    return connection


def prompt_hash(review_text: str, model: str) -> str:
    payload = "\n".join(
        [
            GUIDED_SCHEMA_VERSION,
            GUIDED_PROMPT_VERSION,
            model,
            SYSTEM_PROMPT,
            USER_PROMPT.format(review_text=review_text),
            json.dumps(JSON_SCHEMA, sort_keys=True),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def make_cache_key(row_id: int, model: str, digest: str) -> str:
    return f"{row_id}:{digest}:{model}:{GUIDED_SCHEMA_VERSION}:{GUIDED_PROMPT_VERSION}"


def clean_review_text(value: Any, max_chars: int) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).strip()[:max_chars]


def validate_guided_payload(payload: Any) -> Tuple[Optional[Dict[str, Any]], str]:
    if not isinstance(payload, dict):
        return None, "top_level_not_object"
    if set(payload) != {"aspects", "traveler_segment", "overall_summary"}:
        return None, "top_level_fields"
    aspects = payload.get("aspects")
    if not isinstance(aspects, dict) or set(aspects) != set(ASPECTS):
        return None, "aspect_keys"
    normalized: Dict[str, Any] = {"aspects": {}}
    for aspect in ASPECTS:
        item = aspects.get(aspect)
        if not isinstance(item, dict):
            return None, f"{aspect}:not_object"
        if set(item) != {"mentioned", "sentiment", "score", "evidence"}:
            return None, f"{aspect}:fields"
        mentioned = item.get("mentioned")
        sentiment = item.get("sentiment")
        score = item.get("score")
        evidence = item.get("evidence")
        if not isinstance(mentioned, bool):
            return None, f"{aspect}:mentioned_type"
        if sentiment not in SENTIMENTS:
            return None, f"{aspect}:sentiment"
        if isinstance(score, bool) or not isinstance(score, int) or score not in SCORES:
            return None, f"{aspect}:score"
        if not isinstance(evidence, list) or len(evidence) > 2:
            return None, f"{aspect}:evidence_array"
        if not all(isinstance(item, str) for item in evidence):
            return None, f"{aspect}:evidence_item"
        # The JSON grammar guarantees an array of at most two strings. The
        # 20-word semantic limit is normalized exactly as in the legacy
        # pipeline because JSON Schema cannot express a word count.
        evidence = [" ".join(item.split()[:20]) for item in evidence]
        if not mentioned:
            sentiment = "neu"
            score = 0
            evidence = []
        normalized["aspects"][aspect] = {
            "mentioned": mentioned,
            "sentiment": sentiment,
            "score": score,
            "evidence": evidence,
        }
    segment = payload.get("traveler_segment")
    summary = payload.get("overall_summary")
    if segment not in SEGMENTS:
        return None, "traveler_segment"
    if not isinstance(summary, str):
        return None, "overall_summary_type"
    normalized["traveler_segment"] = segment
    normalized["overall_summary"] = summary
    return normalized, ""


async def request_one(
    client: httpx.AsyncClient,
    profile: Profile,
    endpoint_idx: int,
    row_id: int,
    review_text: str,
    seed: int,
) -> Dict[str, Any]:
    digest = prompt_hash(review_text, profile.model)
    key = make_cache_key(row_id, profile.model, digest)
    started_all = time.perf_counter()
    last_raw = ""
    last_error = ""
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    for attempt in range(1, profile.max_retries + 1):
        payload = {
            "model": profile.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": USER_PROMPT.format(review_text=review_text),
                },
            ],
            "temperature": profile.temperature,
            "max_tokens": profile.max_tokens,
            "seed": seed,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "hotel_aspects_v1",
                    "description": "Six-aspect hotel review extraction.",
                    "schema": JSON_SCHEMA,
                    "strict": True,
                },
            },
        }
        try:
            response = await client.post("/chat/completions", json=payload)
            last_raw = response.text
            if response.status_code != 200:
                last_error = f"http_{response.status_code}:{last_raw[:300]}"
                continue
            response_payload = response.json()
            usage = response_payload.get("usage") or {}
            prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
            completion_tokens = usage.get("completion_tokens", completion_tokens)
            content = (
                response_payload.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            last_raw = content
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError as exc:
                last_error = f"json_parse:{exc.msg}"
                continue
            normalized, schema_error = validate_guided_payload(parsed)
            if normalized is None:
                last_error = f"schema_validation:{schema_error}"
                continue
            return {
                "cache_key": key,
                "row_id": row_id,
                "model": profile.model,
                "prompt_hash": digest,
                "schema_version": GUIDED_SCHEMA_VERSION,
                "prompt_version": GUIDED_PROMPT_VERSION,
                "endpoint_idx": endpoint_idx,
                "ok": 1,
                "response_json": json.dumps(normalized, ensure_ascii=True),
                "raw_text": content,
                "error": "",
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "elapsed_ms": int((time.perf_counter() - started_all) * 1000),
                "attempt_count": attempt,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as exc:  # network/timeout errors are retryable
            last_error = f"{type(exc).__name__}:{exc}"
    return {
        "cache_key": key,
        "row_id": row_id,
        "model": profile.model,
        "prompt_hash": digest,
        "schema_version": GUIDED_SCHEMA_VERSION,
        "prompt_version": GUIDED_PROMPT_VERSION,
        "endpoint_idx": endpoint_idx,
        "ok": 0,
        "response_json": "",
        "raw_text": last_raw[:20_000],
        "error": last_error or "retry_exhaustion",
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "elapsed_ms": int((time.perf_counter() - started_all) * 1000),
        "attempt_count": profile.max_retries,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


CACHE_COLUMNS = (
    "cache_key",
    "row_id",
    "model",
    "prompt_hash",
    "schema_version",
    "prompt_version",
    "endpoint_idx",
    "ok",
    "response_json",
    "raw_text",
    "error",
    "prompt_tokens",
    "completion_tokens",
    "elapsed_ms",
    "attempt_count",
    "created_at",
)


def write_cache_rows(connection: sqlite3.Connection, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    placeholders = ",".join("?" for _ in CACHE_COLUMNS)
    update_columns = [column for column in CACHE_COLUMNS if column != "cache_key"]
    update_sql = ",".join(f"{column}=excluded.{column}" for column in update_columns)
    connection.executemany(
        f"INSERT INTO cache ({','.join(CACHE_COLUMNS)}) VALUES ({placeholders}) "
        f"ON CONFLICT(cache_key) DO UPDATE SET {update_sql}",
        [[row.get(column) for column in CACHE_COLUMNS] for row in rows],
    )
    connection.commit()


def cached_ok_row_ids(
    connection: sqlite3.Connection,
    row_ids: Sequence[int],
) -> set[int]:
    if not row_ids:
        return set()
    result: set[int] = set()
    chunk_size = 900
    for start in range(0, len(row_ids), chunk_size):
        chunk = row_ids[start : start + chunk_size]
        placeholders = ",".join("?" for _ in chunk)
        query = (
            f"SELECT row_id FROM cache WHERE ok=1 "
            f"AND schema_version=? AND prompt_version=? "
            f"AND row_id IN ({placeholders})"
        )
        params: List[Any] = [GUIDED_SCHEMA_VERSION, GUIDED_PROMPT_VERSION, *chunk]
        result.update(int(row[0]) for row in connection.execute(query, params))
    return result


async def process_endpoint(
    endpoint_idx: int,
    endpoint: str,
    shard: pd.DataFrame,
    profile: Profile,
    cache_path: Path,
    resume: bool,
    seed: int,
    max_chars: int,
    position: int,
    stage: str,
) -> Dict[str, Any]:
    connection = init_guided_cache(cache_path)
    row_ids = shard["row_id"].astype(int).tolist()
    hits = cached_ok_row_ids(connection, row_ids) if resume else set()
    uncached = shard[~shard["row_id"].isin(hits)].copy()
    started = time.perf_counter()
    new_rows: List[Dict[str, Any]] = []
    ok_new = 0
    attempts = 0
    headers = {"Authorization": f"Bearer {profile.api_key}"}
    timeout = httpx.Timeout(profile.timeout_sec)
    progress = tqdm(
        total=len(shard),
        initial=len(hits),
        desc=f"{stage} ep{endpoint_idx}",
        unit="row",
        position=position,
        dynamic_ncols=True,
        leave=True,
    )
    progress.set_postfix(cache_hits=len(hits), ok_rate="n/a")
    try:
        async with httpx.AsyncClient(
            base_url=endpoint,
            headers=headers,
            timeout=timeout,
            limits=httpx.Limits(
                max_connections=profile.concurrency,
                max_keepalive_connections=profile.concurrency,
            ),
        ) as client:
            records = list(uncached.itertuples(index=False))
            for start in range(0, len(records), profile.concurrency):
                batch = records[start : start + profile.concurrency]
                tasks = [
                    request_one(
                        client=client,
                        profile=profile,
                        endpoint_idx=endpoint_idx,
                        row_id=int(record.row_id),
                        review_text=clean_review_text(record.review_text, max_chars),
                        seed=seed,
                    )
                    for record in batch
                ]
                results = await asyncio.gather(*tasks)
                write_cache_rows(connection, results)
                new_rows.extend(results)
                ok_new += sum(int(row["ok"]) for row in results)
                attempts += sum(int(row["attempt_count"]) for row in results)
                progress.update(len(results))
                processed = len(new_rows)
                progress.set_postfix(
                    cache_hits=len(hits),
                    ok_rate=f"{ok_new / processed:.4f}" if processed else "n/a",
                )
    finally:
        progress.close()
        connection.close()
    wall = time.perf_counter() - started
    processed = len(new_rows)
    return {
        "endpoint_idx": endpoint_idx,
        "endpoint": endpoint,
        "cache_path": str(cache_path.relative_to(ROOT)),
        "shard_rows": int(len(shard)),
        "cache_hits": int(len(hits)),
        "processed_n": int(processed),
        "ok_n": int(ok_new),
        "ok_rate_new": float(ok_new / processed) if processed else None,
        "attempts_total": int(attempts),
        "mean_attempts": float(attempts / processed) if processed else None,
        "wall_seconds": float(wall),
        "reviews_per_wall_second": float(processed / wall) if wall > 0 else None,
    }


def split_sample_by_endpoint(
    sample: pd.DataFrame,
    endpoints: Sequence[str],
) -> List[pd.DataFrame]:
    n_endpoints = len(endpoints)
    return [
        sample[sample["row_id"].mod(n_endpoints).eq(endpoint_idx)].copy()
        for endpoint_idx in range(n_endpoints)
    ]


def split_pilot(
    shards: Sequence[pd.DataFrame],
    pilot_total: int,
    seed: int,
    cache_dir: Path,
    profile: str,
    resume: bool,
) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
    if pilot_total <= 0:
        return [shard.iloc[0:0].copy() for shard in shards], list(shards)
    rng = np.random.default_rng(seed)
    uncached_by_shard: List[pd.DataFrame] = []
    for endpoint_idx, shard in enumerate(shards):
        path = cache_dir / f"{profile}_ep{endpoint_idx}" / "cache.sqlite"
        connection = init_guided_cache(path)
        hits = (
            cached_ok_row_ids(connection, shard["row_id"].astype(int).tolist())
            if resume
            else set()
        )
        connection.close()
        uncached_by_shard.append(shard[~shard["row_id"].isin(hits)].copy())
    available = sum(len(frame) for frame in uncached_by_shard)
    target = min(pilot_total, available)
    allocations = [0] * len(shards)
    if target:
        ideal = np.asarray(
            [len(frame) for frame in uncached_by_shard], dtype=float
        )
        ideal = target * ideal / ideal.sum()
        allocations = np.floor(ideal).astype(int).tolist()
        remaining = target - sum(allocations)
        order = np.argsort(-(ideal - np.floor(ideal)))
        for idx in order[:remaining]:
            allocations[int(idx)] += 1
    pilots: List[pd.DataFrame] = []
    mains: List[pd.DataFrame] = []
    for endpoint_idx, (shard, uncached, allocation) in enumerate(
        zip(shards, uncached_by_shard, allocations)
    ):
        if allocation:
            chosen = rng.choice(
                uncached["row_id"].to_numpy(dtype=np.int64),
                size=allocation,
                replace=False,
            )
            pilot = shard[shard["row_id"].isin(chosen)].copy()
        else:
            pilot = shard.iloc[0:0].copy()
        pilots.append(pilot)
        mains.append(shard[~shard["row_id"].isin(pilot["row_id"])].copy())
        log(
            f"pilot ep{endpoint_idx}: {len(pilot)} uncached rows "
            f"(shard={len(shard)}, available_uncached={len(uncached)})"
        )
    return pilots, mains


async def run_parallel_stage(
    stage: str,
    shards: Sequence[pd.DataFrame],
    profile: Profile,
    output_dir: Path,
    resume: bool,
    seed: int,
    max_chars: int,
) -> List[Dict[str, Any]]:
    tasks = []
    for endpoint_idx, (endpoint, shard) in enumerate(zip(profile.endpoints, shards)):
        cache_path = (
            output_dir
            / "phase2"
            / "guided_cache"
            / f"{profile.name}_ep{endpoint_idx}"
            / "cache.sqlite"
        )
        tasks.append(
            process_endpoint(
                endpoint_idx=endpoint_idx,
                endpoint=endpoint,
                shard=shard,
                profile=profile,
                cache_path=cache_path,
                resume=resume,
                seed=seed,
                max_chars=max_chars,
                position=endpoint_idx,
                stage=stage,
            )
        )
    return list(await asyncio.gather(*tasks))


def load_guided_results(
    output_dir: Path,
    profile: Profile,
    row_ids: Sequence[int],
) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    selected = set(int(value) for value in row_ids)
    for endpoint_idx in range(len(profile.endpoints)):
        path = (
            output_dir
            / "phase2"
            / "guided_cache"
            / f"{profile.name}_ep{endpoint_idx}"
            / "cache.sqlite"
        )
        connection = init_guided_cache(path)
        frame = pd.read_sql_query(
            "SELECT * FROM cache WHERE schema_version=? AND prompt_version=?",
            connection,
            params=(GUIDED_SCHEMA_VERSION, GUIDED_PROMPT_VERSION),
        )
        connection.close()
        frame = frame[frame["row_id"].isin(selected)]
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    result = result.sort_values(["row_id", "created_at"]).drop_duplicates(
        "row_id", keep="last"
    )
    if len(result) != len(selected):
        missing = len(selected) - len(result)
        raise ValueError(f"guided cache is missing {missing} sampled rows")
    return result.sort_values("row_id").reset_index(drop=True)


def expand_response(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for record in frame.itertuples(index=False):
        parsed: Dict[str, Any] = {}
        try:
            parsed = json.loads(record.response_json) if record.response_json else {}
        except (json.JSONDecodeError, TypeError):
            parsed = {}
        aspects = parsed.get("aspects") if isinstance(parsed, dict) else {}
        output: Dict[str, Any] = {
            "row_id": int(record.row_id),
            f"{prefix}_ok": int(record.ok),
            f"{prefix}_traveler_segment": (
                parsed.get("traveler_segment", "unknown")
                if isinstance(parsed, dict)
                else "unknown"
            ),
        }
        for aspect in ASPECTS:
            item = aspects.get(aspect, {}) if isinstance(aspects, dict) else {}
            output[f"{prefix}_{aspect}_score"] = int(item.get("score", 0) or 0)
            output[f"{prefix}_{aspect}_mentioned"] = bool(
                item.get("mentioned", False)
            )
        rows.append(output)
    return pd.DataFrame(rows)


def load_existing_rows(
    output_dir: Path,
    profile: str,
    row_ids: Sequence[int],
) -> pd.DataFrame:
    path = output_dir / f"aspects_llm_{profile}.parquet"
    needed = [
        "row_id",
        "ok",
        "response_json",
        "traveler_segment",
        *[f"aspect_{aspect}_score" for aspect in ASPECTS],
        *[f"aspect_{aspect}_mentioned" for aspect in ASPECTS],
    ]
    frame = pd.read_parquet(path, columns=needed)
    frame = frame[frame["row_id"].isin(row_ids)].copy()
    renamed = {
        "ok": "legacy_ok",
        "traveler_segment": "legacy_traveler_segment",
        **{
            f"aspect_{aspect}_{suffix}": f"legacy_{aspect}_{suffix}"
            for aspect in ASPECTS
            for suffix in ("score", "mentioned")
        },
    }
    return frame.drop(columns=["response_json"]).rename(columns=renamed)


def safe_pearson(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def agreement_tables(
    output_dir: Path,
    profile: Profile,
    guided_cache: pd.DataFrame,
    row_ids: Sequence[int],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    guided = expand_response(guided_cache, "guided")
    legacy = load_existing_rows(output_dir, profile.name, row_ids)
    joined = legacy.merge(guided, on="row_id", how="inner", validate="one_to_one")
    both_ok = joined["legacy_ok"].eq(1) & joined["guided_ok"].eq(1)
    valid = joined[both_ok]
    rows: List[Dict[str, Any]] = []
    for aspect in ASPECTS:
        rows.append(
            {
                "feature": aspect,
                "feature_type": "aspect_score",
                "n": int(len(valid)),
                "agreement": safe_pearson(
                    valid[f"legacy_{aspect}_score"].to_numpy(),
                    valid[f"guided_{aspect}_score"].to_numpy(),
                ),
                "metric": "pearson_r",
            }
        )
        rows.append(
            {
                "feature": aspect,
                "feature_type": "mentioned",
                "n": int(len(valid)),
                "agreement": float(
                    cohen_kappa_score(
                        valid[f"legacy_{aspect}_mentioned"].astype(bool),
                        valid[f"guided_{aspect}_mentioned"].astype(bool),
                    )
                ),
                "metric": "cohen_kappa",
            }
        )
    rows.append(
        {
            "feature": "traveler_segment",
            "feature_type": "segment",
            "n": int(len(valid)),
            "agreement": float(
                cohen_kappa_score(
                    valid["legacy_traveler_segment"].fillna("unknown"),
                    valid["guided_traveler_segment"].fillna("unknown"),
                )
            ),
            "metric": "cohen_kappa",
        }
    )
    table = pd.DataFrame(rows)
    summary = {
        "sample_n": int(len(joined)),
        "legacy_ok_n": int(joined["legacy_ok"].eq(1).sum()),
        "legacy_ok_rate": float(joined["legacy_ok"].eq(1).mean()),
        "guided_ok_n": int(joined["guided_ok"].eq(1).sum()),
        "guided_ok_rate": float(joined["guided_ok"].eq(1).mean()),
        "both_ok_n": int(both_ok.sum()),
    }
    return table, summary


def parse_legacy_throughput(path: Path) -> Dict[str, Optional[float]]:
    if not path.exists():
        return {"single": None, "combined": None}
    text = path.read_text(encoding="utf-8")

    def find(name: str) -> Optional[float]:
        match = re.search(rf"{re.escape(name)}:\\s*([0-9.]+)", text)
        return float(match.group(1)) if match else None

    return {
        "single": find("throughput_rows_per_sec_single"),
        "combined": find("throughput_rows_per_sec_combined"),
    }


def guided_throughput(
    endpoint_stats: Sequence[Dict[str, Any]],
    sample_n: int,
    dataset_n: int,
) -> Dict[str, Any]:
    wall_values = [
        float(row["wall_seconds"])
        for row in endpoint_stats
        if int(row["processed_n"]) > 0
    ]
    processed = sum(int(row["processed_n"]) for row in endpoint_stats)
    gpu_seconds = sum(
        float(row["wall_seconds"])
        for row in endpoint_stats
        if int(row["processed_n"]) > 0
    )
    overall_wall = max(wall_values) if wall_values else 0.0
    reviews_per_gpu_second = processed / gpu_seconds if gpu_seconds else None
    combined_wall_throughput = processed / overall_wall if overall_wall else None
    projected_gpu_hours = (
        dataset_n / reviews_per_gpu_second / 3600
        if reviews_per_gpu_second
        else None
    )
    projected_wall_hours = (
        dataset_n / combined_wall_throughput / 3600
        if combined_wall_throughput
        else None
    )
    return {
        "sample_n": int(sample_n),
        "processed_uncached_n": int(processed),
        "endpoint_wall_seconds_sum_gpu_seconds": float(gpu_seconds),
        "overall_wall_seconds": float(overall_wall),
        "reviews_per_gpu_second": reviews_per_gpu_second,
        "combined_reviews_per_wall_second": combined_wall_throughput,
        "projected_full_gpu_hours": projected_gpu_hours,
        "projected_full_wall_hours_at_current_endpoints": projected_wall_hours,
    }


def write_report(
    output_dir: Path,
    audit: Dict[str, Any],
    guided: Optional[Dict[str, Any]],
) -> None:
    taxonomy = pd.DataFrame(
        [
            {"failure_class": key, "n": value}
            for key, value in audit["taxonomy"].items()
        ]
    )
    lines = [
        "# Experiment C — Schema Enforcement Audit",
        "",
        "## C1. Legacy enforcement mechanism",
        "",
        (
            "The legacy extraction did **not** use guided decoding, `guided_json`, "
            "or an OpenAI `response_format`. It used prompt constraints, a "
            "best-effort repair parser, default-filling normalization, and retries."
        ),
        "",
        "| mechanism | source evidence |",
        "| --- | --- |",
        *[
            f"| {key} | `{value}` |"
            for key, value in audit["source_evidence"].items()
        ],
        "",
        "- Maximum HTTP generations per row: 3 (`max_retries=3`).",
        "- A row is terminally non-ok only when all attempts fail parsing or an "
        "HTTP/transport error persists. Missing/type/range/category deviations in "
        "parseable objects are normalized rather than marked non-ok.",
        "",
        "## C2. Full-corpus failure taxonomy",
        "",
        f"- Cached rows: {audit['n_cached']:,}",
        f"- OK rows: {audit['n_ok']:,} ({audit['ok_rate']:.4%})",
        f"- Non-ok rows: {audit['n_non_ok']:,}",
        "",
        taxonomy.to_markdown(index=False),
        "",
        (
            "**Retry-distribution limitation:** the legacy cache has no attempt-count "
            "column. The 1,413 terminal failures are known to have exhausted all "
            "three attempts and therefore form a lower bound on rows requiring a "
            "retry. Attempt counts for successful rows are not identifiable from "
            "the persisted data; inventing a full 0/1/2-retry distribution would be "
            "incorrect."
        ),
    ]
    if guided is not None:
        agreement = pd.DataFrame(guided["agreement"])
        endpoint_stats = pd.DataFrame(guided["endpoint_stats"])
        summary = guided["summary"]
        throughput = guided["throughput"]
        legacy_tp = guided["legacy_throughput"]
        lines.extend(
            [
                "",
                "## C3. True guided JSON-schema decoding (50k stratified sample)",
                "",
                f"- Sample n: {summary['sample_n']:,}",
                f"- Legacy OK rate on same rows: {summary['legacy_ok_rate']:.4%}",
                f"- Guided OK rate: {summary['guided_ok_rate']:.4%}",
                f"- Rows OK under both mechanisms: {summary['both_ok_n']:,}",
                "",
                agreement.to_markdown(index=False, floatfmt=".4f"),
                "",
                "### Throughput",
                "",
                endpoint_stats.to_markdown(index=False, floatfmt=".4f"),
                "",
                (
                    f"- Guided reviews/GPU-second: "
                    f"{throughput['reviews_per_gpu_second']:.4f}"
                ),
                (
                    f"- Guided combined reviews/wall-second: "
                    f"{throughput['combined_reviews_per_wall_second']:.4f}"
                ),
                (
                    f"- Legacy reference reviews/GPU-second: "
                    f"{legacy_tp['single'] if legacy_tp['single'] is not None else 'not available'}"
                ),
                (
                    f"- Projected full-corpus GPU-hours: "
                    f"{throughput['projected_full_gpu_hours']:.2f}"
                ),
                (
                    f"- Projected wall-hours with {len(endpoint_stats)} endpoints: "
                    f"{throughput['projected_full_wall_hours_at_current_endpoints']:.2f}"
                ),
                "",
                "## C4. Stop rule",
                "",
                (
                    "The full corpus was **not** re-extracted. Only the stratified "
                    "sample was run; the table above reports the cost projection."
                ),
            ]
        )
    report_path = output_dir / "phase2" / "experiment_c_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"saved: {report_path}")


def guided_output_frame(cache: pd.DataFrame) -> pd.DataFrame:
    expanded = expand_response(cache, "guided")
    output = cache.merge(expanded, on="row_id", how="left", validate="one_to_one")
    return output.sort_values("row_id").reset_index(drop=True)


def main() -> None:
    args = parse_args()
    output_dir = ROOT / "outputs"
    phase2_dir = output_dir / "phase2"
    phase2_dir.mkdir(parents=True, exist_ok=True)
    profile = load_profile(ROOT / "config" / "local_llm.yaml", args.profile, args)

    audit = audit_existing_cache(output_dir, args.profile)
    payload: Dict[str, Any] = {
        "experiment": "C",
        "profile": args.profile,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "legacy_audit": audit,
        "guided_decoding": None,
    }
    if args.audit_only:
        write_json(phase2_dir / "experiment_c_enforcement.json", payload)
        write_report(output_dir, audit, None)
        return

    sample, sample_metadata = select_stratified_sample(
        output_dir,
        sample_size=args.sample_size,
        seed=args.seed,
    )
    shards = split_sample_by_endpoint(sample, profile.endpoints)
    cache_root = phase2_dir / "guided_cache"
    pilots, mains = split_pilot(
        shards=shards,
        pilot_total=args.pilot,
        seed=args.seed,
        cache_dir=cache_root,
        profile=profile.name,
        resume=args.resume,
    )
    log(
        f"guided JSON: endpoints={len(profile.endpoints)}, "
        f"concurrency_per_endpoint={profile.concurrency}, "
        f"total_concurrency={profile.concurrency * len(profile.endpoints)}"
    )
    pilot_stats = asyncio.run(
        run_parallel_stage(
            stage="C3 pilot",
            shards=pilots,
            profile=profile,
            output_dir=output_dir,
            resume=args.resume,
            seed=args.seed,
            max_chars=args.max_chars,
        )
    )
    for row in pilot_stats:
        log(
            f"pilot ep{row['endpoint_idx']}: processed={row['processed_n']}, "
            f"ok_rate={row['ok_rate_new']}, rows/s={row['reviews_per_wall_second']}"
        )
    main_stats = asyncio.run(
        run_parallel_stage(
            stage="C3 main",
            shards=mains,
            profile=profile,
            output_dir=output_dir,
            resume=args.resume,
            seed=args.seed,
            max_chars=args.max_chars,
        )
    )
    endpoint_stats: List[Dict[str, Any]] = []
    for pilot, main in zip(pilot_stats, main_stats):
        processed = int(pilot["processed_n"]) + int(main["processed_n"])
        ok_n = int(pilot["ok_n"]) + int(main["ok_n"])
        wall = float(pilot["wall_seconds"]) + float(main["wall_seconds"])
        endpoint_stats.append(
            {
                "endpoint_idx": int(pilot["endpoint_idx"]),
                "endpoint": pilot["endpoint"],
                "sample_rows": int(pilot["shard_rows"]) + int(main["shard_rows"]),
                "cache_hits": int(pilot["cache_hits"]) + int(main["cache_hits"]),
                "processed_n": processed,
                "ok_rate_new": float(ok_n / processed) if processed else None,
                "mean_attempts": (
                    (
                        int(pilot["attempts_total"])
                        + int(main["attempts_total"])
                    )
                    / processed
                    if processed
                    else None
                ),
                "wall_seconds": wall,
                "reviews_per_wall_second": processed / wall if wall else None,
            }
        )

    guided_cache = load_guided_results(
        output_dir,
        profile,
        sample["row_id"].astype(int).tolist(),
    )
    output_frame = guided_output_frame(guided_cache)
    guided_path = (
        phase2_dir
        / f"aspects_guided_{args.profile}_n{args.sample_size}.parquet"
    )
    output_frame.to_parquet(guided_path, index=False)
    agreement, agreement_summary = agreement_tables(
        output_dir,
        profile,
        guided_cache,
        sample["row_id"].astype(int).tolist(),
    )
    agreement.to_csv(
        phase2_dir / "experiment_c_guided_agreement.csv",
        index=False,
    )
    pd.DataFrame(endpoint_stats).to_csv(
        phase2_dir / "experiment_c_guided_throughput.csv",
        index=False,
    )
    throughput = guided_throughput(
        endpoint_stats,
        sample_n=len(sample),
        dataset_n=sample_metadata["dataset_total_n"],
    )
    legacy_throughput = parse_legacy_throughput(
        output_dir / f"llm_runtime_estimate_{args.profile}.md"
    )
    guided_payload = {
        "sample": sample_metadata,
        "schema_version": GUIDED_SCHEMA_VERSION,
        "prompt_version": GUIDED_PROMPT_VERSION,
        "response_format": "json_schema strict=true",
        "server_structured_outputs": {
            "backend": "xgrammar",
            "disable_any_whitespace": True,
            "rationale": (
                "Prevents Qwen whitespace loops that can otherwise consume "
                "max_tokens while leaving a valid JSON prefix incomplete."
            ),
        },
        "summary": agreement_summary,
        "agreement": agreement.to_dict(orient="records"),
        "pilot_endpoint_stats": pilot_stats,
        "endpoint_stats": endpoint_stats,
        "throughput": throughput,
        "legacy_throughput": legacy_throughput,
        "output_path": str(guided_path.relative_to(ROOT)),
        "full_reextraction_performed": False,
    }
    payload["guided_decoding"] = guided_payload
    write_json(phase2_dir / "experiment_c_enforcement.json", payload)
    write_report(output_dir, audit, guided_payload)
    log(
        f"C3 complete: guided_ok={agreement_summary['guided_ok_rate']:.4%}, "
        f"legacy_ok={agreement_summary['legacy_ok_rate']:.4%}, "
        f"reviews/GPU-s={throughput['reviews_per_gpu_second']:.4f}, "
        f"full projection={throughput['projected_full_gpu_hours']:.2f} GPU-hours"
    )


if __name__ == "__main__":
    main()
