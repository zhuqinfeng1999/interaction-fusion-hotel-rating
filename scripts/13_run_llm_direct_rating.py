#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import httpx
import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation_harness import (  # noqa: E402
    EvaluationHarness,
    PeakRSSMonitor,
    write_json,
)


SEED = 42
N_SPLITS = 5
PROMPT_VERSION = "v1_rating_choice_1to5"


@dataclass
class Profile:
    name: str
    model: str
    api_key: str
    endpoints: List[str]
    timeout_sec: int
    concurrency: int


@dataclass
class RequestItem:
    row_id: int
    fold: int
    mode: str
    prompt: str
    prompt_hash: str
    endpoint_idx: int


def log(message: str) -> None:
    print(message, flush=True)


def load_profile(path: Path, name: str) -> Profile:
    import yaml

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    profiles = payload.get("profiles", {})
    if name not in profiles:
        raise ValueError(f"profile {name!r} not found in {path}")
    raw = profiles[name]
    endpoints = raw.get("endpoints") or [raw["base_url"]]
    return Profile(
        name=name,
        model=str(raw["model"]),
        api_key=str(raw["api_key"]),
        endpoints=[str(endpoint).rstrip("/") for endpoint in endpoints],
        timeout_sec=int(raw.get("timeout_sec", 120)),
        concurrency=int(raw.get("concurrency", 16)),
    )


def parse_modes(raw: str) -> List[str]:
    modes = [item.strip() for item in raw.split(",") if item.strip()]
    if not modes or any(mode not in {"zero", "few"} for mode in modes):
        raise ValueError("--modes must contain zero and/or few")
    return modes


def load_splits(output_dir: Path) -> List[Dict[str, Any]]:
    splits: List[Dict[str, Any]] = []
    for fold in range(1, N_SPLITS + 1):
        arrays = np.load(output_dir / f"splits_fold{fold}.npz")
        splits.append(
            {
                "fold": fold,
                "train_idx": arrays["train_idx"].astype(np.int64),
                "test_idx": arrays["test_idx"].astype(np.int64),
            }
        )
    return splits


def make_stratified_subsample(
    labels: np.ndarray,
    splits: Sequence[Dict[str, Any]],
    sample_size: int,
    output_path: Path,
) -> np.ndarray:
    if output_path.exists():
        selected = np.load(output_path)["row_id"].astype(np.int64)
        if len(selected) != min(sample_size, len(labels)):
            raise ValueError(
                f"saved direct-LLM subsample has {len(selected)} rows, "
                f"expected {min(sample_size, len(labels))}"
            )
        return selected

    target_n = min(sample_size, len(labels))
    rng = np.random.default_rng(SEED)
    strata: List[np.ndarray] = []
    for split in splits:
        test_idx = split["test_idx"]
        for rating in range(1, 6):
            candidates = test_idx[labels[test_idx] == rating]
            strata.append(candidates)

    stratum_sizes = np.asarray([len(item) for item in strata], dtype=np.int64)
    exact_quotas = target_n * stratum_sizes / stratum_sizes.sum()
    quotas = np.floor(exact_quotas).astype(np.int64)
    remaining_quota = target_n - int(quotas.sum())
    remainder_order = np.argsort(-(exact_quotas - quotas))
    for stratum_index in remainder_order[:remaining_quota]:
        quotas[stratum_index] += 1
    selected_parts: List[np.ndarray] = []
    for candidates, count in zip(strata, quotas):
        if count > 0:
            selected_parts.append(
                rng.choice(candidates, size=int(count), replace=False)
            )
    selected = np.unique(np.concatenate(selected_parts))
    if len(selected) != target_n:
        raise RuntimeError(
            f"stratified sampler selected {len(selected)} rows, "
            f"expected {target_n}"
        )
    selected = np.sort(selected.astype(np.int64))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        row_id=selected,
        seed=np.asarray([SEED], dtype=np.int64),
    )
    return selected


def select_exemplars(
    texts: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    fold: int,
    max_chars: int,
) -> str:
    rng = np.random.default_rng(SEED + fold)
    rows: List[str] = []
    for rating in range(1, 6):
        candidates = train_idx[labels[train_idx] == rating]
        if len(candidates) == 0:
            raise ValueError(f"fold {fold} has no rating-{rating} exemplar")
        row_id = int(rng.choice(candidates))
        review = str(texts[row_id])[:max_chars].replace("\n", " ")
        rows.append(f"Review: {review}\nRating: {rating}")
    return "\n\n".join(rows)


def build_prompt(
    text: str,
    mode: str,
    exemplars: str,
    max_chars: int,
) -> str:
    review = text[:max_chars].replace("\n", " ")
    instruction = (
        "Predict the TripAdvisor overall hotel rating for the review. "
        "Return exactly one integer from 1 to 5 and no other text."
    )
    if mode == "zero":
        return f"{instruction}\n\nReview: {review}\nRating:"
    return (
        f"{instruction}\n\nExamples:\n{exemplars}\n\n"
        f"Review: {review}\nRating:"
    )


def open_cache(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            cache_key TEXT PRIMARY KEY,
            row_id INTEGER NOT NULL,
            fold INTEGER NOT NULL,
            mode TEXT NOT NULL,
            model TEXT NOT NULL,
            prompt_version TEXT NOT NULL,
            prompt_hash TEXT NOT NULL,
            ok INTEGER NOT NULL,
            rating INTEGER,
            raw_text TEXT,
            error TEXT,
            prompt_tokens INTEGER,
            completion_tokens INTEGER,
            elapsed_ms INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_prediction_row "
        "ON predictions(row_id, fold, mode)"
    )
    connection.commit()
    return connection


def cache_key(item: RequestItem, model: str) -> str:
    return (
        f"{item.row_id}:{item.fold}:{item.mode}:{item.prompt_hash}:"
        f"{model}:{PROMPT_VERSION}"
    )


def cached_keys(
    connection: sqlite3.Connection,
    items: Sequence[RequestItem],
    model: str,
) -> set[str]:
    keys = [cache_key(item, model) for item in items]
    found: set[str] = set()
    for start in range(0, len(keys), 800):
        batch = keys[start : start + 800]
        placeholders = ",".join("?" for _ in batch)
        query = (
            "SELECT cache_key FROM predictions "
            f"WHERE ok=1 AND cache_key IN ({placeholders})"
        )
        found.update(
            row[0] for row in connection.execute(query, batch).fetchall()
        )
    return found


def parse_rating(raw_text: str) -> int:
    stripped = raw_text.strip()
    if stripped in {"1", "2", "3", "4", "5"}:
        return int(stripped)
    for character in stripped:
        if character in "12345":
            return int(character)
    raise ValueError(f"invalid_rating_response:{stripped[:80]}")


async def request_one(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    item: RequestItem,
    profile: Profile,
) -> Dict[str, Any]:
    started = time.perf_counter()
    raw_text = ""
    error = ""
    prompt_tokens = None
    completion_tokens = None
    for attempt in range(1, 4):
        try:
            async with semaphore:
                response = await client.post(
                    "/chat/completions",
                    json={
                        "model": profile.model,
                        "messages": [
                            {
                                "role": "system",
                                "content": (
                                    "You are a rating classifier. Output only "
                                    "one integer: 1, 2, 3, 4, or 5."
                                ),
                            },
                            {"role": "user", "content": item.prompt},
                        ],
                        "temperature": 0,
                        "max_tokens": 1,
                        "seed": SEED,
                        "structured_outputs": {
                            "choice": ["1", "2", "3", "4", "5"]
                        },
                    },
                )
            response.raise_for_status()
            payload = response.json()
            raw_text = payload["choices"][0]["message"]["content"]
            usage = payload.get("usage") or {}
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
            rating = parse_rating(raw_text)
            return {
                "item": item,
                "ok": 1,
                "rating": rating,
                "raw_text": raw_text,
                "error": "",
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "elapsed_ms": int((time.perf_counter() - started) * 1000),
            }
        except Exception as exc:
            error = f"attempt_{attempt}:{type(exc).__name__}:{exc}"
            if attempt < 3:
                await asyncio.sleep(min(4, 2 ** (attempt - 1)))
    return {
        "item": item,
        "ok": 0,
        "rating": None,
        "raw_text": raw_text,
        "error": error,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "elapsed_ms": int((time.perf_counter() - started) * 1000),
    }


async def run_endpoint_chunk(
    endpoint_idx: int,
    endpoint: str,
    items: Sequence[RequestItem],
    profile: Profile,
    concurrency: int,
    progress: tqdm,
) -> List[Dict[str, Any]]:
    if not items:
        return []
    timeout = httpx.Timeout(profile.timeout_sec)
    headers = {"Authorization": f"Bearer {profile.api_key}"}
    semaphore = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(
        base_url=endpoint,
        headers=headers,
        timeout=timeout,
    ) as client:
        tasks = [
            asyncio.create_task(
                request_one(client, semaphore, item, profile)
            )
            for item in items
        ]
        results: List[Dict[str, Any]] = []
        ok_count = 0
        for task in asyncio.as_completed(tasks):
            result = await task
            results.append(result)
            ok_count += int(result["ok"] == 1)
            progress.update(1)
            progress.set_postfix(
                endpoint=endpoint_idx,
                ok_rate=f"{ok_count / len(results):.3f}",
            )
        return results


def write_cache(
    connection: sqlite3.Connection,
    results: Iterable[Dict[str, Any]],
    profile: Profile,
) -> None:
    rows = []
    for result in results:
        item: RequestItem = result["item"]
        rows.append(
            (
                cache_key(item, profile.model),
                item.row_id,
                item.fold,
                item.mode,
                profile.model,
                PROMPT_VERSION,
                item.prompt_hash,
                result["ok"],
                result["rating"],
                result["raw_text"],
                result["error"],
                result["prompt_tokens"],
                result["completion_tokens"],
                result["elapsed_ms"],
            )
        )
    connection.executemany(
        """
        INSERT OR REPLACE INTO predictions (
            cache_key, row_id, fold, mode, model, prompt_version,
            prompt_hash, ok, rating, raw_text, error, prompt_tokens,
            completion_tokens, elapsed_ms
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    connection.commit()


async def run_uncached(
    items_by_endpoint: Sequence[List[RequestItem]],
    connections: Sequence[sqlite3.Connection],
    profile: Profile,
    concurrency: int,
    write_every: int,
    mode: str,
) -> None:
    progresses = [
        tqdm(
            total=len(items),
            desc=f"{mode} ep{endpoint_idx}",
            unit="rows",
            position=endpoint_idx,
        )
        for endpoint_idx, items in enumerate(items_by_endpoint)
    ]
    offsets = [0] * len(items_by_endpoint)
    try:
        while any(
            offsets[index] < len(items_by_endpoint[index])
            for index in range(len(items_by_endpoint))
        ):
            chunk_tasks = []
            endpoint_indices = []
            for endpoint_idx, items in enumerate(items_by_endpoint):
                start = offsets[endpoint_idx]
                end = min(start + write_every, len(items))
                if start >= end:
                    continue
                endpoint_indices.append(endpoint_idx)
                chunk_tasks.append(
                    run_endpoint_chunk(
                        endpoint_idx,
                        profile.endpoints[endpoint_idx],
                        items[start:end],
                        profile,
                        concurrency,
                        progresses[endpoint_idx],
                    )
                )
                offsets[endpoint_idx] = end
            chunk_results = await asyncio.gather(*chunk_tasks)
            for endpoint_idx, results in zip(
                endpoint_indices,
                chunk_results,
            ):
                write_cache(
                    connections[endpoint_idx],
                    results,
                    profile,
                )
    finally:
        for progress in progresses:
            progress.close()


def read_predictions(
    connections: Sequence[sqlite3.Connection],
    mode: str,
    selected: np.ndarray,
) -> pd.DataFrame:
    frames = []
    for connection in connections:
        frames.append(
            pd.read_sql_query(
                """
                SELECT row_id, fold, rating, ok, prompt_tokens,
                       completion_tokens, elapsed_ms
                FROM predictions
                WHERE mode=? AND model=? AND prompt_version=?
                """,
                connection,
                params=(mode, PROFILE_MODEL, PROMPT_VERSION),
            )
        )
    combined = pd.concat(frames, ignore_index=True)
    combined = combined[
        combined["row_id"].isin(set(selected.tolist()))
    ].sort_values("row_id")
    combined = combined.drop_duplicates("row_id", keep="last")
    missing = np.setdiff1d(selected, combined.loc[combined["ok"] == 1, "row_id"])
    if len(missing):
        raise RuntimeError(
            f"{mode}: {len(missing)} selected rows lack successful predictions"
        )
    return combined[combined["ok"] == 1].copy()


PROFILE_MODEL = ""


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run zero-shot and few-shot Qwen direct rating prediction on a "
            "saved fold-stratified subsample."
        )
    )
    parser.add_argument("--profile", default="qwen25_7b")
    parser.add_argument("--modes", default="zero,few")
    parser.add_argument("--sample-size", type=int, default=100000)
    parser.add_argument("--max-chars", type=int, default=3000)
    parser.add_argument("--exemplar-max-chars", type=int, default=300)
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--endpoints")
    parser.add_argument("--write-every", type=int, default=2000)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    output_dir = ROOT / "outputs"
    profile = load_profile(ROOT / "config" / "local_llm.yaml", args.profile)
    if args.endpoints:
        profile.endpoints = [
            item.strip().rstrip("/")
            for item in args.endpoints.split(",")
            if item.strip()
        ]
    concurrency = args.concurrency or profile.concurrency
    modes = parse_modes(args.modes)
    global PROFILE_MODEL
    PROFILE_MODEL = profile.model
    log(
        f"direct LLM: endpoints={len(profile.endpoints)} "
        f"concurrency_per_endpoint={concurrency} "
        f"total_concurrency={concurrency * len(profile.endpoints)}"
    )

    dataset = pd.read_parquet(
        output_dir / "dataset_clean.parquet",
        columns=[
            "review_text",
            "rating_overall",
            "offering_id",
        ],
    )
    texts = dataset["review_text"].fillna("").astype(str).to_numpy()
    labels = dataset["rating_overall"].to_numpy(dtype=np.int8)
    groups = dataset["offering_id"].to_numpy()
    row_ids = np.arange(len(dataset), dtype=np.int64)
    splits = load_splits(output_dir)
    sample_path = (
        output_dir
        / "phase1"
        / f"direct_llm_subsample_n{args.sample_size}_seed{SEED}.npz"
    )
    selected = make_stratified_subsample(
        labels,
        splits,
        args.sample_size,
        sample_path,
    )
    selected_set = set(selected.tolist())
    fold_by_row = np.empty(len(dataset), dtype=np.int8)
    for split in splits:
        fold_by_row[split["test_idx"]] = split["fold"]

    cache_root = output_dir / "phase1" / "llm_direct_cache" / args.profile
    connections = [
        open_cache(cache_root / f"ep{index}" / "cache.sqlite")
        for index in range(len(profile.endpoints))
    ]
    harness = EvaluationHarness(output_dir)
    run_summary: Dict[str, Any] = {
        "profile": args.profile,
        "model": profile.model,
        "seed": SEED,
        "sample_size": int(len(selected)),
        "sample_path": str(sample_path),
        "selection": (
            "stratified by saved outer fold and integer rating"
        ),
        "prompt_version": PROMPT_VERSION,
        "modes": {},
    }

    try:
        for mode in modes:
            exemplars_by_fold = {
                split["fold"]: select_exemplars(
                    texts,
                    labels,
                    split["train_idx"],
                    split["fold"],
                    args.exemplar_max_chars,
                )
                for split in splits
            }
            items_by_endpoint: List[List[RequestItem]] = [
                [] for _ in profile.endpoints
            ]
            all_items: List[RequestItem] = []
            for row_id in selected:
                fold = int(fold_by_row[row_id])
                prompt = build_prompt(
                    texts[row_id],
                    mode,
                    exemplars_by_fold[fold],
                    args.max_chars,
                )
                hash_value = hashlib.sha256(
                    prompt.encode("utf-8")
                ).hexdigest()
                endpoint_idx = int(row_id % len(profile.endpoints))
                item = RequestItem(
                    row_id=int(row_id),
                    fold=fold,
                    mode=mode,
                    prompt=prompt,
                    prompt_hash=hash_value,
                    endpoint_idx=endpoint_idx,
                )
                all_items.append(item)
                items_by_endpoint[endpoint_idx].append(item)

            uncached_by_endpoint: List[List[RequestItem]] = []
            for endpoint_idx, endpoint_items in enumerate(
                items_by_endpoint
            ):
                hit_keys = (
                    cached_keys(
                        connections[endpoint_idx],
                        endpoint_items,
                        profile.model,
                    )
                    if args.resume
                    else set()
                )
                uncached = [
                    item
                    for item in endpoint_items
                    if cache_key(item, profile.model) not in hit_keys
                ]
                uncached_by_endpoint.append(uncached)
                log(
                    f"{mode} ep{endpoint_idx}: total={len(endpoint_items)} "
                    f"cache_hits={len(hit_keys)} requests={len(uncached)}"
                )

            with PeakRSSMonitor() as memory:
                started = time.perf_counter()
                asyncio.run(
                    run_uncached(
                        uncached_by_endpoint,
                        connections,
                        profile,
                        concurrency,
                        args.write_every,
                        mode,
                    )
                )
                request_seconds = time.perf_counter() - started
                predictions = read_predictions(
                    connections,
                    mode,
                    selected,
                )

            predictions = predictions.set_index("row_id").loc[selected]
            model_id = f"qwen_direct_{mode}_n{len(selected)}"
            fold_records = []
            for split in splits:
                fold = split["fold"]
                test_rows = np.asarray(
                    sorted(
                        selected_set.intersection(
                            split["test_idx"].tolist()
                        )
                    ),
                    dtype=np.int64,
                )
                fold_prediction = predictions.loc[
                    test_rows, "rating"
                ].to_numpy(dtype=float)
                artifact = harness.write_fold(
                    model_id=model_id,
                    fold=fold,
                    row_id=test_rows,
                    offering_id=groups[test_rows],
                    y_true=labels[test_rows].astype(float),
                    y_pred_raw=fold_prediction,
                    preprocessing_seconds=0.0,
                    tuning_seconds=0.0,
                    fit_seconds=0.0,
                    predict_seconds=request_seconds
                    * len(test_rows)
                    / len(selected),
                    peak_memory_mb=memory.peak_mb,
                    selected_hyperparameters={
                        "mode": mode,
                        "temperature": 0,
                        "max_tokens": 1,
                        "structured_outputs": {
                            "choice": ["1", "2", "3", "4", "5"]
                        },
                    },
                    extra={
                        "subsample": True,
                        "sample_size": int(len(selected)),
                        "sample_path": str(sample_path),
                        "prompt_version": PROMPT_VERSION,
                    },
                )
                fold_records.append(artifact.record)
                log(
                    f"{model_id} fold {fold}: "
                    f"MAE={artifact.record['clipped']['mae']:.4f} "
                    f"RMSE={artifact.record['clipped']['rmse']:.4f}"
                )
            metrics = harness.finalize_model(
                model_id,
                {
                    "description": (
                        f"Qwen2.5 direct {mode}-shot integer rating"
                    ),
                    "seed": SEED,
                    "subsample": True,
                    "sample_size": int(len(selected)),
                    "selection": (
                        "stratified by saved outer fold and rating"
                    ),
                    "deployable": True,
                },
            )
            metrics["request_seconds_total"] = request_seconds
            metrics["gpu_hours_total"] = (
                request_seconds * len(profile.endpoints) / 3600
            )
            write_json(
                output_dir / "metrics" / f"{model_id}.json",
                metrics,
            )
            run_summary["modes"][mode] = metrics
    finally:
        for connection in connections:
            connection.close()

    summary_path = (
        output_dir
        / "phase1"
        / f"direct_llm_{args.profile}_summary.json"
    )
    write_json(summary_path, run_summary)
    log(f"saved: {summary_path}")


if __name__ == "__main__":
    main()
