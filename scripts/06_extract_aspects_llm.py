#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import ast
import hashlib
import json
import re
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import threading

import httpx
import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    class DummyTqdm:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def update(self, n: int = 1) -> None:
            return None

        def set_postfix(self, **kwargs) -> None:
            return None

        def close(self) -> None:
            return None

    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else DummyTqdm()

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.aspect_extractor_base import ASPECTS  # noqa: E402

SEED = 42

SCHEMA_VERSION = "v1_aspects6"
PROMPT_VERSION = "v2_strictjson"
FAIL_LOG_LOCK = threading.Lock()

SYSTEM_PROMPT = (
    "You are a strict JSON generator. Output ONLY a single JSON object that "
    "exactly matches the required schema. No explanations, no code blocks."
)

USER_PROMPT_TEMPLATE = """Extract structured aspects from the hotel review.
Return exactly ONE JSON object (not a list). No code blocks. No extra text.
All required fields must appear, even if the review does not mention them.

Required schema:
{
  "aspects": {
    "service": {"mentioned": true/false, "sentiment": "pos|neg|neu", "score": -2|-1|0|1|2, "evidence": ["..."]},
    "cleanliness": {"mentioned": true/false, "sentiment": "pos|neg|neu", "score": -2|-1|0|1|2, "evidence": ["..."]},
    "location": {"mentioned": true/false, "sentiment": "pos|neg|neu", "score": -2|-1|0|1|2, "evidence": ["..."]},
    "rooms": {"mentioned": true/false, "sentiment": "pos|neg|neu", "score": -2|-1|0|1|2, "evidence": ["..."]},
    "value": {"mentioned": true/false, "sentiment": "pos|neg|neu", "score": -2|-1|0|1|2, "evidence": ["..."]},
    "sleep_quality": {"mentioned": true/false, "sentiment": "pos|neg|neu", "score": -2|-1|0|1|2, "evidence": ["..."]}
  },
  "traveler_segment": "business|couple|family|solo|friends|unknown",
  "overall_summary": "one sentence"
}

Rules:
- Evidence must be a STRING ARRAY (no nested arrays), max 2 entries, each <= 20 words.
- If aspect not mentioned: mentioned=false, sentiment="neu", score=0, evidence=[]
- Output MUST be a single JSON object, no list, no markdown, no extra text.

Review:
{review_text}
"""


@dataclass
class Profile:
    name: str
    provider: str
    base_url: str
    endpoints: List[str]
    api_key: str
    model: str
    temperature: float
    max_tokens: int
    timeout_sec: int
    concurrency: int
    max_retries: int


@dataclass
class RowInfo:
    row_id: int
    review_id: Optional[object]
    prompt_hash: str
    cache_key: str
    user_prompt: str
    text_empty: bool
    schema_version: str
    prompt_version: str


def log(message: str) -> None:
    print(message)


def parse_yaml_simple(text: str) -> Dict[str, Any]:
    profiles: Dict[str, Dict[str, Any]] = {}
    current_profile: Optional[str] = None
    current_list_key: Optional[str] = None
    in_profiles = False
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if stripped.endswith(":") and ":" not in stripped[:-1]:
            key = stripped[:-1]
            if indent == 0 and key == "profiles":
                in_profiles = True
                continue
            if in_profiles and indent == 2:
                current_profile = key
                profiles[current_profile] = {}
                current_list_key = None
                continue
            if in_profiles and indent >= 4 and current_profile:
                current_list_key = key
                profiles[current_profile][current_list_key] = []
                continue
        if in_profiles and indent >= 4 and current_profile:
            if stripped.startswith("- ") and current_list_key:
                value = stripped[2:].strip().strip('"').strip("'")
                profiles[current_profile][current_list_key].append(value)
                continue
            if ":" not in stripped:
                continue
            key, value = stripped.split(":", 1)
            current_list_key = None
            value = value.strip().strip('"').strip("'")
            if value.lower() in {"true", "false"}:
                parsed: Any = value.lower() == "true"
            else:
                try:
                    parsed = int(value)
                except ValueError:
                    try:
                        parsed = float(value)
                    except ValueError:
                        parsed = value
            profiles[current_profile][key.strip()] = parsed
    return {"profiles": profiles}


def load_profile(path: Path, name: str) -> Profile:
    if not path.exists():
        raise FileNotFoundError(f"missing config: {path}")
    raw_text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(raw_text)
    except Exception:
        data = parse_yaml_simple(raw_text)
    profiles = data.get("profiles", {})
    if name not in profiles:
        raise ValueError(f"profile '{name}' not found in {path}")
    cfg = profiles[name]
    endpoints: List[str] = []
    raw_endpoints = cfg.get("endpoints")
    if isinstance(raw_endpoints, list):
        endpoints = [str(item).strip() for item in raw_endpoints if str(item).strip()]
    elif isinstance(raw_endpoints, str):
        endpoints = [item.strip() for item in raw_endpoints.split(",") if item.strip()]
    base_url = str(cfg.get("base_url", ""))
    if not endpoints and base_url:
        endpoints = [base_url]
    return Profile(
        name=name,
        provider=str(cfg.get("provider", "openai_compatible")),
        base_url=base_url,
        endpoints=endpoints,
        api_key=str(cfg["api_key"]),
        model=str(cfg["model"]),
        temperature=float(cfg.get("temperature", 0)),
        max_tokens=int(cfg.get("max_tokens", 900)),
        timeout_sec=int(cfg.get("timeout_sec", 120)),
        concurrency=int(cfg.get("concurrency", 16)),
        max_retries=int(cfg.get("max_retries", 3)),
    )


def load_dataset(path: Path) -> pd.DataFrame:
    if path.exists():
        try:
            return pd.read_parquet(path)
        except Exception:
            log("parquet read failed; falling back to CSV.")
    csv_path = path.with_suffix(".csv")
    if csv_path.exists():
        log("parquet not found; loading CSV fallback instead.")
        return pd.read_csv(csv_path, low_memory=False)
    raise FileNotFoundError(f"missing dataset: {path}")


def build_review_text(row: pd.Series, title_col: Optional[str], text_col: str, max_chars: int) -> str:
    title = str(row.get(title_col, "")).strip() if title_col else ""
    text = str(row.get(text_col, "")).strip()
    if title and text:
        combined = f"{title}. {text}"
    else:
        combined = title or text
    combined = combined.strip()
    if max_chars and len(combined) > max_chars:
        combined = combined[:max_chars]
    return combined


def prompt_hash(system_prompt: str, user_prompt: str) -> str:
    payload = f"{system_prompt}\n{user_prompt}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_row_info(
    row: pd.Series,
    row_id: int,
    model: str,
    title_col: Optional[str],
    text_col: str,
    max_chars: int,
) -> RowInfo:
    review_text = build_review_text(row, title_col, text_col, max_chars)
    user_prompt = USER_PROMPT_TEMPLATE.replace("{review_text}", review_text)
    hash_val = prompt_hash(SYSTEM_PROMPT, user_prompt)
    cache_key = f"{row_id}:{hash_val}:{model}:{SCHEMA_VERSION}:{PROMPT_VERSION}"
    review_id = row.get("id") if "id" in row else None
    return RowInfo(
        row_id=row_id,
        review_id=review_id,
        prompt_hash=hash_val,
        cache_key=cache_key,
        user_prompt=user_prompt,
        text_empty=len(review_text.strip()) == 0,
        schema_version=SCHEMA_VERSION,
        prompt_version=PROMPT_VERSION,
    )


def init_cache(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS cache (
            cache_key TEXT PRIMARY KEY,
            row_id INTEGER,
            model TEXT,
            prompt_hash TEXT,
            schema_version TEXT,
            prompt_version TEXT,
            ok INTEGER,
            response_json TEXT,
            raw_text TEXT,
            error TEXT,
            prompt_tokens INTEGER,
            completion_tokens INTEGER,
            elapsed_ms INTEGER,
            created_at TEXT
        )
        """
    )
    existing_cols = {
        row[1] for row in conn.execute("PRAGMA table_info(cache)").fetchall()
    }
    if "schema_version" not in existing_cols:
        conn.execute("ALTER TABLE cache ADD COLUMN schema_version TEXT")
    if "prompt_version" not in existing_cols:
        conn.execute("ALTER TABLE cache ADD COLUMN prompt_version TEXT")
    conn.commit()


def fetch_cache_map(conn: sqlite3.Connection, keys: List[str]) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    if not keys:
        return results
    chunk_size = 500
    for i in range(0, len(keys), chunk_size):
        chunk = keys[i : i + chunk_size]
        placeholders = ",".join("?" * len(chunk))
        query = f"""
            SELECT cache_key, ok, response_json, raw_text, error,
                   prompt_tokens, completion_tokens, elapsed_ms,
                   prompt_hash, model, schema_version, prompt_version
            FROM cache
            WHERE cache_key IN ({placeholders})
        """
        for row in conn.execute(query, chunk).fetchall():
            results[row[0]] = {
                "ok": row[1],
                "response_json": row[2],
                "raw_text": row[3],
                "error": row[4],
                "prompt_tokens": row[5],
                "completion_tokens": row[6],
                "elapsed_ms": row[7],
                "prompt_hash": row[8],
                "model": row[9],
                "schema_version": row[10],
                "prompt_version": row[11],
            }
    return results


def upsert_cache(conn: sqlite3.Connection, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    conn.executemany(
        """
        INSERT OR REPLACE INTO cache (
            cache_key, row_id, model, prompt_hash, schema_version, prompt_version,
            ok, response_json, raw_text, error, prompt_tokens, completion_tokens,
            elapsed_ms, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                row["cache_key"],
                row["row_id"],
                row["model"],
                row["prompt_hash"],
                row.get("schema_version"),
                row.get("prompt_version"),
                row["ok"],
                row.get("response_json"),
                row.get("raw_text"),
                row.get("error"),
                row.get("prompt_tokens"),
                row.get("completion_tokens"),
                row.get("elapsed_ms"),
                row.get("created_at"),
            )
            for row in rows
        ],
    )
    conn.commit()


def _strip_fences(text: str) -> str:
    cleaned = text.strip()
    cleaned = cleaned.replace("```json", "").replace("```", "").strip()
    return cleaned


def _extract_json_snippet(text: str) -> Optional[str]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1:
        return None
    if end == -1 or end <= start:
        return text[start:]
    return text[start : end + 1]


def _repair_common_issues(snippet: str) -> str:
    fixed = re.sub(r"\]\s*,\s*\[", ", ", snippet)
    fixed = re.sub(r",\s*([}\]])", r"\1", fixed)
    brace_diff = fixed.count("{") - fixed.count("}")
    if brace_diff > 0:
        fixed += "}" * brace_diff
    bracket_diff = fixed.count("[") - fixed.count("]")
    if bracket_diff > 0:
        fixed += "]" * bracket_diff
    return fixed


def safe_json_loads(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    cleaned = _strip_fences(text)
    snippet = _extract_json_snippet(cleaned)
    if snippet is None:
        return None

    candidates = [snippet]
    repaired = _repair_common_issues(snippet)
    if repaired != snippet:
        candidates.append(repaired)

    for candidate in candidates:
        for parser in (json.loads, ast.literal_eval):
            try:
                obj = parser(candidate)
            except Exception:
                continue
            if isinstance(obj, list) and obj:
                obj = obj[0]
            if isinstance(obj, dict):
                return obj
    return None


def normalize_evidence(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        items = []
        for item in value:
            if isinstance(item, list):
                items.extend(item)
            else:
                items.append(item)
    else:
        items = [value]
    cleaned: List[str] = []
    for item in items:
        text = str(item).strip()
        if not text:
            continue
        words = text.split()
        if len(words) > 20:
            text = " ".join(words[:20])
        cleaned.append(text)
        if len(cleaned) >= 2:
            break
    return cleaned


def normalize_response(obj: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    default_aspects = {
        aspect: {"mentioned": False, "sentiment": "neu", "score": 0, "evidence": []}
        for aspect in ASPECTS
    }
    if not isinstance(obj, dict):
        return {"aspects": default_aspects, "traveler_segment": "unknown", "overall_summary": ""}
    aspects_in = obj.get("aspects", {})
    normalized_aspects: Dict[str, Dict[str, Any]] = {}
    for aspect in ASPECTS:
        data = aspects_in.get(aspect, {}) if isinstance(aspects_in, dict) else {}
        mentioned = bool(data.get("mentioned", False))
        score = data.get("score", 0)
        try:
            score = int(score)
        except Exception:
            score = 0
        score = max(-2, min(2, score))
        sentiment = str(data.get("sentiment", "neu")).lower()
        if sentiment in {"positive", "pos"}:
            sentiment = "pos"
        elif sentiment in {"negative", "neg"}:
            sentiment = "neg"
        elif sentiment in {"neutral", "neu"}:
            sentiment = "neu"
        else:
            sentiment = "pos" if score > 0 else "neg" if score < 0 else "neu"

        evidence = normalize_evidence(data.get("evidence", []))
        if not mentioned:
            sentiment = "neu"
            score = 0
            evidence = []
        normalized_aspects[aspect] = {
            "mentioned": bool(mentioned),
            "sentiment": sentiment,
            "score": score,
            "evidence": evidence,
        }
    traveler_segment = str(obj.get("traveler_segment", "unknown")).lower()
    allowed_segments = {"business", "couple", "family", "solo", "friends", "unknown"}
    if traveler_segment not in allowed_segments:
        traveler_segment = "unknown"
    overall_summary = str(obj.get("overall_summary", "")).strip()
    return {
        "aspects": normalized_aspects,
        "traveler_segment": traveler_segment,
        "overall_summary": overall_summary,
    }


def build_output_row(
    row_info: RowInfo,
    normalized: Dict[str, Any],
    ok: int,
    error: Optional[str],
    prompt_tokens: Optional[int],
    completion_tokens: Optional[int],
    elapsed_ms: Optional[int],
    model: str,
    prompt_hash: str,
    response_json: str,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "row_index": int(row_info.row_id),
        "row_id": int(row_info.row_id),
        "review_id": row_info.review_id,
        "ok": ok,
        "error": error,
        "prompt_hash": prompt_hash,
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "elapsed_ms": elapsed_ms,
        "traveler_segment": normalized.get("traveler_segment", "unknown"),
        "overall_summary": normalized.get("overall_summary", ""),
        "response_json": response_json,
    }
    aspects = normalized.get("aspects", {})
    for aspect in ASPECTS:
        data = aspects.get(aspect, {})
        try:
            score = int(data.get("score", 0))
        except Exception:
            score = 0
        score = max(-2, min(2, score))
        mentioned = bool(data.get("mentioned", False))
        row[f"aspect_{aspect}_score"] = score
        row[f"aspect_{aspect}_mentioned"] = mentioned
    return row


class OutputWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.csv_path = path.with_suffix(".csv")
        self.use_parquet = False
        self.writer = None
        self.csv_written = False
        try:
            import pyarrow as pa  # type: ignore
            import pyarrow.parquet as pq  # type: ignore

            self.pa = pa
            self.pq = pq
            self.use_parquet = True
        except Exception:
            log("pyarrow not available; writing CSV output instead.")

    def write_rows(self, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return
        df = pd.DataFrame(rows)
        if self.use_parquet:
            table = self.pa.Table.from_pandas(df)
            if self.writer is None:
                self.writer = self.pq.ParquetWriter(self.path, table.schema)
            self.writer.write_table(table)
        else:
            df.to_csv(
                self.csv_path,
                mode="a" if self.csv_written else "w",
                index=False,
                header=not self.csv_written,
            )
            self.csv_written = True

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()


def build_empty_result(row_info: RowInfo, profile: Profile) -> Dict[str, Any]:
    created_at = datetime.utcnow().isoformat()
    normalized = normalize_response(None)
    response_json = json.dumps(normalized, ensure_ascii=True)
    return {
        "cache_key": row_info.cache_key,
        "row_id": row_info.row_id,
        "model": profile.model,
        "prompt_hash": row_info.prompt_hash,
        "schema_version": row_info.schema_version,
        "prompt_version": row_info.prompt_version,
        "ok": 1,
        "response_json": response_json,
        "raw_text": "",
        "error": "empty_text",
        "prompt_tokens": None,
        "completion_tokens": None,
        "elapsed_ms": 0,
        "created_at": created_at,
        "normalized": normalized,
    }


def append_fail_sample(
    path: Path,
    row_info: RowInfo,
    error: str,
    raw_text: str,
    elapsed_ms: Optional[int],
) -> None:
    payload = {
        "row_id": row_info.row_id,
        "error": error,
        "raw_text": raw_text[:1200],
        "prompt_hash": row_info.prompt_hash,
        "elapsed_ms": elapsed_ms,
    }
    with FAIL_LOG_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def request_llm(
    client: httpx.Client,
    row_info: RowInfo,
    profile: Profile,
    seed: int,
    fail_log_path: Optional[Path],
) -> Dict[str, Any]:
    created_at = datetime.utcnow().isoformat()
    if row_info.text_empty:
        return build_empty_result(row_info, profile)
    raw_text = ""
    error = ""
    elapsed_ms: Optional[int] = None
    last_error = ""
    last_prompt_tokens: Optional[int] = None
    last_completion_tokens: Optional[int] = None
    for attempt in range(profile.max_retries):
        suffix = ""
        if attempt > 0:
            suffix = "\n\nONLY OUTPUT JSON. NO EXTRA TEXT."
            if last_error:
                suffix += f"\nPrevious error: {last_error}. Output a complete JSON object."
        user_prompt = row_info.user_prompt + suffix
        payload = {
            "model": profile.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": profile.temperature,
            "max_tokens": profile.max_tokens,
            "seed": seed,
        }
        start = time.perf_counter()
        try:
            response = client.post("/chat/completions", json=payload)
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            raw_text = response.text
            if response.status_code != 200:
                error = f"http_{response.status_code}"
                last_error = error
                continue
            data = response.json()
            content = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            usage = data.get("usage", {}) if isinstance(data, dict) else {}
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
            last_prompt_tokens = prompt_tokens if prompt_tokens is not None else last_prompt_tokens
            last_completion_tokens = (
                completion_tokens if completion_tokens is not None else last_completion_tokens
            )
            parsed = safe_json_loads(content)
            if parsed is None:
                error = "json_parse_failed"
                last_error = error
                raw_text = content
                continue
            normalized = normalize_response(parsed)
            response_json = json.dumps(normalized, ensure_ascii=True)
            return {
                "cache_key": row_info.cache_key,
                "row_id": row_info.row_id,
                "model": profile.model,
                "prompt_hash": row_info.prompt_hash,
                "schema_version": row_info.schema_version,
                "prompt_version": row_info.prompt_version,
                "ok": 1,
                "response_json": response_json,
                "raw_text": content,
                "error": "",
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "elapsed_ms": elapsed_ms,
                "created_at": created_at,
                "normalized": normalized,
            }
        except Exception as exc:
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            error = str(exc)
            raw_text = ""
            last_error = error
    normalized = normalize_response(None)
    response_json = json.dumps(normalized, ensure_ascii=True)
    if fail_log_path is not None:
        append_fail_sample(fail_log_path, row_info, error, raw_text, elapsed_ms)
    return {
        "cache_key": row_info.cache_key,
        "row_id": row_info.row_id,
        "model": profile.model,
        "prompt_hash": row_info.prompt_hash,
        "schema_version": row_info.schema_version,
        "prompt_version": row_info.prompt_version,
        "ok": 0,
        "response_json": response_json,
        "raw_text": raw_text,
        "error": error,
        "prompt_tokens": last_prompt_tokens,
        "completion_tokens": last_completion_tokens,
        "elapsed_ms": elapsed_ms,
        "created_at": created_at,
        "normalized": normalized,
    }


def run_requests(
    rows: List[RowInfo],
    profile: Profile,
    seed: int,
    concurrency: int,
    fail_log_path: Optional[Path],
) -> List[Dict[str, Any]]:
    if not rows:
        return []
    headers = {"Authorization": f"Bearer {profile.api_key}"}
    timeout = httpx.Timeout(profile.timeout_sec)
    with httpx.Client(base_url=profile.base_url, headers=headers, timeout=timeout) as client:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {
                executor.submit(
                    request_llm, client, row_info, profile, seed, fail_log_path
                ): row_info
                for row_info in rows
            }
            results = []
            for future in as_completed(futures):
                results.append(future.result())
            return results


def run_requests_with_progress(
    rows: List[RowInfo],
    profile: Profile,
    seed: int,
    concurrency: int,
    fail_log_path: Optional[Path],
    progress_label: str,
    progress_position: int,
) -> Tuple[List[Dict[str, Any]], float]:
    if not rows:
        return [], 0.0
    headers = {"Authorization": f"Bearer {profile.api_key}"}
    timeout = httpx.Timeout(profile.timeout_sec)
    processed = 0
    ok_count = 0
    start_time = time.perf_counter()
    pbar = tqdm(
        total=len(rows),
        desc=progress_label,
        unit="rows",
        position=progress_position,
        leave=False,
    )
    with httpx.Client(base_url=profile.base_url, headers=headers, timeout=timeout) as client:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {
                executor.submit(
                    request_llm, client, row_info, profile, seed, fail_log_path
                ): row_info
                for row_info in rows
            }
            results = []
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                processed += 1
                if result.get("ok") == 1:
                    ok_count += 1
                elapsed = time.perf_counter() - start_time
                rows_per_sec = processed / elapsed if elapsed > 0 else 0.0
                ok_rate = ok_count / processed if processed else 0.0
                pbar.update(1)
                pbar.set_postfix(
                    rows_per_sec=f"{rows_per_sec:.2f}",
                    cache_hits=0,
                    ok_rate=f"{ok_rate:.3f}",
                )
            pbar.close()
    elapsed_total = time.perf_counter() - start_time
    return results, elapsed_total


def compute_pilot_stats(results: List[Dict[str, Any]], elapsed_s: float) -> Dict[str, Any]:
    processed_n = len(results)
    if processed_n == 0:
        return {
            "processed_n": 0,
            "ok_rate": 0.0,
            "avg_elapsed_ms": None,
            "throughput": 0.0,
            "avg_prompt_tokens": None,
            "avg_completion_tokens": None,
            "elapsed_s": elapsed_s,
        }
    ok_rate = sum(r["ok"] == 1 for r in results) / processed_n
    elapsed_vals = [r["elapsed_ms"] for r in results if r.get("elapsed_ms") is not None]
    avg_elapsed_ms = float(np.mean(elapsed_vals)) if elapsed_vals else None
    prompt_tokens = [r["prompt_tokens"] for r in results if r.get("prompt_tokens") is not None]
    completion_tokens = [
        r["completion_tokens"] for r in results if r.get("completion_tokens") is not None
    ]
    avg_prompt_tokens = float(np.mean(prompt_tokens)) if prompt_tokens else None
    avg_completion_tokens = float(np.mean(completion_tokens)) if completion_tokens else None
    throughput = processed_n / elapsed_s if elapsed_s > 0 else 0.0
    return {
        "processed_n": processed_n,
        "ok_rate": ok_rate,
        "avg_elapsed_ms": avg_elapsed_ms,
        "throughput": throughput,
        "avg_prompt_tokens": avg_prompt_tokens,
        "avg_completion_tokens": avg_completion_tokens,
        "elapsed_s": elapsed_s,
    }


def write_runtime_estimate(
    output_dir: Path,
    profile: Profile,
    endpoints: List[str],
    pilot_n: int,
    stats_by_endpoint: List[Dict[str, Any]],
    total_n_limit: int,
    dataset_total_n: int,
    cached_ok_before_limit_total: int,
    cached_ok_after_full_total: int,
) -> Dict[str, Any]:
    processed_total = sum(s.get("processed_n", 0) for s in stats_by_endpoint)
    throughput_per_endpoint = [s.get("throughput", 0.0) for s in stats_by_endpoint]
    throughput_combined = sum(throughput_per_endpoint)

    remaining_limit = max(0, total_n_limit - cached_ok_before_limit_total - processed_total)
    eta_limit_seconds = remaining_limit / throughput_combined if throughput_combined > 0 else None
    eta_limit_hours = eta_limit_seconds / 3600 if eta_limit_seconds else None
    eta_limit_days = eta_limit_hours / 24 if eta_limit_hours else None

    remaining_full = max(0, dataset_total_n - cached_ok_after_full_total - processed_total)
    eta_full_seconds = remaining_full / throughput_combined if throughput_combined > 0 else None
    eta_full_hours = eta_full_seconds / 3600 if eta_full_seconds else None
    eta_full_days = eta_full_hours / 24 if eta_full_hours else None

    avg_prompt_tokens = [
        s.get("avg_prompt_tokens") for s in stats_by_endpoint if s.get("avg_prompt_tokens") is not None
    ]
    avg_completion_tokens = [
        s.get("avg_completion_tokens")
        for s in stats_by_endpoint
        if s.get("avg_completion_tokens") is not None
    ]
    avg_prompt_tokens_val = float(np.mean(avg_prompt_tokens)) if avg_prompt_tokens else None
    avg_completion_tokens_val = float(np.mean(avg_completion_tokens)) if avg_completion_tokens else None

    lines = [
        "# LLM Runtime Estimate",
        "",
        f"- timestamp_utc: {datetime.utcnow().isoformat()}",
        f"- profile: {profile.name}",
        f"- model: {profile.model}",
        f"- endpoints: {len(endpoints)}",
        f"- pilot_n: {pilot_n}",
        f"- processed_n: {processed_total}",
        f"- avg_prompt_tokens: {avg_prompt_tokens_val}",
        f"- avg_completion_tokens: {avg_completion_tokens_val}",
        f"- throughput_rows_per_sec_single: {np.mean(throughput_per_endpoint) if throughput_per_endpoint else 0.0:.4f}",
        f"- throughput_rows_per_sec_combined: {throughput_combined:.4f}",
        f"- total_n_limit: {total_n_limit}",
        f"- dataset_total_n: {dataset_total_n}",
        f"- cached_ok_before_limit_total: {cached_ok_before_limit_total}",
        f"- cached_ok_total: {cached_ok_after_full_total}",
        f"- cached_ok_after_full_total: {cached_ok_after_full_total}",
        f"- remaining_limit: {remaining_limit}",
        f"- eta_limit_seconds: {eta_limit_seconds}",
        f"- eta_limit_hours: {eta_limit_hours}",
        f"- eta_limit_days: {eta_limit_days}",
        f"- remaining_full: {remaining_full}",
        f"- eta_full_seconds: {eta_full_seconds}",
        f"- eta_full_hours: {eta_full_hours}",
        f"- eta_full_days: {eta_full_days}",
        "",
        "## Endpoint Stats",
        "",
    ]
    for idx, stats in enumerate(stats_by_endpoint):
        lines.extend(
            [
                f"- endpoint_{idx}: {endpoints[idx] if idx < len(endpoints) else 'unknown'}",
                f"  - pilot_uncached_n: {stats.get('pilot_uncached_n', 0)}",
                f"  - processed_n: {stats.get('processed_n', 0)}",
                f"  - pilot_walltime_sec: {stats.get('elapsed_s', 0.0)}",
                f"  - pilot_rows_per_sec: {stats.get('throughput', 0.0):.4f}",
                f"  - ok_rate: {stats.get('ok_rate', 0.0):.4f}",
                f"  - throughput_rows_per_sec: {stats.get('throughput', 0.0):.4f}",
                f"  - cache_ok_before: {stats.get('cache_ok_before', 0)}",
                f"  - cache_ok_after: {stats.get('cache_ok_after', 0)}",
            ]
        )
    lines.append("")

    output_path = output_dir / f"llm_runtime_estimate_{profile.name}.md"
    output_path.write_text("\n".join(lines), encoding="utf-8")
    log(f"saved: {output_path}")
    return {
        "eta_limit_hours": eta_limit_hours,
        "eta_limit_days": eta_limit_days,
        "eta_full_hours": eta_full_hours,
        "eta_full_days": eta_full_days,
        "throughput_combined": throughput_combined,
        "throughput_per_endpoint": throughput_per_endpoint,
        "remaining_limit": remaining_limit,
        "remaining_full": remaining_full,
    }


def count_cached_ok(
    conn: sqlite3.Connection,
    model: str,
    schema_version: str,
    prompt_version: str,
    row_id_max: Optional[int] = None,
) -> int:
    if row_id_max is None:
        row = conn.execute(
            "SELECT COUNT(*) FROM cache WHERE model = ? AND ok = 1 AND schema_version = ? AND prompt_version = ?",
            (model, schema_version, prompt_version),
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT COUNT(*) FROM cache WHERE model = ? AND ok = 1 AND schema_version = ? AND prompt_version = ? AND row_id < ?",
            (model, schema_version, prompt_version, row_id_max),
        ).fetchone()
    return int(row[0]) if row else 0


def build_output_from_cache(
    row_info: RowInfo,
    cached: Dict[str, Any],
    profile: Profile,
) -> Optional[Dict[str, Any]]:
    response_json = cached.get("response_json") or ""
    parsed = safe_json_loads(response_json) if response_json else None
    normalized = normalize_response(parsed)
    if parsed is None and cached.get("ok") == 1:
        return None
    return build_output_row(
        row_info=row_info,
        normalized=normalized,
        ok=int(cached.get("ok", 0)),
        error=cached.get("error"),
        prompt_tokens=cached.get("prompt_tokens"),
        completion_tokens=cached.get("completion_tokens"),
        elapsed_ms=cached.get("elapsed_ms"),
        model=profile.model,
        prompt_hash=row_info.prompt_hash,
        response_json=response_json or json.dumps(normalized, ensure_ascii=True),
    )


def select_uncached_row_infos(
    df: pd.DataFrame,
    profile: Profile,
    conn: sqlite3.Connection,
    title_col: Optional[str],
    text_col: str,
    max_chars: int,
    pilot_n: int,
    seed: int,
    resume: bool,
) -> List[RowInfo]:
    if pilot_n <= 0:
        return []
    rng = np.random.default_rng(seed)
    positions = rng.permutation(len(df))
    selected: List[RowInfo] = []
    chunk_size = 200
    cursor = 0
    while cursor < len(positions) and len(selected) < pilot_n:
        batch_positions = positions[cursor : cursor + chunk_size]
        row_infos: List[RowInfo] = []
        cache_keys: List[str] = []
        for pos in batch_positions:
            row = df.iloc[int(pos)]
            row_id = int(row["row_index"])
            info = build_row_info(
                row=row,
                row_id=row_id,
                model=profile.model,
                title_col=title_col,
                text_col=text_col,
                max_chars=max_chars,
            )
            row_infos.append(info)
            cache_keys.append(info.cache_key)
        cached_map = fetch_cache_map(conn, cache_keys) if resume else {}
        for info in row_infos:
            cached = cached_map.get(info.cache_key)
            if cached and cached.get("ok") == 1:
                continue
            selected.append(info)
            if len(selected) >= pilot_n:
                break
        cursor += chunk_size
    return selected


def run_pilot(
    df: pd.DataFrame,
    profile: Profile,
    conn: sqlite3.Connection,
    title_col: Optional[str],
    text_col: str,
    max_chars: int,
    pilot: int,
    concurrency: int,
    seed: int,
    resume: bool,
    fail_log_path: Optional[Path],
    progress_label: str,
    progress_position: int,
) -> Tuple[Dict[str, Any], int, int]:
    total_n = len(df)
    pilot_n = min(pilot, total_n)
    if pilot_n <= 0:
        return compute_pilot_stats([], 0.0), pilot_n, 0

    uncached = select_uncached_row_infos(
        df=df,
        profile=profile,
        conn=conn,
        title_col=title_col,
        text_col=text_col,
        max_chars=max_chars,
        pilot_n=pilot_n,
        seed=seed,
        resume=resume,
    )
    pilot_uncached_n = len(uncached)
    log(f"pilot: target={pilot_n}, pilot_uncached_n={pilot_uncached_n}, total_n={total_n}")
    if not uncached:
        return compute_pilot_stats([], 0.0), pilot_n, 0

    results, elapsed = run_requests_with_progress(
        uncached,
        profile,
        seed,
        concurrency,
        fail_log_path,
        progress_label=progress_label,
        progress_position=progress_position,
    )
    stats = compute_pilot_stats(results, elapsed)
    upsert_cache(conn, results)
    log(
        "pilot processed {n} rows in {sec:.1f}s => {thr:.2f} rows/s (ok_rate={ok:.3f})".format(
            n=stats["processed_n"],
            sec=elapsed,
            thr=stats["throughput"],
            ok=stats["ok_rate"],
        )
    )
    return stats, pilot_n, pilot_uncached_n


def run_full(
    df: pd.DataFrame,
    profile: Profile,
    conn: sqlite3.Connection,
    title_col: Optional[str],
    text_col: str,
    max_chars: int,
    concurrency: int,
    seed: int,
    resume: bool,
    write_every: int,
    output_path: Path,
    fail_log_path: Optional[Path],
    progress_label: str,
    progress_position: int = 0,
) -> None:
    total_n = len(df)
    writer = OutputWriter(output_path)
    processed = 0
    cache_hits = 0
    ok_count = 0
    seen_row_ids: set[int] = set()
    start_time = time.perf_counter()

    pbar = tqdm(
        total=total_n,
        desc=progress_label,
        unit="rows",
        position=progress_position,
        leave=True,
    )

    for start_idx in range(0, total_n, write_every):
        end_idx = min(total_n, start_idx + write_every)
        batch_positions = list(range(start_idx, end_idx))
        row_infos = []
        cache_keys = []
        for pos in batch_positions:
            row = df.iloc[pos]
            row_id = int(row["row_index"])
            info = build_row_info(
                row=row,
                row_id=row_id,
                model=profile.model,
                title_col=title_col,
                text_col=text_col,
                max_chars=max_chars,
            )
            row_infos.append(info)
            cache_keys.append(info.cache_key)

        cached_map = fetch_cache_map(conn, cache_keys) if resume else {}
        output_rows: List[Dict[str, Any]] = []
        cache_updates: List[Dict[str, Any]] = []
        to_request: List[RowInfo] = []

        for info in row_infos:
            cached = cached_map.get(info.cache_key)
            if cached and cached.get("ok") == 1:
                output_row = build_output_from_cache(info, cached, profile)
                if output_row is not None:
                    output_rows.append(output_row)
                    cache_hits += 1
                    continue
            if info.text_empty:
                result = build_empty_result(info, profile)
                cache_updates.append(result)
                output_rows.append(
                    build_output_row(
                        row_info=info,
                        normalized=result["normalized"],
                        ok=result["ok"],
                        error=result["error"],
                        prompt_tokens=None,
                        completion_tokens=None,
                        elapsed_ms=result["elapsed_ms"],
                        model=profile.model,
                        prompt_hash=info.prompt_hash,
                        response_json=result["response_json"],
                    )
                )
                continue
            to_request.append(info)

        if to_request:
            results = run_requests(to_request, profile, seed, concurrency, fail_log_path)
            cache_updates.extend(results)
            result_map = {result["row_id"]: result for result in results}
            for info in to_request:
                result = result_map.get(info.row_id)
                if result is None:
                    result = {
                        "ok": 0,
                        "error": "missing_result",
                        "prompt_tokens": None,
                        "completion_tokens": None,
                        "elapsed_ms": None,
                        "response_json": json.dumps(normalize_response(None), ensure_ascii=True),
                        "normalized": normalize_response(None),
                    }
                normalized = result.get("normalized") or normalize_response(
                    safe_json_loads(result.get("response_json", ""))
                )
                output_rows.append(
                    build_output_row(
                        row_info=info,
                        normalized=normalized,
                        ok=result["ok"],
                        error=result.get("error"),
                        prompt_tokens=result.get("prompt_tokens"),
                        completion_tokens=result.get("completion_tokens"),
                        elapsed_ms=result.get("elapsed_ms"),
                        model=profile.model,
                        prompt_hash=info.prompt_hash,
                        response_json=result.get("response_json", json.dumps(normalized, ensure_ascii=True)),
                    )
                )

        if cache_updates:
            upsert_cache(conn, cache_updates)
        row_ids = [int(row.get("row_id", -1)) for row in output_rows]
        batch_dups = len(row_ids) - len(set(row_ids))
        cross_dups = len(set(row_ids) & seen_row_ids)
        dup_total = batch_dups + cross_dups
        if dup_total > 0:
            log(f"duplicate row_id detected: {dup_total} rows in {output_path}")
            raise ValueError(f"duplicate row_id detected: {dup_total}")
        seen_row_ids.update(row_ids)
        writer.write_rows(output_rows)
        processed += len(output_rows)
        ok_count += sum(1 for row in output_rows if row.get("ok") == 1)
        elapsed = time.perf_counter() - start_time
        rows_per_sec = processed / elapsed if elapsed > 0 else 0.0
        ok_rate = ok_count / processed if processed else 0.0
        pbar.update(len(output_rows))
        pbar.set_postfix(
            rows_per_sec=f"{rows_per_sec:.2f}",
            cache_hits=cache_hits,
            ok_rate=f"{ok_rate:.3f}",
        )

    writer.close()
    pbar.close()


def read_output_table(path: Path) -> pd.DataFrame:
    if path.exists():
        try:
            return pd.read_parquet(path)
        except Exception:
            log(f"parquet read failed for {path}; trying CSV fallback.")
    csv_path = path.with_suffix(".csv")
    if csv_path.exists():
        return pd.read_csv(csv_path, low_memory=False)
    raise FileNotFoundError(f"missing output part: {path}")


def merge_outputs(part_paths: List[Path], output_path: Path) -> None:
    frames = [read_output_table(path) for path in part_paths]
    merged = pd.concat(frames, ignore_index=True)
    if "row_id" not in merged.columns and "row_index" in merged.columns:
        merged["row_id"] = merged["row_index"].astype(int)
    if "row_id" in merged.columns:
        dup_count = int(merged["row_id"].duplicated().sum())
        if dup_count > 0:
            log(f"duplicate row_id detected in merge: {dup_count}")
            raise ValueError(f"duplicate row_id detected in merge: {dup_count}")
    if "row_index" in merged.columns:
        merged = merged.sort_values("row_index").reset_index(drop=True)
    try:
        merged.to_parquet(output_path, index=False)
    except Exception:
        csv_path = output_path.with_suffix(".csv")
        merged.to_csv(csv_path, index=False)
        log("parquet engine missing; merged output saved as CSV instead.")


def split_counts(total: int, parts: int) -> List[int]:
    if parts <= 0:
        return []
    base, extra = divmod(total, parts)
    return [base + (1 if idx < extra else 0) for idx in range(parts)]


def write_validation_report(df: pd.DataFrame, output_path: Path) -> None:
    rows = len(df)
    ok_rate = float((df["ok"] == 1).mean()) if "ok" in df.columns else None
    if "row_id" not in df.columns and "row_index" in df.columns:
        df = df.copy()
        df["row_id"] = df["row_index"].astype(int)
    row_id_dups = int(df["row_id"].duplicated().sum()) if "row_id" in df.columns else None

    aspect_score_stats: List[str] = []
    aspect_mentioned_stats: List[str] = []
    for aspect in ASPECTS:
        score_col = f"aspect_{aspect}_score"
        mention_col = f"aspect_{aspect}_mentioned"
        if score_col in df.columns:
            scores = pd.to_numeric(df[score_col], errors="coerce")
            score_min = float(scores.min()) if scores.notna().any() else None
            score_max = float(scores.max()) if scores.notna().any() else None
            out_of_range = int(((scores < -2) | (scores > 2)).sum())
            aspect_score_stats.append(
                f"- {score_col}: min={score_min} max={score_max} out_of_range={out_of_range}"
            )
        if mention_col in df.columns:
            values = df[mention_col].dropna()
            if values.empty:
                invalid = 0
                is_bool = True
            else:
                uniques = set(values.unique().tolist())
                allowed = {True, False, 0, 1}
                invalid = sum(1 for v in uniques if v not in allowed)
                is_bool = invalid == 0
            aspect_mentioned_stats.append(
                f"- {mention_col}: bool_like={is_bool} invalid_values={invalid}"
            )

    prompt_missing = int(df["prompt_tokens"].isna().sum()) if "prompt_tokens" in df.columns else None
    completion_missing = (
        int(df["completion_tokens"].isna().sum()) if "completion_tokens" in df.columns else None
    )

    samples = []
    if "response_json" in df.columns:
        for raw in df["response_json"].head(3).astype(str).tolist():
            sample = raw[:200].replace("\n", " ")
            samples.append(sample)

    lines = [
        "# LLM Output Sanity Check",
        "",
        f"- rows: {rows}",
        f"- ok_rate: {ok_rate}",
        f"- row_id_duplicates: {row_id_dups}",
        f"- prompt_tokens_missing: {prompt_missing}",
        f"- completion_tokens_missing: {completion_missing}",
        "",
        "## Aspect Score Ranges",
    ]
    lines.extend(aspect_score_stats or ["- no aspect score columns found"])
    lines.extend(["", "## Aspect Mentioned Bool Check"])
    lines.extend(aspect_mentioned_stats or ["- no aspect mentioned columns found"])
    lines.extend(["", "## response_json Samples"])
    if samples:
        for idx, sample in enumerate(samples, start=1):
            lines.append(f"- sample_{idx}: {sample}")
    else:
        lines.append("- no response_json column found")
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    log(f"saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract aspects with offline LLM.")
    parser.add_argument("--profile", default="qwen25_7b")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--pilot", type=int, default=2000)
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument("--max_tokens", type=int, default=None)
    parser.add_argument("--max_chars", type=int, default=4000)
    parser.add_argument("--endpoints", type=str, default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--write_every", type=int, default=2000)
    parser.add_argument("--output_tag", type=str, default=None)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--resume", dest="resume", action="store_true")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    args = parser.parse_args()

    config_path = ROOT / "config" / "local_llm.yaml"
    profile = load_profile(config_path, args.profile)
    concurrency = args.concurrency or profile.concurrency
    if args.max_tokens is not None:
        profile.max_tokens = args.max_tokens
    output_tag = args.output_tag or profile.name

    output_dir = ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.validate_only:
        output_path = output_dir / f"aspects_llm_{output_tag}.parquet"
        df_out = read_output_table(output_path)
        report_path = output_dir / f"llm_output_sanity_{profile.name}.md"
        write_validation_report(df_out, report_path)
        return

    dataset_path = output_dir / "dataset_clean.parquet"
    df_full = load_dataset(dataset_path).reset_index(drop=False).rename(columns={"index": "row_index"})
    dataset_total_n = len(df_full)

    title_col = None
    if "review_title" in df_full.columns:
        title_col = "review_title"
    elif "title" in df_full.columns:
        title_col = "title"

    if "text" in df_full.columns:
        text_col = "text"
    elif "review_text" in df_full.columns:
        text_col = "review_text"
    else:
        raise ValueError("missing text column (need text or review_text)")

    if args.limit is not None:
        df_full = df_full.iloc[: args.limit].copy()
    total_n_limit = len(df_full)

    endpoints: List[str] = []
    if args.endpoints:
        endpoints = [item.strip() for item in args.endpoints.split(",") if item.strip()]
    elif profile.endpoints and len(profile.endpoints) == 1:
        endpoints = profile.endpoints
    else:
        endpoints = [profile.base_url]

    num_endpoints = len(endpoints)
    if num_endpoints == 0:
        raise ValueError("no endpoints configured")

    total_concurrency = concurrency * num_endpoints
    log(
        f"endpoints={num_endpoints} concurrency_per_endpoint={concurrency} "
        f"total_concurrency={total_concurrency}"
    )

    pilot_total = min(args.pilot, total_n_limit)
    pilot_allocations = split_counts(pilot_total, num_endpoints)

    shard_info: List[Dict[str, Any]] = []
    for ep_idx, endpoint in enumerate(endpoints):
        profile_ep = Profile(
            name=profile.name,
            provider=profile.provider,
            base_url=endpoint,
            endpoints=profile.endpoints,
            api_key=profile.api_key,
            model=profile.model,
            temperature=profile.temperature,
            max_tokens=profile.max_tokens,
            timeout_sec=profile.timeout_sec,
            concurrency=concurrency,
            max_retries=profile.max_retries,
        )
        df_shard = df_full[df_full["row_index"] % num_endpoints == ep_idx].copy()
        df_shard = df_shard.reset_index(drop=True)

        cache_suffix = profile.name if num_endpoints == 1 else f"{profile.name}_ep{ep_idx}"
        cache_dir = output_dir / "llm_cache" / cache_suffix
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / "cache.sqlite"
        fail_log_name = (
            f"llm_fail_samples_{profile.name}.jsonl"
            if num_endpoints == 1
            else f"llm_fail_samples_{profile.name}_ep{ep_idx}.jsonl"
        )
        fail_log_path = output_dir / fail_log_name

        shard_info.append(
            {
                "endpoint_idx": ep_idx,
                "profile": profile_ep,
                "df_shard": df_shard,
                "cache_path": cache_path,
                "fail_log_path": fail_log_path,
                "output_path": output_dir
                / (
                    f"aspects_llm_{output_tag}.parquet"
                    if num_endpoints == 1
                    else f"aspects_llm_{output_tag}_part{ep_idx}.parquet"
                ),
                "progress_label": "llm" if num_endpoints == 1 else f"ep{ep_idx}",
                "progress_position": ep_idx,
                "pilot_label": "pilot" if num_endpoints == 1 else f"pilot-ep{ep_idx}",
                "pilot_position": ep_idx,
                "pilot_n": pilot_allocations[ep_idx] if ep_idx < len(pilot_allocations) else 0,
            }
        )

    def run_pilot_shard(shard: Dict[str, Any]) -> Dict[str, Any]:
        conn = sqlite3.connect(shard["cache_path"])
        init_cache(conn)
        cache_ok_before_limit = count_cached_ok(
            conn, shard["profile"].model, SCHEMA_VERSION, PROMPT_VERSION, total_n_limit
        ) if args.resume else 0
        cache_ok_before_full = count_cached_ok(
            conn, shard["profile"].model, SCHEMA_VERSION, PROMPT_VERSION
        ) if args.resume else 0

        pilot_stats, pilot_n, pilot_uncached_n = run_pilot(
            df=shard["df_shard"],
            profile=shard["profile"],
            conn=conn,
            title_col=title_col,
            text_col=text_col,
            max_chars=args.max_chars,
            pilot=shard["pilot_n"],
            concurrency=concurrency,
            seed=args.seed,
            resume=args.resume,
            fail_log_path=shard["fail_log_path"],
            progress_label=shard["pilot_label"],
            progress_position=shard["pilot_position"],
        )
        cache_ok_after_limit = count_cached_ok(
            conn, shard["profile"].model, SCHEMA_VERSION, PROMPT_VERSION, total_n_limit
        ) if args.resume else cache_ok_before_limit
        cache_ok_after_full = count_cached_ok(
            conn, shard["profile"].model, SCHEMA_VERSION, PROMPT_VERSION
        ) if args.resume else cache_ok_before_full
        conn.close()

        return {
            "endpoint_idx": shard["endpoint_idx"],
            "stats": pilot_stats,
            "pilot_n": pilot_n,
            "pilot_uncached_n": pilot_uncached_n,
            "cache_ok_before_limit": cache_ok_before_limit,
            "cache_ok_after_limit": cache_ok_after_limit,
            "cache_ok_before_full": cache_ok_before_full,
            "cache_ok_after_full": cache_ok_after_full,
        }

    def run_shard(shard: Dict[str, Any]) -> Path:
        conn = sqlite3.connect(shard["cache_path"])
        init_cache(conn)
        run_full(
            df=shard["df_shard"],
            profile=shard["profile"],
            conn=conn,
            title_col=title_col,
            text_col=text_col,
            max_chars=args.max_chars,
            concurrency=concurrency,
            seed=args.seed,
            resume=args.resume,
            write_every=args.write_every,
            output_path=shard["output_path"],
            fail_log_path=shard["fail_log_path"],
            progress_label=shard["progress_label"],
            progress_position=shard["progress_position"],
        )
        conn.close()
        return shard["output_path"]

    async def run_pipeline() -> None:
        pilot_tasks = [asyncio.to_thread(run_pilot_shard, shard) for shard in shard_info]
        pilot_results = await asyncio.gather(*pilot_tasks)
        pilot_results_sorted = sorted(pilot_results, key=lambda item: item["endpoint_idx"])

        stats_by_endpoint: List[Dict[str, Any]] = []
        cached_ok_before_limit_total = 0
        cached_ok_after_full_total = 0
        combined_rows_per_sec = 0.0

        for res in pilot_results_sorted:
            stats = res["stats"]
            combined_rows_per_sec += stats.get("throughput", 0.0)
            cached_ok_before_limit_total += res["cache_ok_before_limit"]
            cached_ok_after_full_total += res["cache_ok_after_full"]
            stats_by_endpoint.append(
                {
                    **stats,
                    "pilot_uncached_n": res["pilot_uncached_n"],
                    "cache_ok_before": res["cache_ok_before_limit"],
                    "cache_ok_after": res["cache_ok_after_limit"],
                }
            )
            log(
                "endpoint {idx} pilot_rows={rows} walltime={sec:.1f}s rows/s={thr:.2f} "
                "ok_rate={ok:.3f} cache_ok_before={c_before} cache_ok_after={c_after}".format(
                    idx=res["endpoint_idx"],
                    rows=stats.get("processed_n", 0),
                    sec=stats.get("elapsed_s", 0.0),
                    thr=stats.get("throughput", 0.0),
                    ok=stats.get("ok_rate", 0.0),
                    c_before=res["cache_ok_before_limit"],
                    c_after=res["cache_ok_after_limit"],
                )
            )
        log(f"pilot combined rows/s={combined_rows_per_sec:.2f}")

        estimate = write_runtime_estimate(
            output_dir=output_dir,
            profile=profile,
            endpoints=endpoints,
            pilot_n=pilot_total,
            stats_by_endpoint=stats_by_endpoint,
            total_n_limit=total_n_limit,
            dataset_total_n=dataset_total_n,
            cached_ok_before_limit_total=cached_ok_before_limit_total,
            cached_ok_after_full_total=cached_ok_after_full_total,
        )
        if estimate["throughput_combined"] > 0:
            log(
                "ETA limit: {hours:.1f}h (~{days:.2f}d), ETA full: {fhours:.1f}h (~{fdays:.2f}d) "
                "at {thr:.2f} rows/s combined".format(
                    hours=estimate["eta_limit_hours"] or 0.0,
                    days=estimate["eta_limit_days"] or 0.0,
                    fhours=estimate["eta_full_hours"] or 0.0,
                    fdays=estimate["eta_full_days"] or 0.0,
                    thr=estimate["throughput_combined"],
                )
            )
        else:
            log("Pilot had no new rows processed; ETA unavailable (all cached or errors).")

        if args.limit is not None and args.limit <= args.pilot:
            log("limit <= pilot; proceeding to output generation with cache.")
        else:
            log("starting full extraction run")

        main_tasks = [asyncio.to_thread(run_shard, shard) for shard in shard_info]
        part_paths = await asyncio.gather(*main_tasks)
        if num_endpoints == 1:
            log(f"done: {part_paths[0]}")
        else:
            merged_path = output_dir / f"aspects_llm_{output_tag}.parquet"
            merge_outputs(part_paths, merged_path)
            log(f"done: {merged_path}")

    asyncio.run(run_pipeline())


if __name__ == "__main__":
    main()
