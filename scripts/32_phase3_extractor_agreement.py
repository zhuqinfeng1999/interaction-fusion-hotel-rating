#!/usr/bin/env python3
"""Phase 3 L4: agreement between guided Qwen and Llama extractors."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.aspect_extractor_base import ASPECTS  # noqa: E402


SEGMENTS = ["business", "couple", "family", "solo", "friends", "unknown"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen-profile", default="qwen25_7b")
    parser.add_argument("--llama-profile", default="llama31_8b")
    parser.add_argument("--qwen-path", type=Path, default=None)
    parser.add_argument("--llama-path", type=Path, default=None)
    return parser.parse_args()


def resolve_unique(path: Path | None, pattern: str) -> Path:
    if path is not None:
        resolved = path if path.is_absolute() else ROOT / path
        if not resolved.exists():
            raise FileNotFoundError(resolved)
        return resolved
    matches = sorted((ROOT / "outputs" / "phase3").glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one {pattern}, found {matches}")
    return matches[0]


def safe_pearson(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def icc_2_1(left: np.ndarray, right: np.ndarray) -> float:
    """Shrout-Fleiss ICC(2,1): two-way random, absolute agreement."""
    ratings = np.column_stack([left, right]).astype(float)
    n, k = ratings.shape
    if n < 2:
        return float("nan")
    grand = float(ratings.mean())
    row_means = ratings.mean(axis=1)
    col_means = ratings.mean(axis=0)
    ms_rows = float(k * np.square(row_means - grand).sum() / (n - 1))
    ms_cols = float(n * np.square(col_means - grand).sum() / (k - 1))
    residual = ratings - row_means[:, None] - col_means[None, :] + grand
    ms_error = float(np.square(residual).sum() / ((n - 1) * (k - 1)))
    denominator = ms_rows + (k - 1) * ms_error + k * (ms_cols - ms_error) / n
    return float((ms_rows - ms_error) / denominator) if denominator else float("nan")


def safe_kappa(left: pd.Series, right: pd.Series) -> float:
    try:
        return float(cohen_kappa_score(left, right))
    except Exception:
        return float("nan")


def evidence_map(value: Any) -> Dict[str, List[str]]:
    try:
        parsed = json.loads(str(value))
    except Exception:
        return {}
    aspects = parsed.get("aspects", {}) if isinstance(parsed, dict) else {}
    result: Dict[str, List[str]] = {}
    for aspect in ASPECTS:
        item = aspects.get(aspect, {}) if isinstance(aspects, dict) else {}
        evidence = item.get("evidence", []) if isinstance(item, dict) else []
        if isinstance(evidence, list):
            result[aspect] = [str(value).strip() for value in evidence[:2] if str(value).strip()]
    return result


def evidence_rates(
    frame: pd.DataFrame, texts: pd.DataFrame, extractor: str
) -> pd.DataFrame:
    joined = frame[["row_id", "ok", "response_json"]].merge(
        texts, on="row_id", how="left", validate="one_to_one"
    )
    counts = {aspect: [0, 0] for aspect in ASPECTS}
    for record in joined[joined["ok"].eq(1)].itertuples(index=False):
        text = str(record.review_text or "").lower()
        for aspect, evidence in evidence_map(record.response_json).items():
            for snippet in evidence:
                counts[aspect][1] += 1
                if snippet.lower() in text:
                    counts[aspect][0] += 1
    rows = []
    for aspect, (hits, total) in counts.items():
        rows.append(
            {
                "extractor": extractor,
                "aspect": aspect,
                "hits": hits,
                "total_evidence": total,
                "hit_rate": hits / total if total else float("nan"),
            }
        )
    total_hits = sum(value[0] for value in counts.values())
    total_evidence = sum(value[1] for value in counts.values())
    rows.append(
        {
            "extractor": extractor,
            "aspect": "overall",
            "hits": total_hits,
            "total_evidence": total_evidence,
            "hit_rate": total_hits / total_evidence if total_evidence else float("nan"),
        }
    )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    output_dir = ROOT / "outputs"
    phase3 = output_dir / "phase3"
    qwen_path = resolve_unique(
        args.qwen_path, f"aspects_guided_{args.qwen_profile}_l1_n*.parquet"
    )
    llama_path = resolve_unique(
        args.llama_path, f"aspects_guided_{args.llama_profile}_l1_n*.parquet"
    )
    columns = [
        "row_id",
        "ok",
        "response_json",
        "traveler_segment",
        *[f"aspect_{aspect}_score" for aspect in ASPECTS],
        *[f"aspect_{aspect}_mentioned" for aspect in ASPECTS],
    ]
    qwen = pd.read_parquet(qwen_path, columns=columns)
    llama = pd.read_parquet(llama_path, columns=columns)
    joined = qwen.merge(
        llama, on="row_id", suffixes=("_qwen", "_llama"), validate="one_to_one"
    )
    both_ok = joined["ok_qwen"].eq(1) & joined["ok_llama"].eq(1)
    valid = joined[both_ok].copy()
    if len(joined) != len(qwen) or len(joined) != len(llama):
        raise ValueError("Qwen/Llama row sets differ")

    score_rows = []
    mention_rows = []
    for aspect in ASPECTS:
        left_score = valid[f"aspect_{aspect}_score_qwen"].to_numpy(dtype=float)
        right_score = valid[f"aspect_{aspect}_score_llama"].to_numpy(dtype=float)
        score_rows.append(
            {
                "aspect": aspect,
                "n": len(valid),
                "icc_2_1": icc_2_1(left_score, right_score),
                "pearson_r": safe_pearson(left_score, right_score),
                "exact_agreement": float(np.mean(left_score == right_score)),
                "mean_qwen": float(left_score.mean()),
                "mean_llama": float(right_score.mean()),
            }
        )
        left_mention = valid[f"aspect_{aspect}_mentioned_qwen"].astype(bool)
        right_mention = valid[f"aspect_{aspect}_mentioned_llama"].astype(bool)
        mention_rows.append(
            {
                "aspect": aspect,
                "n": len(valid),
                "cohen_kappa": safe_kappa(left_mention, right_mention),
                "exact_agreement": float((left_mention == right_mention).mean()),
                "mention_rate_qwen": float(left_mention.mean()),
                "mention_rate_llama": float(right_mention.mean()),
            }
        )
    scores = pd.DataFrame(score_rows)
    mentions = pd.DataFrame(mention_rows)
    segment_qwen = valid["traveler_segment_qwen"].fillna("unknown").astype(str)
    segment_llama = valid["traveler_segment_llama"].fillna("unknown").astype(str)
    segment_kappa = safe_kappa(segment_qwen, segment_llama)
    matrix = confusion_matrix(segment_qwen, segment_llama, labels=SEGMENTS)
    confusion = pd.DataFrame(matrix, index=SEGMENTS, columns=SEGMENTS)
    confusion.index.name = "qwen\\llama"

    texts = pd.read_parquet(
        output_dir / "dataset_clean.parquet", columns=["review_text"]
    ).reset_index(drop=True)
    texts["row_id"] = np.arange(len(texts), dtype=np.int64)
    selected_texts = texts[texts["row_id"].isin(joined["row_id"])].copy()
    evidence = pd.concat(
        [
            evidence_rates(qwen, selected_texts, "qwen"),
            evidence_rates(llama, selected_texts, "llama"),
        ],
        ignore_index=True,
    )

    scores.to_csv(phase3 / "l4_score_agreement.csv", index=False)
    mentions.to_csv(phase3 / "l4_mention_agreement.csv", index=False)
    confusion.to_csv(phase3 / "l4_segment_confusion.csv")
    evidence.to_csv(phase3 / "l4_evidence_hit_rate.csv", index=False)
    payload = {
        "experiment": "L4",
        "qwen_path": str(qwen_path.relative_to(ROOT)),
        "llama_path": str(llama_path.relative_to(ROOT)),
        "sample_n": len(joined),
        "qwen_ok_rate": float(joined["ok_qwen"].eq(1).mean()),
        "llama_ok_rate": float(joined["ok_llama"].eq(1).mean()),
        "both_ok_n": int(both_ok.sum()),
        "segment_kappa": segment_kappa,
        "scores": score_rows,
        "mentions": mention_rows,
        "segment_confusion": confusion.to_dict(),
        "evidence": evidence.to_dict(orient="records"),
    }
    (phase3 / "l4_extractor_agreement.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    lines = [
        "# Phase 3 L4 — Guided Extractor Agreement",
        "",
        f"- sample n: {len(joined):,}",
        f"- both OK n: {int(both_ok.sum()):,}",
        f"- Qwen OK rate: {payload['qwen_ok_rate']:.4%}",
        f"- Llama OK rate: {payload['llama_ok_rate']:.4%}",
        "- decoding match: same prompt, strict JSON schema, temperature=0",
        "",
        "## Aspect scores",
        "",
        scores.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Mention indicators",
        "",
        mentions.to_markdown(index=False, floatfmt=".4f"),
        "",
        f"## Traveler segment (Cohen's kappa = {segment_kappa:.4f})",
        "",
        confusion.to_markdown(),
        "",
        "## Evidence span hit rates",
        "",
        evidence.to_markdown(index=False, floatfmt=".4f"),
        "",
    ]
    report = phase3 / "experiment_l4_agreement.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"saved: {report}")


if __name__ == "__main__":
    main()
