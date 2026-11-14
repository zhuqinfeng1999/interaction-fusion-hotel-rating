#!/usr/bin/env python3
"""Second-round revision (Reviewer 4.1/4.3): does the evidence support the assigned polarity?

An independent judge (Llama-3.1-8B-Instruct, a different model family from the
Qwen extractor) reads ONLY the evidence strings that Qwen quoted for one aspect,
without seeing Qwen's label, and answers (a) whether the excerpts concern that
aspect and (b) which opinion they express. Agreement with Qwen's sentiment label
measures whether the evidence semantically supports the extracted polarity.

Sample: 2,000 reviews drawn uniformly at random (seed 42) from schema-valid rows
with at least one mentioned aspect that carries evidence. Unit: review x aspect.
"""
import argparse
import asyncio
import json
import re
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
MR = ROOT / "outputs" / "r2" / "judge"
MR.mkdir(parents=True, exist_ok=True)
ASPECTS = ["service", "cleanliness", "location", "rooms", "value", "sleep_quality"]
DESC = {"service": "service and staff", "cleanliness": "cleanliness", "location": "location",
        "rooms": "rooms", "value": "value for money", "sleep_quality": "sleep quality (noise, bed comfort, quiet)"}
SYSTEM = "You are a careful annotator of hotel reviews. Answer only from the quoted excerpts."
HEAD = """The excerpts below were quoted from one hotel review.
Target aspect: {desc}
Excerpts: {ev}

"""
USER_REL = HEAD + "Using only these excerpts, do they talk about the hotel's {desc}? Answer yes or no."
USER_POL = HEAD + "Using only these excerpts, what opinion about the {desc} do they express? Answer positive, negative, neutral, or mixed."
CHOICES = {"rel": ["yes", "no"], "pol": ["positive", "negative", "neutral", "mixed"]}
# Guided JSON made Llama emit runaway whitespace on short outputs, so each question is asked
# separately with decoding restricted to the listed answers (as for the zero-shot rater).


def norm(s):
    return re.sub(r"\s+", " ", str(s).lower()).strip()


def build_items(n_reviews, seed):
    ax = pd.read_parquet(OUT / "aspects_llm_qwen25_7b.parquet", columns=["row_id", "ok", "response_json"]).sort_values("row_id")
    ax = ax[ax.ok.eq(1)]
    texts = pd.read_parquet(OUT / "dataset_clean.parquet", columns=["review_text"])["review_text"].fillna("").astype(str).to_numpy()
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ax))
    items = []
    picked = 0
    for pos in order:
        row = ax.iloc[pos]
        try:
            payload = json.loads(row.response_json)
        except Exception:
            continue
        asp = payload.get("aspects", payload)
        cand = []
        for a in ASPECTS:
            d = asp.get(a) or {}
            ev = d.get("evidence") or []
            if isinstance(ev, str):
                ev = [ev]
            ev = [str(e).strip() for e in ev if str(e).strip()]
            if d.get("mentioned") and ev:
                cand.append((a, d.get("sentiment"), d.get("score"), ev))
        if not cand:
            continue
        rid = int(row.row_id)
        rt = norm(texts[rid])
        for a, sent, score, ev in cand:
            items.append({"row_id": rid, "aspect": a, "sentiment": sent, "score": score, "n_spans": len(ev),
                          "evidence": " | ".join(f'"{e}"' for e in ev),
                          "all_spans_hit": all(norm(e) in rt for e in ev)})
        picked += 1
        if picked >= n_reviews:
            break
    return pd.DataFrame(items)


async def judge(items, endpoint, model, concurrency):
    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(base_url=endpoint, headers={"Authorization": "Bearer local-llm"}, timeout=120) as client:
        async def ask(template, choices, r):
            body = {"model": model, "temperature": 0, "max_tokens": 5, "seed": 42,
                    "messages": [{"role": "system", "content": SYSTEM},
                                 {"role": "user", "content": template.format(desc=DESC[r.aspect], ev=r.evidence)}],
                    "structured_outputs": {"choice": choices}}
            for attempt in range(3):
                try:
                    async with sem:
                        resp = await client.post("/chat/completions", json=body)
                    resp.raise_for_status()
                    out = resp.json()["choices"][0]["message"]["content"].strip()
                    if out in choices:
                        return out
                except Exception:  # retry
                    await asyncio.sleep(1 + attempt)
            return None

        async def one(i, r):
            rel = await ask(USER_REL, CHOICES["rel"], r)
            pol = await ask(USER_POL, CHOICES["pol"], r)
            return i, rel, pol
        res = await asyncio.gather(*[one(i, r) for i, r in enumerate(items.itertuples())])
    res.sort()
    return [x[1] for x in res], [x[2] for x in res]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--endpoint", default="http://127.0.0.1:8010/v1")
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--n-reviews", type=int, default=2000)
    p.add_argument("--concurrency", type=int, default=64)
    a = p.parse_args()
    items = build_items(a.n_reviews, 42)
    print("items", len(items), "reviews", items.row_id.nunique(), flush=True)
    rel, pol = asyncio.run(judge(items, a.endpoint, a.model, a.concurrency))
    items["judge_relevant"] = rel
    items["judge_polarity"] = pol
    items.to_parquet(MR / "evidence_judge_items.parquet", index=False)
    ok = items.judge_polarity.notna()
    d = items[ok].copy()
    mp = {"pos": "positive", "neg": "negative", "neu": "neutral"}
    d["qwen_pol"] = d.sentiment.map(mp)
    summary = {"n_items": int(len(items)), "n_judged": int(ok.sum()), "n_reviews": int(items.row_id.nunique())}

    def stats(g):
        rel_rate = float((g.judge_relevant == "yes").mean())
        agree = float((g.judge_polarity == g.qwen_pol).mean())
        nm = g[g.judge_polarity != "mixed"]
        opp = float((((g.qwen_pol == "positive") & (g.judge_polarity == "negative")) | ((g.qwen_pol == "negative") & (g.judge_polarity == "positive"))).mean())
        kap = float(cohen_kappa_score(nm.qwen_pol, nm.judge_polarity)) if nm.qwen_pol.nunique() > 1 else float("nan")
        return {"n": int(len(g)), "relevant_rate": rel_rate, "polarity_agreement": agree,
                "agreement_excl_mixed": float((nm.judge_polarity == nm.qwen_pol).mean()), "mixed_rate": float((g.judge_polarity == "mixed").mean()),
                "opposite_polarity_rate": opp, "kappa_excl_mixed": kap,
                "supported_rate": float(((g.judge_relevant == "yes") & ((g.judge_polarity == g.qwen_pol) | (g.judge_polarity == "mixed"))).mean())}
    summary["overall"] = stats(d)
    summary["by_aspect"] = {asp: stats(g) for asp, g in d.groupby("aspect")}
    summary["by_hit"] = {str(h): stats(g) for h, g in d.groupby("all_spans_hit")}
    summary["confusion"] = pd.crosstab(d.qwen_pol, d.judge_polarity).to_dict()
    (MR / "evidence_judge_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
