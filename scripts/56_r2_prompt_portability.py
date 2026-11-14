#!/usr/bin/env python3
"""Second-round revision (Reviewer 3, prompt portability): does stating the polarity
convention in the prompt remove Llama's score inversion?

Re-extracts 10,000 rows of the 200,041-row extractor-comparison sample with
Llama-3.1-8B-Instruct under the identical guided JSON schema, system prompt,
decoding settings and review truncation as the Phase 3 comparison. The only
change is one added sentence that defines the score convention.
"""
import argparse
import asyncio
import importlib.util
import json
import sys
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
MR = ROOT / "outputs" / "r2" / "portability"
MR.mkdir(parents=True, exist_ok=True)
spec = importlib.util.spec_from_file_location("g18", ROOT / "scripts" / "18_phase2_schema_enforcement.py")
G = importlib.util.module_from_spec(spec)
sys.modules["g18"] = G
spec.loader.exec_module(G)
ASPECTS = list(G.ASPECTS)
ADDED = ("The score is the polarity of the opinion: -2 very negative, -1 negative, 0 neutral, "
         "1 positive, 2 very positive. Its sign must agree with the sentiment label.\n")
first, rest = G.USER_PROMPT.split("\n", 1)
USER_REVISED = first + "\n" + ADDED + rest
EXPECTED = {"neg": -1, "neu": 0, "pos": 1}


def icc21(x, y):
    m = np.column_stack([x, y]).astype(float)
    n, k = m.shape
    g = m.mean()
    ssr = k * ((m.mean(1) - g) ** 2).sum()
    ssc = n * ((m.mean(0) - g) ** 2).sum()
    sse = ((m - g) ** 2).sum() - ssr - ssc
    msr, msc, mse = ssr / (n - 1), ssc / (k - 1), sse / ((n - 1) * (k - 1))
    return float((msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n))


def expand(js):
    d = json.loads(js)
    a = d["aspects"]
    rec = {"segment": d.get("traveler_segment")}
    for asp in ASPECTS:
        rec[f"{asp}_m"] = bool(a[asp]["mentioned"])
        rec[f"{asp}_s"] = int(a[asp]["score"])
        rec[f"{asp}_sent"] = a[asp]["sentiment"]
    return rec


async def run(rows, texts, endpoint, model, concurrency, max_chars):
    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(base_url=endpoint, headers={"Authorization": "Bearer local-llm"}, timeout=180) as client:
        async def one(rid):
            review = G.clean_review_text(texts[rid], max_chars)
            body = {"model": model, "temperature": 0.0, "max_tokens": 900, "seed": 42,
                    "messages": [{"role": "system", "content": G.SYSTEM_PROMPT},
                                 {"role": "user", "content": USER_REVISED.format(review_text=review)}],
                    "response_format": {"type": "json_schema", "json_schema": {
                        "name": "hotel_aspects_v1", "description": "Six-aspect hotel review extraction.",
                        "schema": G.JSON_SCHEMA, "strict": True}}}
            for attempt in range(3):
                try:
                    async with sem:
                        r = await client.post("/chat/completions", json=body)
                    r.raise_for_status()
                    content = r.json()["choices"][0]["message"]["content"]
                    norm, err = G.validate_guided_payload(json.loads(content))
                    if norm is not None:
                        return rid, json.dumps(norm)
                except Exception:
                    await asyncio.sleep(1 + attempt)
            return rid, None
        out = await asyncio.gather(*[one(int(r)) for r in rows])
    return dict(out)


def metrics(frame, src, ref="qwen"):
    rows = []
    for asp in ASPECTS:
        s, sent, m = frame[f"{src}_{asp}_s"], frame[f"{src}_{asp}_sent"], frame[f"{src}_{asp}_m"]
        cons = float((np.sign(s) == sent.map(EXPECTED)).mean())
        rec = {"source": src, "aspect": asp, "consistency": cons, "mention_rate": float(m.mean())}
        if src != ref:
            rs, rm = frame[f"{ref}_{asp}_s"], frame[f"{ref}_{asp}_m"]
            rec["icc_vs_qwen"] = icc21(s, rs)
            rec["pearson_vs_qwen"] = float(np.corrcoef(s, rs)[0, 1])
            rec["mention_kappa_vs_qwen"] = float(cohen_kappa_score(m, rm))
            both = m & rm
            rec["sentiment_agree_both_mentioned"] = float((sent[both] == frame.loc[both, f"{ref}_{asp}_sent"]).mean())
        rows.append(rec)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--endpoint", default="http://127.0.0.1:8010/v1")
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--n", type=int, default=10000)
    p.add_argument("--concurrency", type=int, default=64)
    p.add_argument("--max-chars", type=int, default=4000)
    a = p.parse_args()
    cols = ["row_id", "ok", "response_json"]
    ll = pd.read_parquet(OUT / "phase3" / "aspects_guided_llama31_8b_l1_n200041.parquet", columns=cols)
    qw = pd.read_parquet(OUT / "phase3" / "aspects_guided_qwen25_7b_l1_n200041.parquet", columns=cols)
    both = ll[ll.ok.eq(1)].merge(qw[qw.ok.eq(1)], on="row_id", suffixes=("_llama", "_qwen"))
    rng = np.random.default_rng(42)
    sample = np.sort(rng.choice(both.row_id.to_numpy(), size=a.n, replace=False))
    texts = pd.read_parquet(OUT / "dataset_clean.parquet", columns=["review_text"])["review_text"].fillna("").astype(str).to_numpy()
    cache = MR / "llama_revised_prompt.parquet"
    if cache.exists():
        got = pd.read_parquet(cache)
    else:
        res = asyncio.run(run(sample, texts, a.endpoint, a.model, a.concurrency, a.max_chars))
        got = pd.DataFrame({"row_id": list(res.keys()), "response_json": list(res.values())})
        got.to_parquet(cache, index=False)
    valid = got.response_json.notna()
    print("revised valid rate", float(valid.mean()), flush=True)
    sub = both[both.row_id.isin(got.loc[valid, "row_id"])].merge(got[valid].rename(columns={"response_json": "response_json_rev"}), on="row_id")
    frame = pd.DataFrame({"row_id": sub.row_id})
    for src, col in [("qwen", "response_json_qwen"), ("llama", "response_json_llama"), ("llamarev", "response_json_rev")]:
        ex = pd.DataFrame([expand(j) for j in sub[col]])
        ex.columns = [f"{src}_{c}" for c in ex.columns]
        frame = pd.concat([frame, ex.reset_index(drop=True)], axis=1)
    res = pd.DataFrame(metrics(frame, "qwen") + metrics(frame, "llama") + metrics(frame, "llamarev"))
    res.to_csv(MR / "portability_metrics.csv", index=False)
    seg = {"llama_vs_qwen": float(cohen_kappa_score(frame.llama_segment, frame.qwen_segment)),
           "llamarev_vs_qwen": float(cohen_kappa_score(frame.llamarev_segment, frame.qwen_segment))}
    summary = {"n_sample": int(a.n), "n_valid_revised": int(valid.sum()), "n_compared": int(len(frame)), "segment_kappa": seg,
               "added_sentence": ADDED.strip()}
    (MR / "portability_summary.json").write_text(json.dumps(summary, indent=1))
    print(res.round(4).to_string())
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
