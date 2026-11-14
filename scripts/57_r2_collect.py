#!/usr/bin/env python3
"""Collect every number that the revised manuscript and response letter need from the second-round revision runs."""
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
MR = ROOT / "outputs" / "r2"
vals, notes = {}, {}


def boot_ci(diff, seed=42, n=10_000, chunk=250):
    rng = np.random.default_rng(seed)
    o = []
    for s in range(0, n, chunk):
        m = min(chunk, n - s)
        o.append(diff[rng.integers(0, len(diff), size=(m, len(diff)))].mean(axis=1))
    return [float(x) for x in np.quantile(np.concatenate(o), [0.025, 0.975])]


def preds(root, mid):
    # Scripts 13 and 09 write to outputs/, the r2 scripts to outputs/r2: look in both.
    folder = next((r / "predictions" / mid for r in (root, MR, OUT) if (r / "predictions" / mid).exists()), root / "predictions" / mid)
    fr = pd.concat([pd.read_parquet(p) for p in sorted(folder.glob("fold*.parquet"))])
    return fr.sort_values("row_id").reset_index(drop=True)


def paired(a, b):
    m = a[["row_id", "offering_id", "y_true", "y_pred_clipped"]].merge(b[["row_id", "y_pred_clipped"]], on="row_id", suffixes=("_a", "_b"))
    h = m.assign(ea=(m.y_pred_clipped_a - m.y_true).abs(), eb=(m.y_pred_clipped_b - m.y_true).abs()).groupby("offering_id")[["ea", "eb"]].mean()
    d = (h.ea - h.eb).to_numpy()
    return float(d.mean()), boot_ci(d), float(stats.wilcoxon(d).pvalue), int(len(d))


def agg(mid, root=MR):
    p = next((r / "metrics" / f"{mid}.json" for r in (root, OUT) if (r / "metrics" / f"{mid}.json").exists()), None)
    if p is None:
        return None
    m = json.load(open(p))["aggregate"]["clipped"]
    return m["mean"]["mae"], m["std"]["mae"], m["mean"]["rmse"], m["std"]["rmse"]


def pm(x, sd):
    return f"{x:.4f} $\\pm$ {sd:.4f}"


inter = preds(OUT, "phase4_u0_interaction_fusion")
text = preds(OUT, "phase4_u0_text_only_tfidf_ridge")

# zero-shot full corpus
zs = agg("qwen_direct_zero_n877808")
if zs:
    vals.update(ZS_MAE=f"{zs[0]:.4f}", ZS_RMSE=f"{zs[2]:.4f}", ZS_MAE_SD=pm(zs[0], zs[1]), ZS_RMSE_SD=pm(zs[2], zs[3]))
    zp = preds(MR, "qwen_direct_zero_n877808")
    err = (zp.y_pred_clipped - zp.y_true).abs()
    vals["ZS_EXACT"] = f"{100 * (err < 0.5).mean():.1f}"
    vals["ZS_MISS2"] = f"{100 * (err >= 2).mean():.1f}"
    ie = (inter.y_pred_clipped - inter.y_true).abs()
    notes["fusion_miss2_share"] = float((ie >= 1.5).mean()), float((ie >= 2).mean())
    zpath = next(r / "metrics" / "qwen_direct_zero_n877808.json" for r in (MR, OUT) if (r / "metrics" / "qwen_direct_zero_n877808.json").exists())
    m = json.load(open(zpath))
    req = m.get("request_seconds_total")
    # In our run the 100,000 rows of the earlier subsample were served from cache; adjust if you ran all rows fresh.
    n_new = 877808 - 100000
    per_review = req * 2 / n_new
    vals["ZS_PER1K"] = f"{per_review * 1000:.0f}"
    vals["ZS_GPUH"] = f"{per_review * 877808 / 3600:.1f}"
    notes["zs_request_seconds"] = req
    notes["zs_vs_inter"] = paired(zp, inter)
    notes["zs_vs_text"] = paired(zp, text)
    conf = pd.crosstab(zp.y_true.astype(int), zp.y_pred_clipped.round().astype(int))
    conf.to_csv(MR / "analysis" / "zero_shot_confusion.csv")
    notes["zs_by_true"] = zp.assign(e=err).groupby(zp.y_true.astype(int)).e.mean().round(4).to_dict()
    # subsample check: same 100k rows as the earlier report
    sub = np.load(MR / "phase1" / "direct_llm_subsample_n100000_seed42.npz")["row_id"] if (MR / "phase1" / "direct_llm_subsample_n100000_seed42.npz").exists() else np.load(OUT / "phase1" / "direct_llm_subsample_n100000_seed42.npz")["row_id"]
    s = zp[zp.row_id.isin(set(sub.tolist()))]
    notes["zs_100k_rows_mae"] = float((s.y_pred_clipped - s.y_true).abs().mean())

# text + true aspects
ta = agg("mr_u0_text_true_aspects_reference")
if ta:
    vals.update(TA_MAE=f"{ta[0]:.4f}", TA_RMSE=f"{ta[2]:.4f}", TA_MAE_SD=pm(ta[0], ta[1]), TA_RMSE_SD=pm(ta[2], ta[3]))

# no sleep quality
ns = agg("mr_u0_a9_interaction_without_sleep_quality")
if ns:
    vals.update(NS_MAE_SD=pm(ns[0], ns[1]), NS_RMSE_SD=pm(ns[2], ns[3]))
    d = ns[0] - 0.4361788349099216
    vals["NS_DELTA"] = f"{d:+.4f}"
    nsp = preds(MR, "mr_u0_a9_interaction_without_sleep_quality")
    md, ci, p, nh = paired(nsp, inter)
    vals["NS_DELTA_CI"] = f"${md:+.4f}$ ({ci[0]:.4f}, {ci[1]:.4f})"
    notes["ns_paired"] = (md, ci, p, nh)

# BERT tuned
sel = MR / "tuning" / "bert_lr" / "selection.json"
if sel.exists():
    s = json.load(open(sel))
    notes["bert_selection"] = s
    lr = s["selected_learning_rate"]
    tag = f"{lr:.0e}".replace("+", "").replace("-", "m")
    b = agg(f"bert_base_uncased_mr_lr{tag}_regression")
    notes["bert_tag"] = tag
    if b:
        vals.update(BERT_MAE=f"{b[0]:.4f}", BERT_MAE_SD=pm(b[0], b[1]), BERT_RMSE_SD=pm(b[2], b[3]), BERT_RMSE=f"{b[2]:.4f}")

# judge
jp = MR / "judge" / "evidence_judge_summary.json"
if jp.exists():
    j = json.load(open(jp))
    o = j["overall"]
    vals.update(J_N=f"{j['n_judged']:,}", J_REL=f"{100 * o['relevant_rate']:.1f}", J_AGREE=f"{100 * o['polarity_agreement']:.1f}",
                J_KAPPA=f"{o['kappa_excl_mixed']:.2f}", J_OPP=f"{100 * o['opposite_polarity_rate']:.1f}", J_MIXED=f"{100 * o['mixed_rate']:.1f}")
    vals["JUDGE_AGREE"] = vals["J_AGREE"]
    notes["judge"] = j

# portability
pp = MR / "portability" / "portability_metrics.csv"
if pp.exists():
    t = pd.read_csv(pp)
    g = lambda src, asp, col: float(t[(t.source == src) & (t.aspect == asp)][col].iloc[0])
    vals["P_CONS_OLD"] = f"{g('llama', 'service', 'consistency'):.2f}"
    vals["P_CONS_NEW"] = f"{g('llamarev', 'service', 'consistency'):.2f}"
    vals["P_ICC_OLD"] = f"{g('llama', 'service', 'icc_vs_qwen'):.2f}"
    vals["P_ICC_NEW"] = f"{g('llamarev', 'service', 'icc_vs_qwen'):.2f}"
    notes["portability"] = t.round(4).to_dict(orient="records")
    notes["portability_summary"] = json.load(open(MR / "portability" / "portability_summary.json"))

json.dump({"values": vals, "notes": notes}, open(MR / "analysis" / "collected_values.json", "w"), indent=1, default=float)
print(json.dumps({"values": vals, "notes": notes}, indent=1, default=float))
