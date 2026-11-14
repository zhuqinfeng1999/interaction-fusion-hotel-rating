#!/usr/bin/env python3
"""Second-round revision quick analyses on existing held-out predictions (no refits).

Q1 routing rule for zero-extraction reviews (Reviewer 3)
Q2 Booking.com gain by extraction coverage (Reviewer 4.2)
Q3 aspect fidelity split by mention + polarity direction vs reviewer sub-ratings (Reviewer 4.1/4.3/4.5)
Outputs: outputs/r2/analysis/*.csv|json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
AN = ROOT / "outputs" / "r2" / "analysis"
AN.mkdir(parents=True, exist_ok=True)
ASPECTS = ["service", "cleanliness", "location", "rooms", "value", "sleep_quality"]
SC = [f"aspect_{a}_score" for a in ASPECTS]
MC = [f"aspect_{a}_mentioned" for a in ASPECTS]
RC = [f"rating_{a}" for a in ASPECTS]


def boot_ci(diff, seed=42, n=10_000, chunk=250):
    rng = np.random.default_rng(seed)
    out = []
    for s in range(0, n, chunk):
        m = min(chunk, n - s)
        idx = rng.integers(0, len(diff), size=(m, len(diff)))
        out.append(diff[idx].mean(axis=1))
    out = np.concatenate(out)
    return [float(x) for x in np.quantile(out, [0.025, 0.975])]


def paired(frame, a, b):
    """Hotel-weighted paired MAE difference a-b on clipped predictions."""
    h = frame.assign(ea=(frame[a] - frame.y).abs(), eb=(frame[b] - frame.y).abs()).groupby("offering_id")[["ea", "eb"]].mean()
    d = (h.ea - h.eb).to_numpy()
    w = stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided", method="auto") if not np.allclose(d, 0) else None
    return {"n_hotels": int(len(d)), "mean_diff": float(d.mean()), "ci95": boot_ci(d), "wilcoxon_p": float(w.pvalue) if w else 1.0}


def fold_metrics(frame, col):
    rows = []
    for f, g in frame.groupby("fold"):
        e = g[col] - g.y
        rows.append((float(e.abs().mean()), float(np.sqrt((e ** 2).mean()))))
    a = np.array(rows)
    return {"mae_mean": a[:, 0].mean(), "mae_sd": a[:, 0].std(ddof=1), "rmse_mean": a[:, 1].mean(), "rmse_sd": a[:, 1].std(ddof=1)}


def load_pred(mid):
    fr = pd.concat([pd.read_parquet(p) for p in sorted((OUT / "predictions" / mid).glob("fold*.parquet"))])
    return fr.sort_values("row_id").reset_index(drop=True)


# ---------------- extraction (structured columns only; failed rows -> zeros, as in 41)
ax = pd.read_parquet(OUT / "aspects_llm_qwen25_7b.parquet", columns=["row_id", "ok", "traveler_segment", *SC, *MC]).sort_values("row_id").reset_index(drop=True)
failed = ~ax["ok"].eq(1)
ax.loc[failed, SC] = 0
ax.loc[failed, MC] = False
S = ax[SC].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(float)
M = ax[MC].fillna(False).to_numpy(bool)
n_ment = M.sum(1)
ds = pd.read_parquet(OUT / "dataset_clean.parquet", columns=["offering_id", "rating_overall", *RC]).reset_index(drop=True)
res = {}

# ---------------- Q1 routing
t = load_pred("phase4_u0_text_only_tfidf_ridge")
s = load_pred("phase4_u0_text_llm_simple_fusion")
i = load_pred("phase4_u0_interaction_fusion")
assert (t.row_id.values == i.row_id.values).all() and (s.row_id.values == i.row_id.values).all()
fr = pd.DataFrame({"row_id": t.row_id, "offering_id": t.offering_id, "fold": t.fold, "y": t.y_true,
                   "text": t.y_pred_clipped, "simple": s.y_pred_clipped, "inter": i.y_pred_clipped})
fr["nm"] = n_ment[fr.row_id.values]
zero = fr.nm.eq(0)
fr["routed_inter"] = np.where(zero, fr.text, fr.inter)
fr["routed_simple"] = np.where(zero, fr.text, fr.simple)
q1 = {"n_zero": int(zero.sum()), "share_zero": float(zero.mean()), "n_failed_extraction": int(failed.sum()),
      "failed_within_zero": int((failed.values[fr.row_id.values] & zero.values).sum())}
for c in ["text", "simple", "inter", "routed_inter", "routed_simple"]:
    q1[c] = fold_metrics(fr, c)
    q1[c + "_zero_stratum_mae"] = float((fr.loc[zero, c] - fr.loc[zero, "y"]).abs().mean())
q1["paired_routed_inter_minus_inter"] = paired(fr, "routed_inter", "inter")
q1["paired_routed_inter_minus_text"] = paired(fr, "routed_inter", "text")
# per-stratum by number of mentioned aspects (for reference)
strata = []
for k in range(7):
    g = fr[fr.nm == k]
    strata.append({"n_mentioned": k, "n": int(len(g)), "share": float(len(g) / len(fr)),
                   "mae_text": float((g.text - g.y).abs().mean()), "mae_inter": float((g.inter - g.y).abs().mean()),
                   "mae_simple": float((g.simple - g.y).abs().mean())})
q1["strata"] = strata
res["Q1_routing"] = q1
pd.DataFrame(strata).to_csv(AN / "q1_tripadvisor_strata.csv", index=False)
print(json.dumps(q1, indent=1, default=float))

# ---------------- Q3 fidelity by mention and polarity direction
rows = []
for k, a in enumerate(ASPECTS):
    r = ds[RC[k]].to_numpy(float)
    present = ~np.isnan(r) & ax["ok"].eq(1).to_numpy()
    mapped = np.clip(3 + S[:, k], 1, 5)
    for label, mask in [("all", present), ("mentioned", present & M[:, k]), ("not_mentioned", present & ~M[:, k])]:
        x, yv = mapped[mask], r[mask]
        rec = {"aspect": a, "subset": label, "n": int(mask.sum()), "mae": float(np.abs(x - yv).mean()),
               "rmse": float(np.sqrt(((x - yv) ** 2).mean())),
               "pearson": float(stats.pearsonr(x, yv)[0]) if x.std() > 0 else float("nan"),
               "spearman": float(stats.spearmanr(x, yv)[0]) if x.std() > 0 else float("nan"),
               "mean_subrating": float(yv.mean())}
        if label == "mentioned":
            pol = np.sign(S[mask, k]).astype(int)              # -1 neg, 0 neutral, +1 pos
            ref = np.where(yv <= 2, -1, np.where(yv >= 4, 1, 0))  # reviewer's own sub-rating
            rec["direction_agreement"] = float((pol == ref).mean())
            rec["kappa_direction"] = float(cohen_kappa_score(pol, ref))
            nz = pol != 0
            rec["share_nonneutral"] = float(nz.mean())
            rec["sign_agreement_nonneutral"] = float((pol[nz] == ref[nz]).mean())
            rec["p_high_given_pos"] = float((yv[pol > 0] >= 4).mean())
            rec["p_low_given_neg"] = float((yv[pol < 0] <= 2).mean())
            rec["opposite_sign_rate"] = float(((pol == 1) & (ref == -1) | (pol == -1) & (ref == 1)).mean())
        rows.append(rec)
q3 = pd.DataFrame(rows)
q3.to_csv(AN / "q3_aspect_fidelity_by_mention.csv", index=False)
print(q3.to_string())
res["Q3_mention_rates"] = {a: float(M[:, k].mean()) for k, a in enumerate(ASPECTS)}

# ---------------- Q2 Booking by coverage
bp = pd.read_parquet(OUT / "phase4" / "booking_u0_predictions.parquet")
print(bp.head(3).T)
print(bp.model.unique())
bx = pd.read_parquet(OUT / "phase3" / "aspects_guided_booking_qwen_guided_n200035.parquet")
print([c for c in bx.columns][:40])
res_path = AN / "quick_analyses.json"
res_path.write_text(json.dumps(res, indent=1, default=float))
print("saved", res_path)
