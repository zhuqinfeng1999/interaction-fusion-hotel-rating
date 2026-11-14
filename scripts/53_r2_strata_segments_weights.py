#!/usr/bin/env python3
"""Second-round revision analyses, part 2 (no refits).

Q3b fidelity table on the exact Table 3 population (all rows with the sub-rating present)
Q2  Booking.com gain by number of mentioned aspects + TripAdvisor gain reweighted to Booking coverage
Q4  traveler-segment lexical cue check (Reviewer 4.4)
Q5  segment-specific effective score slopes in the U0 interaction model (Reviewer 6)
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
AN = ROOT / "outputs" / "r2" / "analysis"
ASPECTS = ["service", "cleanliness", "location", "rooms", "value", "sleep_quality"]
SEGMENTS = ["business", "couple", "family", "solo", "friends", "unknown"]
SC = [f"aspect_{a}_score" for a in ASPECTS]
MC = [f"aspect_{a}_mentioned" for a in ASPECTS]
RC = [f"rating_{a}" for a in ASPECTS]
res = {}


def boot_ci(diff, seed=42, n=10_000, chunk=250):
    rng = np.random.default_rng(seed)
    out = []
    for s in range(0, n, chunk):
        m = min(chunk, n - s)
        out.append(diff[rng.integers(0, len(diff), size=(m, len(diff)))].mean(axis=1))
    return [float(x) for x in np.quantile(np.concatenate(out), [0.025, 0.975])]


def load_extraction(path, ok_filter=True):
    ax = pd.read_parquet(path, columns=["row_id", "ok", "traveler_segment", *SC, *MC]).sort_values("row_id").reset_index(drop=True)
    failed = ~ax["ok"].eq(1)
    ax.loc[failed, SC] = 0
    ax.loc[failed, MC] = False
    ax.loc[failed, "traveler_segment"] = "unknown"
    return ax, failed.to_numpy()


ax, failed = load_extraction(OUT / "aspects_llm_qwen25_7b.parquet")
S = ax[SC].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(float)
M = ax[MC].fillna(False).to_numpy(bool)
seg = ax["traveler_segment"].fillna("unknown").astype(str).str.lower()
seg = seg.where(seg.isin(SEGMENTS), "unknown").to_numpy()
ds = pd.read_parquet(OUT / "dataset_clean.parquet", columns=["offering_id", "rating_overall", "review_text", *RC]).reset_index(drop=True)

# ---------------- Q3b: Table 3 population
rows = []
for k, a in enumerate(ASPECTS):
    r = ds[RC[k]].to_numpy(float)
    present = ~np.isnan(r)
    mapped = np.clip(3 + S[:, k], 1, 5)
    for label, mask in [("all", present), ("mentioned", present & M[:, k]), ("not_mentioned", present & ~M[:, k])]:
        x, yv = mapped[mask], r[mask]
        rec = {"aspect": a, "subset": label, "n": int(mask.sum()), "share_of_present": float(mask.sum() / present.sum()),
               "mae": float(np.abs(x - yv).mean()), "rmse": float(np.sqrt(((x - yv) ** 2).mean())),
               "pearson": float(stats.pearsonr(x, yv)[0]) if x.std() > 0 else float("nan"),
               "spearman": float(stats.spearmanr(x, yv)[0]) if x.std() > 0 else float("nan")}
        if label == "mentioned":
            pol = np.sign(S[mask, k]).astype(int)
            ref = np.where(yv <= 2, -1, np.where(yv >= 4, 1, 0))
            rec["direction_agreement"] = float((pol == ref).mean())
            rec["kappa_direction"] = float(cohen_kappa_score(pol, ref))
            rec["opposite_sign_rate"] = float((((pol == 1) & (ref == -1)) | ((pol == -1) & (ref == 1))).mean())
        rows.append(rec)
q3 = pd.DataFrame(rows)
# pooled over aspects (mentioned only)
pol_all, ref_all, x_all, y_all = [], [], [], []
for k in range(6):
    r = ds[RC[k]].to_numpy(float)
    mask = ~np.isnan(r) & M[:, k]
    pol_all.append(np.sign(S[mask, k]).astype(int)); yv = r[mask]
    ref_all.append(np.where(yv <= 2, -1, np.where(yv >= 4, 1, 0)))
    x_all.append(np.clip(3 + S[mask, k], 1, 5)); y_all.append(yv)
pol_all, ref_all, x_all, y_all = map(np.concatenate, (pol_all, ref_all, x_all, y_all))
res["Q3_pooled_mentioned"] = {"n": int(len(pol_all)), "direction_agreement": float((pol_all == ref_all).mean()),
                              "kappa": float(cohen_kappa_score(pol_all, ref_all)),
                              "opposite_sign_rate": float((((pol_all == 1) & (ref_all == -1)) | ((pol_all == -1) & (ref_all == 1))).mean()),
                              "mae": float(np.abs(x_all - y_all).mean()), "pearson": float(stats.pearsonr(x_all, y_all)[0])}
q3.to_csv(AN / "q3b_aspect_fidelity_table3_population.csv", index=False)
print(q3.round(4).to_string())
print(res["Q3_pooled_mentioned"])

# ---------------- Q2 Booking by coverage
bp = pd.read_parquet(OUT / "phase4" / "booking_u0_predictions.parquet")
wide = bp.pivot_table(index=["row_id", "offering_id", "fold", "y_true"], columns="model", values="prediction").reset_index()
bx, bfailed = load_extraction(OUT / "phase3" / "aspects_guided_booking_qwen_guided_n200035.parquet")
bx = bx.set_index("row_id")
wide["nm"] = bx.loc[wide.row_id, MC].fillna(False).to_numpy(bool).sum(1)
assert len(wide) == 200035, len(wide)
ta = pd.read_csv(AN / "q1_tripadvisor_strata.csv")
brow = []
for k in range(7):
    g = wide[wide.nm == k]
    if len(g) == 0:
        continue
    e_t = (g.text_only - g.y_true).abs().mean(); e_i = (g.interaction_fusion - g.y_true).abs().mean(); e_s = (g.simple_fusion - g.y_true).abs().mean()
    brow.append({"n_mentioned": k, "n": int(len(g)), "share": float(len(g) / len(wide)), "mae_text": e_t, "mae_inter": e_i, "mae_simple": e_s,
                 "rel_gain_inter": float((e_t - e_i) / e_t)})
bstr = pd.DataFrame(brow)
bstr.to_csv(AN / "q2_booking_strata.csv", index=False)
print(bstr.round(4).to_string())
ta["rel_gain_inter"] = (ta.mae_text - ta.mae_inter) / ta.mae_text
print(ta.round(4).to_string())
# TripAdvisor gain reweighted to Booking's coverage distribution (stratum weights = Booking shares)
w = bstr.set_index("n_mentioned")["share"].reindex(range(7)).fillna(0)
tt = ta.set_index("n_mentioned")
mae_t_rw = float((w * tt.mae_text).sum()); mae_i_rw = float((w * tt.mae_inter).sum())
res["Q2"] = {"booking_mean_mentioned": float(wide.nm.mean()), "tripadvisor_mean_mentioned": float((tt.n * tt.index).sum() / tt.n.sum()),
             "booking_share_zero": float((wide.nm == 0).mean()), "booking_share_ge3": float((wide.nm >= 3).mean()),
             "tripadvisor_share_ge3": float(tt.loc[3:, "share"].sum()),
             "tripadvisor_reweighted_to_booking": {"mae_text": mae_t_rw, "mae_inter": mae_i_rw, "rel_gain": (mae_t_rw - mae_i_rw) / mae_t_rw},
             "booking_overall_rel_gain": float(((wide.text_only - wide.y_true).abs().mean() - (wide.interaction_fusion - wide.y_true).abs().mean()) / (wide.text_only - wide.y_true).abs().mean()),
             "booking_rel_gain_ge3": float(((wide[wide.nm >= 3].text_only - wide[wide.nm >= 3].y_true).abs().mean() - (wide[wide.nm >= 3].interaction_fusion - wide[wide.nm >= 3].y_true).abs().mean()) / (wide[wide.nm >= 3].text_only - wide[wide.nm >= 3].y_true).abs().mean()) if (wide.nm >= 3).any() else None}
print(json.dumps(res["Q2"], indent=1))

# ---------------- Q4 segment lexical cues
cues = {
    "business": r"\b(business trip|on business|for business|conference|work trip|for work|colleague|colleagues|client|clients|meeting|meetings|business traveler|business traveller)\b",
    "couple": r"\b(my wife|my husband|my boyfriend|my girlfriend|my partner|my fianc[eé]e?|anniversary|honeymoon|romantic|my spouse)\b",
    "family": r"\b(kids|children|child|my son|my daughter|our son|our daughter|family|toddler|baby|teenagers?)\b",
    "friends": r"\b(friends|girlfriends trip|girls trip|girls weekend|buddies|mates|group of friends)\b",
    "solo": r"\b(traveling alone|travelling alone|by myself|on my own|solo|alone)\b",
}
text = ds["review_text"].fillna("").str.lower()
cue_hits = {s: text.str.contains(p, regex=True).to_numpy() for s, p in cues.items()}
tab = []
for s_lab in SEGMENTS:
    mask = seg == s_lab
    rec = {"segment": s_lab, "n": int(mask.sum())}
    for c in cues:
        rec[f"cue_{c}"] = float(cue_hits[c][mask].mean())
    rec["any_cue"] = float(np.any(np.vstack([cue_hits[c][mask] for c in cues]), axis=0).mean())
    tab.append(rec)
q4 = pd.DataFrame(tab)
q4.to_csv(AN / "q4_segment_cues.csv", index=False)
print(q4.round(3).to_string())

# ---------------- Q5 effective score slopes, U0 interaction model
coef = pd.read_csv(OUT / "phase5" / "S1_coefficient_tables.csv")
inter = pd.read_csv(OUT / "phase5" / "S1_segment_aspect_interactions.csv")
ci = coef[coef.model == "phase4_u0_interaction_fusion"]
print(ci.block.unique(), ci.feature.unique()[:30])
z = np.hstack([S, M.astype(float)])
U = pd.get_dummies(pd.Series(seg)).reindex(columns=SEGMENTS, fill_value=False).to_numpy(float)
slopes = []
for fold in range(1, 6):
    tr = np.load(OUT / f"splits_fold{fold}.npz")["train_idx"]
    sd_s = S[tr].std(axis=0)
    for k, a in enumerate(ASPECTS):
        main = ci[(ci.fold == fold) & (ci.feature == f"score_{a}")]
        if len(main) != 1:
            raise SystemExit(f"main effect not found for {a}: {ci[(ci.fold == fold)].feature.unique()}")
        beta = float(main.estimate.iloc[0])
        for j, s_lab in enumerate(SEGMENTS):
            p = S[tr, k] * U[tr, j]
            sd_p = p.std()
            g = inter[(inter.fold == fold) & (inter.segment == s_lab) & (inter.aspect == a) & (inter.interaction_type == "score")]
            gam = float(g.estimate.iloc[0])
            slopes.append({"fold": fold, "aspect": a, "segment": s_lab, "beta_main_std": beta, "gamma_std": gam,
                           "gamma_ci_low": float(g.bootstrap_ci_low.iloc[0]), "gamma_ci_high": float(g.bootstrap_ci_high.iloc[0]),
                           "slope_raw": beta / sd_s[k] + gam / sd_p})
sl = pd.DataFrame(slopes)
sl.to_csv(AN / "q5_effective_slopes_by_fold.csv", index=False)
summ = sl.groupby(["aspect", "segment"]).agg(slope_mean=("slope_raw", "mean"), slope_sd=("slope_raw", "std"),
                                             gamma_mean=("gamma_std", "mean"),
                                             gamma_ci_excl0_all_folds=("gamma_ci_low", lambda v: None)).reset_index()
excl = sl.assign(ex=lambda d: (d.gamma_ci_low > 0) | (d.gamma_ci_high < 0)).groupby(["aspect", "segment"]).ex.all().reset_index()
summ = summ.drop(columns="gamma_ci_excl0_all_folds").merge(excl, on=["aspect", "segment"])
summ.to_csv(AN / "q5_effective_slopes_summary.csv", index=False)
print(summ.pivot(index="aspect", columns="segment", values="slope_mean").reindex(ASPECTS)[SEGMENTS].round(4).to_string())
print(summ.pivot(index="aspect", columns="segment", values="slope_sd").reindex(ASPECTS)[SEGMENTS].round(4).to_string())
print(summ.pivot(index="aspect", columns="segment", values="ex").reindex(ASPECTS)[SEGMENTS].to_string())
(AN / "more_analyses.json").write_text(json.dumps(res, indent=1, default=float))
