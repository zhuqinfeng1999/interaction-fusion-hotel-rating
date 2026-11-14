#!/usr/bin/env python3
"""Build S5_File.xlsx (additional analyses of the minor revision) from the saved result files."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MR = ROOT / "outputs" / "r2"
AN = MR / "analysis"
out = MR / "S5_File.xlsx"
sheets = {}
readme = []


def add(name, df, desc):
    sheets[name] = df
    readme.append({"sheet": name, "contents": desc})


add("fidelity_by_mention", pd.read_csv(AN / "q3b_aspect_fidelity_table3_population.csv"),
    "Extracted polarity clip(3+s,1,5) against platform sub-ratings, on all rated reviews and split by whether the extractor marked the aspect as mentioned. Sign agreement groups sub-ratings 1-2 / 3 / 4-5.")
j = json.load(open(MR / "judge" / "evidence_judge_summary.json"))
rows = [{"group": "overall", **j["overall"]}] + [{"group": f"aspect={k}", **v} for k, v in j["by_aspect"].items()] + \
       [{"group": f"all_spans_literal_hit={k}", **v} for k, v in j["by_hit"].items()]
add("judge_summary", pd.DataFrame(rows),
    "Llama-3.1-8B-Instruct judge on the evidence quoted by the Qwen extractor (2,000 random reviews; unit = review x aspect). relevant_rate: judge says the evidence concerns the aspect. polarity_agreement: judge polarity equals Qwen sentiment label. kappa excludes records rated mixed.")
add("judge_confusion", pd.DataFrame(j["confusion"]).rename_axis("qwen_label").reset_index(),
    "Rows: Qwen sentiment label; columns: judge polarity.")
add("prompt_portability", pd.read_csv(MR / "portability" / "portability_metrics.csv"),
    "10,000 reviews of the extractor-comparison sample. qwen = Qwen guided; llama = Llama guided with the original prompt; llamarev = Llama guided with the added convention sentence (S4 File). consistency = share of rows where sign(score) matches the sentiment label.")
add("strata_tripadvisor", pd.read_csv(AN / "q1_tripadvisor_strata.csv"),
    "TripAdvisor held-out MAE by number of mentioned aspects (common tuning protocol).")
add("strata_booking", pd.read_csv(AN / "q2_booking_strata.csv"),
    "Booking.com held-out MAE (1-5 scale) by number of mentioned aspects.")
q1 = json.load(open(AN / "quick_analyses.json"))["Q1_routing"]
r = []
for k in ["text", "simple", "inter", "routed_inter", "routed_simple"]:
    r.append({"model": k, **q1[k]})
add("routing_rule", pd.DataFrame(r),
    "Routing rule: reviews with no mentioned aspect (7,598) scored by text-only, all others by the fusion model. Fold mean and SD over the five outer folds.")
add("segment_phrase_rates", pd.read_csv(AN / "q4_segment_cues.csv"),
    "Share of reviews containing a travel-party phrase, by extracted segment. Phrase lists are in the sheet segment_phrase_lists.")
cues = {
    "business": "business trip | on business | for business | conference | work trip | for work | colleague(s) | client(s) | meeting(s) | business traveler/traveller",
    "couple": "my wife | my husband | my boyfriend | my girlfriend | my partner | my fiance(e) | anniversary | honeymoon | romantic | my spouse",
    "family": "kids | children | child | my son | my daughter | our son | our daughter | family | toddler | baby | teenager(s)",
    "friends": "friends | girlfriends trip | girls trip | girls weekend | buddies | mates | group of friends",
    "solo": "traveling alone | travelling alone | by myself | on my own | solo | alone",
}
add("segment_phrase_lists", pd.DataFrame([{"segment": k, "phrases (whole words, case-insensitive)": v} for k, v in cues.items()]), "Phrase lists used for the segment check.")
add("segment_weights_by_fold", pd.read_csv(AN / "q5_effective_slopes_by_fold.csv"),
    "Interaction-fusion model: main-effect and segment-interaction coefficients (standardized) and the summed weight on the raw polarity scale, per fold.")
add("segment_weights_summary", pd.read_csv(AN / "q5_effective_slopes_summary.csv"),
    "Mean and SD across folds of the raw-scale weights (Table in the manuscript). ex = the interaction's bootstrap CI excludes zero in all five folds.")
cz = AN / "zero_shot_confusion.csv"
if cz.exists():
    add("zero_shot_confusion", pd.read_csv(cz), "Zero-shot Qwen2.5-7B-Instruct on all 877,808 reviews: rows true rating, columns predicted rating.")
models = []
for mid, label in [("qwen_direct_zero_n877808", "Qwen2.5-7B-Instruct zero-shot (all reviews)"),
                   ("mr_u0_text_true_aspects_reference", "Text + true aspects (not deployable)"),
                   ("mr_u0_a9_interaction_without_sleep_quality", "A9: interaction fusion without sleep-quality columns")]:
    p = next((r / "metrics" / f"{mid}.json" for r in (MR, ROOT / "outputs") if (r / "metrics" / f"{mid}.json").exists()), MR / "missing")
    if p.exists():
        m = json.load(open(p))
        for f in m["folds"]:
            models.append({"model": label, "model_id": mid, "fold": f["fold"], "mae": f["clipped"]["mae"], "rmse": f["clipped"]["rmse"],
                           "selected": json.dumps(f.get("selected_hyperparameters", {}).get("alpha", f.get("selected_hyperparameters", {}).get("learning_rate")))})
for p in sorted(list((MR / "metrics").glob("bert_base_uncased_mr_lr*_regression.json")) + list((ROOT / "outputs" / "metrics").glob("bert_base_uncased_mr_lr*_regression.json"))):
    m = json.load(open(p))
    for f in m["folds"]:
        models.append({"model": "BERT-base fine-tuned (selected learning rate)", "model_id": p.stem, "fold": f["fold"],
                       "mae": f["clipped"]["mae"], "rmse": f["clipped"]["rmse"], "selected": f["selected_hyperparameters"]["learning_rate"]})
add("new_models_per_fold", pd.DataFrame(models), "Per-fold held-out metrics (predictions clipped to [1,5]) of the models added in this revision.")
sel = MR / "tuning" / "bert_lr" / "selection.json"
if sel.exists():
    add("bert_lr_sweep", pd.DataFrame(json.load(open(sel))["rows"]), "BERT-base learning-rate sweep on the grouped validation split of outer fold 1 (same split as RoBERTa).")

with pd.ExcelWriter(out) as w:
    pd.DataFrame(readme).to_excel(w, sheet_name="README", index=False)
    for k, v in sheets.items():
        v.to_excel(w, sheet_name=k[:31], index=False)
print("wrote", out, list(sheets))
