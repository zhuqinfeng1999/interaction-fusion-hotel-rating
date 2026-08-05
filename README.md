<div align="center">

# Interaction Fusion of Text and LLM-Extracted Aspects

### Hotel Rating Prediction at Scale — A Verifiable Study on TripAdvisor

[![License: MIT](https://img.shields.io/badge/Code-MIT-blue.svg)](LICENSE)
[![Data: CC BY 4.0](https://img.shields.io/badge/Data-CC%20BY%204.0-lightgrey.svg)](LICENSE-DATA)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-3776AB.svg)](https://www.python.org/)
<!-- Add the OSF DOI badge here after the deposit has been created and a DOI assigned. -->

**Xue Zhu**<sup>1</sup> · **Qinfeng Zhu**<sup>2</sup>

<sup>1</sup> School of Culture and Tourism, Wuxi University of Technology · <sup>2</sup> Department of Computer Science, University of Liverpool

</div>

---

## Overview

This repository contains the complete implementation and release-ready artefacts for a study of
whether structured aspect signals extracted offline by a large language model improve hotel
rating prediction, and in what form they should be supplied to the predictor.

The study is built around three commitments:

- **Leakage resistance.** All evaluation uses 5-fold `GroupKFold` grouped by hotel
  identifier, so no property appears in both training and test data. The exact split is
  released and hash-verified.
- **Auditability.** Every extraction is schema-validated with logged failures, evidence
  spans are checked against the source text, and the extraction step is validated against
  grammar-constrained decoding and against a second open-weight model.
- **Fair comparison.** Every model in the primary analysis is tuned by one identical
  procedure. The originally submitted analysis, which did not do this, is retained
  alongside the corrected one rather than replaced.

## Quick start

The commands below assume Ubuntu, no Docker, and that the two upstream corpora have been
obtained from their distributors. They do not download or redistribute review text.

```bash
# 1. Create the reference environment
conda env create -f environment.yml
conda activate hotel-fusion

# 2. Put reviews.csv and offerings.csv in data_ori/, then preprocess
python scripts/00_sanity_check.py
python scripts/01_prepare_dataset.py

# 3. Reconstruct and verify the canonical leakage-resistant folds
python splits/expand_splits.py --verify
python splits/expand_splits.py --output-dir outputs

# 4. Reproduce the non-LLM baselines
python scripts/03_run_baselines.py
```

Start the two local vLLM servers as described in [Local LLM inference](#local-llm-inference),
then run a small uncached pilot before committing to the full extraction:

```bash
ENDPOINTS="http://localhost:8000/v1,http://localhost:8001/v1"

python scripts/06_extract_aspects_llm.py --profile qwen25_7b \
  --pilot 500 --limit 500 --concurrency 16 --resume --endpoints "$ENDPOINTS"

python scripts/06_extract_aspects_llm.py --profile qwen25_7b \
  --pilot 2000 --concurrency 16 --resume --endpoints "$ENDPOINTS"

python scripts/06_extract_aspects_llm.py --profile qwen25_7b --validate-only
python scripts/07_run_llm_aspects_analysis.py --profile qwen25_7b \
  --with-text --max_features 200000
```

For the manuscript's unified protocol, continue with scripts 41–47 using the exact
commands in [`docs/REPRODUCE.md`](docs/REPRODUCE.md). Validate the final release with
`splits/expand_splits.py --verify`, the extraction sanity report, and the SHA-256 manifests.

---

## Key results

Held-out performance on 877,808 TripAdvisor reviews of 3,945 properties, 5-fold
`GroupKFold` by property, predictions clipped to [1, 5], mean ± SD across folds.

| Method | MAE | RMSE |
| :--- | :---: | :---: |
| **Panel A — originally submitted protocol** | | |
| Text-only (TF-IDF + Ridge) | 0.4644 ± 0.0059 | 0.6099 ± 0.0074 |
| LLM aspects only | 0.5127 ± 0.0071 | 0.6862 ± 0.0083 |
| Simple fusion (Text + LLM) | 0.4676 ± 0.0054 | 0.6210 ± 0.0064 |
| Residual-on-LLM | 0.4488 ± 0.0058 | 0.5970 ± 0.0067 |
| Upper bound (true aspects)† | 0.4322 ± 0.0072 | 0.6354 ± 0.0134 |
| **Interaction Fusion** | **0.4385 ± 0.0061** | **0.5797 ± 0.0069** |
| **Panel B — unified tuning protocol** | | |
| Text-only (TF-IDF + Ridge) | 0.4625 ± 0.0065 | 0.6052 ± 0.0080 |
| Simple fusion (Text + LLM) | 0.4364 ± 0.0061 | 0.5771 ± 0.0069 |
| Residual-on-LLM | 0.4423 ± 0.0062 | 0.5885 ± 0.0070 |
| **Interaction Fusion** | **0.4362 ± 0.0058** | **0.5767 ± 0.0063** |
| **Panel C — neural text baselines** | | |
| RoBERTa-base, fine-tuned | 0.3685 ± 0.0081 | 0.5365 ± 0.0099 |
| BERT-base (CLS) + interaction block | 0.4119 ± 0.0065 | 0.5466 ± 0.0082 |

<sup>† Not deployable at inference time; consumes platform-provided aspect sub-ratings.</sup>

**What this shows.** Supplying the structured channel reduces MAE by 5.7 % against a
strong sparse text baseline, and the gain replicates under a second extraction model and
on an independent Booking.com corpus. The *functional form* in which the channel enters
matters far less: under unified tuning, additive concatenation and segment-conditioned
interaction differ by 0.0030 MAE in the hotel-weighted paired contrast, roughly one tenth
of the gain either achieves over text alone. A fine-tuned transformer attains lower error
than any configuration proposed here, at roughly twelve times the fitting cost and without
inspectable coefficients.

Full per-fold metrics, paired significance tests, and the regularization audit are in
[`results/`](results/) and in the three Supporting Information workbooks.

---

## Repository layout

```
.
├── config/                    Runtime configuration
├── src/                       Library modules: extraction, features, evaluation harness
├── scripts/                   Retained numbered scripts for the reported pipeline
├── prompts/                   The exact extraction prompt as deployed
├── splits/                    Canonical hash-verified fold assignment
├── results/                   CSV tables backing every number in the manuscript
├── supporting_information/    S1, S2, S3 workbooks as submitted to the journal
├── figures/                   Manuscript figures
└── docs/                      REPRODUCE.md and DATA.md
```

Large derived artefacts — sanitized extraction outputs, evidence offsets, consolidated
per-review predictions, and full coefficient tables — are prepared for a separate data
deposit. Its DOI will be added after upload. See [Data](#data).

---

## Installation

Requires Python 3.10. GPU is needed only for the extraction and transformer stages; every
ridge model in the paper runs on CPU.

```bash
git clone https://github.com/zhuqinfeng1999/interaction-fusion-hotel-rating.git
cd interaction-fusion-hotel-rating
conda env create -f environment.yml
conda activate hotel-fusion
```

Or with pip:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`sqlite3` is supplied by Python's standard library. The client pipeline uses pandas,
NumPy, SciPy, scikit-learn, PyArrow, httpx, PyYAML, tqdm, matplotlib, psutil, PyTorch,
and Transformers. Install vLLM separately in the GPU-serving environment:

```bash
pip install vllm
```

## Local LLM inference

The extractor calls an OpenAI-compatible HTTP API. On the reference Ubuntu workstation
(2 × RTX 5090D, 24 GB each), one independent vLLM server is run per GPU. Set the model
folder to your complete Qwen2.5-7B-Instruct checkout; `--served-model-name` keeps the API
model identifier consistent with `config/local_llm.yaml`.

```bash
MODEL_DIR="$HOME/.cache/huggingface/qwen25_7b"

# Terminal 1: GPU 0, port 8000
CUDA_VISIBLE_DEVICES=0 vllm serve "$MODEL_DIR" \
  --host localhost --port 8000 \
  --served-model-name Qwen/Qwen2.5-7B-Instruct \
  --dtype auto --gpu-memory-utilization 0.90 \
  --max-model-len 4096 --max-num-seqs 64 \
  --api-key local-llm

# Terminal 2: GPU 1, port 8001
CUDA_VISIBLE_DEVICES=1 vllm serve "$MODEL_DIR" \
  --host localhost --port 8001 \
  --served-model-name Qwen/Qwen2.5-7B-Instruct \
  --dtype auto --gpu-memory-utilization 0.90 \
  --max-model-len 4096 --max-num-seqs 64 \
  --api-key local-llm
```

Suggested starting points are conservative; reduce sequence count or context length if
the server runs out of memory.

| Model size | GPUs | `max-model-len` | `max-num-seqs` | Notes |
| :--- | :---: | ---: | ---: | :--- |
| 7B/8B | one model replica per 24 GB GPU | 4096 | 64 | Reference configuration |
| 14B | one replica per GPU if it fits | 2048–4096 | 16–32 | Consider a quantized checkpoint; lower both values first |

Health does not require authorization, but `/v1/*` does:

```bash
curl http://localhost:8000/health
curl http://localhost:8001/health

curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer local-llm" \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen2.5-7B-Instruct","messages":[{"role":"user","content":"Return one JSON object."}],"temperature":0,"max_tokens":32}'
```

---

## Data

### Upstream corpora

Neither corpus is redistributed here. Both are public third-party releases obtained from
their distributors.

| Corpus | Source | Licence | Scale |
| :--- | :--- | :--- | :--- |
| TripAdvisor (primary) | [Kaggle](https://www.kaggle.com/datasets/joebeachcapital/hotel-reviews) | Consult the distributor's current terms; not redistributed here | 877,808 cleaned reviews, 3,945 properties, 2001–2012 |
| Booking.com (external validation) | [Kaggle](https://www.kaggle.com/datasets/jiashenliu/515k-hotel-reviews-data-in-europe) · [HF mirror](https://huggingface.co/datasets/Dricz/515k-Hotel-Reviews-In-Europe) | The mirror and Kaggle page carry different licence labels; see `docs/DATA.md` | 515,738 source rows; 514,935 cleaned reviews, 1,493 properties, 2015–2017 |

Place the TripAdvisor files at `data_ori/reviews.csv` and `data_ori/offerings.csv`, then run
`scripts/01_prepare_dataset.py`. The Booking.com file is fetched and checksum-verified
automatically by `scripts/37_phase3_booking_prepare.py`.

### Released derived artefacts

Prepared for a companion OSF deposit under CC BY 4.0. The public OSF link and DOI will
be added here only after the deposit has been created:

| Directory | Contents |
| :--- | :--- |
| `extraction/` | Sanitized structured outputs for both extraction models on all corpora — aspect polarity, mention indicators, traveler segment, schema-validity flag |
| `extraction/*_evidence_offsets.parquet` | Character offsets locating each evidence span within the cleaned review text |
| `predictions/` | Held-out predictions for every model reported in the paper, one column per model |
| `coefficients/` | Complete coefficient tables with cluster-robust standard errors and bootstrap intervals |
| `diagnostics/` | Regularization response curves and extraction-quality statistics |

**No review text is redistributed anywhere in this release.** Evidence spans are published
as offsets against the cleaned review text produced by `scripts/01_prepare_dataset.py`; to
recover the text, obtain the upstream corpus and apply the offsets. The offsets were
recovered by case-normalized literal substring matching, resolving to the first occurrence
where a span matches more than once; unmatched items are recorded with `start = end = -1`.

### Extraction models

Model weights are not redistributed. The pipeline pins:

| Role | Model |
| :--- | :--- |
| Primary extractor | `Qwen/Qwen2.5-7B-Instruct` |
| Second extractor (sensitivity analysis) | `meta-llama/Llama-3.1-8B-Instruct` |
| Neural text baselines | `roberta-base`, `bert-base-uncased` |

---

## Reproducing the results

The full pipeline is staged. Each stage is resumable and caches its outputs.

```bash
# 1. Prepare the corpus and the canonical splits
python scripts/01_prepare_dataset.py
python splits/expand_splits.py --verify       # confirms the split hash

# 2. Offline aspect extraction (GPU; ~62 GPU-hours on 2 × RTX 5090D)
#    Skip this and use the released extraction outputs instead if you prefer.
python scripts/06_extract_aspects_llm.py --profile qwen25_7b

# 3. Primary models under the unified tuning protocol (CPU; ~1 hour)
python scripts/41_phase4_u0_models.py --stage primary --folds 1,2,3,4,5 --inner-jobs 5

# 4. Ablations (CPU)
python scripts/41_phase4_u0_models.py --stage ablations --variants A2,A3,A5,A7,A8

# 5. External validation on Booking.com (CPU after extraction)
python scripts/42_phase4_booking_u0.py --folds 1,2,3,4,5 --inner-jobs 5

# 6. Consolidate every reported table
python scripts/43_phase4_consolidate.py
```

To skip extraction entirely, download `extraction/` from the OSF deposit into `outputs/`
and start at step 3. Full commands, expected runtimes, and hardware notes are in
[`docs/REPRODUCE.md`](docs/REPRODUCE.md).

**Determinism note.** Repeated extraction under an identical configuration reproduces the
complete structured record exactly for 65.6 % of reviews; individual fields agree across
runs for 98.0–99.9 %. Exact reproduction of the published numbers therefore requires the
released sanitized structured outputs rather than re-running the extractor.

---

## Scope of this release

This repository reproduces the results reported in the manuscript. It is not a dump of
every experiment conducted during the study. Exploratory model variants that the paper
does not report — alternative text channels, cross-fitted representation experiments,
permutation controls, and sample-size scaling studies — are omitted, as are regenerable
caches, model checkpoints, and any artefact containing review text.

---

## Outputs and validation

The working pipeline writes regenerable artefacts under `outputs/`; the public package
contains only the filtered files in `release/github/` and `release/osf/`.

| Path | Producer | Contents / validation |
| :--- | :--- | :--- |
| `outputs/dataset_clean.parquet` | `01_prepare_dataset.py` | Cleaned joined corpus; never published because it contains review text |
| `outputs/splits_fold{1..5}.npz` | `splits/expand_splits.py` or `02_make_splits.py` | Train/test row indices; verify against the released assignment hash |
| `outputs/results_baselines.{json,md}` | `03_run_baselines.py` | Text-only and true-aspect upper-bound fold metrics |
| `outputs/error_analysis_baselines.md` | `03b_error_analysis_baselines.py` | Long-tail and aspect-missingness diagnostics |
| `outputs/aspects_llm_{profile}.parquet` | `06_extract_aspects_llm.py` | Full structured extraction with auditable `row_id`; the working copy may contain non-public audit fields |
| `outputs/llm_cache/{profile}_ep*/cache.sqlite` | `06_extract_aspects_llm.py` | Versioned, resumable endpoint-local caches; not published |
| `outputs/llm_fail_samples_{profile}_ep*.jsonl` | `06_extract_aspects_llm.py` | Truncated failed-response audit samples; private, not published |
| `outputs/llm_runtime_estimate_{profile}.md` | `06_extract_aspects_llm.py` | Per-endpoint pilot throughput, token use, and full-corpus ETA |
| `outputs/llm_output_sanity_{profile}.md` | `06_extract_aspects_llm.py --validate-only` | Key uniqueness, types, ranges, token coverage, and sample audit |
| `outputs/results_llm_aspects_{profile}.{json,md}` | `07_run_llm_aspects_analysis.py` | LLM-only, mixed, residual, interaction, and subset results |
| `outputs/aspect_extraction_quality_{profile}.md` | `07_run_llm_aspects_analysis.py` | MAE/RMSE and correlation against platform aspect ratings |
| `outputs/join_alignment_check_{profile}.md` | `07_run_llm_aspects_analysis.py` | Evidence-to-review substring alignment audit |
| `outputs/aspect_importance_{profile}.md` and figure | `07_run_llm_aspects_analysis.py` | Standardized coefficient ranking and plot |
| `outputs/segment_preference_{profile}.md` and heatmap | `07_run_llm_aspects_analysis.py` | Segment-wise aspect means, mention rates, and heatmap |
| `outputs/run_profile_{profile}_timing.md` | `07_run_llm_aspects_analysis.py` | Stage-level wall time and cache-hit profiling |
| `release/github/supporting_information/S{1,2,3}_File.xlsx` | scripts 48/49 | Journal Supporting Information; independently uploadable and hash-tracked |
| `release/{github,osf}/MANIFEST.csv` | `49_phase6b_prepare_release.py` | Byte size and SHA-256 for every other file in each release tree |

## Reproducibility

- Sampling, split construction, and bootstrap procedures use seed 42 unless an explicit
  determinism seed (42, 43, or 44) is shown.
- The canonical fold object is grouped by `offering_id`; its int8 assignment hashes to
  `dfe3a434fb00c894218309c332627dc0131a2db3cc2cd241727fc712a4803942`.
- Rating predictions are clipped to [1, 5] for the reported primary metrics.
- LLM caches are isolated by endpoint and keyed by model, prompt hash,
  `schema_version=v1_aspects6`, and `prompt_version=v2_strictjson`. `--resume` reuses only
  compatible successful rows.
- The public extraction Parquets remove raw responses, evidence strings, summaries,
  prompts, latency fields, and review text. Evidence is released only as character offsets.

## Troubleshooting

- **401 Unauthorized from vLLM:** `/v1/*` requires
  `Authorization: Bearer local-llm`; `/health` does not. Ensure the key in
  `config/local_llm.yaml` matches `vllm serve --api-key`.
- **Tokenizer/config files missing:** a local model directory must include the complete
  tokenizer, config, index, and weight shards. Do not point vLLM at a partial checkout.
- **Hugging Face downloads are slow:** authenticate with `huggingface-cli login`, accept
  gated-model terms where required, and use resumable snapshot downloads.
- **Extraction `ok_rate=0` or `json_parse_failed`:** confirm `max_tokens=900`, inspect the
  private failure JSONL, and verify the served model name. The extractor repairs common
  array/truncation defects, fills missing schema fields, and retries before failing a row.
- **Only one GPU is busy:** pass both comma-separated endpoints. `--concurrency` is per
  endpoint, so total concurrency doubles in the two-server setup. Check both ports with
  `curl /health` and the processes with `nvidia-smi`.
- **Analysis shows low GPU utilization:** scripts 07 and 41–49 are CPU/I/O intensive;
  low GPU utilization is expected. Lower `--max_features` or disable `--with-text` if RAM
  is constrained.
- **Cache appears stale:** prompt/schema-version changes intentionally cause misses.
  Preserve old caches for audit and resume with the current versioned directories.

---

## Licence

| Component | Licence |
| :--- | :--- |
| Source code in this repository | [MIT](LICENSE) |
| Derived data tables in this repository and in the OSF deposit | [CC BY 4.0](LICENSE-DATA) |
| Upstream TripAdvisor corpus | Consult the distributor's current terms — not redistributed here |
| Upstream Booking.com corpus | See `docs/DATA.md` and the distributor pages — not redistributed here |

Licence labels for the upstream corpora are recorded as documentation of how those
releases are distributed. They are not a determination of rights in the underlying
observations, which were collected from public review pages by third parties unconnected
with this study.

---

## Citation

```bibtex
@misc{zhu2026interaction,
  title   = {Interaction fusion of text and {LLM}-extracted aspects improves hotel
             rating prediction at scale: A verifiable study on {TripAdvisor}},
  author  = {Zhu, Xue and Zhu, Qinfeng},
  year    = {2026},
  note    = {Manuscript under review}
}
```

Please also cite the upstream corpora and the deposited artefacts
(the DOI will be added after deposit) if you use them.

---

## Acknowledgements

This work builds on [vLLM](https://github.com/vllm-project/vllm),
[scikit-learn](https://scikit-learn.org/), and
[Hugging Face Transformers](https://github.com/huggingface/transformers). We thank the
maintainers of the public review corpora analysed here.
