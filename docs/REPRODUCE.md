# Reproducing the Reported Results

## Environment

```bash
conda env create -f environment.yml
conda activate hotel-fusion
```

Python 3.10 is the reference environment. Ridge/diagnostic stages are CPU and I/O intensive; low GPU utilization is expected. GPUs are used only for extraction and transformer training.

## Upstream data

Obtain the corpora from their distributors as documented in `DATA.md`. No review text is included in this release.

## Canonical split

```bash
python splits/expand_splits.py --verify
python splits/expand_splits.py --output-dir outputs
```

Expected assignment SHA-256: `dfe3a434fb00c894218309c332627dc0131a2db3cc2cd241727fc712a4803942`.

## Commands

```bash
python scripts/00_sanity_check.py
python scripts/01_prepare_dataset.py
python splits/expand_splits.py --output-dir outputs

# Optional GPU extraction (use released sanitized extraction data for exact reproduction)
python scripts/06_extract_aspects_llm.py --profile qwen25_7b --resume

# Published/original protocol and diagnostics
python scripts/08_phase0_reproduce.py
python scripts/17_phase2_schema_forensics.py
python scripts/18_phase2_schema_enforcement.py
python scripts/28_phase3_regularization_curves.py --stage all

# Unified U0 primary models and ablations
python scripts/41_phase4_u0_models.py --stage primary --folds 1,2,3,4,5 --inner-jobs 5
python scripts/41_phase4_u0_models.py --stage ablations --variants A2,A3,A5,A7,A8 --folds 1,2,3,4,5 --inner-jobs 5

# Booking external validation
python scripts/37_phase3_booking_prepare.py
python scripts/42_phase4_booking_u0.py --folds 1,2,3,4,5 --inner-jobs 5

# Consolidation and U0 inference
python scripts/43_phase4_consolidate.py
python scripts/44_phase5_discrepancy_strata.py
python scripts/45_phase5_e2_u0_missingness.py
python scripts/46_phase5_u0_coefficient_tables.py --bootstrap-resamples 10000 --folds 1,2,3,4,5
python scripts/47_finalize_phase5.py
python scripts/48_phase6a_assemble.py
```

## Second-revision analyses (PONE-D-26-23918R1)

All outputs go to `outputs/r2/`, except scripts 13 and 09, which keep writing to `outputs/`.
The GPU steps need a vLLM server per model. Serve Llama-3.1-8B-Instruct with
`--served-model-name meta-llama/Llama-3.1-8B-Instruct --port 8010`, or pass `--model` and
`--endpoint` to scripts 55 and 56.

```bash
# Zero-shot Qwen rater on all reviews (reuses cached rows of the earlier 100k subsample)
python scripts/13_run_llm_direct_rating.py --modes zero --sample-size 877808 --concurrency 48 --resume

# BERT-base learning-rate sweep (same fold-1 validation split as RoBERTa), then 5 folds at the selected rate
python scripts/54_r2_bert_lr_sweep.py --stage sweep --learning-rates 1e-5,2e-5,3e-5,5e-5
python scripts/54_r2_bert_lr_sweep.py --stage select
python scripts/09_train_transformer_regression.py --model-name bert-base-uncased \
  --model-tag bert_base_uncased_mr_lr5em05 --learning-rate 5e-5 --folds 1,2,3,4,5 --resume

# Text + true aspects reference (TA) and ablation A9 (no sleep-quality columns), U0 protocol
python scripts/51_r2_u0_new_variants.py --variant TA --inner-jobs 5
python scripts/51_r2_u0_new_variants.py --variant NS --inner-jobs 5

# Analyses on held-out predictions (no refits)
python scripts/52_r2_fidelity_routing.py
python scripts/53_r2_strata_segments_weights.py

# Llama-3.1-8B evidence judge and prompt-portability test
python scripts/55_r2_evidence_judge.py
python scripts/56_r2_prompt_portability.py

# Collect numbers and build S5 File
python scripts/57_r2_collect.py
python scripts/58_r2_s5_workbook.py
```

Replace `5e-5` and the tag in the BERT command with the rate printed by `--stage select`.
Script 55 writes item-level files that contain quoted evidence; keep them local. The public
version on OSF (`diagnostics/evidence_judge_r2.parquet`) keeps only labels and row keys.

## Expected runtime

- Preprocessing/split expansion: minutes on local SSD storage.
- Full Qwen extraction: approximately 31 wall-clock hours with two independent RTX 5090D endpoints at the measured pilot throughput (about 62 aggregate GPU-hours); hardware and server batching materially affect this value.
- U0 sparse models: approximately one to several CPU-hours depending on core count and TF-IDF cache state.
- Transformer baselines and 10,000-hotel bootstrap inference: many compute-hours; use the resumable fold/cache options.
- Phase 6 release assembly: minutes; CPU only.

Exact published numbers require the released sanitized structured extraction fields because stochastic re-extraction is not record-level deterministic.

## Retained scripts

- `00_sanity_check.py` — raw-file schema check
- `01_prepare_dataset.py` — TripAdvisor preprocessing used by every analysis
- `02_make_splits.py` — original grouped split construction
- `03_run_baselines.py` — published TF-IDF and true-aspect baselines
- `03b_error_analysis_baselines.py` — reported baseline error diagnostics
- `06_extract_aspects_llm.py` — deployed Qwen extraction and cache protocol
- `07_run_llm_aspects_analysis.py` — published legacy fusion and extraction-quality analysis
- `08_phase0_reproduce.py` — published-protocol reproduction gate
- `09_train_transformer_regression.py` — reported neural text baselines
- `13_run_llm_direct_rating.py` — reported zero-shot direct-LLM baseline
- `14_summarize_phase1.py` — neural/direct baseline consolidation
- `17_phase2_schema_forensics.py` — schema-validity forensic analysis
- `18_phase2_schema_enforcement.py` — true guided-JSON validation
- `20_phase2_regularization_forensics.py` — reported protocol-sensitivity models
- `20b_phase2_r_orchestrator.sh` — orchestrator for reported protocol sensitivity
- `21_phase2_controls_ablations.py` — feature and fitting utilities imported by common-alpha script 26
- `23_phase2_transformer_tuning.py` — reported transformer tuning audit
- `23b_phase2_t_orchestrator.sh` — orchestrator for reported transformer tuning
- `24_finalize_phase2.py` — Phase 2 reported-table consolidation
- `25_phase2_claim_sensitivity.py` — reported claim/protocol sensitivity
- `26_phase2_common_alpha_sensitivity.py` — reported common-alpha sensitivity
- `27_phase3_g0_alpha_audit.py` — published-alpha provenance audit
- `28_phase3_regularization_curves.py` — 615-row alpha-response diagnostic
- `28b_validate_phase3_g1.py` — alpha-curve validation gate
- `29_phase3_l_sample.py` — canonical extractor-sensitivity sample
- `30_phase3_determinism.py` — three-seed extraction determinism study
- `31_phase3_guided_extract.py` — guided extraction for sensitivity/external validation
- `32_phase3_extractor_agreement.py` — Qwen-Llama agreement and evidence grounding
- `33_phase3_extractor_downstream.py` — seven reported L5 downstream models
- `34_phase3_coefficient_tables.py` — statistical inference utilities imported by U0 coefficient script 46
- `35_finalize_phase3.py` — Phase 3 reported-table consolidation
- `37_phase3_booking_prepare.py` — external-corpus acquisition and preprocessing
- `38_phase3_booking_analyze.py` — initial Booking external validation
- `39_phase3_w_after_qwen.sh` — Booking extraction/analysis orchestrator
- `40_phase3_llama_after_qwen.sh` — Llama sensitivity orchestrator
- `41_phase4_u0_models.py` — primary U0 models and manuscript ablations
- `42_phase4_booking_u0.py` — U0 external validation
- `43_phase4_consolidate.py` — manuscript-ready Phase 4 tables
- `44_phase5_discrepancy_strata.py` — published/simple-fusion discrepancy and U0 strata
- `45_phase5_e2_u0_missingness.py` — U0 segment-masking curve
- `46_phase5_u0_coefficient_tables.py` — complete U0 regression inference
- `47_finalize_phase5.py` — Phase 5 consistency report
- `48_phase6a_assemble.py` — Supporting Information workbook assembly
- `49_phase6b_prepare_release.py` — sanitized public-release assembly and verification
- `51_r2_u0_new_variants.py`: text + true aspects reference and ablation A9 under U0
- `52_r2_fidelity_routing.py`: routing rule for empty extractions; fidelity by mention
- `53_r2_strata_segments_weights.py`: fidelity table, coverage strata, segment phrases, segment-specific weights
- `54_r2_bert_lr_sweep.py`: BERT-base learning-rate selection
- `55_r2_evidence_judge.py`: independent judge on the quoted evidence
- `56_r2_prompt_portability.py`: Llama re-extraction with a stated polarity convention
- `57_r2_collect.py`: collects the numbers reported in the revision
- `58_r2_s5_workbook.py`: builds S5 File

## Dropped scripts

- `04_aspect_pipeline_dummy.py` — dummy extractor smoke test; not in manuscript
- `05_segment_heterogeneity.py` — dummy-extractor exploratory analysis
- `10_extract_transformer_embeddings.py` — frozen-representation experiment excluded from manuscript
- `11_run_representation_fusion.py` — frozen/SVD representation variants excluded from manuscript
- `12_extract_tfidf_svd_embeddings.py` — TF-IDF+SVD channel excluded from manuscript
- `15_train_end_to_end_fusion.py` — end-to-end fusion variants excluded from manuscript
- `16_run_phase1_remaining.sh` — orchestrates excluded representation/end-to-end variants
- `19_phase2_crossfit_representations.py` — cross-fitted representation experiment excluded from manuscript
- `19b_phase2_gpu_orchestrator.sh` — orchestrates excluded cross-fitted representations
- `21b_phase2_pa_orchestrator.sh` — orchestrates excluded controls
- `22_phase2_missingness_error_analysis.py` — superseded by U0-consistent Phase 5 scripts 44/45
- `22b_phase2_ed_orchestrator.sh` — orchestrates superseded missingness analysis
- `24b_phase2_finalize_orchestrator.sh` — depends on the superseded Phase 2 missingness script
- `36_phase3_orchestrate_remaining.sh` — depends on superseded Phase 3 coefficient script 34
