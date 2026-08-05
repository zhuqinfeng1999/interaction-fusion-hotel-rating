# Data and Privacy

## Upstream corpora

No upstream review text is redistributed.

### TripAdvisor

- Obtain the Hotel Reviews corpus from its distributor and place `reviews.csv` and `offerings.csv` under `data_ori/`.
- Required review fields include `title`, `text`, nested `ratings`, nested `author`, and `offering_id`; offering metadata comes from `offerings.csv`.
- The distributor labels the database under the [Open Database License (ODbL) v1.0](https://opendatacommons.org/licenses/odbl/1-0/) and its contents under the [Database Contents License (DbCL) v1.0](https://opendatacommons.org/licenses/dbcl/1-0/). These labels document how the release is distributed rather than determining rights in the underlying observations.
- The repository did **not** persist an authoritative TripAdvisor download timestamp. The local CSV mtime of 2023-08-30 is archive metadata and must not be described as the download date.
- After `scripts/01_prepare_dataset.py`, the accepted cleaned corpus has 877,808 reviews from 3,945 hotels.

### Booking.com external validation

- Source used: Hugging Face release `Dricz/515k-Hotel-Reviews-In-Europe`, commit `cdaea79962c5d667dd5177956ef0353001f6ade3`.
- Persisted download time: `2026-08-02T06:40:24.716734+00:00`.
- Source SHA-256: `a4810c2757934f0a826a1b16a437eb67a38be45b1a22ad56772afce0b6c11af9`.
- The source has 515,738 rows; preprocessing retains 514,935 reviews, 1,493 properties, dated 2015-08-04 through 2017-08-03.
- The Hugging Face mirror and Kaggle release have matching data identity in the project's provenance audit but display different licence labels. Those labels are documented, not interpreted as proof that either uploader can relicense the underlying reviews.

## OSF deposit

- Public project: <https://osf.io/ps54c/>
- DOI: <https://doi.org/10.17605/OSF.IO/PS54C>
- `extraction/`: structured numeric/categorical extraction fields only. The only string column is `traveler_segment`.
- `predictions/`: consolidated held-out predictions. TripAdvisor model columns are clipped to [1,5].
- Booking `y_true` and unsuffixed model columns use the native score scale; `y_true_rescaled_1_5` and columns ending `__rescaled_1_5` provide the manuscript's 1-5 scale.
- `coefficients/` and `diagnostics/`: complete machine-readable inference and quality tables.

## Evidence-offset convention

Aspect IDs are `0=service`, `1=cleanliness`, `2=location`, `3=rooms`, `4=value`, `5=sleep_quality`.
Offsets are Python character indices into the cleaned `review_text` produced by `scripts/01_prepare_dataset.py`. Matching uses `review_text.lower().find(evidence.lower())`; if multiple matches occur, the first is retained. Unmatched evidence items remain in the table with `start=end=-1` and `matched=false`.

- `tripadvisor_qwen25_7b_full_evidence_offsets.parquet`: 3,527,105/3,846,102 matched (91.705966%); first occurrence used for 29,829 ambiguous matched excerpts.
- `tripadvisor_qwen25_7b_guided_200k_evidence_offsets.parquet`: 954,130/983,771 matched (96.987002%); first occurrence used for 7,030 ambiguous matched excerpts.
- `tripadvisor_llama31_8b_guided_200k_evidence_offsets.parquet`: 1,173,099/1,214,393 matched (96.599618%); first occurrence used for 20,709 ambiguous matched excerpts.

The evidence strings themselves, response JSON, raw model output, prompts attached to rows, cleaned review text, and all upstream CSVs are excluded.
