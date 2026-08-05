#!/usr/bin/env bash
# Start Booking external extraction only after the L3 Qwen guided output passes.
# Conda's activation/deactivation hooks may legitimately probe unset optional
# variables, so do not enable nounset before activating the project env.
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
eval "$(conda shell.bash hook)"
conda activate "${HOTEL_CONDA_ENV:-hotel-fusion}"
L3_OUTPUT="outputs/phase3/aspects_guided_qwen25_7b_l1_n200041.parquet"
L3_STATS="outputs/phase3/l_guided_runtime_qwen25_7b_l1_n200041.json"

while [[ ! -f "$L3_OUTPUT" || ! -f "$L3_STATS" ]]; do
  echo "[$(date --iso-8601=seconds)] W waiting for completed L3 Qwen artifacts"
  sleep 60
done

python - <<'PY'
import json
from pathlib import Path
import pandas as pd

output = Path("outputs/phase3/aspects_guided_qwen25_7b_l1_n200041.parquet")
stats_path = Path("outputs/phase3/l_guided_runtime_qwen25_7b_l1_n200041.json")
stats = json.loads(stats_path.read_text())
if int(stats["sample_n"]) != 200041 or float(stats["ok_rate"]) < 0.99:
    raise RuntimeError(f"L3 Qwen validation failed: {stats}")
frame = pd.read_parquet(output, columns=["row_id", "ok"])
if len(frame) != 200041 or frame["row_id"].duplicated().any():
    raise RuntimeError("L3 Qwen parquet cardinality/key check failed")
print(f"L3 Qwen validation PASS: ok_rate={frame['ok'].eq(1).mean():.4%}")
PY

echo "[$(date --iso-8601=seconds)] starting Booking guided Qwen extraction"
python scripts/31_phase3_guided_extract.py \
  --extractor qwen \
  --profile qwen25_7b \
  --sample-file outputs/phase3/w_sample_n200000_seed42.npz \
  --dataset-file outputs/phase3/booking_dataset_clean.parquet \
  --cache-root outputs/phase3/w_guided_cache \
  --output-tag booking_qwen_guided_n200035 \
  --runtime-prefix w_guided_runtime \
  --experiment W2 \
  --stage-label "W Booking Qwen guided" \
  --endpoints "http://localhost:8000/v1,http://localhost:8001/v1" \
  --concurrency 16 \
  --resume

echo "[$(date --iso-8601=seconds)] starting Booking corrected-protocol models"
python scripts/38_phase3_booking_analyze.py \
  --qwen-path outputs/phase3/aspects_guided_booking_qwen_guided_n200035.parquet \
  --max-features 200000 \
  --resume
echo "[$(date --iso-8601=seconds)] W external validation complete"
