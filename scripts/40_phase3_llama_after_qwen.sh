#!/usr/bin/env bash
# Safely switch the two local vLLM endpoints from Qwen to Llama only after
# both the external-Qwen branch and the verified model download are complete.
set -eo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /absolute/path/to/Llama-3.1-8B-Instruct" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="$1"
PHASE3="$ROOT/outputs/phase3"
W_PARQUET="$PHASE3/aspects_guided_booking_qwen_guided_n200035.parquet"
W_STATS="$PHASE3/w_guided_runtime_booking_qwen_guided_n200035.json"
mkdir -p "$PHASE3"
cd "$ROOT"

eval "$(conda shell.bash hook)"
conda activate "${HOTEL_CONDA_ENV:-hotel-fusion}"

timestamp() {
  date --iso-8601=seconds
}

model_is_complete() {
  MODEL_DIR="$MODEL_DIR" python - <<'PY' >/dev/null 2>&1
import hashlib
import os
from pathlib import Path

root = Path(os.environ["MODEL_DIR"])
expected = {
    "model-00001-of-00004.safetensors": (4976698672, "2b1879f356aed350030bb40eb45ad362c89d9891096f79a3ab323d3ba5607668"),
    "model-00002-of-00004.safetensors": (4999802720, "09d433f650646834a83c580877bd60c6d1f88f7755305c12576b5c7058f9af15"),
    "model-00003-of-00004.safetensors": (4915916176, "fc1cdddd6bfa91128d6e94ee73d0ce62bfcdb7af29e978ddcab30c66ae9ea7fa"),
    "model-00004-of-00004.safetensors": (1168138808, "92ecfe1a2414458b4821ac8c13cf8cb70aed66b5eea8dc5ad9eeb4ff309d6d7b"),
}
required = [
    "config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
]
if any(not (root / name).is_file() for name in required):
    raise SystemExit(1)
for name, (size, digest) in expected.items():
    path = root / name
    if not path.is_file() or path.stat().st_size != size:
        raise SystemExit(1)
    if path.with_suffix(path.suffix + ".aria2").exists():
        raise SystemExit(1)
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            hasher.update(block)
    if hasher.hexdigest() != digest:
        raise SystemExit(1)
PY
}

while [[ ! -f "$W_PARQUET" || ! -f "$W_STATS" ]]; do
  echo "[$(timestamp)] Llama switch waiting for completed Booking/Qwen extraction"
  sleep 120
done

W_PARQUET="$W_PARQUET" W_STATS="$W_STATS" python - <<'PY'
import json
import os
from pathlib import Path

import pandas as pd

parquet_path = Path(os.environ["W_PARQUET"])
stats = json.loads(Path(os.environ["W_STATS"]).read_text(encoding="utf-8"))
if int(stats.get("sample_n", -1)) != 200035 or float(stats.get("ok_rate", 0)) < 0.99:
    raise RuntimeError(f"Booking/Qwen stats validation failed: {stats}")
frame = pd.read_parquet(parquet_path, columns=["row_id", "ok"])
if len(frame) != 200035 or frame["row_id"].duplicated().any():
    raise RuntimeError("Booking/Qwen parquet cardinality/key validation failed")
if float(frame["ok"].eq(1).mean()) < 0.99:
    raise RuntimeError("Booking/Qwen parquet OK rate below 0.99")
print(f"Booking/Qwen extraction validation PASS: ok={frame['ok'].eq(1).mean():.4%}")
PY

while ! model_is_complete; do
  echo "[$(timestamp)] Llama switch waiting for verified model files"
  sleep 120
done
echo "[$(timestamp)] prerequisites PASS: Booking/Qwen extraction complete and Llama model verified"

mapfile -t QWEN_PIDS < <(pgrep -f "vllm serve .*/qwen25_7b" || true)
if [[ ${#QWEN_PIDS[@]} -gt 0 ]]; then
  echo "[$(timestamp)] stopping Qwen vLLM pids=${QWEN_PIDS[*]}"
  kill -TERM "${QWEN_PIDS[@]}"
  for _ in $(seq 1 60); do
    alive=0
    for pid in "${QWEN_PIDS[@]}"; do
      kill -0 "$pid" 2>/dev/null && alive=1
    done
    [[ "$alive" -eq 0 ]] && break
    sleep 5
  done
fi
if pgrep -f "vllm serve .*/qwen25_7b" >/dev/null; then
  echo "Qwen vLLM did not stop cleanly; refusing to start overlapping servers" >&2
  exit 1
fi

VLLM_BIN="$(command -v vllm)"

endpoint_is_llama() {
  local port="$1"
  curl -fsS --max-time 10 \
    -H 'Authorization: Bearer local-llm' \
    "http://localhost:${port}/v1/models" \
    | EXPECTED_MODEL="$MODEL_DIR" python -c '
import json, os, sys
payload = json.load(sys.stdin)
ids = {str(item.get("id", "")) for item in payload.get("data", [])}
raise SystemExit(0 if os.environ["EXPECTED_MODEL"] in ids else 1)
'
}

for endpoint_idx in 0 1; do
  port=$((8000 + endpoint_idx))
  log_path="$PHASE3/llama_vllm_ep${endpoint_idx}.log"
  pid_path="$PHASE3/llama_vllm_ep${endpoint_idx}.pid"
  if endpoint_is_llama "$port" 2>/dev/null; then
    echo "[$(timestamp)] reusing healthy Llama endpoint=$endpoint_idx port=$port"
    continue
  fi
  if curl -fsS --max-time 5 "http://localhost:${port}/health" >/dev/null 2>&1; then
    echo "port $port is occupied by a healthy non-Llama endpoint; refusing to overlap models" >&2
    exit 1
  fi
  if [[ -f "$pid_path" ]]; then
    stale_pid="$(cat "$pid_path")"
    if kill -0 "$stale_pid" 2>/dev/null; then
      echo "[$(timestamp)] stopping stale Llama endpoint=$endpoint_idx pid=$stale_pid"
      kill -TERM "$stale_pid"
      for _ in $(seq 1 30); do
        kill -0 "$stale_pid" 2>/dev/null || break
        sleep 2
      done
    fi
  fi
  echo "[$(timestamp)] starting Llama endpoint=$endpoint_idx gpu=$endpoint_idx port=$port"
  nohup env CUDA_VISIBLE_DEVICES="$endpoint_idx" "$VLLM_BIN" serve "$MODEL_DIR" \
    --host localhost \
    --port "$port" \
    --dtype auto \
    --gpu-memory-utilization 0.90 \
    --max-model-len 4096 \
    --max-num-seqs 64 \
    --api-key local-llm \
    --structured-outputs-config '{"backend":"xgrammar","disable_any_whitespace":true}' \
    >"$log_path" 2>&1 &
  echo "$!" >"$pid_path"
done

for endpoint_idx in 0 1; do
  port=$((8000 + endpoint_idx))
  ready=0
  for _ in $(seq 1 180); do
    if curl -fsS --max-time 5 "http://localhost:${port}/health" >/dev/null; then
      ready=1
      break
    fi
    sleep 5
  done
  if [[ "$ready" -ne 1 ]]; then
    echo "Llama endpoint $endpoint_idx failed health check; see $PHASE3/llama_vllm_ep${endpoint_idx}.log" >&2
    exit 1
  fi
  curl -fsS --max-time 10 \
    -H 'Authorization: Bearer local-llm' \
    "http://localhost:${port}/v1/models" >/dev/null
  echo "[$(timestamp)] Llama endpoint $endpoint_idx ready"
done

python scripts/31_phase3_guided_extract.py \
  --extractor llama \
  --profile llama31_8b \
  --model "$MODEL_DIR" \
  --sample-file outputs/phase3/v_sample_n20_seed42.npz \
  --cache-root outputs/phase3/l_smoke_cache \
  --output-tag llama31_8b_smoke_n20 \
  --runtime-prefix l_smoke_runtime \
  --experiment L2_SMOKE \
  --stage-label "Llama guided smoke" \
  --endpoints "http://localhost:8000/v1,http://localhost:8001/v1" \
  --concurrency 4 \
  --resume

python - <<'PY'
import json
from pathlib import Path

import pandas as pd

phase3 = Path("outputs/phase3")
stats = json.loads(
    (phase3 / "l_smoke_runtime_llama31_8b_smoke_n20.json").read_text(
        encoding="utf-8"
    )
)
frame = pd.read_parquet(
    phase3 / "aspects_guided_llama31_8b_smoke_n20.parquet",
    columns=["row_id", "ok"],
)
if len(frame) != 20 or frame["row_id"].duplicated().any():
    raise RuntimeError("Llama smoke cardinality/key validation failed")
if float(stats.get("ok_rate", 0)) < 0.95 or float(frame["ok"].eq(1).mean()) < 0.95:
    raise RuntimeError(f"Llama guided smoke OK rate failed: {stats}")
print(f"Llama guided smoke PASS: ok={frame['ok'].eq(1).mean():.4%}")
PY

python scripts/31_phase3_guided_extract.py \
  --extractor llama \
  --profile llama31_8b \
  --model "$MODEL_DIR" \
  --endpoints "http://localhost:8000/v1,http://localhost:8001/v1" \
  --concurrency 16 \
  --resume

python scripts/32_phase3_extractor_agreement.py
python scripts/33_phase3_extractor_downstream.py
date --iso-8601=seconds >"$PHASE3/phase3_l_complete.txt"
echo "[$(timestamp)] Phase 3 L2/L4/L5/L6 complete"
