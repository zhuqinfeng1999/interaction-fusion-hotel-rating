#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
LOG_DIR="outputs/phase2/logs"
GATE_JSON="outputs/phase2/experiment_r_reduced_runtime_gate.json"
LOG_FILE="$LOG_DIR/experiment_r_reduced.log"
mkdir -p "$LOG_DIR"

# Inner CV is parallelized explicitly in Python.  Keep BLAS libraries at one
# thread per worker so three inner folds do not silently oversubscribe the CPU.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

echo "[$(date --iso-8601=seconds)] Experiment R reduced protocol: fold-1 runtime gate" \
  | tee -a "$LOG_FILE"
fold1_started="$(date +%s)"
"$PYTHON_BIN" scripts/20_phase2_regularization_forensics.py \
  --mode all \
  --folds 1 \
  --inner-jobs 3 \
  --alpha-screen-max-iterations 250 \
  --alpha-refine-half-decades 0.5 \
  --lambda-coordinate-passes 2 \
  --resume \
  2>&1 | tee -a "$LOG_FILE"
fold1_finished="$(date +%s)"
fold1_seconds="$((fold1_finished - fold1_started))"

decision="$($PYTHON_BIN - "$fold1_seconds" "$GATE_JSON" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

fold1_seconds = int(sys.argv[1])
path = Path(sys.argv[2])
projected_seconds = fold1_seconds * 5
threshold_seconds = 12 * 3600
decision = "continue_all_folds" if projected_seconds <= threshold_seconds else "stop_over_12h"
payload = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "protocol": "approved_reduced_v1",
    "fold1_wall_seconds": fold1_seconds,
    "fold1_wall_hours": fold1_seconds / 3600,
    "projected_five_fold_seconds": projected_seconds,
    "projected_five_fold_hours": projected_seconds / 3600,
    "threshold_hours": 12.0,
    "decision": decision,
    "projection_rule": "fold-1 observed wall time multiplied by five",
}
path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(decision)
PY
)"

echo "[$(date --iso-8601=seconds)] fold-1 wall=${fold1_seconds}s; gate=$decision" \
  | tee -a "$LOG_FILE"
if [[ "$decision" != "continue_all_folds" ]]; then
  echo "[$(date --iso-8601=seconds)] stopping before folds 2-5 due to 12-hour gate" \
    | tee -a "$LOG_FILE"
  exit 75
fi

echo "[$(date --iso-8601=seconds)] gate passed; running folds 2-5" \
  | tee -a "$LOG_FILE"
"$PYTHON_BIN" scripts/20_phase2_regularization_forensics.py \
  --mode all \
  --folds 2,3,4,5 \
  --inner-jobs 3 \
  --alpha-screen-max-iterations 250 \
  --alpha-refine-half-decades 0.5 \
  --lambda-coordinate-passes 2 \
  --resume \
  2>&1 | tee -a "$LOG_FILE"

echo "[$(date --iso-8601=seconds)] Experiment R complete" | tee -a "$LOG_FILE"
