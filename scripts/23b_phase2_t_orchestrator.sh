#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
LOG_DIR="outputs/phase2/logs"
mkdir -p "$LOG_DIR"

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

echo "[$(date --iso-8601=seconds)] T1 learning-rate sweep on two GPUs" \
  | tee -a "$LOG_DIR/experiment_t_orchestrator.log"
"$PYTHON_BIN" scripts/23_phase2_transformer_tuning.py \
  --stage sweep --learning-rates 1e-5,2e-5 --device cuda:0 --num-workers 2 --resume \
  2>&1 | tee -a "$LOG_DIR/experiment_t_sweep_gpu0.log" &
gpu0_job=$!
"$PYTHON_BIN" scripts/23_phase2_transformer_tuning.py \
  --stage sweep --learning-rates 3e-5,5e-5 --device cuda:1 --num-workers 2 --resume \
  2>&1 | tee -a "$LOG_DIR/experiment_t_sweep_gpu1.log" &
gpu1_job=$!
wait "$gpu0_job"
wait "$gpu1_job"

"$PYTHON_BIN" scripts/23_phase2_transformer_tuning.py --stage select \
  2>&1 | tee -a "$LOG_DIR/experiment_t_orchestrator.log"

read -r selected_lr selected_tag rerun_required <<<"$($PYTHON_BIN - <<'PY'
import json
from pathlib import Path
p = json.loads(Path('outputs/phase2/experiment_t1_selection.json').read_text())
lr = float(p['selected_learning_rate'])
tag = f'{lr:.0e}'.replace('+', '').replace('-', 'm')
print(f'{lr:.12g} {tag} {int(p["requires_five_fold_rerun"])}')
PY
)"

if [[ "$rerun_required" == "1" ]]; then
  selected_model_tag="roberta_base_phase2_lr${selected_tag}"
  echo "[$(date --iso-8601=seconds)] T1 selected $selected_lr; rerunning five folds" \
    | tee -a "$LOG_DIR/experiment_t_orchestrator.log"
  "$PYTHON_BIN" scripts/09_train_transformer_regression.py \
    --model-name roberta-base --model-tag "$selected_model_tag" \
    --device cuda:0 --folds 1,3,5 --learning-rate "$selected_lr" \
    --num-workers 2 --resume --skip-token-audit \
    2>&1 | tee -a "$LOG_DIR/experiment_t_selected_gpu0.log" &
  selected_gpu0=$!
  "$PYTHON_BIN" scripts/09_train_transformer_regression.py \
    --model-name roberta-base --model-tag "$selected_model_tag" \
    --device cuda:1 --folds 2,4 --learning-rate "$selected_lr" \
    --num-workers 2 --resume --skip-token-audit \
    2>&1 | tee -a "$LOG_DIR/experiment_t_selected_gpu1.log" &
  selected_gpu1=$!
  wait "$selected_gpu0"
  wait "$selected_gpu1"
  "$PYTHON_BIN" scripts/09_train_transformer_regression.py \
    --model-name roberta-base --model-tag "$selected_model_tag" \
    --device cuda:0 --folds 1,2,3,4,5 --learning-rate "$selected_lr" \
    --num-workers 2 --resume --skip-token-audit \
    2>&1 | tee -a "$LOG_DIR/experiment_t_selected_finalize.log"
else
  selected_model_tag="roberta_base"
  echo "[$(date --iso-8601=seconds)] T1 retained published learning rate 2e-5" \
    | tee -a "$LOG_DIR/experiment_t_orchestrator.log"
fi

max512_tag="roberta_base_phase2_maxlen512_${selected_tag}"
echo "[$(date --iso-8601=seconds)] T2 fold-1 max_length=512" \
  | tee -a "$LOG_DIR/experiment_t_orchestrator.log"
"$PYTHON_BIN" scripts/09_train_transformer_regression.py \
  --model-name roberta-base --model-tag "$max512_tag" \
  --device cuda:0 --folds 1 --max-length 512 \
  --train-batch-size 8 --eval-batch-size 16 --gradient-accumulation 8 \
  --learning-rate "$selected_lr" --num-workers 2 --resume \
  2>&1 | tee -a "$LOG_DIR/experiment_t_maxlen512.log"

"$PYTHON_BIN" scripts/23_phase2_transformer_tuning.py \
  --stage finalize --selected-model-tag "$selected_model_tag" \
  --max512-model-tag "$max512_tag" \
  2>&1 | tee -a "$LOG_DIR/experiment_t_orchestrator.log"
echo "[$(date --iso-8601=seconds)] Experiment T complete" \
  | tee -a "$LOG_DIR/experiment_t_orchestrator.log"
