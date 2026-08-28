#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
RUN_ROOT="$PROJECT_ROOT/train/runs/seventh_train"
SPLIT_ROOT="$PROJECT_ROOT/external/data/splits/seventh_train"
IMAGE_ROOT="$PROJECT_ROOT/external/data/images"
SIXTH_WEIGHTS="$SCRIPT_DIR/best_model_six_train.weights.h5"
PYTHON="${PYTHON:-python3}"
BATCH_SIZE="${BATCH_SIZE:-96}"
WORKERS="${WORKERS:-16}"
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"

required=(
  "$SIXTH_WEIGHTS"
  "$SPLIT_ROOT/fixed_split_manifest.json"
  "$SPLIT_ROOT/train.txt"
  "$SPLIT_ROOT/legacy_val.txt"
  "$SPLIT_ROOT/new_val.txt"
  "$SPLIT_ROOT/combined_val.txt"
  "$IMAGE_ROOT/dragon"
  "$IMAGE_ROOT/peak"
  "$IMAGE_ROOT/soar"
)
for path in "${required[@]}"; do
  if [[ ! -e "$path" ]]; then
    relative="${path#"$PROJECT_ROOT/"}"
    echo "[seventh_train] missing required external resource: $relative" >&2
    exit 1
  fi
done

mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/checkpoints" "$RUN_ROOT/status"
exec 9>"$RUN_ROOT/.formal.lock"
if ! flock -n 9; then
  echo "[seventh_train] another formal process owns the lock" >&2
  exit 1
fi

echo "[seventh_train] start=$(date --iso-8601=seconds)"
echo "INITIALIZATION=sixth_best_weights"
echo "INITIALIZATION_WEIGHTS=train/scripts/best_model_six_train.weights.h5"
echo "RESUME_MODEL_WEIGHTS=false"
echo "RESTORE_SIXTH_OPTIMIZER=false"
echo "RESTORE_SIXTH_EPOCH=false"
echo "HASH_CHECK_REQUIRED=false"

if (( BATCH_SIZE < 96 || BATCH_SIZE % 96 != 0 )); then
  echo "Selected batch size violates 96-sample quota blocks: $BATCH_SIZE" >&2
  exit 1
fi
echo "SELECTED_BATCH_SIZE=$BATCH_SIZE"

HARD_POOL="$RUN_ROOT/hard_samples/hard_pool.json"
if [[ ! -f "$HARD_POOL" ]]; then
  echo "[seventh_train] hard pool missing; generating it from the relative external dataset"
  "$PYTHON" "$SCRIPT_DIR/hard_sample_pool_seventh.py" \
    --batch-size "${HARD_BATCH_SIZE:-192}" --workers "$WORKERS"
fi

resume=()
if [[ -f "$RUN_ROOT/checkpoints/formal/training_state/checkpoint" ]]; then
  resume=(--resume)
  echo "[seventh_train] resume_from_own_checkpoint=true"
else
  echo "[seventh_train] fresh_epoch_1_with_new_adam=true"
fi
"$PYTHON" "$SCRIPT_DIR/train_seventh.py" --experiment formal --batch-size "$BATCH_SIZE" \
  --max-epochs 100 --workers "$WORKERS" --initial-lr 1e-4 "${resume[@]}"
echo "[seventh_train] training_complete=$(date --iso-8601=seconds)"
touch "$RUN_ROOT/status/formal_training_complete.ok"
