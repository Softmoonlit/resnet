#!/usr/bin/env bash
set -euo pipefail

# Stage B Round 2: image size sweep (AMP only)
# Usage:
#   bash scripts/run_stage_b_round2.sh
#   bash scripts/run_stage_b_round2.sh --dataset tiny-imagenet --data-dir ./data/tiny-imagenet --batch-size 32 --workers 4 --epochs 5
#   bash scripts/run_stage_b_round2.sh --mode full --dataset tiny-imagenet --data-dir ./data/tiny-imagenet

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATASET="tiny-imagenet"
DATA_DIR="${PROJECT_ROOT}/data/tiny-imagenet"
BATCH_SIZE="32"
WORKERS="4"
EPOCHS="5"
SEED="42"
IMG_LIST="224 256"
MODE="minimal"

while [[ $# -gt 0 ]]; do
  key="$1"
  case "$key" in
    --dataset)
      DATASET="$2"
      shift; shift
      ;;
    --data-dir)
      DATA_DIR="$2"
      shift; shift
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift; shift
      ;;
    --workers)
      WORKERS="$2"
      shift; shift
      ;;
    --epochs)
      EPOCHS="$2"
      shift; shift
      ;;
    --seed)
      SEED="$2"
      shift; shift
      ;;
    --img-list)
      IMG_LIST="$2"
      shift; shift
      ;;
    --mode)
      MODE="$2"
      shift; shift
      ;;
    *)
      echo "Unknown arg: $1"
      exit 1
      ;;
  esac
done

if [[ "$MODE" == "minimal" ]]; then
  IMG_LIST="224 256"
elif [[ "$MODE" == "full" ]]; then
  IMG_LIST="160 192 224 256"
else
  echo "[ERROR] Invalid --mode: ${MODE}. Use minimal or full."
  exit 1
fi

if [[ "${DATA_DIR}" != /* ]]; then
  DATA_DIR="${PROJECT_ROOT}/${DATA_DIR#./}"
fi

if [[ ! -d "${DATA_DIR}/train" || ! -d "${DATA_DIR}/val" ]]; then
  echo "[ERROR] Dataset path invalid: ${DATA_DIR}"
  echo "[ERROR] Required folders: ${DATA_DIR}/train and ${DATA_DIR}/val"
  exit 1
fi

cd "${PROJECT_ROOT}"

echo "[INFO] Mode: ${MODE}"
echo "[INFO] IMG_LIST: ${IMG_LIST}"

for img in ${IMG_LIST}; do
  run_name="b2_img${img}_bs${BATCH_SIZE}_w${WORKERS}"
  echo "[RUN] img_size=${img}, batch=${BATCH_SIZE}, workers=${WORKERS}, epochs=${EPOCHS}"
  python3 "${SCRIPT_DIR}/train_resnet18.py" \
    --dataset "${DATASET}" \
    --data-dir "${DATA_DIR}" \
    --epochs "${EPOCHS}" \
    --img-size "${img}" \
    --batch-size "${BATCH_SIZE}" \
    --workers "${WORKERS}" \
    --amp \
    --pin-memory \
    --seed "${SEED}" \
    --run-name "${run_name}"
done

echo "[DONE] Stage B Round 2 completed."
