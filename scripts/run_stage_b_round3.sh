#!/usr/bin/env bash
set -euo pipefail

# Stage B Round 3: dataloader sweep (AMP only)
# Usage:
#   bash scripts/run_stage_b_round3.sh
#   bash scripts/run_stage_b_round3.sh --dataset tiny-imagenet --data-dir ./data/tiny-imagenet --batch-size 32 --img-size 224 --epochs 5
#   bash scripts/run_stage_b_round3.sh --mode full --dataset tiny-imagenet --data-dir ./data/tiny-imagenet

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATASET="tiny-imagenet"
DATA_DIR="${PROJECT_ROOT}/data/tiny-imagenet"
BATCH_SIZE="32"
IMG_SIZE="224"
EPOCHS="5"
SEED="42"
WORKERS_LIST="4 6"
PREFETCH_FACTOR="2"
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
    --img-size)
      IMG_SIZE="$2"
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
    --workers-list)
      WORKERS_LIST="$2"
      shift; shift
      ;;
    --prefetch-factor)
      PREFETCH_FACTOR="$2"
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
  WORKERS_LIST="4 6"
elif [[ "$MODE" == "full" ]]; then
  WORKERS_LIST="2 4 6"
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
echo "[INFO] WORKERS_LIST: ${WORKERS_LIST}"

for workers in ${WORKERS_LIST}; do
  run_name="b3_img${IMG_SIZE}_bs${BATCH_SIZE}_w${workers}"
  echo "[RUN] img_size=${IMG_SIZE}, batch=${BATCH_SIZE}, workers=${workers}, epochs=${EPOCHS}, prefetch=${PREFETCH_FACTOR}"
  python3 "${SCRIPT_DIR}/train_resnet18.py" \
    --dataset "${DATASET}" \
    --data-dir "${DATA_DIR}" \
    --epochs "${EPOCHS}" \
    --img-size "${IMG_SIZE}" \
    --batch-size "${BATCH_SIZE}" \
    --workers "${workers}" \
    --amp \
    --pin-memory \
    --prefetch-factor "${PREFETCH_FACTOR}" \
    --seed "${SEED}" \
    --run-name "${run_name}"
done

echo "[DONE] Stage B Round 3 completed."
