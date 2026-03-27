#!/usr/bin/env bash
set -euo pipefail

# Stage B minimal retest orchestrator (AMP only)
# Runs three minimal checks in sequence:
# 1) Batch-size retest: 32 and 48
# 2) Image-size retest: 224 and 256
# 3) Workers retest: 4 and 6
#
# Usage:
#   bash scripts/run_stage_b_minimal_retest.sh --dataset tiny-imagenet --data-dir ./data/tiny-imagenet --epochs 5

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATASET="tiny-imagenet"
DATA_DIR="${PROJECT_ROOT}/data/tiny-imagenet"
BATCH_SIZE="32"
WORKERS="4"
IMG_SIZE="224"
EPOCHS="5"
SEED="42"

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
    *)
      echo "Unknown arg: $1"
      exit 1
      ;;
  esac
done

if [[ "${DATA_DIR}" != /* ]]; then
  DATA_DIR="${PROJECT_ROOT}/${DATA_DIR#./}"
fi

if [[ ! -d "${DATA_DIR}/train" || ! -d "${DATA_DIR}/val" ]]; then
  echo "[ERROR] Dataset path invalid: ${DATA_DIR}"
  echo "[ERROR] Required folders: ${DATA_DIR}/train and ${DATA_DIR}/val"
  exit 1
fi

cd "${PROJECT_ROOT}"

echo "[STEP 1/3] Batch-size minimal retest (32,48)"
bash "${SCRIPT_DIR}/benchmark_matrix.sh" \
  --mode minimal \
  --dataset "${DATASET}" \
  --data-dir "${DATA_DIR}" \
  --img-size "${IMG_SIZE}" \
  --workers "${WORKERS}" \
  --seed "${SEED}" \
  --batch-list "32 48"

echo "[STEP 2/3] Image-size minimal retest (224,256)"
bash "${SCRIPT_DIR}/run_stage_b_round2.sh" \
  --mode minimal \
  --dataset "${DATASET}" \
  --data-dir "${DATA_DIR}" \
  --batch-size "${BATCH_SIZE}" \
  --workers "${WORKERS}" \
  --epochs "${EPOCHS}" \
  --seed "${SEED}" \
  --img-list "224 256"

echo "[STEP 3/3] Workers minimal retest (4,6)"
bash "${SCRIPT_DIR}/run_stage_b_round3.sh" \
  --mode minimal \
  --dataset "${DATASET}" \
  --data-dir "${DATA_DIR}" \
  --batch-size "${BATCH_SIZE}" \
  --img-size "${IMG_SIZE}" \
  --epochs "${EPOCHS}" \
  --seed "${SEED}" \
  --workers-list "4 6"

echo "[DONE] Stage B minimal retest finished."
