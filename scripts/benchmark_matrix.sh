#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/benchmark_matrix.sh --dataset imagenet100 --data-dir ./data/imagenet100
#   bash scripts/benchmark_matrix.sh --mode full --dataset tiny-imagenet --data-dir ./data/tiny-imagenet

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATASET="imagenet100"
DATA_DIR="${PROJECT_ROOT}/data/imagenet100"
IMG_SIZE="224"
WORKERS="4"
SEED="42"
WARMUP_STEPS="100"
BENCHMARK_STEPS="300"
BATCH_LIST="8 16 24 32 48 64"
OUT_DIR="logs/csv"
MODE="minimal"
USER_BATCH_LIST=""

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
    --img-size)
      IMG_SIZE="$2"
      shift; shift
      ;;
    --workers)
      WORKERS="$2"
      shift; shift
      ;;
    --seed)
      SEED="$2"
      shift; shift
      ;;
    --warmup-steps)
      WARMUP_STEPS="$2"
      shift; shift
      ;;
    --benchmark-steps)
      BENCHMARK_STEPS="$2"
      shift; shift
      ;;
    --batch-list)
      BATCH_LIST="$2"
      USER_BATCH_LIST="$2"
      shift; shift
      ;;
    --out-dir)
      OUT_DIR="$2"
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

if [[ -z "$USER_BATCH_LIST" && "$MODE" == "minimal" ]]; then
  BATCH_LIST="32 48"
elif [[ "$MODE" == "full" ]]; then
  :
else
  echo "[ERROR] Invalid --mode: ${MODE}. Use minimal or full."
  exit 1
fi

# Resolve relative paths against project root so the script works from any cwd.
if [[ "${DATA_DIR}" != /* ]]; then
  DATA_DIR="${PROJECT_ROOT}/${DATA_DIR#./}"
fi

if [[ "${OUT_DIR}" != /* ]]; then
  OUT_DIR="${PROJECT_ROOT}/${OUT_DIR#./}"
fi

if [[ ! -d "${DATA_DIR}/train" || ! -d "${DATA_DIR}/val" ]]; then
  echo "[ERROR] Dataset path invalid: ${DATA_DIR}"
  echo "[ERROR] Required folders: ${DATA_DIR}/train and ${DATA_DIR}/val"
  exit 1
fi

cd "${PROJECT_ROOT}"

mkdir -p "$OUT_DIR"
TS=$(date +"%Y%m%d_%H%M%S")
SUMMARY_FILE="$OUT_DIR/benchmark_matrix_${DATASET}_${TS}.csv"

echo "timestamp,dataset,img_size,batch_size,amp,workers,status,images_per_sec,max_memory_mb,raw" > "$SUMMARY_FILE"

echo "[INFO] Summary: $SUMMARY_FILE"
echo "[INFO] Mode: $MODE"
echo "[INFO] BATCH_LIST: $BATCH_LIST"

run_case() {
  local batch_size="$1"
  local amp_name="amp"

  local run_name="bench_${DATASET}_${amp_name}_bs${batch_size}_img${IMG_SIZE}_${TS}"
  local cmd="python3 \"${SCRIPT_DIR}/train_resnet18.py\" \
    --dataset ${DATASET} \
    --data-dir ${DATA_DIR} \
    --img-size ${IMG_SIZE} \
    --batch-size ${batch_size} \
    --workers ${WORKERS} \
    --seed ${SEED} \
    --benchmark-only \
    --warmup-steps ${WARMUP_STEPS} \
    --benchmark-steps ${BENCHMARK_STEPS} \
    --run-name ${run_name} \
    --amp"

  echo "[RUN] amp=${amp_name} batch=${batch_size}"

  set +e
  output=$(eval "$cmd" 2>&1)
  ret=$?
  set -e

  if [[ $ret -ne 0 ]]; then
    status="fail"
    ips=""
    mem=""
  else
    status="ok"
    ips=$(echo "$output" | sed -n 's/.*images_per_sec=\([0-9.]*\).*/\1/p' | tail -n 1)
    mem=$(echo "$output" | sed -n 's/.*max_memory_mb=\([0-9.]*\).*/\1/p' | tail -n 1)
  fi

  one_line=$(echo "$output" | tr '\n' ' ' | tr ',' ';' | sed 's/\"//g')
  echo "${TS},${DATASET},${IMG_SIZE},${batch_size},${amp_name},${WORKERS},${status},${ips},${mem},\"${one_line}\"" >> "$SUMMARY_FILE"

  if [[ "$status" == "fail" ]]; then
    echo "[WARN] Failed at amp=${amp_name}, batch=${batch_size}."
    echo "[WARN] Last error lines:"
    echo "$output" | tail -n 8
  fi
}

for bs in $BATCH_LIST; do
  run_case "$bs"
done

echo "[DONE] Benchmark matrix completed."
