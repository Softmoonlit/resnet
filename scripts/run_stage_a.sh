#!/usr/bin/env bash
set -euo pipefail

# Stage A quick validation on CIFAR-10
# Usage:
#   bash scripts/run_stage_a.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

python3 "${SCRIPT_DIR}/train_resnet18.py" \
  --dataset cifar10 \
  --data-dir "${PROJECT_ROOT}/data/cifar10" \
  --epochs 10 \
  --img-size 224 \
  --batch-size 32 \
  --workers 4 \
  --amp \
  --pin-memory \
  --seed 42
