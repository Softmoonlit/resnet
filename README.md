# ResNet18 Training Toolkit for Jetson Orin Nano 8G

This repository provides a practical training toolkit to execute the staged plan in `jetson_orin_nano_resnet18_training_plan.md`.

## 1. Project Layout

```text
resnet/
  |- data/
  |  |- cifar10/
  |  |- imagenet100/
  |- logs/
  |  |- runs/
  |  |- csv/
  |- checkpoints/
  |- scripts/
  |  |- train_resnet18.py
  |  |- benchmark_matrix.sh
  |  |- summarize_results.py
  |  |- run_stage_a.sh
  |- requirements.txt
  |- README.md
  |- jetson_orin_nano_resnet18_training_plan.md
```

## 2. Copy To Jetson

Copy this folder to your board, for example:

```bash
scp -r ./resnet <user>@<jetson_ip>:~
```

## 3. Environment Preparation on Jetson

```bash
cd ~/resnet
python3 -m pip install -U pip
python3 -m pip install -r requirements.txt
```

Use NVIDIA-provided Jetson-compatible `torch` and `torchvision` if pip install from source index is not suitable.

## 4. Set Performance Mode Before Training

```bash
sudo nvpmodel -m 0
sudo jetson_clocks
```

In a second terminal, collect telemetry:

```bash
tegrastats --interval 1000 | tee logs/tegrastats.log
```

If Tiny-ImageNet `val` is still in the original `val/images` layout, convert it once before training:

```bash
python3 scripts/fix_tiny_imagenet_val.py --data-dir ./data/tiny-imagenet
```

## 5. Stage A (Quick Validation)

```bash
bash scripts/run_stage_a.sh
```

Or run directly:

```bash
python3 scripts/train_resnet18.py \
  --dataset cifar10 \
  --data-dir ./data/cifar10 \
  --epochs 10 \
  --img-size 224 \
  --batch-size 32 \
  --workers 4 \
  --amp \
  --pin-memory \
  --seed 42
```

## 6. Stage B Batch Matrix Benchmark

```bash
bash scripts/benchmark_matrix.sh \
  --dataset imagenet100 \
  --data-dir ./data/imagenet100 \
  --img-size 224 \
  --workers 4
```

This runs AMP-only benchmark sweeps over batch sizes and writes a matrix CSV in `logs/csv`.

### Stage B Minimal Retest (Recommended)

```bash
bash scripts/run_stage_b_minimal_retest.sh \
  --dataset tiny-imagenet \
  --data-dir ./data/tiny-imagenet \
  --epochs 5
```

This performs the minimal set in one command:
- batch-size: 32, 48
- img-size: 224, 256
- workers: 4, 6

If you want broader coverage, switch individual scripts to full mode using `--mode full`.

### Stage B Round 2 (Image Size Sweep, One Command)

```bash
bash scripts/run_stage_b_round2.sh \
  --dataset tiny-imagenet \
  --data-dir ./data/tiny-imagenet \
  --batch-size 32 \
  --workers 4 \
  --epochs 5
```

### Stage B Round 3 (DataLoader Sweep, One Command)

```bash
bash scripts/run_stage_b_round3.sh \
  --dataset tiny-imagenet \
  --data-dir ./data/tiny-imagenet \
  --batch-size 32 \
  --img-size 224 \
  --epochs 5
```

## 7. Stage C Long Training Example

```bash
python3 scripts/train_resnet18.py \
  --dataset imagenet100 \
  --data-dir ./data/imagenet100 \
  --epochs 90 \
  --img-size 224 \
  --batch-size <best_batch> \
  --workers 4 \
  --amp \
  --pin-memory \
  --label-smoothing 0.1 \
  --warmup-epochs 5 \
  --save-every 5 \
  --seed 42
```

## 8. Output and Reporting

Training logs:
- TensorBoard: `logs/runs/<run_name>`
- CSV per run: `logs/csv/<run_name>.csv`
- Checkpoints: `checkpoints/<run_name>/best.pth`, `last.pth`

Aggregate CSVs into one ranking table:

```bash
python3 scripts/summarize_results.py --csv-dir logs/csv --output logs/csv/summary_report.csv
```

## 9. Main Script Arguments

`train_resnet18.py` supports:
- `--dataset` (`cifar10 | imagenet100 | tiny-imagenet`)
- `--data-dir`
- `--epochs`
- `--batch-size`
- `--img-size`
- `--workers`
- `--amp`
- `--grad-accum-steps`
- `--lr`, `--base-lr`, `--weight-decay`, `--momentum`
- `--label-smoothing`
- `--save-every`
- `--seed`
- `--warmup-epochs`
- `--pin-memory`, `--prefetch-factor`
- `--benchmark-only`, `--warmup-steps`, `--benchmark-steps`

Each epoch logs:
- `train_loss`
- `val_top1`
- `val_top5`
- `epoch_time`
- `images_per_sec`
- `max_memory_mb`
