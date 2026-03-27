#!/usr/bin/env python3
import argparse
import csv
import math
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover
    SummaryWriter = None


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    val_top1: float
    val_top5: float
    epoch_time: float
    images_per_sec: float
    max_memory_mb: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ResNet18 on Jetson-friendly settings")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "imagenet100", "tiny-imagenet"])
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision training")
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=-1.0, help="If < 0, auto-scale from base-lr")
    parser.add_argument("--base-lr", type=float, default=0.1, help="Base LR for linear scaling")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--benchmark-only", action="store_true", help="Run warmup+benchmark steps and exit")
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--benchmark-steps", type=int, default=300)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = False
    cudnn.benchmark = True


def topk_accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1, 5)) -> Tuple[float, ...]:
    maxk = min(max(topk), output.size(1))
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    result = []
    for k in topk:
        k = min(k, output.size(1))
        correct_k = correct[:k].reshape(-1).float().sum(0)
        result.append((correct_k * (100.0 / batch_size)).item())
    return tuple(result)


def build_transforms(dataset: str, img_size: int):
    if dataset == "cifar10":
        train_tf = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(img_size, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
            ]
        )
        val_tf = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
            ]
        )
    else:
        train_tf = transforms.Compose(
            [
                transforms.RandomResizedCrop(img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
        val_tf = transforms.Compose(
            [
                transforms.Resize(int(img_size * 256 / 224)),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return train_tf, val_tf


def build_datasets(dataset: str, data_dir: str, img_size: int):
    train_tf, val_tf = build_transforms(dataset, img_size)

    if dataset == "cifar10":
        train_set = datasets.CIFAR10(root=data_dir, train=True, transform=train_tf, download=True)
        val_set = datasets.CIFAR10(root=data_dir, train=False, transform=val_tf, download=True)
        num_classes = 10
    else:
        train_root = Path(data_dir) / "train"
        val_root = Path(data_dir) / "val"
        if not train_root.exists() or not val_root.exists():
            raise FileNotFoundError(
                f"Expected ImageFolder structure in {data_dir}: train/ and val/ are required."
            )
        train_set = datasets.ImageFolder(root=str(train_root), transform=train_tf)
        val_set = datasets.ImageFolder(root=str(val_root), transform=val_tf)
        if set(train_set.classes) != set(val_set.classes):
            if len(val_set.classes) == 1 and val_set.classes[0] == "images":
                raise ValueError(
                    "Invalid Tiny-ImageNet val layout detected: val/ is being read as a single class 'images'. "
                    "Please reorganize val images into class subfolders using val_annotations.txt."
                )
            raise ValueError(
                "Train/val class mismatch in ImageFolder dataset. "
                f"train_classes={len(train_set.classes)}, val_classes={len(val_set.classes)}"
            )
        num_classes = len(train_set.classes)

    return train_set, val_set, num_classes


def build_dataloader(dataset, batch_size: int, workers: int, pin_memory: bool, prefetch_factor: int, shuffle: bool):
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": workers,
        "pin_memory": pin_memory,
        "drop_last": shuffle,
    }
    if workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(**kwargs)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp_enabled: bool,
    grad_accum_steps: int,
) -> Tuple[float, int]:
    model.train()
    running_loss = 0.0
    sample_count = 0

    optimizer.zero_grad(set_to_none=True)
    for step, (images, target) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
            output = model(images)
            loss = criterion(output, target) / grad_accum_steps

        scaler.scale(loss).backward()

        if step % grad_accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        running_loss += loss.item() * grad_accum_steps * images.size(0)
        sample_count += images.size(0)

    if len(loader) % grad_accum_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    return running_loss / max(sample_count, 1), sample_count


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float, int]:
    model.eval()
    top1_sum = 0.0
    top5_sum = 0.0
    total = 0

    for images, target in loader:
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        output = model(images)
        top1, top5 = topk_accuracy(output, target, topk=(1, 5))

        batch_size = images.size(0)
        top1_sum += top1 * batch_size
        top5_sum += top5 * batch_size
        total += batch_size

    if total == 0:
        return 0.0, 0.0, 0
    return top1_sum / total, top5_sum / total, total


@torch.no_grad()
def benchmark_steps(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    warmup_steps: int,
    benchmark_steps_count: int,
    amp_enabled: bool,
) -> Tuple[float, float]:
    model.train()
    criterion = nn.CrossEntropyLoss()

    consumed = 0
    step_iter = iter(loader)

    def next_batch():
        nonlocal step_iter
        try:
            return next(step_iter)
        except StopIteration:
            step_iter = iter(loader)
            return next(step_iter)

    for _ in range(max(warmup_steps, 0)):
        images, target = next_batch()
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
            output = model(images)
            _ = criterion(output, target)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.time()

    for _ in range(max(benchmark_steps_count, 1)):
        images, target = next_batch()
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
            output = model(images)
            _ = criterion(output, target)
        consumed += images.size(0)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.time() - t0
    ips = consumed / max(elapsed, 1e-8)
    return ips, elapsed


def build_scheduler(optimizer: optim.Optimizer, epochs: int, warmup_epochs: int):
    if epochs <= 0:
        return None

    def lr_lambda(epoch_idx: int):
        if warmup_epochs > 0 and epoch_idx < warmup_epochs:
            return float(epoch_idx + 1) / float(max(1, warmup_epochs))
        progress = (epoch_idx - warmup_epochs) / float(max(1, epochs - warmup_epochs))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best_top1: float,
    args: argparse.Namespace,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "best_top1": best_top1,
            "args": vars(args),
        },
        path,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(args.amp and device.type == "cuda")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"{args.dataset}_bs{args.batch_size}_img{args.img_size}_{timestamp}"

    log_root = Path(args.log_dir)
    csv_dir = log_root / "csv"
    tb_dir = log_root / "runs" / run_name
    csv_dir.mkdir(parents=True, exist_ok=True)

    csv_path = csv_dir / f"{run_name}.csv"
    ckpt_dir = Path(args.checkpoint_dir) / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_set, val_set, num_classes = build_datasets(args.dataset, args.data_dir, args.img_size)

    train_loader = build_dataloader(
        train_set,
        batch_size=args.batch_size,
        workers=args.workers,
        pin_memory=args.pin_memory,
        prefetch_factor=args.prefetch_factor,
        shuffle=True,
    )
    val_loader = build_dataloader(
        val_set,
        batch_size=args.batch_size,
        workers=args.workers,
        pin_memory=args.pin_memory,
        prefetch_factor=args.prefetch_factor,
        shuffle=False,
    )

    model = models.resnet18(weights=None, num_classes=num_classes).to(device)

    eff_batch = args.batch_size * max(1, args.grad_accum_steps)
    lr = args.lr if args.lr > 0 else args.base_lr * (eff_batch / 256.0)

    criterion = nn.CrossEntropyLoss(label_smoothing=max(args.label_smoothing, 0.0))
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = build_scheduler(optimizer, args.epochs, args.warmup_epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    writer = SummaryWriter(log_dir=str(tb_dir)) if SummaryWriter is not None else None

    print("=" * 80)
    print(f"Run name         : {run_name}")
    print(f"Device           : {device}")
    print(f"Dataset          : {args.dataset}")
    print(f"Num classes      : {num_classes}")
    print(f"Train samples    : {len(train_set)}")
    print(f"Val samples      : {len(val_set)}")
    print(f"Batch size       : {args.batch_size}")
    print(f"Grad accum steps : {args.grad_accum_steps}")
    print(f"Effective batch  : {eff_batch}")
    print(f"AMP enabled      : {amp_enabled}")
    print(f"Learning rate    : {lr:.6f}")
    print(f"CSV log          : {csv_path}")
    print(f"Checkpoint dir   : {ckpt_dir}")
    print("=" * 80)

    if args.benchmark_only:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        ips, elapsed = benchmark_steps(
            model,
            train_loader,
            device,
            warmup_steps=args.warmup_steps,
            benchmark_steps_count=args.benchmark_steps,
            amp_enabled=amp_enabled,
        )
        max_mem = 0.0
        if device.type == "cuda":
            max_mem = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        print(
            f"[BENCHMARK] warmup_steps={args.warmup_steps} benchmark_steps={args.benchmark_steps} "
            f"elapsed={elapsed:.2f}s images_per_sec={ips:.2f} max_memory_mb={max_mem:.2f}"
        )
        return

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow(
            [
                "epoch",
                "train_loss",
                "val_top1",
                "val_top5",
                "epoch_time",
                "images_per_sec",
                "max_memory_mb",
                "lr",
            ]
        )

        best_top1 = -1.0
        for epoch in range(1, args.epochs + 1):
            epoch_start = time.time()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

            train_loss, train_seen = train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                scaler,
                device,
                amp_enabled,
                args.grad_accum_steps,
            )

            val_top1, val_top5, _ = validate(model, val_loader, device)

            epoch_time = time.time() - epoch_start
            images_per_sec = train_seen / max(epoch_time, 1e-8)
            max_memory_mb = (
                torch.cuda.max_memory_allocated(device) / (1024 * 1024) if device.type == "cuda" else 0.0
            )

            current_lr = optimizer.param_groups[0]["lr"]
            metrics = EpochMetrics(
                epoch=epoch,
                train_loss=train_loss,
                val_top1=val_top1,
                val_top5=val_top5,
                epoch_time=epoch_time,
                images_per_sec=images_per_sec,
                max_memory_mb=max_memory_mb,
            )

            writer_csv.writerow(
                [
                    metrics.epoch,
                    f"{metrics.train_loss:.6f}",
                    f"{metrics.val_top1:.4f}",
                    f"{metrics.val_top5:.4f}",
                    f"{metrics.epoch_time:.4f}",
                    f"{metrics.images_per_sec:.4f}",
                    f"{metrics.max_memory_mb:.2f}",
                    f"{current_lr:.8f}",
                ]
            )
            f.flush()

            if writer is not None:
                writer.add_scalar("train/loss", metrics.train_loss, epoch)
                writer.add_scalar("val/top1", metrics.val_top1, epoch)
                writer.add_scalar("val/top5", metrics.val_top5, epoch)
                writer.add_scalar("perf/images_per_sec", metrics.images_per_sec, epoch)
                writer.add_scalar("perf/epoch_time", metrics.epoch_time, epoch)
                writer.add_scalar("perf/max_memory_mb", metrics.max_memory_mb, epoch)
                writer.add_scalar("train/lr", current_lr, epoch)

            print(
                f"Epoch [{epoch:03d}/{args.epochs:03d}] "
                f"loss={metrics.train_loss:.4f} "
                f"top1={metrics.val_top1:.2f} top5={metrics.val_top5:.2f} "
                f"time={metrics.epoch_time:.2f}s ips={metrics.images_per_sec:.2f} "
                f"max_mem={metrics.max_memory_mb:.1f}MB lr={current_lr:.6f}"
            )

            if math.isnan(metrics.train_loss):
                raise RuntimeError("train_loss is NaN. Stop run to avoid invalid results.")

            if metrics.val_top1 > best_top1:
                best_top1 = metrics.val_top1
                save_checkpoint(ckpt_dir / "best.pth", model, optimizer, scaler, epoch, best_top1, args)

            if args.save_every > 0 and epoch % args.save_every == 0:
                save_checkpoint(ckpt_dir / f"epoch_{epoch:03d}.pth", model, optimizer, scaler, epoch, best_top1, args)

            if scheduler is not None:
                scheduler.step()

        save_checkpoint(ckpt_dir / "last.pth", model, optimizer, scaler, args.epochs, best_top1, args)
        print(f"Training finished. Best top1={best_top1:.2f}")

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
