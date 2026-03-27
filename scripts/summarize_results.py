#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
from statistics import mean


def load_epoch_csv(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def summarize_run(rows):
    if not rows:
        return None
    best_top1 = max(float(r["val_top1"]) for r in rows)
    best_top5 = max(float(r["val_top5"]) for r in rows)
    avg_ips = mean(float(r["images_per_sec"]) for r in rows)
    avg_epoch_time = mean(float(r["epoch_time"]) for r in rows)
    max_mem = max(float(r["max_memory_mb"]) for r in rows)
    last = rows[-1]
    return {
        "epochs": len(rows),
        "best_top1": best_top1,
        "best_top5": best_top5,
        "avg_images_per_sec": avg_ips,
        "avg_epoch_time": avg_epoch_time,
        "max_memory_mb": max_mem,
        "final_train_loss": float(last["train_loss"]),
    }


def main():
    parser = argparse.ArgumentParser(description="Summarize training CSV logs into a single ranking table")
    parser.add_argument("--csv-dir", type=str, default="logs/csv")
    parser.add_argument("--output", type=str, default="logs/csv/summary_report.csv")
    args = parser.parse_args()

    csv_dir = Path(args.csv_dir)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    candidates = sorted([p for p in csv_dir.glob("*.csv") if not p.name.startswith("summary_report")])

    summary_rows = []
    for path in candidates:
        rows = load_epoch_csv(path)
        if not rows:
            continue
        if "epoch" not in rows[0]:
            # Skip benchmark matrix files.
            continue

        stats = summarize_run(rows)
        if stats is None:
            continue

        summary_rows.append(
            {
                "run_file": path.name,
                "epochs": stats["epochs"],
                "best_top1": f"{stats['best_top1']:.4f}",
                "best_top5": f"{stats['best_top5']:.4f}",
                "avg_images_per_sec": f"{stats['avg_images_per_sec']:.4f}",
                "avg_epoch_time": f"{stats['avg_epoch_time']:.4f}",
                "max_memory_mb": f"{stats['max_memory_mb']:.2f}",
                "final_train_loss": f"{stats['final_train_loss']:.6f}",
            }
        )

    summary_rows.sort(key=lambda x: (float(x["best_top1"]), float(x["avg_images_per_sec"])), reverse=True)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "rank",
            "run_file",
            "epochs",
            "best_top1",
            "best_top5",
            "avg_images_per_sec",
            "avg_epoch_time",
            "max_memory_mb",
            "final_train_loss",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, row in enumerate(summary_rows, start=1):
            writer.writerow({"rank": idx, **row})

    print(f"Wrote summary to: {out_path}")
    print(f"Included runs: {len(summary_rows)}")


if __name__ == "__main__":
    main()
