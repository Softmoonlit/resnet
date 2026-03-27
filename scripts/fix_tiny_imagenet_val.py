#!/usr/bin/env python3
import argparse
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reorganize Tiny-ImageNet val set into ImageFolder format: val/<class>/<image>."
    )
    parser.add_argument("--data-dir", type=str, default="data/tiny-imagenet", help="Tiny-ImageNet root path")
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of moving (uses more disk but keeps the original val/images intact)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.data_dir)
    val_dir = root / "val"
    images_dir = val_dir / "images"
    ann_path = val_dir / "val_annotations.txt"

    if not val_dir.exists() or not images_dir.exists() or not ann_path.exists():
        raise FileNotFoundError(
            f"Expected Tiny-ImageNet val files under {val_dir}: images/ and val_annotations.txt are required."
        )

    # If class folders already exist and images folder is empty/missing, assume fixed.
    class_dirs = [p for p in val_dir.iterdir() if p.is_dir() and p.name != "images"]
    if class_dirs and not any(images_dir.iterdir()):
        print("val layout already looks fixed; nothing to do.")
        return

    moved = 0
    with ann_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            filename, wnid = parts[0], parts[1]
            src = images_dir / filename
            dst_dir = val_dir / wnid
            dst = dst_dir / filename
            dst_dir.mkdir(parents=True, exist_ok=True)

            if not src.exists():
                # Skip if already moved/copied in a previous run.
                if dst.exists():
                    continue
                raise FileNotFoundError(f"Missing validation image: {src}")

            if args.copy:
                shutil.copy2(src, dst)
            else:
                shutil.move(str(src), str(dst))
            moved += 1

    if not args.copy:
        # Keep filesystem clean once everything is moved.
        try:
            images_dir.rmdir()
        except OSError:
            pass

    print(f"Done. Processed images: {moved}")
    print("Now val should be readable as ImageFolder classes.")


if __name__ == "__main__":
    main()
