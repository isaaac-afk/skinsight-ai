"""
Prepare ISIC 2019 data for SkinSight AI.

Reads ISIC_2019_Training_GroundTruth.csv (one-hot diagnosis columns) and the
folder of training JPEGs, then sorts images into the layout the training
script expects:

    data/isic/train/<class>/*.jpg
    data/isic/val/<class>/*.jpg

Optionally caps each class to a balanced subset (recommended for CPU training).

ISIC 2019 ground-truth columns:
    image, MEL, NV, BCC, AK, BKL, DF, VASC, SCC, UNK

  MEL  = Melanoma
  NV   = Melanocytic nevus
  BCC  = Basal cell carcinoma
  AK   = Actinic keratosis
  BKL  = Benign keratosis
  DF   = Dermatofibroma
  VASC = Vascular lesion
  SCC  = Squamous cell carcinoma
  UNK  = Unknown / none of the above (skipped)

Usage (defaults match the recommended Windows layout):
    python prepare_isic2019.py \
        --csv  data/ISIC_2019_Training_GroundTruth.csv \
        --images data/ISIC_2019_Training_Input \
        --out data/isic \
        --per-class 1500 \
        --val-frac 0.2
"""

import argparse
import csv
import random
import shutil
from pathlib import Path

CLASS_COLUMNS = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
# UNK is intentionally excluded.


def find_image(images_dir: Path, image_id: str) -> Path | None:
    """ISIC ids in the CSV have no extension; files are .jpg (sometimes .JPG)."""
    for ext in (".jpg", ".jpeg", ".JPG", ".JPEG", ".png"):
        candidate = images_dir / f"{image_id}{ext}"
        if candidate.exists():
            return candidate
    return None


def parse_groundtruth(csv_path: Path):
    """Yield (image_id, class_name) for each row whose label is a known class."""
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = row["image"]
            # one-hot: find the column set to 1.0
            label = None
            for cls in CLASS_COLUMNS:
                val = row.get(cls, "0")
                try:
                    if float(val) == 1.0:
                        label = cls
                        break
                except ValueError:
                    continue
            if label is not None:
                yield image_id, label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/ISIC_2019_Training_GroundTruth.csv")
    ap.add_argument("--images", default="data/ISIC_2019_Training_Input")
    ap.add_argument("--out", default="data/isic")
    ap.add_argument("--per-class", type=int, default=1500,
                    help="Max images per class (0 = use all). Balances the set.")
    ap.add_argument("--val-frac", type=float, default=0.2,
                    help="Fraction of each class held out for validation.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--copy", action="store_true",
                    help="Copy files instead of the default (also copy). Kept for clarity.")
    args = ap.parse_args()

    random.seed(args.seed)

    csv_path = Path(args.csv)
    images_dir = Path(args.images)
    out_dir = Path(args.out)

    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")
    if not images_dir.exists():
        raise SystemExit(f"Images folder not found: {images_dir}")

    # Group image ids by class.
    by_class: dict[str, list[str]] = {c: [] for c in CLASS_COLUMNS}
    missing = 0
    for image_id, label in parse_groundtruth(csv_path):
        src = find_image(images_dir, image_id)
        if src is None:
            missing += 1
            continue
        by_class[label].append(image_id)

    if missing:
        print(f"Warning: {missing} labeled images not found on disk (skipped).")

    print("Available per class:")
    for c in CLASS_COLUMNS:
        print(f"  {c:<5} {len(by_class[c])}")

    # Cap + split + copy.
    total_train = total_val = 0
    for cls, ids in by_class.items():
        if not ids:
            print(f"  (skipping {cls}: no images)")
            continue
        random.shuffle(ids)
        if args.per_class > 0:
            ids = ids[: args.per_class]

        n_val = max(1, int(len(ids) * args.val_frac))
        val_ids = ids[:n_val]
        train_ids = ids[n_val:]

        for split, split_ids in (("train", train_ids), ("val", val_ids)):
            dest_dir = out_dir / split / cls
            dest_dir.mkdir(parents=True, exist_ok=True)
            for image_id in split_ids:
                src = find_image(images_dir, image_id)
                shutil.copy2(src, dest_dir / src.name)

        total_train += len(train_ids)
        total_val += len(val_ids)
        print(f"  {cls:<5} -> train {len(train_ids):>5} | val {len(val_ids):>5}")

    print(f"\nDone. Total: train {total_train} | val {total_val}")
    print(f"Output layout under: {out_dir.resolve()}")
    print("Now you can run:  python skinsight_model.py")


if __name__ == "__main__":
    main()
