"""
Convert GTSDB (German Traffic Sign Detection Benchmark) into YOLO
format for sign LOCATION only — every sign class collapses to one
"traffic_sign" id (Phase 1 scope). Sign TYPE classification is a
separate Phase 2 step using GTSRB on the cropped regions.

GTSDB's raw layout: a folder of .ppm images + a single gt.txt with
lines like:
    00000.ppm;774;411;815;446;11

    (filename;x1;y1;x2;y2;class_id)

Usage:
    python scripts/convert_gtsdb_to_yolo.py \
        --gtsdb_root /kaggle/working/data/gtsdb \
        --out_root /kaggle/working/data/phase1_yolo/gtsdb \
        --val_fraction 0.15
"""
import argparse
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.phase1_classes import GTSDB_SIGN_ID


def convert(gtsdb_root: Path, out_root: Path, val_fraction: float, seed: int = 0):
    gt_file = gtsdb_root / "gt.txt"
    img_dir = gtsdb_root  # GTSDB usually keeps images alongside gt.txt; adjust if yours nests them

    boxes_by_image = defaultdict(list)
    for line in gt_file.read_text().strip().splitlines():
        fname, x1, y1, x2, y2, _cls_id = line.split(";")
        boxes_by_image[fname].append((float(x1), float(y1), float(x2), float(y2)))

    image_names = sorted(boxes_by_image.keys())
    # GTSDB also has plenty of images with NO signs at all (good negatives) —
    # include those too, not just images with a gt.txt entry
    all_ppm = sorted(p.name for p in img_dir.glob("*.ppm"))
    for name in all_ppm:
        if name not in boxes_by_image:
            boxes_by_image[name] = []
    image_names = sorted(boxes_by_image.keys())

    random.Random(seed).shuffle(image_names)
    n_val = int(len(image_names) * val_fraction)
    split_map = {name: ("val" if i < n_val else "train") for i, name in enumerate(image_names)}

    counts = {"train": 0, "val": 0}
    for fname in tqdm(image_names, desc="GTSDB"):
        split = split_map[fname]
        img_path = img_dir / fname
        if not img_path.exists():
            continue
        with Image.open(img_path) as im:
            w, h = im.size

        lines = []
        for (x1, y1, x2, y2) in boxes_by_image[fname]:
            cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
            bw, bh = (x2 - x1) / w, (y2 - y1) / h
            lines.append(f"{GTSDB_SIGN_ID} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        out_dir = out_root / "labels" / split
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(fname).stem
        (out_dir / f"{stem}.txt").write_text("\n".join(lines))
        counts[split] += 1

    print(f"GTSDB converted: train={counts['train']} val={counts['val']} "
          f"(images with zero signs are kept as negatives, not dropped)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gtsdb_root", required=True, type=Path)
    ap.add_argument("--out_root", required=True, type=Path)
    ap.add_argument("--val_fraction", type=float, default=0.15)
    args = ap.parse_args()
    convert(args.gtsdb_root, args.out_root, args.val_fraction)


if __name__ == "__main__":
    main()
