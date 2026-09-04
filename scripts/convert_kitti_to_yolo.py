"""
Convert KITTI 2D object detection labels into Phase 1 YOLO format
(car / van_truck / tram / pedestrian / cyclist — see
configs/phase1_classes.py). KITTI's raw layout: training/image_2/*.png
+ training/label_2/*.txt, one line per object:
    type truncated occluded alpha x1 y1 x2 y2 h w l x y z ry

Usage:
    python scripts/convert_kitti_to_yolo.py \
        --kitti_root /kaggle/working/data/kitti \
        --out_root /kaggle/working/data/phase1_yolo/kitti \
        --val_fraction 0.1
"""
import argparse
import random
from pathlib import Path

from PIL import Image
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.phase1_classes import map_kitti_class


def convert(kitti_root: Path, out_root: Path, val_fraction: float, seed: int = 0):
    img_dir = kitti_root / "training" / "image_2"
    lbl_dir = kitti_root / "training" / "label_2"

    image_files = sorted(img_dir.glob("*.png"))
    random.Random(seed).shuffle(image_files)
    n_val = int(len(image_files) * val_fraction)
    split_map = {f: ("val" if i < n_val else "train") for i, f in enumerate(image_files)}

    counts = {"train": 0, "val": 0}
    for img_path in tqdm(image_files, desc="KITTI"):
        split = split_map[img_path]
        stem = img_path.stem
        lbl_path = lbl_dir / f"{stem}.txt"
        if not lbl_path.exists():
            continue

        with Image.open(img_path) as im:
            w, h = im.size

        lines = []
        for row in lbl_path.read_text().strip().splitlines():
            parts = row.split()
            cls_name = parts[0]
            cls_id = map_kitti_class(cls_name)
            if cls_id is None:
                continue
            x1, y1, x2, y2 = map(float, parts[4:8])
            cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
            bw, bh = (x2 - x1) / w, (y2 - y1) / h
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        out_dir = out_root / "labels" / split
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{stem}.txt").write_text("\n".join(lines))
        counts[split] += 1

    print(f"KITTI converted: train={counts['train']} val={counts['val']}")
    print("NOTE: KITTI has no traffic-light/sign boxes and no train/bus "
          "distinction (Van/Truck both -> 'truck') — expected, not a bug.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitti_root", required=True, type=Path)
    ap.add_argument("--out_root", required=True, type=Path)
    ap.add_argument("--val_fraction", type=float, default=0.1)
    args = ap.parse_args()
    convert(args.kitti_root, args.out_root, args.val_fraction)


if __name__ == "__main__":
    main()
