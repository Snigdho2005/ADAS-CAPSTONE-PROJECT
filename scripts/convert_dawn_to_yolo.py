"""
Convert DAWN (adverse weather: Fog, Rain, Sand, Snow subfolders, each
PASCAL VOC XML annotated) into unified YOLO format, plus a per-image
weather-condition CSV (reuses the same schema as bdd_to_yolo.py's
scene_attributes csv so your WeatherHead training code doesn't need
separate logic for this source).

DAWN is deliberately kept as a held-out ROBUSTNESS EVAL set rather
than blended into training by default — the point of your "Scene
Robustness Across Conditions" feature is measuring performance on
weather buckets, which is easiest to defend if at least some of DAWN
is never trained on. Use --split_for_training if you want some of it
in the training mix too.

Usage:
    python scripts/convert_dawn_to_yolo.py \
        --dawn_root /kaggle/working/data/DAWN \
        --out_root /kaggle/working/data/unified_yolo/dawn
"""
import argparse
import csv
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.unified_classes import DAWN_TO_UNIFIED, map_class

WEATHER_FOLDERS = {"Fog": "foggy", "Rain": "rainy", "Sand": "sandy", "Snow": "snowy"}


def convert(dawn_root: Path, out_root: Path, split_for_training: bool, val_fraction: float):
    lbl_dir = out_root / "labels" / ("train" if split_for_training else "eval_only")
    lbl_dir.mkdir(parents=True, exist_ok=True)
    scene_rows = []
    n_written = 0

    for folder_name, weather_label in WEATHER_FOLDERS.items():
        folder = dawn_root / folder_name
        if not folder.exists():
            print(f"WARNING: {folder} not found, skipping — check your DAWN folder names")
            continue

        # DAWN's typical layout: <Weather>/<Weather>/*.jpg + PASCAL_VOC/*.xml
        img_files = sorted(folder.rglob("*.jpg")) + sorted(folder.rglob("*.png"))
        anno_dir_candidates = list(folder.rglob("PASCAL_VOC"))
        anno_dir = anno_dir_candidates[0] if anno_dir_candidates else folder

        for img_path in tqdm(img_files, desc=f"DAWN/{folder_name}"):
            stem = img_path.stem
            xml_path = anno_dir / f"{stem}.xml"

            lines = []
            if xml_path.exists():
                tree = ET.parse(xml_path)
                root = tree.getroot()
                size = root.find("size")
                w = float(size.find("width").text)
                h = float(size.find("height").text)
                for obj in root.findall("object"):
                    cls_name = obj.find("name").text.strip().lower()
                    cls_id = map_class(cls_name, DAWN_TO_UNIFIED)
                    if cls_id is None:
                        continue
                    bnd = obj.find("bndbox")
                    x1, y1 = float(bnd.find("xmin").text), float(bnd.find("ymin").text)
                    x2, y2 = float(bnd.find("xmax").text), float(bnd.find("ymax").text)
                    cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
                    bw, bh = (x2 - x1) / w, (y2 - y1) / h
                    lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            else:
                # no box annotations for this image — still useful as a
                # weather-classification / robustness-eval sample even
                # without detection boxes
                pass

            (lbl_dir / f"{stem}.txt").write_text("\n".join(lines))
            scene_rows.append({"image": img_path.name, "weather": weather_label})
            n_written += 1

    csv_path = out_root / "scene_attributes.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "weather"])
        writer.writeheader()
        writer.writerows(scene_rows)

    print(f"DAWN converted: {n_written} images across {len(WEATHER_FOLDERS)} weather buckets")
    print(f"Scene attributes written to {csv_path}")
    if not split_for_training:
        print("Kept as eval-only (default) — pass --split_for_training to blend some into the training set.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dawn_root", required=True, type=Path)
    ap.add_argument("--out_root", required=True, type=Path)
    ap.add_argument("--split_for_training", action="store_true")
    ap.add_argument("--val_fraction", type=float, default=0.2)
    args = ap.parse_args()
    convert(args.dawn_root, args.out_root, args.split_for_training, args.val_fraction)


if __name__ == "__main__":
    main()
