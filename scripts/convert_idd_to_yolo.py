"""
Convert IDD-Detection (IIIT-Hyderabad, https://idd.insaan.iiit.ac.in)
into unified YOLO-format labels. IDD's raw annotations are PASCAL VOC
XML per image, organized under IDD_Detection/{JPEGImages,Annotations}.

Usage:
    python scripts/convert_idd_to_yolo.py \
        --idd_root /kaggle/working/data/IDD_Detection \
        --out_root /kaggle/working/data/unified_yolo/idd

NOTE: IDD's exact folder layout has shifted across dataset releases —
verify Annotations/JPEGImages subfolder names against what you actually
downloaded and adjust IMG_DIR/ANNO_DIR below if they differ.
"""
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.unified_classes import IDD_TO_UNIFIED, map_class


def convert_split(idd_root: Path, out_root: Path, split: str):
    img_dir = idd_root / "JPEGImages" / split
    anno_dir = idd_root / "Annotations" / split
    lbl_out = out_root / "labels" / split
    lbl_out.mkdir(parents=True, exist_ok=True)

    xml_files = sorted(anno_dir.rglob("*.xml"))
    n_written, n_skipped = 0, 0

    for xml_path in tqdm(xml_files, desc=f"IDD {split}"):
        stem = xml_path.stem
        # IDD nests by scene/subfolder; mirror the same relative path for images
        rel = xml_path.relative_to(anno_dir).with_suffix(".jpg")
        img_path = img_dir / rel
        if not img_path.exists():
            n_skipped += 1
            continue

        tree = ET.parse(xml_path)
        root = tree.getroot()
        size = root.find("size")
        w = float(size.find("width").text)
        h = float(size.find("height").text)

        lines = []
        for obj in root.findall("object"):
            cls_name = obj.find("name").text.strip()
            cls_id = map_class(cls_name, IDD_TO_UNIFIED)
            if cls_id is None:
                continue
            bnd = obj.find("bndbox")
            x1, y1 = float(bnd.find("xmin").text), float(bnd.find("ymin").text)
            x2, y2 = float(bnd.find("xmax").text), float(bnd.find("ymax").text)
            cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
            bw, bh = (x2 - x1) / w, (y2 - y1) / h
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        out_txt = lbl_out / f"{stem}.txt"
        out_txt.write_text("\n".join(lines))
        n_written += 1

    print(f"[IDD {split}] wrote {n_written} label files, skipped {n_skipped} (missing image)")
    return n_written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idd_root", required=True, type=Path)
    ap.add_argument("--out_root", required=True, type=Path)
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    args = ap.parse_args()

    for split in args.splits:
        convert_split(args.idd_root, args.out_root, split)


if __name__ == "__main__":
    main()
