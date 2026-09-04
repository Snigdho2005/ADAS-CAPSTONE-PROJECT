"""
Merge the separately-converted IDD / KITTI / Waymo / DAWN datasets
(each already in unified-class YOLO format from their own convert_*
script) into one combined dataset directory, via symlinks (not copies
— these datasets are large, no need to duplicate the actual image
bytes).

Produces:
    unified_yolo/merged/images/train/<source>__<stem>.jpg  (symlinks)
    unified_yolo/merged/labels/train/<source>__<stem>.txt   (symlinks)
    unified_yolo/merged/data.yaml
    unified_yolo/merged/source_manifest.csv   (which source each image came from — useful for
                                                per-source breakdown in your results tables,
                                                e.g. "accuracy on IDD-only subset")

Usage:
    python scripts/merge_datasets.py \
        --unified_root /kaggle/working/data/unified_yolo \
        --include idd kitti dawn            # omit waymo unless you actually ran its converter
        --out_name merged
"""
import argparse
import csv
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.unified_classes import UNIFIED_CLASSES

# where each source's raw images live, relative to its own dataset root —
# adjust these if your directory layout differs from the converters' defaults
IMAGE_ROOT_HINTS = {
    "idd": "IDD_Detection/JPEGImages",
    "kitti": "kitti/training/image_2",
    "dawn": "DAWN",           # DAWN images stay nested per-weather-folder; script searches recursively
    "waymo": "unified_yolo/waymo/images",  # waymo converter already writes images into unified_yolo
}


import shutil


def link_or_copy(src: Path, dst: Path):
    if dst.exists():
        return
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copyfile(src.resolve(), dst)


def find_image(stem: str, source: str, raw_data_root: Path, unified_root: Path):
    """Best-effort image lookup across the different layouts each
    source uses. Falls back to a recursive search — slow for very
    large sources, but only runs once at merge time."""
    if source == "waymo":
        for split in ["train", "val"]:
            p = unified_root / "waymo" / "images" / split / f"{stem}.jpg"
            if p.exists():
                return p
        return None

    hint_dir = raw_data_root / IMAGE_ROOT_HINTS[source]
    for ext in [".jpg", ".png"]:
        direct = hint_dir / f"{stem}{ext}"
        if direct.exists():
            return direct
    matches = list(hint_dir.rglob(f"{stem}.jpg")) + list(hint_dir.rglob(f"{stem}.png"))
    return matches[0] if matches else None


def merge(unified_root: Path, raw_data_root: Path, include: list, out_name: str):
    out_root = unified_root / out_name
    manifest_rows = []

    for split in ["train", "val"]:
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)

        for source in include:
            lbl_dir = unified_root / source / "labels" / split
            if not lbl_dir.exists():
                print(f"skip {source}/{split}: {lbl_dir} not found (did you run its convert_* script?)")
                continue

            n_linked, n_missing_img = 0, 0
            for txt_path in lbl_dir.glob("*.txt"):
                stem = txt_path.stem
                img_path = find_image(stem, source, raw_data_root, unified_root)
                if img_path is None:
                    n_missing_img += 1
                    continue

                new_stem = f"{source}__{stem}"
                img_link = out_root / "images" / split / f"{new_stem}{img_path.suffix}"
                lbl_link = out_root / "labels" / split / f"{new_stem}.txt"

                link_or_copy(img_path, img_link)
                link_or_copy(txt_path, lbl_link)

                manifest_rows.append({"file": new_stem, "source": source, "split": split})
                n_linked += 1

            print(f"{source}/{split}: linked {n_linked}, {n_missing_img} missing images")

    with open(out_root / "source_manifest.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "source", "split"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    yaml_lines = [f"path: {out_root}", "train: images/train", "val: images/val", "names:"]
    for idx, name in sorted(UNIFIED_CLASSES.items()):
        yaml_lines.append(f"  {idx}: {name}")
    (out_root / "data.yaml").write_text("\n".join(yaml_lines))

    print(f"\nMerged dataset written to {out_root}")
    print(f"Sources included: {include}")
    print(f"Total linked samples: {len(manifest_rows)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unified_root", required=True, type=Path,
                     help="parent dir containing idd/, kitti/, dawn/, waymo/ from the individual converters")
    ap.add_argument("--raw_data_root", type=Path, default=None,
                     help="parent dir containing the original IDD_Detection/, kitti/, DAWN/ folders "
                          "(defaults to --unified_root's parent if not given)")
    ap.add_argument("--include", nargs="+", default=["idd", "kitti", "dawn"],
                     choices=["idd", "kitti", "dawn", "waymo"])
    ap.add_argument("--out_name", default="merged")
    args = ap.parse_args()

    raw_root = args.raw_data_root or args.unified_root.parent
    merge(args.unified_root, raw_root, args.include, args.out_name)


if __name__ == "__main__":
    main()
