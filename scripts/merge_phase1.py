"""
Phase 1 merge: combines converted KITTI (object detection) and GTSDB
(sign location) into one training set via symlinks. Much simpler than
scripts/merge_datasets.py (which handles IDD/DAWN/Waymo too) — kept
separate on purpose so Phase 1 has a minimal, easy-to-explain pipeline.

Usage:
    python scripts/merge_phase1.py \
        --phase1_root /kaggle/working/data/phase1_yolo \
        --kitti_images /kaggle/working/data/kitti/training/image_2 \
        --gtsdb_images /kaggle/working/data/gtsdb \
        --out_name merged
"""
import argparse
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.phase1_classes import PHASE1_CLASSES


import shutil


def link_or_copy(src: Path, dst: Path):
    if dst.exists():
        return
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copyfile(src.resolve(), dst)


def merge(phase1_root: Path, kitti_images: Path, gtsdb_images: Path, out_name: str, include_sources: list = None):
    out_root = phase1_root / out_name
    include_sources = include_sources or ["kitti", "gtsdb"]
    # try multiple extensions per source since the actual file type can
    # vary depending on which converter produced the images (e.g. the
    # NDJSON-based KITTI converter downloads .jpg, while raw KITTI
    # PNGs use .png)
    all_sources = {
        "kitti": (phase1_root / "kitti" / "labels", kitti_images, [".png", ".jpg", ".jpeg"]),
        "gtsdb": (phase1_root / "gtsdb" / "labels", gtsdb_images, [".ppm", ".png", ".jpg"]),
    }
    sources = {k: v for k, v in all_sources.items() if k in include_sources}

    for split in ["train", "val"]:
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)

        for source_name, (lbl_root, img_root, exts) in sources.items():
            lbl_dir = lbl_root / split
            if not lbl_dir.exists():
                print(f"skip {source_name}/{split}: {lbl_dir} not found "
                      f"(did you run convert_{source_name}_to_yolo.py?)")
                continue

            n_linked, n_missing = 0, 0
            for txt_path in lbl_dir.glob("*.txt"):
                stem = txt_path.stem
                img_path = None
                for ext in exts:
                    candidate = img_root / f"{stem}{ext}"
                    if candidate.exists():
                        img_path = candidate
                        break
                if img_path is None:
                    n_missing += 1
                    continue

                new_stem = f"{source_name}__{stem}"
                img_link = out_root / "images" / split / f"{new_stem}{img_path.suffix}"
                lbl_link = out_root / "labels" / split / f"{new_stem}.txt"
                link_or_copy(img_path, img_link)
                link_or_copy(txt_path, lbl_link)
                n_linked += 1

            print(f"{source_name}/{split}: linked {n_linked}, {n_missing} missing images")

    yaml_lines = [f"path: {out_root}", "train: images/train", "val: images/val", "names:"]
    for idx, name in sorted(PHASE1_CLASSES.items()):
        yaml_lines.append(f"  {idx}: {name}")
    (out_root / "data.yaml").write_text("\n".join(yaml_lines))
    print(f"\nMerged Phase 1 dataset written to {out_root}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase1_root", required=True, type=Path,
                     help="parent dir containing kitti/ and gtsdb/ from the individual converters")
    ap.add_argument("--kitti_images", required=True, type=Path)
    ap.add_argument("--gtsdb_images", required=True, type=Path)
    ap.add_argument("--sources", nargs="+", default=["kitti", "gtsdb"], choices=["kitti", "gtsdb"],
                     help="sources to include in the merged set (e.g. --sources kitti)")
    ap.add_argument("--out_name", default="merged")
    args = ap.parse_args()
    merge(args.phase1_root, args.kitti_images, args.gtsdb_images, args.out_name, include_sources=args.sources)


if __name__ == "__main__":
    main()
