"""
Convert an Ultralytics Platform KITTI NDJSON export into Phase 1 YOLO
format. This is a DIFFERENT source/format than raw KITTI —
convert_kitti_to_yolo.py assumes local PNG images + label_2/*.txt on
disk; this script instead:
  - reads one JSON object per line (first line is dataset metadata,
    the rest are per-image entries)
  - downloads each image from its (signed, expiring) CDN url
  - boxes are ALREADY normalized YOLO format [class, cx, cy, w, h] —
    no coordinate math needed, just class-id remapping
  - "split" field ("train"/"val") is already assigned, no need to
    do our own train/val random split like the raw-KITTI converter did

Ultralytics' 8-class KITTI taxonomy -> our Phase 1 6-class taxonomy:
    car -> car, van/truck -> van_truck, tram -> tram,
    pedestrian/person_sitting -> pedestrian, cyclist -> cyclist,
    misc -> dropped (no Phase 1 equivalent)

Usage:
    python scripts/convert_kitti_ndjson_to_yolo.py \
        --ndjson_path /kaggle/working/kitti.ndjson \
        --out_root /kaggle/working/data/phase1_yolo/kitti \
        --max_workers 16

The image download step is the slow part (7,481 images) — this uses a
thread pool since it's I/O-bound. Signed URLs expire eventually, so
don't sit on this file for weeks before running it.
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Ultralytics' KITTI class ids (from the ndjson's own "class_names" metadata)
# -> our Phase 1 class ids (configs/phase1_classes.py)
NDJSON_ID_TO_PHASE1 = {
    0: 0,   # car -> car
    1: 1,   # van -> van_truck
    2: 1,   # truck -> van_truck
    3: 3,   # pedestrian -> pedestrian
    4: 3,   # person_sitting -> pedestrian
    5: 4,   # cyclist -> cyclist
    6: 2,   # tram -> tram
    7: None,  # misc -> dropped, no Phase 1 equivalent
}


def download_one(entry, img_out_dir, timeout=20):
    stem = Path(entry["file"]).stem
    ext = Path(entry["file"]).suffix or ".jpg"
    img_path = img_out_dir / f"{stem}{ext}"
    if img_path.exists() and img_path.stat().st_size > 0:
        return stem, True  # already downloaded, skip re-fetching
    try:
        resp = requests.get(entry["url"], timeout=timeout)
        resp.raise_for_status()
        img_path.write_bytes(resp.content)
        return stem, True
    except Exception as e:
        return stem, False


def write_label(entry, lbl_out_dir):
    stem = Path(entry["file"]).stem
    lines = []
    for box in entry.get("annotations", {}).get("boxes", []):
        ndjson_cls, cx, cy, w, h = box
        phase1_cls = NDJSON_ID_TO_PHASE1.get(int(ndjson_cls))
        if phase1_cls is None:
            continue
        lines.append(f"{phase1_cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    (lbl_out_dir / f"{stem}.txt").write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ndjson_path", required=True, type=Path)
    ap.add_argument("--out_root", required=True, type=Path)
    ap.add_argument("--max_workers", type=int, default=16)
    args = ap.parse_args()

    entries = []
    with open(args.ndjson_path) as f:
        for line in f:
            d = json.loads(line)
            if d.get("type") == "image":
                entries.append(d)

    print(f"Loaded {len(entries)} image entries")

    for split in ["train", "val"]:
        (args.out_root / "labels" / split).mkdir(parents=True, exist_ok=True)
    img_out_dir = args.out_root / "images_all"
    img_out_dir.mkdir(parents=True, exist_ok=True)

    # write labels first (cheap, no network needed)
    for entry in tqdm(entries, desc="Writing labels"):
        split = entry["split"]
        write_label(entry, args.out_root / "labels" / split)

    # download images in parallel (the slow, network-bound part) — all
    # into ONE flat folder regardless of split, since merge_phase1.py
    # matches images to labels by filename stem using a single image
    # root, with the split determined by which labels/<split>/ folder
    # the stem's .txt lives in
    n_ok, n_failed = 0, 0
    failed_files = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {}
        for entry in entries:
            futures[pool.submit(download_one, entry, img_out_dir)] = entry["file"]

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Downloading images"):
            stem, ok = fut.result()
            if ok:
                n_ok += 1
            else:
                n_failed += 1
                failed_files.append(futures[fut])

    print(f"\nDone. Downloaded {n_ok} images, {n_failed} failed.")
    if failed_files:
        fail_log = args.out_root / "failed_downloads.txt"
        fail_log.write_text("\n".join(failed_files))
        print(f"Failed filenames written to {fail_log} — re-run this script to retry them "
              f"(already-downloaded images are skipped, not re-fetched).")
    print(f"\nImages written (flat, both splits) to {img_out_dir}")
    print(f"Labels written to {args.out_root}/labels/{{train,val}}")
    print(f"\nFor merge_phase1.py, pass:  --kitti_images {img_out_dir}")


if __name__ == "__main__":
    main()
