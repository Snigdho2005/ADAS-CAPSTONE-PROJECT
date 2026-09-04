"""
Convert Waymo Open Dataset (camera images + 2D box labels) to unified
YOLO format.

HEAVIER DEPENDENCY THAN THE OTHER CONVERTERS: Waymo ships as TFRecord
files and needs the `waymo-open-dataset` pip package (TensorFlow-based)
just to parse the protobufs — this is the main integration cost
flagged in the earlier discussion. Install with:

    pip install waymo-open-dataset-tf-2-11-0  # match the TF version note in Waymo's docs

Usage:
    python scripts/convert_waymo_to_yolo.py \
        --waymo_tfrecord_dir /kaggle/working/data/waymo/training \
        --out_root /kaggle/working/data/unified_yolo/waymo \
        --max_segments 20   # Waymo is huge; cap how many .tfrecord segments to process

Given the scope tradeoff discussed earlier, consider running this with
a small --max_segments first (or skipping Waymo entirely) rather than
processing the full ~1TB training set on a Kaggle/Colab budget.
"""
import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.unified_classes import WAYMO_TO_UNIFIED, map_class

try:
    import tensorflow as tf
    from waymo_open_dataset import dataset_pb2 as open_dataset
    from waymo_open_dataset.utils import frame_utils
except ImportError:
    tf = None


WAYMO_TYPE_NAMES = {
    1: "TYPE_VEHICLE", 2: "TYPE_PEDESTRIAN", 3: "TYPE_SIGN", 4: "TYPE_CYCLIST",
}
FRONT_CAMERA_NAME = 1  # open_dataset.CameraName.FRONT


def convert(tfrecord_dir: Path, out_root: Path, max_segments: int, split: str):
    if tf is None:
        raise ImportError(
            "waymo-open-dataset / tensorflow not installed. This converter is "
            "optional given the scope discussion — install with "
            "`pip install waymo-open-dataset-tf-2-11-0` or skip Waymo."
        )

    img_out = out_root / "images" / split
    lbl_out = out_root / "labels" / split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    tfrecord_files = sorted(tfrecord_dir.glob("*.tfrecord"))[:max_segments]
    n_frames = 0

    for seg_path in tqdm(tfrecord_files, desc="Waymo segments"):
        dataset = tf.data.TFRecordDataset(str(seg_path), compression_type="")
        for frame_idx, data in enumerate(dataset):
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))

            front_image = next(
                (im for im in frame.images if im.name == FRONT_CAMERA_NAME), None
            )
            if front_image is None:
                continue

            img_array = tf.image.decode_jpeg(front_image.image).numpy()
            h, w = img_array.shape[:2]
            stem = f"{seg_path.stem}_{frame_idx:04d}"

            import cv2
            cv2.imwrite(str(img_out / f"{stem}.jpg"), cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))

            front_labels = next(
                (cl for cl in frame.camera_labels if cl.name == FRONT_CAMERA_NAME), None
            )
            lines = []
            if front_labels is not None:
                for lbl in front_labels.labels:
                    type_name = WAYMO_TYPE_NAMES.get(lbl.type)
                    if type_name is None:
                        continue
                    cls_id = map_class(type_name, WAYMO_TO_UNIFIED)
                    if cls_id is None:
                        continue
                    box = lbl.box
                    cx, cy = box.center_x / w, box.center_y / h
                    bw, bh = box.length / w, box.width / h
                    lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            (lbl_out / f"{stem}.txt").write_text("\n".join(lines))
            n_frames += 1

    print(f"Waymo converted: {n_frames} frames from {len(tfrecord_files)} segments")
    print("NOTE: Waymo boxes are TYPE_VEHICLE (undifferentiated car/truck/bus) "
          "and TYPE_SIGN (no light-vs-sign split) — coarser than your other "
          "sources for those classes, expected given Waymo's own taxonomy.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--waymo_tfrecord_dir", required=True, type=Path)
    ap.add_argument("--out_root", required=True, type=Path)
    ap.add_argument("--max_segments", type=int, default=20,
                     help="cap the number of .tfrecord segments processed — Waymo is huge")
    ap.add_argument("--split", default="train", choices=["train", "val"])
    args = ap.parse_args()
    convert(args.waymo_tfrecord_dir, args.out_root, args.max_segments, args.split)


if __name__ == "__main__":
    main()
