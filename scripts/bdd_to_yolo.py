"""
Convert BDD100K detection JSON labels into YOLO-format .txt files
(one .txt per image, normalized xywh) plus separate CSVs for the
scene-level attributes (weather / time-of-day / scene) used by the
Weather-Condition Classification head, and lane/drivable-area masks
left as-is (already PNG masks in the BDD100K download).

Usage:
    python bdd_to_yolo.py --bdd_root /kaggle/working/data/bdd100k \
                           --out_root /kaggle/working/data/bdd100k_yolo
"""
import argparse
import json
import os
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

# BDD100K detection classes we care about -> our unified class ids
# (traffic light / traffic sign are localization-only classes here;
#  sign *type* comes from the GTSRB classifier downstream, traffic
#  light *state* is a separate attribute head, not a detection class)
CLASS_MAP = {
    "car": 0,
    "truck": 1,
    "bus": 2,
    "motorcycle": 3,
    "bicycle": 4,
    "rider": 5,       # cyclist/motorcyclist on the vehicle
    "person": 6,       # pedestrian
    "traffic light": 7,
    "traffic sign": 8,
    "train": 9,
}
NUM_CLASSES = len(CLASS_MAP)

# BDD100K also gives per-traffic-light attribute "trafficLightColor": red/green/yellow/none
TL_COLOR_MAP = {"red": 0, "yellow": 1, "green": 2, "none": -1}


def convert_split(bdd_root: Path, out_root: Path, split: str):
    label_file = bdd_root / "labels" / f"det_{split}.json"
    img_dir = bdd_root / "images" / "100k" / split
    lbl_out_dir = out_root / "labels" / split
    lbl_out_dir.mkdir(parents=True, exist_ok=True)

    with open(label_file) as f:
        data = json.load(f)

    scene_rows = []
    tl_rows = []

    for entry in tqdm(data, desc=f"Converting {split}"):
        name = entry["name"]
        img_path = img_dir / name
        if not img_path.exists():
            continue
        with Image.open(img_path) as im:
            w, h = im.size

        attrs = entry.get("attributes", {})
        scene_rows.append({
            "image": name,
            "weather": attrs.get("weather", "undefined"),
            "timeofday": attrs.get("timeofday", "undefined"),
            "scene": attrs.get("scene", "undefined"),
        })

        yolo_lines = []
        for obj in entry.get("labels", []):
            cat = obj.get("category")
            if cat not in CLASS_MAP or "box2d" not in obj:
                continue
            box = obj["box2d"]
            x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
            cx = (x1 + x2) / 2 / w
            cy = (y1 + y2) / 2 / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            cls_id = CLASS_MAP[cat]
            yolo_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            if cat == "traffic light":
                color = obj.get("attributes", {}).get("trafficLightColor", "none")
                tl_rows.append({
                    "image": name, "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "color": color, "color_id": TL_COLOR_MAP.get(color, -1),
                })

        out_txt = lbl_out_dir / (Path(name).stem + ".txt")
        out_txt.write_text("\n".join(yolo_lines))

    pd.DataFrame(scene_rows).to_csv(out_root / f"scene_attributes_{split}.csv", index=False)
    pd.DataFrame(tl_rows).to_csv(out_root / f"traffic_light_states_{split}.csv", index=False)
    print(f"[{split}] wrote {len(scene_rows)} label files, "
          f"{len(tl_rows)} traffic-light attribute rows")


def write_data_yaml(out_root: Path, bdd_root: Path):
    yaml_content = f"""# Ultralytics-style data config
path: {out_root}
train: {bdd_root}/images/100k/train
val: {bdd_root}/images/100k/val
names:
"""
    for name, idx in sorted(CLASS_MAP.items(), key=lambda kv: kv[1]):
        yaml_content += f"  {idx}: {name}\n"
    (out_root / "bdd100k.yaml").write_text(yaml_content)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bdd_root", required=True, type=Path)
    ap.add_argument("--out_root", required=True, type=Path)
    args = ap.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val"]:
        convert_split(args.bdd_root, args.out_root, split)
    write_data_yaml(args.out_root, args.bdd_root)
    print("Done. YOLO-format labels + data yaml written to", args.out_root)


if __name__ == "__main__":
    main()
