# ADAS Hybrid Detector — Data Pipeline (IDD + KITTI + DAWN)

Waymo is intentionally excluded (see project discussion — heavy TFRecord/
TensorFlow dependency, US-only imagery, not aligned with the Indian-
conditions focus). `scripts/convert_waymo_to_yolo.py` still exists if you
change your mind later, but it's not part of this workflow.

## 0. Download the raw datasets

| Dataset | Source | Notes |
|---|---|---|
| IDD-Detection | https://idd.insaan.iiit.ac.in (free registration) | Also grab IDD-Segmentation if using the drivable-area head |
| KITTI | https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d, or `kaggle datasets download -d klemenko/kitti-dataset` | Confirm the mirror includes `label_2/`, not just images |
| DAWN | https://data.mendeley.com/datasets/766ygrbt8y/3 | No registration needed |

Unzip so you have:
```
/kaggle/working/data/
├── IDD_Detection/{JPEGImages,Annotations}/...
├── kitti/training/{image_2,label_2}/...
└── DAWN/{Fog,Rain,Sand,Snow}/...
```

**Before running full conversions**: point each `convert_*_to_yolo.py` at
a small subset first, open 5-10 of the produced `.txt` files next to
their source images, and eyeball that the boxes land in sensible places.
Dataset folder layouts shift between releases and these converters
encode the commonly-documented structure, not something verified
against your specific download.

## 1. Convert each source to unified YOLO-format labels

```bash
python scripts/convert_idd_to_yolo.py \
    --idd_root /kaggle/working/data/IDD_Detection \
    --out_root /kaggle/working/data/unified_yolo/idd

python scripts/convert_kitti_to_yolo.py \
    --kitti_root /kaggle/working/data/kitti \
    --out_root /kaggle/working/data/unified_yolo/kitti \
    --val_fraction 0.1

python scripts/convert_dawn_to_yolo.py \
    --dawn_root /kaggle/working/data/DAWN \
    --out_root /kaggle/working/data/unified_yolo/dawn
    # kept eval-only by default (no --split_for_training) — see note below
```

## 2. Merge into one training set

```bash
python scripts/merge_datasets.py \
    --unified_root /kaggle/working/data/unified_yolo \
    --raw_data_root /kaggle/working/data \
    --include idd kitti dawn \
    --out_name merged
```

This symlinks images (no duplication) into
`unified_yolo/merged/{images,labels}/{train,val}/`, and writes
`source_manifest.csv` mapping every file back to its source dataset —
useful later for a per-source breakdown table in your results section
(e.g. "accuracy on IDD-only subset vs KITTI-only subset").

**On DAWN's eval-only default**: your "Scene Robustness Across
Conditions" feature is strongest as a result if at least some weather
data was never trained on — that's what makes the robustness number
meaningful rather than circular. If you want more training volume and
are OK folding some DAWN into training, re-run its converter with
`--split_for_training` and re-run the merge with `dawn` included in
`--include`; just keep a note in your writeup about which weather
images were held out vs trained on.

## 3. Train

```bash
python scripts/train.py \
    --data_root /kaggle/working/data/unified_yolo/merged \
    --images_root /kaggle/working/data/unified_yolo/merged/images \
    --variant hybrid_s --epochs 30 --num_classes 12
```

Note `--num_classes 12` — the unified taxonomy (`configs/unified_classes.py`)
has 12 classes, not BDD100K's original 10 (adds `autorickshaw` and
`animal` for IDD).

`train.py` currently wires up `BDD100KDataset`; for the merged set, swap
in `UnifiedYoloDataset` from `data/unified_dataset.py` — same collate_fn
signature, so it's a one-line change in `build_dataloaders()`.

## 4. Evaluate DAWN separately (robustness table)

Run inference/eval against `unified_yolo/dawn/labels/eval_only/` on its
own to get your per-weather-condition numbers — this is the artifact
that backs up the "Scene Robustness Across Conditions" feature rather
than a single blended metric.

## 5. Everything else is unchanged

`benchmark_fps.py`, `infer_dashcam.py`, `train_gtsrb.py`,
`train_rtdetr_baseline.py` all work the same as before — just point
`--num_classes` / `--weights` at whichever checkpoint you trained on
the merged data.
