#!/bin/bash
# Dataset setup for the Hybrid CNN-Transformer ADAS project
# Run this inside a Kaggle notebook (with internet ON) or Colab.
set -e

DATA_ROOT="${1:-/kaggle/working/data}"
mkdir -p "$DATA_ROOT"
cd "$DATA_ROOT"

echo "==> Setting up Kaggle API (needs kaggle.json uploaded / KAGGLE_USERNAME+KAGGLE_KEY env vars)"
pip install -q kaggle

# ---------------------------------------------------------------------------
# 1. BDD100K (mirrored on Kaggle — much faster than the official server)
#    Contains: 100k images, det_train.json / det_val.json (boxes for vehicles,
#    peds, cyclists, traffic lights, signs), lane + drivable-area masks,
#    weather/timeofday/scene attributes per image.
# ---------------------------------------------------------------------------
echo "==> Downloading BDD100K (images + labels)"
mkdir -p bdd100k && cd bdd100k
kaggle datasets download -d solesensei/solesensei_bdd100k -p . --unzip
# NOTE: if this specific mirror is stale/removed, search "bdd100k" on Kaggle
# Datasets and swap the slug above, or fall back to the official portal:
#   https://bdd-data.berkeley.edu/  (Download tab, requires free registration)
cd "$DATA_ROOT"

# ---------------------------------------------------------------------------
# 2. GTSRB (German Traffic Sign Recognition Benchmark) — 43-class classifier
#    used in the detect-then-classify sign pipeline.
# ---------------------------------------------------------------------------
echo "==> Downloading GTSRB"
mkdir -p gtsrb && cd gtsrb
kaggle datasets download -d meowmeowmeowmeowmeow/gtsrb-german-traffic-sign -p . --unzip
cd "$DATA_ROOT"

echo "==> Done. Directory layout:"
find "$DATA_ROOT" -maxdepth 2 -type d

echo ""
echo "Next step: run scripts/bdd_to_yolo.py to convert BDD100K JSON labels"
echo "into YOLO-format .txt files before training."
