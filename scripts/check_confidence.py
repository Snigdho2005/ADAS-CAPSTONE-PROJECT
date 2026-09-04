"""
Helper script to inspect confidence scores produced by a trained model checkpoint.

Usage:
    python scripts/check_confidence.py --weights runs/phase1_hybrid_s/best.pt --image data/phase1_yolo/merged/images/val/kitti__001712f9117b42901313183312742d79.jpg
    python scripts/check_confidence.py --weights runs/phase1_hybrid_s/best.pt --video 1.mov
"""
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.hybrid_model import HybridADASModel
from configs.phase1_classes import PHASE1_CLASSES


def inspect_image(model, img_bgr, device, img_size=640):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (img_size, img_size))
    img_t = torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0

    with torch.no_grad():
        preds = model(img_t)
        cls_logits = preds["cls_logits"][0]
        boxes = preds["boxes"][0]

    scores = cls_logits.sigmoid()
    max_scores, max_ids = scores.max(-1)

    top_scores, top_idx = max_scores.topk(10)
    
    print("\n--- Model Confidence Inspection ---")
    print(f"Global Maximum Confidence in image: {max_scores.max().item():.4f} ({max_scores.max().item()*100:.1f}%)")
    print(f"Anchors with conf > 0.10: {(max_scores > 0.10).sum().item()}")
    print(f"Anchors with conf > 0.25: {(max_scores > 0.25).sum().item()}")
    print(f"Anchors with conf > 0.50: {(max_scores > 0.50).sum().item()}")
    print(f"Anchors with conf > 0.75: {(max_scores > 0.75).sum().item()}")

    print("\nTop 10 Detected Objects:")
    for rank, idx in enumerate(top_idx, 1):
        score = max_scores[idx].item()
        cls_id = max_ids[idx].item()
        cls_name = PHASE1_CLASSES.get(cls_id, f"class_{cls_id}")
        box = boxes[idx].cpu().numpy().round(1)
        print(f"  #{rank:2d}: {cls_name:<15s} conf={score:.4f} ({score*100:5.1f}%)  box=[{box[0]}, {box[1]}, {box[2]}, {box[3]}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="path to model checkpoint (.pt)")
    ap.add_argument("--image", default=None, help="path to image file")
    ap.add_argument("--video", default=None, help="path to video file (inspects 1st frame)")
    ap.add_argument("--variant", default="hybrid_s")
    ap.add_argument("--img_size", type=int, default=640)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = len(PHASE1_CLASSES)
    
    model = HybridADASModel(num_classes=num_classes, use_drivable_head=False, use_weather_head=False).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    if args.image:
        img_bgr = cv2.imread(args.image)
        inspect_image(model, img_bgr, device, img_size=args.img_size)
    elif args.video:
        cap = cv2.VideoCapture(args.video)
        ret, frame = cap.read()
        cap.release()
        if ret:
            print(f"Inspecting first frame of video: {args.video}")
            inspect_image(model, frame, device, img_size=args.img_size)
        else:
            print(f"Error opening video: {args.video}")
    else:
        print("Please specify --image or --video")


if __name__ == "__main__":
    main()
