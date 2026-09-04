"""
Phase 1 visual inference — draws bounding boxes + class labels directly
on video frames and saves an output .mp4, so you have something to
actually show in a review (rather than infer_dashcam.py's terminal-only
readout, which was built for the later Phase 2 taxonomy anyway).

Kaggle usage (run as a notebook cell with the ! prefix, or in a script):
    !python scripts/infer_phase1_visual.py \
        --video /kaggle/input/your-dataset/dashcam.mp4 \
        --weights /kaggle/working/runs/phase1_hybrid_s/best.pt \
        --variant hybrid_s \
        --out /kaggle/working/output_annotated.mp4

Kaggle notebooks can't pop up a video player mid-run — after this
finishes, the output file lands in /kaggle/working/, and you play it
via the Output tab (Kaggle renders .mp4 files there), or download it
and play it locally.
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

VARIANT_PRESETS = {
    "cnn_baseline": dict(width_mult=0.50, depth_mult=0.33, num_transformer_layers=0),
    "hybrid_n":     dict(width_mult=0.25, depth_mult=0.33, num_transformer_layers=1),
    "hybrid_s":     dict(width_mult=0.50, depth_mult=0.33, num_transformer_layers=2),
    "hybrid_m":     dict(width_mult=0.75, depth_mult=0.67, num_transformer_layers=4),
}

# distinct BGR colors per class, for readability on screen
CLASS_COLORS = {
    0: (60, 180, 75),    # car - green
    1: (0, 130, 200),    # van_truck - blue
    2: (245, 130, 48),   # tram - orange
    3: (230, 25, 75),    # pedestrian - red
    4: (255, 225, 25),   # cyclist - yellow
    5: (145, 30, 180),   # traffic_sign - purple
}


def build_model(variant, num_classes):
    cfg = VARIANT_PRESETS[variant]
    return HybridADASModel(
        num_classes=num_classes, use_drivable_head=False, use_weather_head=False, **cfg
    )


def draw_detections(frame, boxes, classes, scores):
    for box, cls_id, score in zip(boxes, classes, scores):
        x1, y1, x2, y2 = [int(v) for v in box]
        color = CLASS_COLORS.get(int(cls_id), (200, 200, 200))
        label = f"{PHASE1_CLASSES.get(int(cls_id), '?')} {score:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def nms_fast(boxes, scores, iou_threshold=0.45):
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-7)
        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]
    return keep


def run(video_path, weights_path, variant, out_path, conf_thresh=0.35,
        img_size=640, max_frames=None, nms_thresh=0.45):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    num_classes = len(PHASE1_CLASSES)
    model = build_model(variant, num_classes).to(device)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {w0}x{h0} @ {fps:.1f}fps, {total_frames} frames")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w0, h0))

    frame_idx = 0
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if max_frames and frame_idx > max_frames:
                break

            img = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (img_size, img_size))
            img_t = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0

            preds = model(img_t)
            cls_logits = preds["cls_logits"][0]
            boxes = preds["boxes"][0]

            scores_all = cls_logits.sigmoid()
            max_scores, max_ids = scores_all.max(-1)
            keep_mask = max_scores > conf_thresh

            kept_boxes = boxes[keep_mask].cpu().numpy()
            kept_classes = max_ids[keep_mask].cpu().numpy()
            kept_scores = max_scores[keep_mask].cpu().numpy()

            if len(kept_boxes):
                nms_indices = nms_fast(kept_boxes, kept_scores, iou_threshold=nms_thresh)
                kept_boxes = kept_boxes[nms_indices]
                kept_classes = kept_classes[nms_indices]
                kept_scores = kept_scores[nms_indices]

                sx, sy = w0 / img_size, h0 / img_size
                kept_boxes[:, [0, 2]] *= sx
                kept_boxes[:, [1, 3]] *= sy

            frame = draw_detections(frame, kept_boxes, kept_classes, kept_scores)
            cv2.putText(frame, f"Frame {frame_idx}/{total_frames}  |  {len(kept_boxes)} objects",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

            writer.write(frame)

            if frame_idx % 50 == 0:
                print(f"  processed {frame_idx}/{total_frames} frames "
                      f"({len(kept_boxes)} detections this frame)")

    cap.release()
    writer.release()
    print(f"\nDone. Annotated video saved to: {out_path}")
    print("On Kaggle: check the 'Output' tab of this notebook session to view/download it.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--variant", default="hybrid_s", choices=list(VARIANT_PRESETS.keys()))
    ap.add_argument("--out", default="/kaggle/working/output_annotated.mp4")
    ap.add_argument("--conf_thresh", type=float, default=0.35)
    ap.add_argument("--img_size", type=int, default=640)
    ap.add_argument("--max_frames", type=int, default=None,
                     help="cap frames processed, useful for a quick preview before running the full video")
    args = ap.parse_args()
    run(args.video, args.weights, args.variant, args.out,
        conf_thresh=args.conf_thresh, img_size=args.img_size, max_frames=args.max_frames)


if __name__ == "__main__":
    main()
