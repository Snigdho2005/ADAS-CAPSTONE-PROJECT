"""
Pure-transformer baseline: RT-DETR (via Ultralytics), trained on the
same BDD100K YOLO-format data your hybrid model uses. This gives you
the third point on your accuracy-vs-speed axis:

    pure CNN (cnn_baseline)  --  hybrid (hybrid_s/m)  --  pure transformer (RT-DETR)

RT-DETR needs `data/bdd100k.yaml` from bdd_to_yolo.py — same format
Ultralytics YOLO uses, so no extra conversion needed.

Usage:
    pip install ultralytics
    python scripts/train_rtdetr_baseline.py \
        --data_yaml /kaggle/working/data/bdd100k_yolo/bdd100k.yaml \
        --epochs 30 --imgsz 640

This also runs the FPS benchmark at the end (batch size 1, matching
your hybrid model's benchmarking script) so the numbers are directly
comparable in your ablation table.
"""
import argparse
import time

import torch
from ultralytics import RTDETR


def train(args):
    model = RTDETR(args.model_variant)  # e.g. "rtdetr-l.pt" pretrained checkpoint
    model.train(
        data=args.data_yaml,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch_size,
        device=0 if torch.cuda.is_available() else "cpu",
        project=args.out_dir,
        name="rtdetr_baseline",
    )
    return model


def benchmark_fps(model, imgsz=640, n_iters=100, device="cuda"):
    """Batch-size-1 FPS benchmark, mirroring how you'll benchmark the
    hybrid model, so the numbers in your comparison table line up."""
    dummy = torch.randn(1, 3, imgsz, imgsz).to(device)
    model_pt = model.model.to(device).eval()

    with torch.no_grad():
        for _ in range(10):  # warmup
            model_pt(dummy)
        torch.cuda.synchronize() if device == "cuda" else None

        t0 = time.time()
        for _ in range(n_iters):
            model_pt(dummy)
        torch.cuda.synchronize() if device == "cuda" else None
        dt = time.time() - t0

    fps = n_iters / dt
    latency_ms = (dt / n_iters) * 1000
    return fps, latency_ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_yaml", required=True)
    ap.add_argument("--model_variant", default="rtdetr-l.pt",
                     help="rtdetr-l.pt (larger, ~32M) or rtdetr-x.pt; there's no official 'small' RT-DETR checkpoint")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch_size", type=int, default=8)  # RT-DETR is memory-heavy, smaller batch than the hybrid model
    ap.add_argument("--out_dir", default="/kaggle/working/runs")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = train(args)

    fps, latency_ms = benchmark_fps(model, imgsz=args.imgsz, device=device)
    print(f"\nRT-DETR baseline ({args.model_variant}) on {device}:")
    print(f"  FPS: {fps:.1f}  |  Latency: {latency_ms:.1f} ms  (batch size 1, {args.imgsz}px)")
    print("  -> compare directly against your hybrid model's benchmark_fps.py output")


if __name__ == "__main__":
    main()
