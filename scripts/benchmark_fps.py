"""
Batch-size-1 FPS/latency benchmark for the hybrid model, using the
exact same protocol as scripts/train_rtdetr_baseline.py's
benchmark_fps() so the two numbers are directly comparable.

Usage:
    python scripts/benchmark_fps.py --variant hybrid_s --weights runs/hybrid_s/best.pt
    python scripts/benchmark_fps.py --variant cnn_baseline --weights runs/cnn_baseline/best.pt
    python scripts/benchmark_fps.py --variant hybrid_m --weights runs/hybrid_m/best.pt

Run this for every variant + the RT-DETR baseline to build the
accuracy-vs-speed table for your ablation study.
"""
import argparse
import time
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.hybrid_model import build_model_variant


def benchmark(model, imgsz=640, n_iters=100, device="cuda", warmup=10, fp16=False):
    dummy = torch.randn(1, 3, imgsz, imgsz).to(device)
    if fp16 and device == "cuda":
        dummy = dummy.half()
        model = model.half()
    model = model.to(device).eval()

    with torch.no_grad():
        for _ in range(warmup):
            model(dummy)
        if device == "cuda":
            torch.cuda.synchronize()

        t0 = time.time()
        for _ in range(n_iters):
            model(dummy)
        if device == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0

    fps = n_iters / dt
    latency_ms = (dt / n_iters) * 1000
    return fps, latency_ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="hybrid_s",
                     choices=["cnn_baseline", "hybrid_n", "hybrid_s", "hybrid_m"])
    ap.add_argument("--weights", default=None, help="optional; benchmark works with random weights too")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--num_classes", type=int, default=12)
    ap.add_argument("--no_drivable", action="store_true", help="disable drivable area head")
    ap.add_argument("--no_weather", action="store_true", help="disable weather classification head")
    ap.add_argument("--fp16", action="store_true", help="benchmark with FP16 (closer to a real TensorRT deployment)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_drivable = not args.no_drivable
    use_weather = not args.no_weather
    num_classes = args.num_classes
    state_dict = None

    if args.weights:
        ckpt = torch.load(args.weights, map_location=device)
        state_dict = ckpt if not isinstance(ckpt, dict) or "state_dict" not in ckpt else ckpt["state_dict"]

        if "det_head.blocks.P3.cls_branch.2.weight" in state_dict:
            num_classes = state_dict["det_head.blocks.P3.cls_branch.2.weight"].shape[0]

        use_drivable = any(k.startswith("drivable_head.") for k in state_dict.keys())
        use_weather = any(k.startswith("weather_head.") for k in state_dict.keys())

    model = build_model_variant(
        args.variant,
        num_classes=num_classes,
        use_drivable_head=use_drivable,
        use_weather_head=use_weather,
    )
    if state_dict is not None:
        model.load_state_dict(state_dict)

    use_fp16 = args.fp16 and device == "cuda"
    fps, latency_ms = benchmark(model, imgsz=args.imgsz, device=device, fp16=use_fp16)
    params_m = sum(p.numel() for p in model.parameters()) / 1e6

    print(f"variant={args.variant}  params={params_m:.2f}M  device={device}  "
          f"fp16={use_fp16}")
    print(f"FPS: {fps:.1f}  |  Latency: {latency_ms:.1f} ms  (batch size 1, {args.imgsz}px)")


if __name__ == "__main__":
    main()
