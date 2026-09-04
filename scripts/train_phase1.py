"""
Phase 1 training: object detection (KITTI) + sign location (GTSDB)
only. No drivable-area or weather heads — those come in Phase 2 along
with IDD/DAWN. Reuses the same backbone/neck/head architecture, just
with use_drivable_head=False, use_weather_head=False and the smaller
6-class Phase 1 taxonomy.

Usage:
    python scripts/train_phase1.py \
        --data_root /kaggle/working/data/phase1_yolo/merged \
        --variant hybrid_s --epochs 30 --batch_size 16
"""
import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.hybrid_model import HybridADASModel
from models.losses import DetectionLoss
from data.phase1_dataset import Phase1Dataset, collate_fn
from configs.phase1_classes import NUM_PHASE1_CLASSES

VARIANT_PRESETS = {
    "cnn_baseline": dict(width_mult=0.50, depth_mult=0.33, num_transformer_layers=0),
    "hybrid_n":     dict(width_mult=0.25, depth_mult=0.33, num_transformer_layers=1),
    "hybrid_s":     dict(width_mult=0.50, depth_mult=0.33, num_transformer_layers=2),
    "hybrid_m":     dict(width_mult=0.75, depth_mult=0.67, num_transformer_layers=4),
}


def build_model(variant):
    cfg = VARIANT_PRESETS[variant]
    return HybridADASModel(
        num_classes=NUM_PHASE1_CLASSES,
        use_drivable_head=False,
        use_weather_head=False,
        **cfg,
    )


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="output of merge_phase1.py")
    ap.add_argument("--variant", default="hybrid_s", choices=list(VARIANT_PRESETS.keys()))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--img_size", type=int, default=640)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--out_dir", default="runs")
    ap.add_argument("--patience", type=int, default=7, help="Early stopping patience")
    ap.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    ap.add_argument("--no_amp", action="store_true", help="Disable Automatic Mixed Precision (AMP)")
    return ap.parse_args()


def run_epoch(model, loader, loss_fn, optimizer, scaler, device, train=True, use_amp=True):
    model.train(train)
    total_loss, n_batches = 0.0, 0
    max_conf_epoch, total_boxes_above_35 = 0.0, 0

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        boxes_list, classes_list = batch["boxes"], batch["classes"]

        with torch.set_grad_enabled(train):
            with torch.amp.autocast("cuda", enabled=(device == "cuda" and use_amp)):
                preds = model(images)
                losses = loss_fn(preds, boxes_list, classes_list)
                loss = losses["loss"]

            if train:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                scaler.step(optimizer)
                scaler.update()

            if not train:
                scores = preds["cls_logits"].sigmoid()
                max_conf_epoch = max(max_conf_epoch, scores.max().item())
                total_boxes_above_35 += (scores.max(-1).values > 0.35).sum().item()

        total_loss += loss.item()
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    if not train:
        return avg_loss, max_conf_epoch, total_boxes_above_35
    return avg_loss


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    out_dir = Path(args.out_dir) / f"phase1_{args.variant}"
    out_dir.mkdir(parents=True, exist_ok=True)

    use_amp = not args.no_amp and device == "cuda"

    train_ds = Phase1Dataset(args.data_root, split="train", img_size=args.img_size)
    val_ds = Phase1Dataset(args.data_root, split="val", img_size=args.img_size)

    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": (device == "cuda"),
        "persistent_workers": (args.num_workers > 0),
    }

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, drop_last=True, **loader_kwargs
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, **loader_kwargs
    )
    print(f"train images: {len(train_ds)}  val images: {len(val_ds)} | AMP FP16: {use_amp}")

    model = build_model(args.variant).to(device)
    print(f"{args.variant}: {model.count_params()/1e6:.2f}M params, {NUM_PHASE1_CLASSES} classes")

    loss_fn = DetectionLoss(num_classes=NUM_PHASE1_CLASSES)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val = float("inf")
    epochs_since_improvement = 0
    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss = run_epoch(model, train_loader, loss_fn, optimizer, scaler, device, train=True, use_amp=use_amp)
        val_loss, max_conf, boxes_35 = run_epoch(model, val_loader, loss_fn, optimizer, scaler, device, train=False, use_amp=use_amp)
        scheduler.step()
        print(f"[phase1_{args.variant}] epoch {epoch+1:02d}/{args.epochs:02d} "
              f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"max_val_conf={max_conf:.4f} ({max_conf*100:5.1f}%) boxes>0.35={boxes_35} ({time.time()-t0:.1f}s)", flush=True)

        if val_loss < best_val:
            best_val = val_loss
            epochs_since_improvement = 0
            torch.save(model.state_dict(), out_dir / "best.pt")
        else:
            epochs_since_improvement += 1

        torch.save(model.state_dict(), out_dir / "last.pt")

        if epochs_since_improvement >= args.patience:
            print(f"No val improvement for {args.patience} epochs — stopping early at epoch {epoch+1}.")
            break

    print(f"Done. Best val loss {best_val:.4f}, checkpoints in {out_dir}")


if __name__ == "__main__":
    main()
