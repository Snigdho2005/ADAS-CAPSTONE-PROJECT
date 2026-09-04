"""
Training entrypoint.

Default dataset source is the merged IDD+KITTI+DAWN set (see
README_DATA_PIPELINE.md for the full conversion pipeline):

    python scripts/train.py \
        --dataset unified \
        --data_root /kaggle/working/data/unified_yolo/merged \
        --variant hybrid_s --epochs 30 --batch_size 16 --img_size 640 --num_classes 12

BDD100K is still supported via --dataset bdd100k (10 classes):

    python scripts/train.py \
        --dataset bdd100k \
        --data_root /kaggle/working/data/bdd100k_yolo \
        --images_root /kaggle/working/data/bdd100k/images/100k \
        --variant hybrid_s --epochs 30 --num_classes 10

Swap --variant cnn_baseline / hybrid_n / hybrid_s / hybrid_m to run the
ablation sweep (this is your central result — accuracy vs FPS as
transformer layers are added).
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

from models.hybrid_model import build_model_variant
from models.losses import DetectionLoss, AuxLosses
from data.bdd_dataset import BDD100KDataset, collate_fn as bdd_collate_fn
from data.unified_dataset import UnifiedYoloDataset, collate_fn as unified_collate_fn


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="unified", choices=["unified", "bdd100k"],
                     help="'unified' = merged IDD+KITTI+DAWN set (default), 'bdd100k' = original BDD100K pipeline")
    ap.add_argument("--data_root", required=True,
                     help="unified_yolo/merged (for --dataset unified) or output of bdd_to_yolo.py (for --dataset bdd100k)")
    ap.add_argument("--images_root", default=None, help="required only for --dataset bdd100k (bdd100k/images/100k)")
    ap.add_argument("--variant", default="hybrid_s",
                     choices=["cnn_baseline", "hybrid_n", "hybrid_s", "hybrid_m"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--img_size", type=int, default=640)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--num_classes", type=int, default=12,
                     help="12 for the unified IDD/KITTI/DAWN taxonomy (default); use 10 for --dataset bdd100k")
    ap.add_argument("--out_dir", default="/kaggle/working/runs")
    ap.add_argument("--use_wandb", action="store_true")
    return ap.parse_args()


def build_dataloaders(args, device):
    data_root = Path(args.data_root)

    if args.dataset == "unified":
        train_ds = UnifiedYoloDataset(data_root, split="train", img_size=args.img_size)
        val_ds = UnifiedYoloDataset(data_root, split="val", img_size=args.img_size)
        collate = unified_collate_fn
    else:
        if args.images_root is None:
            raise ValueError("--images_root is required for --dataset bdd100k")
        images_root = Path(args.images_root)
        train_ds = BDD100KDataset(
            images_dir=images_root / "train",
            labels_dir=data_root / "labels" / "train",
            scene_csv=data_root / "scene_attributes_train.csv",
            img_size=args.img_size,
        )
        val_ds = BDD100KDataset(
            images_dir=images_root / "val",
            labels_dir=data_root / "labels" / "val",
            scene_csv=data_root / "scene_attributes_val.csv",
            img_size=args.img_size,
        )
        collate = bdd_collate_fn

    loader_kwargs = {
        "num_workers": getattr(args, "num_workers", 4),
        "pin_memory": (device == "cuda"),
        "persistent_workers": (getattr(args, "num_workers", 4) > 0),
    }

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               collate_fn=collate, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate, **loader_kwargs)
    return train_loader, val_loader


def run_epoch(model, loader, det_loss_fn, aux_loss_fn, optimizer, scaler, device, train=True, use_amp=True):
    model.train(train)
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        boxes_list = batch["boxes"]
        classes_list = batch["classes"]
        drivable = batch["drivable_mask"].to(device, non_blocking=True)
        weather = batch["weather_label"].to(device, non_blocking=True)
        timeofday = batch["timeofday_label"].to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            with torch.amp.autocast("cuda", enabled=(device == "cuda" and use_amp)):
                preds = model(images)
                det_losses = det_loss_fn(preds, boxes_list, classes_list)
                aux_losses = aux_loss_fn(preds, drivable, weather, timeofday)

                loss = det_losses["loss"]
                for v in aux_losses.values():
                    loss = loss + 0.3 * v  # down-weight aux losses vs. the main detection task

            if train:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                scaler.step(optimizer)
                scaler.update()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    out_dir = Path(args.out_dir) / args.variant
    out_dir.mkdir(parents=True, exist_ok=True)

    use_amp = device == "cuda"

    if args.use_wandb:
        import wandb
        wandb.init(project="adas-hybrid-detector", name=args.variant, config=vars(args))

    train_loader, val_loader = build_dataloaders(args, device)

    model = build_model_variant(args.variant, num_classes=args.num_classes).to(device)
    print(f"{args.variant}: {model.count_params()/1e6:.2f}M params | AMP FP16: {use_amp}")

    det_loss_fn = DetectionLoss(num_classes=args.num_classes)
    aux_loss_fn = AuxLosses()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val = float("inf")
    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss = run_epoch(model, train_loader, det_loss_fn, aux_loss_fn, optimizer, scaler, device, train=True, use_amp=use_amp)
        val_loss = run_epoch(model, val_loader, det_loss_fn, aux_loss_fn, optimizer, scaler, device, train=False, use_amp=use_amp)
        scheduler.step()
        dt = time.time() - t0

        print(f"[{args.variant}] epoch {epoch+1}/{args.epochs} "
              f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} ({dt:.1f}s)")

        if args.use_wandb:
            wandb.log({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": scheduler.get_last_lr()[0]})

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), out_dir / "best.pt")

        torch.save(model.state_dict(), out_dir / "last.pt")

    print(f"Done. Best val loss {best_val:.4f}, checkpoints in {out_dir}")


if __name__ == "__main__":
    main()
