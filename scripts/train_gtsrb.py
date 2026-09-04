"""
GTSRB (German Traffic Sign Recognition Benchmark) classifier training.
This is the "classify" half of the detect-then-classify sign pipeline:
BDD100K's "traffic sign" detection boxes get cropped and fed through
this trained classifier to get the actual sign type (stop/yield/speed
limit/etc, 43 classes).

Expects the Kaggle GTSRB layout:
    gtsrb/Train/<class_id>/*.png
    gtsrb/Test/*.png + Test.csv (path,ClassId)

Usage:
    python scripts/train_gtsrb.py --data_root /kaggle/working/data/gtsrb --epochs 15
"""
import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

GTSRB_CLASS_NAMES = {
    0: "Speed limit 20", 1: "Speed limit 30", 2: "Speed limit 50", 3: "Speed limit 60",
    4: "Speed limit 70", 5: "Speed limit 80", 6: "End speed limit 80", 7: "Speed limit 100",
    8: "Speed limit 120", 9: "No overtaking", 10: "No overtaking (trucks)",
    11: "Priority at next intersection", 12: "Priority road", 13: "Yield", 14: "Stop",
    15: "No vehicles", 16: "No trucks", 17: "No entry", 18: "General caution",
    19: "Dangerous curve left", 20: "Dangerous curve right", 21: "Double curve",
    22: "Bumpy road", 23: "Slippery road", 24: "Road narrows right",
    25: "Road work", 26: "Traffic signals", 27: "Pedestrians", 28: "Children crossing",
    29: "Bicycles crossing", 30: "Ice/snow", 31: "Wild animals crossing",
    32: "End of all limits", 33: "Turn right ahead", 34: "Turn left ahead",
    35: "Ahead only", 36: "Go straight or right", 37: "Go straight or left",
    38: "Keep right", 39: "Keep left", 40: "Roundabout mandatory",
    41: "End no overtaking", 42: "End no overtaking (trucks)",
}
NUM_CLASSES = len(GTSRB_CLASS_NAMES)


class GTSRBDataset(Dataset):
    def __init__(self, root, split="train", img_size=64, transform=None):
        self.root = Path(root)
        self.img_size = img_size
        self.transform = transform
        self.samples = []

        if split == "train":
            train_dir = self.root / "Train"
            for cls_dir in sorted(train_dir.iterdir()):
                if not cls_dir.is_dir():
                    continue
                cls_id = int(cls_dir.name)
                for img_path in cls_dir.glob("*.png"):
                    self.samples.append((img_path, cls_id))
        else:
            test_csv = self.root / "Test.csv"
            df = pd.read_csv(test_csv)
            path_col = "Path" if "Path" in df.columns else "path"
            label_col = "ClassId" if "ClassId" in df.columns else "class_id"
            for _, row in df.iterrows():
                self.samples.append((self.root / row[path_col], int(row[label_col])))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def build_transforms(img_size, train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def build_model(num_classes=NUM_CLASSES, arch="resnet18", pretrained=True):
    if arch == "resnet18":
        m = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights="IMAGENET1K_V1" if pretrained else None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    else:
        raise ValueError(arch)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--arch", default="resnet18", choices=["resnet18", "efficientnet_b0"])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--img_size", type=int, default=64)
    ap.add_argument("--out", default="/kaggle/working/runs/gtsrb_classifier.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = GTSRBDataset(args.data_root, "train", args.img_size, build_transforms(args.img_size, True))
    test_ds = GTSRBDataset(args.data_root, "test", args.img_size, build_transforms(args.img_size, False))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = build_model(arch=args.arch).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
            correct += (out.argmax(1) == labels).sum().item()
            total += imgs.size(0)
        scheduler.step()
        train_acc = correct / total

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                out = model(imgs)
                val_correct += (out.argmax(1) == labels).sum().item()
                val_total += imgs.size(0)
        val_acc = val_correct / val_total

        print(f"epoch {epoch+1}/{args.epochs} train_loss={train_loss/total:.4f} "
              f"train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), args.out)

    print(f"Done. Best val_acc={best_acc:.4f}, saved to {args.out}")


if __name__ == "__main__":
    main()
