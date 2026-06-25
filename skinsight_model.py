"""
SkinSight AI — ISIC lesion image classifier (educational ML demo).

NOT a medical device. Outputs are informational classifications across
dermatology image categories, not diagnoses. Always surface a
"consult a dermatologist" message to end users.

Stack: PyTorch + timm (EfficientNet-B0 transfer learning).
Handles ISIC class imbalance via weighted sampling + weighted loss.
"""

import os
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
import timm
from sklearn.metrics import classification_report, roc_auc_score
import numpy as np

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
DATA_DIR = "data/isic"          # expects data/isic/train/<class>/*.jpg and data/isic/val/<class>/*.jpg
MODEL_NAME = "efficientnet_b0"
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15
LR = 3e-4
WEIGHT_DECAY = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_PATH = "skinsight_efficientnet_b0.pt"

# ImageNet normalization (pretrained backbone expects this)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
def build_transforms():
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),       # lesions have no canonical orientation
        transforms.ColorJitter(0.1, 0.1, 0.1),
        transforms.RandomAffine(degrees=20),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(int(IMG_SIZE * 1.14)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return train_tf, eval_tf


def make_weighted_sampler(dataset):
    """Oversample minority classes — ISIC is heavily skewed toward nevi."""
    targets = np.array([s[1] for s in dataset.samples])
    class_counts = np.bincount(targets)
    class_weights = 1.0 / np.clip(class_counts, 1, None)
    sample_weights = class_weights[targets]
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    ), class_counts


def build_loaders():
    train_tf, eval_tf = build_transforms()
    train_ds = ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_tf)
    val_ds = ImageFolder(os.path.join(DATA_DIR, "val"), transform=eval_tf)

    sampler, class_counts = make_weighted_sampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True)
    return train_loader, val_loader, train_ds.classes, class_counts


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------
def build_model(num_classes):
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=num_classes)
    return model.to(DEVICE)


# ----------------------------------------------------------------------
# Train / eval
# ----------------------------------------------------------------------
def train():
    train_loader, val_loader, classes, class_counts = build_loaders()
    num_classes = len(classes)
    print(f"Classes: {classes}")
    print(f"Train counts: {dict(zip(classes, class_counts.tolist()))}")

    model = build_model(num_classes)

    # Weighted loss as a second defense against imbalance + label smoothing
    inv = 1.0 / np.clip(class_counts, 1, None)
    weights = torch.tensor(inv / inv.sum() * num_classes, dtype=torch.float32, device=DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_auc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        running = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            running += loss.item() * imgs.size(0)
        scheduler.step()
        train_loss = running / len(train_loader.dataset)

        auc, report = evaluate(model, val_loader, classes)
        print(f"Epoch {epoch+1:02d} | loss {train_loss:.4f} | macro-AUC {auc:.4f}")

        if auc > best_auc:
            best_auc = auc
            torch.save({
                "state_dict": model.state_dict(),
                "classes": classes,
                "model_name": MODEL_NAME,
                "img_size": IMG_SIZE,
            }, CKPT_PATH)
            print(f"  ↳ saved checkpoint (macro-AUC {auc:.4f})")

    print(f"\nBest macro-AUC: {best_auc:.4f}")
    print(report)


@torch.no_grad()
def evaluate(model, loader, classes):
    model.eval()
    all_probs, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(DEVICE)
        probs = torch.softmax(model(imgs), dim=1).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.numpy())
    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    preds = probs.argmax(1)

    # one-vs-rest macro AUC (robust to imbalance)
    try:
        auc = roc_auc_score(
            np.eye(len(classes))[labels], probs,
            multi_class="ovr", average="macro",
        )
    except ValueError:
        auc = float("nan")

    report = classification_report(labels, preds, target_names=classes, zero_division=0)
    return auc, report


# ----------------------------------------------------------------------
# Inference (used by the Flask backend later)
# ----------------------------------------------------------------------
class Predictor:
    def __init__(self, ckpt_path=CKPT_PATH):
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        self.classes = ckpt["classes"]
        self.model = timm.create_model(ckpt["model_name"], pretrained=False,
                                       num_classes=len(self.classes)).to(DEVICE)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        _, self.tf = build_transforms()

    @torch.no_grad()
    def predict(self, pil_image):
        x = self.tf(pil_image.convert("RGB")).unsqueeze(0).to(DEVICE)
        probs = torch.softmax(self.model(x), dim=1).cpu().numpy()[0]
        ranked = sorted(zip(self.classes, probs.tolist()),
                        key=lambda p: p[1], reverse=True)
        return {
            "predictions": [{"label": c, "confidence": round(p, 4)} for c, p in ranked],
            "top_label": ranked[0][0],
            "top_confidence": round(ranked[0][1], 4),
            "disclaimer": ("Informational classification only. This is NOT a "
                           "medical diagnosis. Consult a dermatologist for any "
                           "skin concern."),
        }


if __name__ == "__main__":
    train()
