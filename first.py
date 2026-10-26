"""
Custom CNN baseline for Yellow-Rust-19 image classification.

Expected data layout:
Yellow-Rust-19/
    class_1/
        image001.jpg
        ...
    class_2/
        ...
    ...

Important:
- Point --data-dir at the directory whose immediate subdirectories are the class names.
- This first experiment trains on Yellow-Rust-19 only.
- NWRD is a segmentation dataset with masks, so do not mix its masks/images into
  this classifier until image-level class labels have been defined reliably.
- Splits are made before any augmentation. Keep duplicate/near-duplicate source
  images in the same split if the dataset provides source/group IDs.
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.preprocessing import label_binarize


def parse_args():
    parser = argparse.ArgumentParser(description="Train a custom CNN on Yellow-Rust-19.")
    parser.add_argument("--data-dir", required=True, help="Folder containing one subfolder per class.")
    parser.add_argument("--output-dir", default="runs/yellow_rust_cnn", help="Where to save model and metrics.")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0, help="0 is a safe default on macOS.")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CustomCNN(nn.Module):
    """Compact baseline CNN with adaptive pooling and a classification head."""

    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.35),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def choose_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_loader(dataset, indices, batch_size, shuffle, workers):
    return DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=False,
    )


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    all_labels, all_probs, all_preds = [], [], []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = loss_fn(logits, labels)
            if training:
                loss.backward()
                optimizer.step()

        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        total_loss += loss.item() * labels.size(0)
        all_labels.extend(labels.detach().cpu().numpy().tolist())
        all_probs.extend(probs.detach().cpu().numpy().tolist())
        all_preds.extend(preds.detach().cpu().numpy().tolist())

    return (
        total_loss / max(1, len(loader.dataset)),
        np.asarray(all_labels),
        np.asarray(all_probs),
        np.asarray(all_preds),
    )


def calculate_metrics(y_true, y_probs, y_pred, class_names):
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "classification_report": classification_report(
            y_true, y_pred, labels=list(range(len(class_names))),
            target_names=class_names, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=list(range(len(class_names)))
        ).tolist(),
    }

    # Multiclass one-vs-rest metrics. These require enough examples from each
    # class in the evaluation partition.
    try:
        y_bin = label_binarize(y_true, classes=list(range(len(class_names))))
        if len(class_names) == 2:
            # label_binarize returns one column for binary labels.
            result["auroc_ovr_macro"] = float(roc_auc_score(y_true, y_probs[:, 1]))
            result["pr_auc_macro"] = float(average_precision_score(y_true, y_probs[:, 1]))
        else:
            result["auroc_ovr_macro"] = float(
                roc_auc_score(y_true, y_probs, multi_class="ovr", average="macro")
            )
            result["pr_auc_macro"] = float(
                average_precision_score(y_bin, y_probs, average="macro")
            )
    except (ValueError, IndexError) as exc:
        result["auroc_pr_auc_note"] = (
            "AUROC/PR-AUC could not be calculated for this split: " + str(exc)
        )
    return result


def main():
    args = parse_args()
    set_seed(args.seed)

    data_dir = Path(args.data_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        raise SystemExit(f"Dataset folder not found: {data_dir}")

    # Augmentation is applied only to training images. Validation/test images
    # receive deterministic resizing and normalization.
    train_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(12),
        transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.10),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    # ImageFolder expects one subfolder per class. Create two dataset views so
    # only training samples receive augmentation.
    train_view = datasets.ImageFolder(str(data_dir), transform=train_transform)
    eval_view = datasets.ImageFolder(str(data_dir), transform=eval_transform)

    if len(train_view.classes) < 2:
        raise SystemExit(
            "Found fewer than two class folders. --data-dir must point to the "
            "directory whose immediate subfolders are the image classes."
        )
    if len(train_view) < 10:
        raise SystemExit(f"Only found {len(train_view)} images. Check the dataset path.")

    targets = np.asarray(train_view.targets)
    indices = np.arange(len(targets))

    # 70% train, 15% validation, 15% test, stratified by class.
    train_idx, temp_idx = train_test_split(
        indices, test_size=0.30, random_state=args.seed, stratify=targets
    )
    temp_targets = targets[temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, random_state=args.seed, stratify=temp_targets
    )

    class_names = train_view.classes
    print(f"Dataset: {data_dir}")
    print(f"Classes ({len(class_names)}): {class_names}")
    print(f"Images: {len(targets)} | train={len(train_idx)} | val={len(val_idx)} | test={len(test_idx)}")
    print("Warning: if the dataset includes duplicate/augmented versions of images,")
    print("group them by original source before splitting to avoid data leakage.")

    train_loader = make_loader(train_view, train_idx, args.batch_size, True, args.workers)
    val_loader = make_loader(eval_view, val_idx, args.batch_size, False, args.workers)
    test_loader = make_loader(eval_view, test_idx, args.batch_size, False, args.workers)

    device = choose_device()
    print(f"Device: {device}")
    model = CustomCNN(num_classes=len(class_names)).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    best_val_loss = float("inf")
    best_path = output_dir / "best_custom_cnn.pt"
    history = []

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss, _, _, _ = run_epoch(model, train_loader, loss_fn, device, optimizer)
        val_loss, y_val, p_val, pred_val = run_epoch(model, val_loader, loss_fn, device)
        scheduler.step(val_loss)
        val_acc = accuracy_score(y_val, pred_val)
        val_f1 = f1_score(y_val, pred_val, average="macro", zero_division=0)

        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_accuracy": float(val_acc),
            "val_macro_f1": float(val_f1),
            "seconds": round(time.time() - start, 1),
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f} | val_macro_f1={val_f1:.4f} | "
            f"{row['seconds']}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "image_size": args.image_size,
                "architecture": "CustomCNN",
            }, best_path)

    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Load best validation checkpoint, then evaluate the held-out test set once.
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, y_test, p_test, pred_test = run_epoch(model, test_loader, loss_fn, device)
    metrics = calculate_metrics(y_test, p_test, pred_test, class_names)
    metrics["test_loss"] = float(test_loss)
    metrics["class_names"] = class_names
    metrics["split_sizes"] = {
        "train": int(len(train_idx)), "validation": int(len(val_idx)), "test": int(len(test_idx))
    }

    with open(output_dir / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nHELD-OUT TEST RESULTS")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    if "auroc_ovr_macro" in metrics:
        print(f"Macro one-vs-rest AUROC: {metrics['auroc_ovr_macro']:.4f}")
        print(f"Macro PR-AUC: {metrics['pr_auc_macro']:.4f}")
    elif "auroc_pr_auc_note" in metrics:
        print(metrics["auroc_pr_auc_note"])
    print("\nClassification report:")
    print(classification_report(
        y_test, pred_test, labels=list(range(len(class_names))),
        target_names=class_names, zero_division=0
    ))
    print(f"\nSaved best model: {best_path}")
    print(f"Saved metrics: {output_dir / 'test_metrics.json'}")
    print("This is an initial random stratified split. For publication-quality results,")
    print("use grouped splits by original plant/field/source when that metadata is available.")


if __name__ == "__main__":
    main()
