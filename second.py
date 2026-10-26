
#!/usr/bin/env python3
"""
Weather-based wheat yellow rust classifier.

Examples:
  python train_weather_nn.py --csv merged_data/nwrd_weather_merged.csv --target disease_label

Expected target:
  Binary: 0 = absent, 1 = present
  Multiclass: integer or categorical disease classes

Important:
  The target must be a genuine disease label, not a weather variable.
"""

from pathlib import Path
import argparse
import json
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


WEATHER_FEATURES = [
    "t2m",   # 2 m air temperature (usually Kelvin in ERA5)
    "d2m",   # 2 m dew point (usually Kelvin in ERA5)
    "tp",    # Total precipitation (units depend on ERA5 product)
    "u10",   # 10 m east-west wind component
    "v10",   # 10 m north-south wind component
]


class WeatherClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.25),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.15),

            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.network(x)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv",
        default="merged_data/nwrd_weather_merged.csv",
        help="CSV containing weather features and disease labels",
    )
    parser.add_argument(
        "--target",
        default="disease_label",
        help="Column containing the known disease label",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--output", default="runs/weather_nn")

    return parser.parse_args()


def main():
    args = parse_args()

    csv_path = Path(args.csv)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}\n"
            "Check the CSV path or use --csv to specify it."
        )

    df = pd.read_csv(csv_path)

    print(f"\nDevice: {DEVICE}")
    print(f"Dataset: {csv_path}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")

    if args.target not in df.columns:
        raise ValueError(
            f"\nTarget column '{args.target}' does not exist.\n"
            f"Available columns: {df.columns.tolist()}\n\n"
            "Add a verified disease label to the dataset, then rerun "
            "with --target YOUR_LABEL_COLUMN. Weather measurements alone "
            "cannot train a supervised disease classifier."
        )

    features = [c for c in WEATHER_FEATURES if c in df.columns]

    if len(features) < 3:
        raise ValueError(
            "At least three weather features are required. "
            f"Found: {features}"
        )

    print(f"Weather features: {features}")
    print(f"Target column: {args.target}")

    # Keep rows with a known target.
    df = df.dropna(subset=[args.target]).copy()

    # Remove missing/invalid numeric targets and weather-only rows.
    X_df = df[features].apply(pd.to_numeric, errors="coerce")
    X_df = X_df.replace([np.inf, -np.inf], np.nan)

    valid = X_df.notna().any(axis=1)
    df = df.loc[valid].copy()
    X_df = X_df.loc[valid].copy()

    # Encode class labels.
    encoder = LabelEncoder()
    y = encoder.fit_transform(df[args.target].astype(str))

    classes, counts = np.unique(y, return_counts=True)

    print("\nClass distribution:")
    for class_id, count in zip(classes, counts):
        print(f"  {encoder.classes_[class_id]}: {count}")

    num_classes = len(encoder.classes_)

    if num_classes < 2:
        raise ValueError("Training requires at least two disease classes.")

    if counts.min() < 3:
        raise ValueError(
            "Each class needs at least 3 labelled rows for this split. "
            "Collect more labelled examples before training."
        )

    if len(df) < 30:
        raise ValueError(
            f"Only {len(df)} labelled rows are available. "
            "This is too little for a meaningful neural-network evaluation. "
            "Collect more labelled observations first."
        )

    # Split before fitting preprocessing to avoid test-set leakage.
    X_train_df, X_temp_df, y_train, y_temp = train_test_split(
        X_df,
        y,
        test_size=0.30,
        random_state=SEED,
        stratify=y,
    )

    X_val_df, X_test_df, y_val, y_test = train_test_split(
        X_temp_df,
        y_temp,
        test_size=0.50,
        random_state=SEED,
        stratify=y_temp,
    )

    # Fit preprocessing using training data only.
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_train = scaler.fit_transform(imputer.fit_transform(X_train_df))
    X_val = scaler.transform(imputer.transform(X_val_df))
    X_test = scaler.transform(imputer.transform(X_test_df))

    def to_tensor(array):
        return torch.tensor(array, dtype=torch.float32)

    def make_loader(X, y_values, shuffle=False):
        dataset = TensorDataset(
            to_tensor(X),
            torch.tensor(y_values, dtype=torch.long),
        )
        return DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=shuffle,
        )

    train_loader = make_loader(X_train, y_train, shuffle=True)
    val_loader = make_loader(X_val, y_val)
    test_loader = make_loader(X_test, y_test)

    model = WeatherClassifier(
        input_size=len(features),
        num_classes=num_classes,
    ).to(DEVICE)

    # Compensate for class imbalance in the training set.
    train_counts = np.bincount(y_train, minlength=num_classes)
    weights = len(y_train) / (
        num_classes * np.maximum(train_counts, 1)
    )

    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=DEVICE)
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-4,
    )

    best_val_loss = float("inf")
    best_state = None
    patience = 15
    epochs_without_improvement = 0

    print("\nStarting training...")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_total = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            train_loss_total += loss.item() * len(xb)

        train_loss = train_loss_total / len(train_loader.dataset)

        model.eval()
        val_loss_total = 0.0
        val_predictions = []
        val_actual = []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb)
                loss = criterion(logits, yb)

                val_loss_total += loss.item() * len(xb)
                val_predictions.extend(logits.argmax(1).cpu().numpy())
                val_actual.extend(yb.cpu().numpy())

        val_loss = val_loss_total / len(val_loader.dataset)
        val_accuracy = accuracy_score(val_actual, val_predictions)

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d}/{args.epochs} | "
                f"train loss: {train_loss:.4f} | "
                f"val loss: {val_loss:.4f} | "
                f"val accuracy: {val_accuracy:.4f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid model.")

    model.load_state_dict(best_state)
    model.eval()

    # Final evaluation on held-out test data.
    y_pred = []
    y_true = []

    with torch.no_grad():
        for xb, yb in test_loader:
            logits = model(xb.to(DEVICE))
            y_pred.extend(logits.argmax(1).cpu().numpy())
            y_true.extend(yb.numpy())

    labels = list(range(num_classes))
    class_names = [str(c) for c in encoder.classes_]

    metrics = {
        "device": str(DEVICE),
        "rows_used": int(len(df)),
        "features": features,
        "target": args.target,
        "classes": class_names,
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "test_macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "best_validation_loss": float(best_val_loss),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=labels
        ).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        ),
    }

    print("\n===== TEST RESULTS =====")
    print(f"Accuracy: {metrics['test_accuracy']:.4f}")
    print(f"Macro F1: {metrics['test_macro_f1']:.4f}")
    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=class_names,
            zero_division=0,
        )
    )
    print("Confusion matrix:")
    print(np.array(metrics["confusion_matrix"]))

    # Save the model and the fitted preprocessing objects.
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_size": len(features),
            "num_classes": num_classes,
            "features": features,
            "classes": class_names,
            "target": args.target,
            "imputer_statistics": imputer.statistics_.tolist(),
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
        },
        output_dir / "weather_classifier.pt",
    )

    with open(output_dir / "label_encoder.json", "w") as f:
        json.dump(class_names, f, indent=2)

    with open(output_dir / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved model: {output_dir / 'weather_classifier.pt'}")
    print(f"Saved metrics: {output_dir / 'test_metrics.json'}")


if __name__ == "__main__":
    main()
