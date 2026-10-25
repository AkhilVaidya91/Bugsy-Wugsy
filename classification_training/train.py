# python train.py \
#   --csv data/bug_crops.csv \
#   --output-dir runs/bug_classifier_comparison \
#   --models efficientnet_v2_m mobilenet_v3_small \
#   --warmup-epochs 3 \
#   --finetune-epochs 15 \
#   --batch-size 16

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import (
    EfficientNet_V2_M_Weights,
    MobileNet_V3_Small_Weights,
)

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedGroupKFold


CLASSES = ["1", "2", "3", "other"]
ARCHITECTURES = ["efficientnet_v2_m", "mobilenet_v3_small"]
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def split_grouped(df):
    """Keep crops from the same source photograph in one split."""

    outer = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=42
    )

    train_val_idx, test_idx = next(
        outer.split(
            df,
            df.label_id,
            groups=df.source_image_id,
        )
    )

    train_val = df.iloc[train_val_idx].copy().reset_index(drop=True)
    test = df.iloc[test_idx].copy().reset_index(drop=True)

    inner = StratifiedGroupKFold(
        n_splits=4, shuffle=True, random_state=43
    )

    train_idx, val_idx = next(
        inner.split(
            train_val,
            train_val.label_id,
            groups=train_val.source_image_id,
        )
    )

    train = train_val.iloc[train_idx].copy().reset_index(drop=True)
    val = train_val.iloc[val_idx].copy().reset_index(drop=True)

    return train, val, test


def load_manifest(csv_path):
    df = pd.read_csv(csv_path)

    required = {
        "crop_path",
        "label",
        "source_image_id",
        "bbox_width",
        "bbox_height",
        "yolo_confidence",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing CSV columns: {sorted(missing)}")

    df["label"] = df["label"].astype(str).str.strip().str.lower()

    unknown = sorted(set(df.label) - set(CLASSES))
    if unknown:
        raise ValueError(
            f"Unknown labels {unknown}. Expected {CLASSES}."
        )

    df["label_id"] = df.label.map(
        {name: i for i, name in enumerate(CLASSES)}
    )

    df["crop_path"] = df.crop_path.map(
        lambda p: str(
            (
                Path(p)
                if Path(p).is_absolute()
                else csv_path.parent / p
            ).resolve()
        )
    )

    df["bbox_width"] = pd.to_numeric(
        df.bbox_width, errors="raise"
    ).clip(lower=0)

    df["bbox_height"] = pd.to_numeric(
        df.bbox_height, errors="raise"
    ).clip(lower=0)

    df["yolo_confidence"] = pd.to_numeric(
        df.yolo_confidence, errors="raise"
    ).fillna(0)

    if "aspect_ratio" not in df.columns:
        df["aspect_ratio"] = (
            df.bbox_width / df.bbox_height.clip(lower=1)
        )

    df["aspect_ratio"] = pd.to_numeric(
        df.aspect_ratio, errors="raise"
    ).fillna(1).clip(lower=0)

    # Log-transform box dimensions to reduce scale differences.
    df["meta_w"] = np.log1p(df.bbox_width)
    df["meta_h"] = np.log1p(df.bbox_height)
    df["meta_ar"] = df.aspect_ratio
    df["meta_conf"] = df.yolo_confidence

    for path in df.crop_path:
        if not Path(path).is_file():
            raise FileNotFoundError(f"Crop not found: {path}")

    return df


class CropDataset(Dataset):
    def __init__(self, df, meta_mean, meta_std, train=False):
        self.df = df.reset_index(drop=True)
        self.meta_mean = meta_mean
        self.meta_std = meta_std

        augmentations = []

        if train:
            augmentations = [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(8),
                transforms.ColorJitter(
                    brightness=0.12,
                    contrast=0.12,
                ),
            ]

        self.transform = transforms.Compose(
            augmentations
            + [
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ]
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Preserve aspect ratio and pad to 224 x 224.
        with Image.open(row.crop_path) as im:
            im = ImageOps.pad(
                im.convert("RGB"),
                (224, 224),
                color=(0, 0, 0),
            )

        image = self.transform(im)

        metadata = np.array(
            [
                row.meta_w,
                row.meta_h,
                row.meta_ar,
                row.meta_conf,
            ],
            dtype=np.float32,
        )

        metadata = torch.tensor(
            (metadata - self.meta_mean) / self.meta_std,
            dtype=torch.float32,
        )

        return image, metadata, int(row.label_id)


class MultiResolutionClassifier(nn.Module):
    """
    Extract features at five spatial scales.
    Fuse the visual features with bounding-box metadata.
    """

    def __init__(self, arch):
        super().__init__()
        self.arch = arch

        if arch == "efficientnet_v2_m":
            net = models.efficientnet_v2_m(
                weights=EfficientNet_V2_M_Weights.DEFAULT
            )
            self.features = net.features

            # Last block at each spatial scale: 112, 56, 28, 14, 7.
            self.ends = [1, 2, 3, 5, 8]
            channels = [24, 48, 80, 176, 1280]
            self.finetune_start = 7

        elif arch == "mobilenet_v3_small":
            net = models.mobilenet_v3_small(
                weights=MobileNet_V3_Small_Weights.DEFAULT
            )
            self.features = net.features

            # Approximate spatial scales: 112, 56, 28, 14, 7.
            self.ends = [0, 1, 2, 4, 9]
            channels = [16, 16, 24, 40, 96]
            self.finetune_start = 9

        else:
            raise ValueError(f"Unknown architecture: {arch}")

        # Project each scale to a compact 64-dimensional vector.
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(c, 64),
                nn.ReLU(),
            )
            for c in channels
        ])

        # Four metadata features:
        # width, height, aspect ratio, YOLO confidence.
        self.meta_branch = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Dropout(0.15),
        )

        self.head = nn.Sequential(
            nn.Linear(5 * 64 + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(128, 4),
        )

        self.freeze_backbone()

    def freeze_backbone(self):
        for p in self.features.parameters():
            p.requires_grad = False

    def unfreeze_last_block(self):
        self.freeze_backbone()

        layers = list(self.features.children())

        for layer in layers[self.finetune_start:]:
            for p in layer.parameters():
                p.requires_grad = True

    def forward(self, image, metadata):
        x = image
        feature_vectors = []

        end_to_slot = {
            end: slot
            for slot, end in enumerate(self.ends)
        }

        for i, layer in enumerate(self.features):
            x = layer(x)

            if i in end_to_slot:
                slot = end_to_slot[i]
                feature_vectors.append(
                    self.projections[slot](x)
                )

        visual_features = torch.cat(
            feature_vectors, dim=1
        )

        metadata_features = self.meta_branch(metadata)

        combined = torch.cat(
            [visual_features, metadata_features],
            dim=1,
        )

        return self.head(combined)


def evaluate(model, loader, device):
    model.eval()

    true_labels = []
    predictions = []
    probabilities = []

    with torch.no_grad():
        for images, metadata, labels in loader:
            logits = model(
                images.to(device),
                metadata.to(device),
            )

            probs = torch.softmax(
                logits, dim=1
            ).cpu().numpy()

            probabilities.extend(probs)
            predictions.extend(probs.argmax(axis=1))
            true_labels.extend(labels.numpy())

    macro_f1 = f1_score(
        true_labels,
        predictions,
        labels=list(range(4)),
        average="macro",
        zero_division=0,
    )

    return (
        macro_f1,
        np.asarray(true_labels),
        np.asarray(predictions),
        np.asarray(probabilities),
    )


def train_one(
    arch,
    train_df,
    val_df,
    test_df,
    meta_mean,
    meta_std,
    args,
    out_dir,
    device,
):
    model = MultiResolutionClassifier(arch).to(device)

    train_loader = DataLoader(
        CropDataset(train_df, meta_mean, meta_std, train=True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = DataLoader(
        CropDataset(val_df, meta_mean, meta_std),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    test_loader = DataLoader(
        CropDataset(test_df, meta_mean, meta_std),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # Conservative class balancing using inverse square-root frequency.
    counts = np.bincount(
        train_df.label_id, minlength=4
    )

    class_weights = 1.0 / np.sqrt(
        np.maximum(counts, 1)
    )
    class_weights /= class_weights.mean()

    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(
            class_weights,
            dtype=torch.float32,
            device=device,
        )
    )

    best_f1 = -1.0
    best_state = None

    def fit_phase(epochs, optimizer, phase):
        nonlocal best_f1, best_state

        for epoch in range(epochs):
            model.train()

            # Keep BatchNorm running statistics fixed in the backbone.
            model.features.eval()

            for images, metadata, labels in train_loader:
                images = images.to(device)
                metadata = metadata.to(device)
                labels = labels.to(device)

                optimizer.zero_grad(set_to_none=True)

                logits = model(images, metadata)
                loss = criterion(logits, labels)

                loss.backward()
                optimizer.step()

            val_f1, _, _, _ = evaluate(
                model, val_loader, device
            )

            print(
                f"{arch} | {phase} | "
                f"epoch {epoch + 1}/{epochs} | "
                f"validation macro-F1={val_f1:.4f}"
            )

            if val_f1 > best_f1:
                best_f1 = val_f1

                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }

                torch.save(
                    {
                        "state_dict": best_state,
                        "architecture": arch,
                        "classes": CLASSES,
                    },
                    out_dir / f"{arch}_best.pt",
                )

    # Stage 1: train the new classification layers.
    head_params = [
        p for name, p in model.named_parameters()
        if not name.startswith("features.")
        and p.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        head_params,
        lr=args.head_lr,
        weight_decay=1e-4,
    )

    fit_phase(
        args.warmup_epochs,
        optimizer,
        "head-only",
    )

    # Stage 2: fine-tune the final backbone block.
    model.load_state_dict(best_state)
    model.unfreeze_last_block()

    head_params = [
        p for name, p in model.named_parameters()
        if not name.startswith("features.")
        and p.requires_grad
    ]

    backbone_params = [
        p for name, p in model.named_parameters()
        if name.startswith("features.")
        and p.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        [
            {
                "params": head_params,
                "lr": args.head_lr / 10,
            },
            {
                "params": backbone_params,
                "lr": args.backbone_lr,
            },
        ],
        weight_decay=1e-4,
    )

    fit_phase(
        args.finetune_epochs,
        optimizer,
        "fine-tune",
    )

    # Evaluate the best validation checkpoint on the test set.
    model.load_state_dict(best_state)

    _, y_true, y_pred, probs = evaluate(
        model, test_loader, device
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(4)),
        target_names=CLASSES,
        output_dict=True,
        zero_division=0,
    )

    (out_dir / f"{arch}_test_report.json").write_text(
        json.dumps(report, indent=2)
    )

    pd.DataFrame(
        confusion_matrix(
            y_true, y_pred, labels=list(range(4))
        ),
        index=CLASSES,
        columns=CLASSES,
    ).to_csv(out_dir / f"{arch}_confusion_matrix.csv")

    predictions = test_df[
        ["crop_path", "source_image_id", "label"]
    ].copy()

    predictions["predicted_label"] = [
        CLASSES[i] for i in y_pred
    ]

    for i, cls in enumerate(CLASSES):
        predictions[f"prob_{cls}"] = probs[:, i]

    predictions.to_csv(
        out_dir / f"{arch}_test_predictions.csv",
        index=False,
    )

    return {
        "model": arch,
        "test_accuracy": accuracy_score(y_true, y_pred),
        "test_balanced_accuracy": balanced_accuracy_score(
            y_true, y_pred
        ),
        "test_macro_f1": f1_score(
            y_true,
            y_pred,
            labels=list(range(4)),
            average="macro",
            zero_division=0,
        ),
        **{
            f"recall_{cls}": report[cls]["recall"]
            for cls in CLASSES
        },
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/bug_classifier_comparison"),
    )
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--finetune-epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=ARCHITECTURES,
        default=ARCHITECTURES,
    )

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_manifest(args.csv)
    train_df, val_df, test_df = split_grouped(df)

    meta_cols = [
        "meta_w", "meta_h", "meta_ar", "meta_conf"
    ]

    # Fit metadata scaling on training data only.
    meta_mean = train_df[meta_cols].mean().to_numpy(
        dtype=np.float32
    )
    meta_std = train_df[meta_cols].std().to_numpy(
        dtype=np.float32
    )
    meta_std[meta_std < 1e-6] = 1.0

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")
    print(
        f"Crops: train={len(train_df)}, "
        f"validation={len(val_df)}, test={len(test_df)}"
    )
    print(
        "Training class counts:",
        train_df.label.value_counts()
        .reindex(CLASSES, fill_value=0).to_dict(),
    )

    results = []

    for arch in args.models:
        results.append(
            train_one(
                arch,
                train_df,
                val_df,
                test_df,
                meta_mean,
                meta_std,
                args,
                args.output_dir,
                device,
            )
        )

    comparison = pd.DataFrame(results).sort_values(
        "test_macro_f1",
        ascending=False,
    )

    comparison.to_csv(
        args.output_dir / "model_comparison.csv",
        index=False,
    )

    print("\nFinal model comparison:")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()