
from __future__ import annotations

import json
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import (
    efficientnet_v2_m,
    EfficientNet_V2_M_Weights,
)

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["1", "2", "3", "Insect"]
NUM_CLASSES = len(CLASS_NAMES)

IMAGE_SIZE = 224
BATCH_SIZE = 4
HEAD_ONLY_EPOCHS = 3
FINETUNE_EPOCHS = 17
TOTAL_EPOCHS = HEAD_ONLY_EPOCHS + FINETUNE_EPOCHS

HEAD_LR = 1e-3
BACKBONE_LR = 1e-5
WEIGHT_DECAY = 1e-4
DROPOUT = 0.35

BBOX_MARGIN = 0.05
NUM_WORKERS = 0  # Reliable default on macOS

SEED = 42

# Numeric feature normalization.
LOG_AREA_DIVISOR = 20.0

# Standard ImageNet normalization for the selected weights.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ============================================================
# REPRODUCIBILITY AND DEVICE
# ============================================================

def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        print("Using Apple Metal Performance Shaders (MPS).")
        return torch.device("mps")

    print("MPS unavailable. Using CPU.")
    return torch.device("cpu")


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def square_resize(image: Image.Image) -> Image.Image:
    """
    Preserve aspect ratio, pad to a square, then resize to 224x224.
    """
    image = image.convert("RGB")

    return ImageOps.pad(
        image,
        (IMAGE_SIZE, IMAGE_SIZE),
        method=Image.Resampling.BILINEAR,
        color=(0, 0, 0),
        centering=(0.5, 0.5),
    )


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    """
    Return RGB float32 tensor in [0, 1], shape (3, 224, 224).
    Normalization happens inside the model.
    """
    image = square_resize(image)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def augment_crop(image: Image.Image) -> Image.Image:
    """
    Mild augmentation for a small insect dataset.
    Avoid aggressive geometric transformations.
    """
    if random.random() < 0.5:
        image = ImageOps.mirror(image)

    if random.random() < 0.5:
        image = ImageEnhance.Brightness(image).enhance(
            random.uniform(0.90, 1.10)
        )

    if random.random() < 0.5:
        image = ImageEnhance.Contrast(image).enhance(
            random.uniform(0.90, 1.10)
        )

    if random.random() < 0.5:
        image = ImageEnhance.Color(image).enhance(
            random.uniform(0.90, 1.10)
        )

    return image


# ============================================================
# YOLO DATASET
# ============================================================

class YOLOCropDataset(Dataset):
    """
    Convert each YOLO bounding box into one classification sample.

    Expected structure:
        split/images/image.jpg
        split/labels/image.txt

    Each YOLO line:
        class_id x_center y_center width height

    Coordinates are normalized to [0, 1].
    """

    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, split: str, training: bool = False):
        self.split_dir = ROOT / split
        self.images_dir = self.split_dir / "images"
        self.labels_dir = self.split_dir / "labels"
        self.training = training

        if not self.images_dir.is_dir():
            raise FileNotFoundError(
                f"Missing image directory: {self.images_dir}"
            )

        if not self.labels_dir.is_dir():
            raise FileNotFoundError(
                f"Missing label directory: {self.labels_dir}"
            )

        self.samples = []
        missing_labels = 0
        invalid_boxes = 0

        image_paths = sorted(
            p for p in self.images_dir.iterdir()
            if p.is_file()
            and p.suffix.lower() in self.IMAGE_EXTENSIONS
        )

        for image_path in image_paths:
            label_path = self.labels_dir / f"{image_path.stem}.txt"

            if not label_path.exists():
                missing_labels += 1
                continue

            try:
                with Image.open(image_path) as im:
                    image_width, image_height = im.size
            except Exception as exc:
                print(f"Skipping unreadable image {image_path}: {exc}")
                continue

            try:
                lines = label_path.read_text(
                    encoding="utf-8"
                ).splitlines()
            except Exception as exc:
                print(f"Skipping unreadable label {label_path}: {exc}")
                continue

            for line_number, line in enumerate(lines, start=1):
                line = line.strip()
                if not line:
                    continue

                parts = line.split()

                if len(parts) < 5:
                    invalid_boxes += 1
                    continue

                try:
                    class_id = int(parts[0])
                    xc, yc, bw, bh = map(float, parts[1:5])
                except ValueError:
                    invalid_boxes += 1
                    continue

                if not 0 <= class_id < NUM_CLASSES:
                    invalid_boxes += 1
                    continue

                if not all(
                    math.isfinite(v) for v in (xc, yc, bw, bh)
                ):
                    invalid_boxes += 1
                    continue

                if bw <= 0 or bh <= 0:
                    invalid_boxes += 1
                    continue

                # Convert normalized YOLO xywh to pixel xyxy.
                x1 = (xc - bw / 2.0) * image_width
                y1 = (yc - bh / 2.0) * image_height
                x2 = (xc + bw / 2.0) * image_width
                y2 = (yc + bh / 2.0) * image_height

                # Clip the actual bounding box to the image.
                x1 = max(0.0, min(x1, image_width))
                y1 = max(0.0, min(y1, image_height))
                x2 = max(0.0, min(x2, image_width))
                y2 = max(0.0, min(y2, image_height))

                if x2 <= x1 or y2 <= y1:
                    invalid_boxes += 1
                    continue

                # Scale features describe the actual bbox, not
                # the expanded crop used for visual classification.
                box_area = (x2 - x1) * (y2 - y1)
                image_area = float(image_width * image_height)
                relative_area = box_area / image_area

                # Expand the crop slightly to retain context.
                margin_x = (x2 - x1) * BBOX_MARGIN
                margin_y = (y2 - y1) * BBOX_MARGIN

                crop_x1 = max(0, int(math.floor(x1 - margin_x)))
                crop_y1 = max(0, int(math.floor(y1 - margin_y)))
                crop_x2 = min(
                    image_width, int(math.ceil(x2 + margin_x))
                )
                crop_y2 = min(
                    image_height, int(math.ceil(y2 + margin_y))
                )

                if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
                    invalid_boxes += 1
                    continue

                self.samples.append({
                    "image_path": str(image_path),
                    "box": (
                        crop_x1, crop_y1, crop_x2, crop_y2
                    ),
                    "label": class_id,
                    "area_pixels": float(box_area),
                    "relative_area": float(relative_area),
                })

        print(f"\n[{split}]")
        print(f"Images found: {len(image_paths)}")
        print(f"Valid object crops: {len(self.samples)}")
        print(f"Images without labels: {missing_labels}")
        print(f"Invalid annotations: {invalid_boxes}")

        counts = Counter(s["label"] for s in self.samples)
        for class_id, name in enumerate(CLASS_NAMES):
            print(f"  Class {class_id} ({name}): {counts[class_id]} crops")

        if not self.samples:
            raise RuntimeError(
                f"No valid object crops found in {self.split_dir}. "
                "Check the YOLO labels and folder structure."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        with Image.open(sample["image_path"]) as im:
            image = im.convert("RGB")
            image = image.crop(sample["box"])

        if self.training:
            image = augment_crop(image)

        image_tensor = image_to_tensor(image)

        # Two scale features:
        # 1. log absolute area in original-image pixels
        # 2. bbox area / original image area
        scale_features = torch.tensor(
            [
                math.log1p(sample["area_pixels"])
                / LOG_AREA_DIVISOR,
                sample["relative_area"],
            ],
            dtype=torch.float32,
        )

        label = torch.tensor(
            sample["label"], dtype=torch.long
        )

        return image_tensor, scale_features, label


# ============================================================
# SCALE-AWARE EFFICIENTNETV2-M
# ============================================================

class ScaleAwareEfficientNetV2M(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()

        weights = EfficientNet_V2_M_Weights.IMAGENET1K_V1
        backbone = efficientnet_v2_m(weights=weights)

        self.features = backbone.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Normalization is part of this model's forward pass.
        self.register_buffer(
            "imagenet_mean",
            torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "imagenet_std",
            torch.tensor(IMAGENET_STD).view(1, 3, 1, 1),
        )

        visual_features = backbone.classifier[1].in_features
        assert visual_features == 1280

        self.classifier = nn.Sequential(
            nn.Linear(visual_features + 2, 512),
            nn.SiLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(512, num_classes),
        )

        # Keep pretrained BatchNorm affine parameters frozen.
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                for parameter in module.parameters():
                    parameter.requires_grad = False

        # Start by training the new head only.
        for parameter in self.features.parameters():
            parameter.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)

        # Always freeze BatchNorm running statistics.
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

        return self

    def unfreeze_backbone(self):
        for parameter in self.features.parameters():
            parameter.requires_grad = True

        # Re-freeze BatchNorm affine parameters.
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                for parameter in module.parameters():
                    parameter.requires_grad = False

    def forward(
        self,
        images: torch.Tensor,
        scale_features: torch.Tensor,
    ) -> torch.Tensor:
        # Input images are RGB float tensors in [0, 1].
        x = (images - self.imagenet_mean) / self.imagenet_std

        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)

        # Concatenate visual and original-scale features.
        x = torch.cat((x, scale_features), dim=1)

        # Return logits, not probabilities.
        return self.classifier(x)


# ============================================================
# TRAINING AND EVALUATION
# ============================================================

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()

    total_loss = 0.0
    total_samples = 0
    correct = 0

    for images, scale, labels in loader:
        images = images.to(device)
        scale = scale.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images, scale)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()

    return (
        total_loss / max(total_samples, 1),
        correct / max(total_samples, 1),
    )


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()

    all_labels = []
    all_predictions = []
    all_probabilities = []
    total_loss = 0.0
    total_samples = 0

    criterion = nn.CrossEntropyLoss()

    for images, scale, labels in loader:
        images = images.to(device)
        scale = scale.to(device)
        labels = labels.to(device)

        logits = model(images, scale)
        loss = criterion(logits, labels)
        probabilities = torch.softmax(logits, dim=1)
        predictions = logits.argmax(dim=1)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        all_labels.extend(labels.cpu().tolist())
        all_predictions.extend(predictions.cpu().tolist())
        all_probabilities.extend(probabilities.cpu().tolist())

    y_true = np.asarray(all_labels)
    y_pred = np.asarray(all_predictions)
    y_prob = np.asarray(all_probabilities)

    metrics = {
        "loss": total_loss / max(total_samples, 1),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "weighted_f1": f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=list(range(NUM_CLASSES)),
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=list(range(NUM_CLASSES)),
        ).tolist(),
    }

    return metrics, y_true, y_pred, y_prob


def save_confusion_matrix(matrix, path: Path):
    fig, ax = plt.subplots(figsize=(7, 6))

    ax.imshow(matrix, interpolation="nearest")
    ax.set(
        xticks=np.arange(NUM_CLASSES),
        yticks=np.arange(NUM_CLASSES),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        xlabel="Predicted class",
        ylabel="True class",
        title="Test Confusion Matrix",
    )

    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center")

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ============================================================
# EXPORT
# ============================================================

def export_models(model, device):
    model.eval()

    pt_path = OUTPUT_DIR / "scale_aware_efficientnet_v2_m.pt"
    onnx_path = OUTPUT_DIR / "scale_aware_efficientnet_v2_m.onnx"

    checkpoint = {
        "model_state_dict": model.cpu().state_dict(),
        "class_names": CLASS_NAMES,
        "num_classes": NUM_CLASSES,
        "image_size": IMAGE_SIZE,
        "model_name": "ScaleAwareEfficientNetV2-M",
        "scale_feature_names": [
            "log1p_bbox_area_pixels_div_20",
            "bbox_area_fraction_of_original_image",
        ],
        "log_area_divisor": LOG_AREA_DIVISOR,
        "bbox_margin": BBOX_MARGIN,
        "imagenet_mean": IMAGENET_MEAN,
        "imagenet_std": IMAGENET_STD,
    }

    torch.save(checkpoint, pt_path)

    model.to(device)
    model.eval()

    # Export on CPU for compatibility with the ONNX exporter.
    export_model = model.cpu().eval()

    dummy_images = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    dummy_scale = torch.tensor([[0.5, 0.1]], dtype=torch.float32)

    # Legacy TorchScript-based exporter is used deliberately for
    # compatibility with this model and broad ONNX tooling support.
    torch.onnx.export(
        export_model,
        (dummy_images, dummy_scale),
        str(onnx_path),
        input_names=["images", "scale_features"],
        output_names=["logits"],
        dynamic_axes={
            "images": {0: "batch_size"},
            "scale_features": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        opset_version=17,
        dynamo=False,
    )

    model.to(device)

    print(f"\nPyTorch checkpoint saved: {pt_path}")
    print(f"ONNX model saved: {onnx_path}")
    print("ONNX inputs: images, scale_features")
    print("ONNX output: logits")


# ============================================================
# MAIN
# ============================================================

def main():
    set_seed()

    device = get_device()
    print(f"Device: {device}")

    train_dataset = YOLOCropDataset("train", training=True)
    test_dataset = YOLOCropDataset("test", training=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    model = ScaleAwareEfficientNetV2M().to(device)

    # Calculate class weights from training crops only.
    counts = Counter(s["label"] for s in train_dataset.samples)
    class_counts = torch.tensor(
        [counts.get(i, 0) for i in range(NUM_CLASSES)],
        dtype=torch.float32,
    )

    if (class_counts == 0).any():
        missing = [
            CLASS_NAMES[i]
            for i in range(NUM_CLASSES)
            if class_counts[i] == 0
        ]
        raise RuntimeError(
            f"Training set has no examples for classes: {missing}"
        )

    # Inverse-square-root weighting is less extreme than 1 / count.
    class_weights = class_counts.rsqrt()
    class_weights = class_weights / class_weights.mean()
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device)
    )

    optimizer = torch.optim.AdamW(
        [
            {
                "params": list(model.features.parameters()),
                "lr": BACKBONE_LR,
            },
            {
                "params": list(model.classifier.parameters()),
                "lr": HEAD_LR,
            },
        ],
        weight_decay=WEIGHT_DECAY,
    )

    total_epochs = TOTAL_EPOCHS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_epochs
    )

    history = []

    print("\nStarting training...")
    print(f"Head-only epochs: {HEAD_ONLY_EPOCHS}")
    print(f"Fine-tuning epochs: {FINETUNE_EPOCHS}")

    for epoch in range(total_epochs):
        if epoch == HEAD_ONLY_EPOCHS:
            print("\nUnfreezing backbone for fine-tuning.")
            model.unfreeze_backbone()

        train_loss, train_accuracy = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        scheduler.step()

        record = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
        }
        history.append(record)

        print(
            f"Epoch {epoch + 1:02d}/{total_epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"train_accuracy={train_accuracy:.4f}"
        )

    # Save the final epoch, without selecting a model based on test data.
    with (OUTPUT_DIR / "training_history.json").open("w") as f:
        json.dump(history, f, indent=2)

    print("\nEvaluating once on the held-out test set...")
    metrics, y_true, y_pred, y_prob = evaluate(
        model, test_loader, device
    )

    print(f"Test loss:       {metrics['loss']:.4f}")
    print(f"Test accuracy:   {metrics['accuracy']:.4f}")
    print(f"Test macro-F1:   {metrics['macro_f1']:.4f}")
    print(f"Test weighted-F1:{metrics['weighted_f1']:.4f}")

    print("\nPer-class results:")
    print(
        json.dumps(metrics["classification_report"], indent=2)
    )

    with (OUTPUT_DIR / "test_metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    save_confusion_matrix(
        np.asarray(metrics["confusion_matrix"]),
        OUTPUT_DIR / "confusion_matrix.png",
    )

    # Restore model to CPU for checkpointing and export.
    export_models(model, device)

    print("\nAll outputs saved in:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
