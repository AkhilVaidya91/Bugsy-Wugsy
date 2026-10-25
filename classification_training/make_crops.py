# python make_crops.py \
#   --dataset bugs_dataset \
#   --output-dir data

import argparse
import math
import re
from pathlib import Path

import pandas as pd
from PIL import Image


# YOLO class id -> classifier label used by train.py.
YOLO_TO_LABEL = {0: "1", 1: "2", 2: "3", 3: "other"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Same 1.2x crop expansion as detection_to_dataset/infer.py.
EXPANSION = 1.2


def source_image_id(image_path):
    """Strip the Roboflow export suffix so copies of one photo share an id."""
    return re.sub(r"_(jpe?g|png|bmp|webp)\.rf\..*$", "", image_path.stem)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=Path, default=Path("bugs_dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("data"))

    args = parser.parse_args()

    crop_dir = args.output_dir / "crops"
    crop_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    skipped = 0

    # train.py makes its own grouped split, so pool every Roboflow split.
    for split in ["train", "valid", "test"]:
        images_dir = args.dataset / split / "images"
        labels_dir = args.dataset / split / "labels"

        if not images_dir.is_dir():
            continue

        for image_path in sorted(images_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            label_path = labels_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                continue

            with Image.open(image_path) as im:
                image = im.convert("RGB")

            image_w, image_h = image.size

            lines = label_path.read_text(encoding="utf-8").splitlines()

            for i, line in enumerate(lines):
                parts = line.split()
                if len(parts) != 5:
                    skipped += int(bool(parts))
                    continue

                class_id = int(parts[0])
                if class_id not in YOLO_TO_LABEL:
                    skipped += 1
                    continue

                xc, yc, bw, bh = map(float, parts[1:])

                # Original (unexpanded) box size in pixels.
                bbox_w = bw * image_w
                bbox_h = bh * image_h
                cx = xc * image_w
                cy = yc * image_h

                if bbox_w <= 0 or bbox_h <= 0:
                    skipped += 1
                    continue

                crop_x1 = max(0, math.floor(cx - bbox_w * EXPANSION / 2))
                crop_y1 = max(0, math.floor(cy - bbox_h * EXPANSION / 2))
                crop_x2 = min(image_w, math.ceil(cx + bbox_w * EXPANSION / 2))
                crop_y2 = min(image_h, math.ceil(cy + bbox_h * EXPANSION / 2))

                if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
                    skipped += 1
                    continue

                label = YOLO_TO_LABEL[class_id]
                crop_name = f"{split}_{image_path.stem}_det{i:03d}_{label}.jpg"

                image.crop(
                    (crop_x1, crop_y1, crop_x2, crop_y2)
                ).save(crop_dir / crop_name, quality=95)

                rows.append({
                    "crop_path": f"crops/{crop_name}",
                    "label": label,
                    "source_image_id": source_image_id(image_path),
                    "bbox_width": bbox_w,
                    "bbox_height": bbox_h,
                    # Ground-truth boxes have no detector confidence.
                    "yolo_confidence": 1.0,
                    "aspect_ratio": bbox_w / max(bbox_h, 1),
                })

    df = pd.DataFrame(rows)
    csv_path = args.output_dir / "bug_crops.csv"
    df.to_csv(csv_path, index=False)

    print(f"Crops saved: {len(df)} -> {crop_dir}")
    print(f"Skipped annotations: {skipped}")
    print(f"Source images: {df.source_image_id.nunique()}")
    print("Class counts:", df.label.value_counts().to_dict())
    print(f"Manifest saved: {csv_path}")


if __name__ == "__main__":
    main()
