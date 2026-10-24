from pathlib import Path
import csv
import cv2
import math
from ultralytics import YOLO

# Paths and settings
MODEL_PATH = "best.pt"
IMAGE_DIR = Path("input_images")
OUTPUT_DIR = Path("output_crops")
CROP_DIR = OUTPUT_DIR / "crops"

IMG_SIZE = 640
CONF_THRESHOLD = 0.25
EXPANSION = 1.2

CROP_DIR.mkdir(parents=True, exist_ok=True)
model = YOLO(MODEL_PATH)

image_paths = sorted(
    p for p in IMAGE_DIR.iterdir()
    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
)

csv_path = OUTPUT_DIR / "predictions.csv"

fields = [
    "source_image", "crop_path", "detection_id", "class_id",
    "class_name", "confidence", "image_width", "image_height",
    "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
    "bbox_width", "bbox_height", "bbox_area",
    "crop_x1", "crop_y1", "crop_x2", "crop_y2",
    "crop_width", "crop_height", "crop_area"
]

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()

    for result in model.predict(
        source=[str(p) for p in image_paths],
        imgsz=IMG_SIZE,
        conf=CONF_THRESHOLD,
        stream=True,
        verbose=False
    ):
        image = result.orig_img  # Original image, in BGR format
        image_h, image_w = image.shape[:2]

        if result.boxes is None:
            continue

        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        source_name = Path(result.path).name

        for i, (box, confidence, class_id) in enumerate(
            zip(boxes, confidences, classes), start=1
        ):
            x1, y1, x2, y2 = box.tolist()
            bbox_w, bbox_h = x2 - x1, y2 - y1
            bbox_area = bbox_w * bbox_h

            # Expand the box by 1.2x around its centre
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            crop_x1 = max(0, math.floor(cx - bbox_w * EXPANSION / 2))
            crop_y1 = max(0, math.floor(cy - bbox_h * EXPANSION / 2))
            crop_x2 = min(image_w, math.ceil(cx + bbox_w * EXPANSION / 2))
            crop_y2 = min(image_h, math.ceil(cy + bbox_h * EXPANSION / 2))

            crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
            if crop.size == 0:
                continue

            crop_h, crop_w = crop.shape[:2]
            class_name = str(result.names[class_id]).replace("/", "_").replace("\\", "_")

            crop_name = f"{Path(source_name).stem}_det{i:03d}_{class_name}.jpg"
            crop_path = CROP_DIR / crop_name

            if not cv2.imwrite(str(crop_path), crop):
                print(f"Could not save crop: {crop_path}")
                continue

            writer.writerow({
                "source_image": source_name,
                "crop_path": str(crop_path),
                "detection_id": i,
                "class_id": class_id,
                "class_name": class_name,
                "confidence": float(confidence),
                "image_width": image_w,
                "image_height": image_h,
                "bbox_x1": x1, "bbox_y1": y1,
                "bbox_x2": x2, "bbox_y2": y2,
                "bbox_width": bbox_w,
                "bbox_height": bbox_h,
                "bbox_area": bbox_area,
                "crop_x1": crop_x1, "crop_y1": crop_y1,
                "crop_x2": crop_x2, "crop_y2": crop_y2,
                "crop_width": crop_w,
                "crop_height": crop_h,
                "crop_area": crop_w * crop_h
            })

print(f"Crops saved to: {CROP_DIR}")
print(f"Metadata saved to: {csv_path}")