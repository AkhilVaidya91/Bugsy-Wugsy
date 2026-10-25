import math
import shutil
from pathlib import Path

from PIL import Image
from ultralytics import YOLO

DATA_YAML = Path("data.yaml")     # path to your yaml
DATASET_DIR = Path("dataset")     # folder containing your train/test images
EPOCHS = 80
MAX_IMGSZ = 640                  # cap on resolution
OUTPUT_DIR = Path("output")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def main():
    sizes = []
    for p in DATASET_DIR.rglob("*"):
        if p.suffix.lower() in IMG_EXTS:
            with Image.open(p) as im:
                sizes.append(max(im.size))
    imgsz = min(math.ceil(max(sizes) / 32) * 32, MAX_IMGSZ)
    print(f"Training at imgsz={imgsz}")

    model = YOLO("yolo26s.pt")
    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        patience=50,
        imgsz=imgsz,
        batch=-1,
        cache="ram",
        project=str(Path("runs").resolve()),
        name="train",
        exist_ok=True,
    )

    OUTPUT_DIR.mkdir(exist_ok=True)
    shutil.copy2(model.trainer.best, OUTPUT_DIR / "best.pt")
    print(f"Saved: {(OUTPUT_DIR / 'best.pt').resolve()}")


if __name__ == "__main__":
    main()