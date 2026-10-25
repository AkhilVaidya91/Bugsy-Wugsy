"""Storage: save annotated images and append rows to a CSV log."""
from __future__ import annotations

import csv
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

CLASSES = ("1", "2", "3", "other")
PALETTE = {
    "1": (60, 180, 75),
    "2": (230, 120, 40),
    "3": (60, 60, 230),
    "other": (200, 60, 200),
}
_CSV_HEADER = [
    "timestamp", "image_filename",
    "count_1", "count_2", "count_3", "count_other", "total",
]


# ---------------------------------------------------------------------------
# Annotation helper
# ---------------------------------------------------------------------------

def draw_annotations(
    bgr_frame: np.ndarray,
    detections,  # List[Detection] from detector.py
    classifications: List[tuple],  # [(label, confidence), ...]
    counts: Dict[str, int],
) -> np.ndarray:
    """Return a copy of *bgr_frame* with bounding boxes and counts drawn on it."""
    out = bgr_frame.copy()
    h, w = out.shape[:2]
    thick = max(2, int(max(h, w) / 600))
    font_scale = max(0.4, max(h, w) / 3000)

    for det, (label, conf) in zip(detections, classifications):
        color = PALETTE.get(label, (140, 140, 140))
        x1, y1, x2, y2 = int(det.x1), int(det.y1), int(det.x2), int(det.y2)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)
        text = f"{label} {conf:.2f}"
        cv2.putText(
            out, text,
            (x1, max(14, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            max(1, thick - 1),
            cv2.LINE_AA,
        )

    # Count banner at the top
    banner = "  ".join(f"{c}:{counts.get(c, 0)}" for c in CLASSES)
    banner += f"   TOTAL:{sum(counts.values())}"
    bh = max(30, int(36 * font_scale * 2))
    cv2.rectangle(out, (0, 0), (w, bh), (0, 0, 0), -1)
    cv2.putText(
        out, banner,
        (8, int(bh * 0.7)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale * 1.2,
        (255, 255, 255),
        max(1, thick - 1),
        cv2.LINE_AA,
    )
    return out


# ---------------------------------------------------------------------------
# Storage manager
# ---------------------------------------------------------------------------

class Storage:
    """Saves annotated frames and writes CSV rows to *out_dir*."""

    def __init__(self, out_dir: str):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._csv_path = self.out_dir / "counts.csv"
        self._ensure_csv_header()

    # ------------------------------------------------------------------
    def _ensure_csv_header(self) -> None:
        if not self._csv_path.exists() or self._csv_path.stat().st_size == 0:
            with open(self._csv_path, "w", newline="") as f:
                csv.writer(f, lineterminator="\n").writerow(_CSV_HEADER)

    def save(
        self,
        annotated_frame: np.ndarray,
        counts: Dict[str, int],
        jpeg_quality: int = 85,
    ) -> Path:
        """
        Write the annotated JPEG and append a CSV row.

        Returns the saved image path.
        """
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        img_name = f"frame_{stamp}.jpg"
        img_path = self.out_dir / img_name

        cv2.imwrite(
            str(img_path),
            annotated_frame,
            [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
        )

        self._append_csv(stamp, img_name, counts)
        return img_path

    def _append_csv(
        self, timestamp: str, filename: str, counts: Dict[str, int]
    ) -> None:
        row = [
            timestamp,
            filename,
            counts.get("1", 0),
            counts.get("2", 0),
            counts.get("3", 0),
            counts.get("other", 0),
            sum(counts.values()),
        ]
        with open(self._csv_path, "a", newline="") as f:
            csv.writer(f, lineterminator="\n").writerow(row)
