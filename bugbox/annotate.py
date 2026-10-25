from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

PALETTE = [(60, 180, 75), (230, 120, 40), (60, 60, 230), (200, 60, 200), (60, 200, 200), (140, 140, 140)]


def annotate(img: np.ndarray, dets, counts: dict, total: int, labeler, max_side: int = 1600) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    thick = max(2, int(max(h, w) / 600))
    scale = max(0.5, max(h, w) / 1800)
    names = labeler.targets + [labeler.other]
    for d in dets:
        c = labeler.canonical(d.label)
        color = PALETTE[names.index(c) % len(PALETTE)]
        x1, y1, x2, y2 = [int(v) for v in d.box]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)
        mark = {"cloud": "*", "cache": "*"}.get(getattr(d, "source", "local"), "")
        cv2.putText(out, f"{c} {d.confidence:.2f}{mark}", (x1, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                    scale * 0.6, color, max(1, thick - 1), cv2.LINE_AA)
    banner = "  ".join(f"{k}: {v}" for k, v in counts.items()) + f"   TOTAL: {total}"
    bh = int(40 * scale) + 10
    cv2.rectangle(out, (0, 0), (w, bh), (0, 0, 0), -1)
    cv2.putText(out, banner, (10, int(bh * 0.7)), cv2.FONT_HERSHEY_SIMPLEX, scale * 0.8, (255, 255, 255),
                max(1, thick - 1), cv2.LINE_AA)
    s = min(1.0, max_side / max(h, w))
    if s < 1.0:
        out = cv2.resize(out, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return out


def save_jpeg(path: Path, img: np.ndarray, quality: int = 90):
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, quality])
