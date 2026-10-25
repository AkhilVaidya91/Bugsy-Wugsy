"""Detection: RF-DETR Small inference with a local .pt weights file."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np


@dataclass
class Detection:
    """Single bounding-box detection from the detector."""
    x1: float   # top-left pixel x
    y1: float   # top-left pixel y
    x2: float   # bottom-right pixel x
    y2: float   # bottom-right pixel y
    confidence: float

    @property
    def w(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def h(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2


class Detector:
    """Runs RF-DETR Small on RGB frames using a locally downloaded .pt file."""

    def __init__(self, weights: str, confidence: float = 0.4):
        path = Path(weights).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Detector weights not found: {path}")

        try:
            from rfdetr import RFDETRSmall
        except ImportError as exc:
            raise RuntimeError(
                "Install the rfdetr package first:  pip install rfdetr"
            ) from exc

        self.conf = confidence
        # pretrain_weights loads a locally saved .pt checkpoint
        self.model = RFDETRSmall(pretrain_weights=str(path))

    def detect(self, rgb_frame: np.ndarray) -> List[Detection]:
        """
        Parameters
        ----------
        rgb_frame : np.ndarray
            H×W×3 uint8 array in RGB colour order (as returned by Picamera2).

        Returns
        -------
        List[Detection]
            One entry per insect found above the confidence threshold.
        """
        result = self.model.predict(rgb_frame, threshold=self.conf)
        if isinstance(result, (list, tuple)):
            result = result[0] if result else None
        if result is None:
            return []

        xyxy = np.asarray(getattr(result, "xyxy", []))
        confs = np.asarray(getattr(result, "confidence", []))

        out: List[Detection] = []
        for i, box in enumerate(xyxy):
            x1, y1, x2, y2 = map(float, box)
            score = float(confs[i]) if i < len(confs) else 0.0
            out.append(Detection(x1, y1, x2, y2, score))
        return out
