"""Classifier: ONNX EfficientNetV2-M multi-resolution crop classifier.

The model was exported from classification_training/train.py.
It expects two ONNX inputs:
  - image  : float32 [1, 3, 224, 224]  (ImageNet-normalised, RGB channel order)
  - scale  : float32 [1, 2]            (log1p(bbox_w), log1p(bbox_h))

It returns logits float32 [1, 4] for classes ["1", "2", "3", "other"].

``classify()`` accepts a **BGR** crop (OpenCV convention) and converts
to RGB internally before normalisation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps

CLASSES = ("1", "2", "3", "other")

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class Classifier:
    """Wraps the ONNX classifier session for single-crop inference."""

    def __init__(self, weights: str):
        path = Path(weights).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Classifier weights not found: {path}")

        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "Install onnxruntime first:  pip install onnxruntime"
            ) from exc

        self.session = ort.InferenceSession(
            str(path), providers=["CPUExecutionProvider"]
        )
        inputs = self.session.get_inputs()
        self._image_input = inputs[0].name
        self._scale_input = inputs[1].name
        self._output = self.session.get_outputs()[0].name

    def classify(
        self,
        bgr_crop: np.ndarray,
        bbox_w: float,
        bbox_h: float,
    ) -> Tuple[str, float]:
        """
        Parameters
        ----------
        bgr_crop : np.ndarray
            BGR uint8 crop as returned by OpenCV indexing.
        bbox_w, bbox_h : float
            Bounding-box width and height in pixels (used as size metadata).

        Returns
        -------
        (class_label, confidence) e.g. ("1", 0.87)
        """
        # BGR → RGB → PIL, preserve aspect ratio, pad to 224×224
        rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
        pil = ImageOps.pad(Image.fromarray(rgb), (224, 224), color=(0, 0, 0))

        # Float32 image tensor: [1, 3, 224, 224]
        arr = np.asarray(pil, dtype=np.float32) / 255.0
        arr = (arr - _MEAN) / _STD
        arr = np.transpose(arr, (2, 0, 1))[np.newaxis, ...]  # [1, C, H, W]

        # Scale metadata: log1p(w), log1p(h)
        scale = np.log1p(
            np.array([[max(bbox_w, 0.0), max(bbox_h, 0.0)]], dtype=np.float32)
        )

        logits = self.session.run(
            [self._output],
            {self._image_input: arr, self._scale_input: scale},
        )[0]  # [1, 4]

        # Softmax
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = (exp / exp.sum(axis=1, keepdims=True))[0]

        idx = int(np.argmax(probs))
        return CLASSES[idx], float(probs[idx])
