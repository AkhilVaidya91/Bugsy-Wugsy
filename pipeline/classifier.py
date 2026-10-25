"""Classifier: ONNX scale-aware EfficientNetV2-M crop classifier.

The bundled model is exported from
``scale_aware/Bugs Detection.yolo26/train.py``. It expects:
- two ONNX inputs:
- images        : float32 [1, 3, 224, 224] (RGB values in [0, 1])
- scale_features: float32 [1, 2]            (normalised area features)

It returns logits float32 [1, 4] for classes ["1", "2", "3", "other"].

``classify()`` accepts a **BGR** crop (OpenCV convention) and converts
to RGB internally. ImageNet normalisation is performed inside the ONNX model.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps

CLASSES = ("1", "2", "3", "other")
_LOG_AREA_DIVISOR = 20.0


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
        if len(inputs) != 2 or inputs[0].name != "images" or inputs[1].name != "scale_features":
            raise RuntimeError(
                "Classifier model contract mismatch: expected ONNX inputs "
                "'images' and 'scale_features'. Use the scale-aware exported model."
            )
        self._image_input = inputs[0].name
        self._scale_input = inputs[1].name
        self._output = self.session.get_outputs()[0].name

    def classify(
        self,
        bgr_crop: np.ndarray,
        bbox_w: float,
        bbox_h: float,
        frame_w: int,
        frame_h: int,
    ) -> Tuple[str, float]:
        """
        Parameters
        ----------
        bgr_crop : np.ndarray
            BGR uint8 crop as returned by OpenCV indexing.
        bbox_w, bbox_h : float
            Original detector bounding-box dimensions in pixels.
        frame_w, frame_h : int
            Original frame dimensions in pixels.

        Returns
        -------
        (class_label, confidence) e.g. ("1", 0.87)
        """
        if frame_w <= 0 or frame_h <= 0:
            raise ValueError("frame dimensions must be positive")

        # BGR → RGB → PIL, preserve aspect ratio, pad to 224×224.
        # The training pipeline uses bilinear resizing and the model performs
        # ImageNet normalization inside its forward pass.
        rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
        pil = ImageOps.pad(
            Image.fromarray(rgb),
            (224, 224),
            method=Image.Resampling.BILINEAR,
            color=(0, 0, 0),
            centering=(0.5, 0.5),
        )

        # Float32 image tensor: [1, 3, 224, 224], RGB in [0, 1].
        arr = np.asarray(pil, dtype=np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))[np.newaxis, ...]  # [1, C, H, W]

        # Match scale_aware/train.py exactly:
        # log1p(original bbox area) / 20 and bbox area / original frame area.
        area = max(bbox_w, 0.0) * max(bbox_h, 0.0)
        scale = np.array(
            [[np.log1p(area) / _LOG_AREA_DIVISOR,
              area / float(frame_w * frame_h)]],
            dtype=np.float32,
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
