"""Camera module: continuous streaming from Picamera2 with autofocus."""
from __future__ import annotations

import time

import numpy as np


class Camera:
    """Wraps Picamera2 for continuous RGB frame streaming."""

    def __init__(self, width: int = 1920, height: int = 1080, af: bool = True):
        from libcamera import controls
        from picamera2 import Picamera2

        self._controls = controls
        self.cam = Picamera2()
        cfg = self.cam.create_preview_configuration(
            main={"size": (width, height), "format": "RGB888"}
        )
        self.cam.configure(cfg)
        self.cam.start()
        time.sleep(2.0)  # let AE/AWB settle

        if af:
            self._run_autofocus()

    # ------------------------------------------------------------------
    def _run_autofocus(self) -> None:
        from libcamera import controls

        self.cam.set_controls({
            "AfMode": controls.AfModeEnum.Continuous,
            "AfRange": controls.AfRangeEnum.Normal,
        })
        # Give the continuous AF a moment to settle
        time.sleep(1.5)

    def grab_frame(self) -> np.ndarray:
        """Return the latest frame as an RGB uint8 array (H, W, 3)."""
        return self.cam.capture_array("main")

    def close(self) -> None:
        try:
            self.cam.stop()
            self.cam.close()
        except Exception:
            pass
