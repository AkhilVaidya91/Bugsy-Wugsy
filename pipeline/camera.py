"""Camera module: continuous streaming from Picamera2 with autofocus.

Colour order
------------
Picamera2 with format ``RGB888`` returns pixels as R, G, B in memory.
``grab_frame()`` therefore returns an **RGB** uint8 array (H, W, 3).

  • Pass it directly to RF-DETR (which expects RGB).
  • Convert to BGR with ``cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)``
    before any OpenCV drawing or ``cv2.imwrite`` call.
  • The classifier's ``classify()`` accepts BGR crops and converts
    to RGB internally before normalisation.

Autofocus strategy
------------------
Continuous AF hunts during streaming and can leave the lens at a wrong
position for many frames.  Instead we run a **blocking one-shot AF
cycle** at startup (identical to the reference capture script), read
the settled lens position, then lock the camera to Manual mode at that
position for the rest of the session.
"""
from __future__ import annotations

import logging
import time

import numpy as np

log = logging.getLogger(__name__)


class Camera:
    """Wraps Picamera2 for continuous RGB frame streaming."""

    def __init__(self, width: int = 1920, height: int = 1080, af: bool = True):
        from libcamera import controls
        from picamera2 import Picamera2

        self.cam = Picamera2()

        # RGB888 → capture_array returns R, G, B byte order (confirmed by
        # Picamera2 docs). Convert to BGR in main.py before any OpenCV ops.
        cfg = self.cam.create_preview_configuration(
            main={"size": (width, height), "format": "RGB888"}
        )
        self.cam.configure(cfg)
        self.cam.start()
        time.sleep(2.0)  # let AE / AWB settle

        if af:
            self._focus_and_lock(controls)

    # ------------------------------------------------------------------
    def _focus_and_lock(self, controls) -> None:
        """Run one blocking AF cycle, then lock the lens at that position."""
        self.cam.set_controls({
            "AfMode": controls.AfModeEnum.Auto,
            "AfRange": controls.AfRangeEnum.Normal,
        })

        ok = self.cam.autofocus_cycle()          # blocks until AF completes
        lens = self.cam.capture_metadata().get("LensPosition")
        log.info("AF %s, lens position %s", "ok" if ok else "FAILED", lens)

        if ok and lens is not None:
            # Lock focus so streaming frames don't suffer from AF hunting
            self.cam.set_controls({
                "AfMode": controls.AfModeEnum.Manual,
                "LensPosition": lens,
            })
        else:
            log.warning("AF failed – continuing with camera default focus")

    # ------------------------------------------------------------------
    def grab_frame(self) -> np.ndarray:
        """Return the latest frame as an **RGB** uint8 array (H, W, 3)."""
        return self.cam.capture_array("main")

    def close(self) -> None:
        try:
            self.cam.stop()
            self.cam.close()
        except Exception:
            pass
