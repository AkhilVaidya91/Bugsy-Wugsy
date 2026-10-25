"""Camera module: continuous streaming from Picamera2 with autofocus.

Colour order
------------
Picamera2's ``RGB888`` capture array is stored in BGR byte order (the
OpenCV-compatible layout). ``grab_frame()`` therefore returns a **BGR**
uint8 array (H, W, 3).

  • Convert to RGB before passing a frame to RF-DETR.
  • Use the frame directly for OpenCV drawing and ``cv2.imwrite``.
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

        # RGB888 capture_array uses BGR byte order, which is also the
        # convention used by OpenCV and the classifier crop interface.
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
        """Return the latest frame as a **BGR** uint8 array (H, W, 3)."""
        return self.cam.capture_array("main")

    def close(self) -> None:
        try:
            self.cam.stop()
            self.cam.close()
        except Exception:
            pass
