"""Camera module: continuous streaming from Picamera2 with autofocus.

Colour order
------------
Picamera2 with format ``BGR888`` stores pixels as B, G, R in memory —
the native byte order for OpenCV.  ``grab_frame()`` therefore returns a
BGR uint8 array (H, W, 3) that can be passed directly to cv2 functions
or annotated and saved with ``cv2.imwrite``.  Convert to RGB only when
needed by a model that expects RGB (e.g. RF-DETR).

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
    """Wraps Picamera2 for continuous BGR frame streaming."""

    def __init__(self, width: int = 1920, height: int = 1080, af: bool = True):
        from libcamera import controls
        from picamera2 import Picamera2

        self.cam = Picamera2()

        # BGR888 → capture_array gives native OpenCV byte order (B, G, R)
        cfg = self.cam.create_preview_configuration(
            main={"size": (width, height), "format": "BGR888"}
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
