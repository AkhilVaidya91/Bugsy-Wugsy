"""
pipeline/main.py – continuous insect-counting pipeline for Raspberry Pi 5.

Usage
-----
python -m pipeline.main \\
    --detector  models/weights.pt \\
    --classifier models/scale_effic.onnx \\
    --out-dir   captures \\
    --server    http://192.168.1.50:5000/upload \\
    --save-every 10 \\
    --det-conf  0.4 \\
    --lights-pins 26

The ring lights (relay board on GPIO) are switched on before the camera
starts, so autofocus runs under the same light as the frames you analyse,
and are switched off again when the pipeline stops.

Press Ctrl-C to stop.
"""
from __future__ import annotations

import argparse
import logging
import signal
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict

import cv2
import numpy as np

from .camera import Camera
from .classifier import CLASSES, Classifier
from .detector import Detector
from .display import EInk
from .lights import Lights, parse_pins
from .storage import Storage, draw_annotations
from .uploader import Uploader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pipeline")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _empty_counts() -> Dict[str, int]:
    return {c: 0 for c in CLASSES}


def _smooth_counts(history: deque) -> Dict[str, int]:
    """Median-smooth counts across the recent frame history."""
    import statistics
    keys = CLASSES
    return {
        k: int(round(statistics.median([c.get(k, 0) for c in history])))
        for k in keys
    }


def _on_sigterm(signum, frame):  # noqa: ARG001
    """Treat SIGTERM (systemd stop, `kill`) like Ctrl-C so the lights are switched off."""
    raise KeyboardInterrupt


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    signal.signal(signal.SIGTERM, _on_sigterm)

    # Lights first: creating the object also forces the relay OFF, so a relay
    # left on by an earlier run is switched off while the models load.
    lights = Lights(
        pins=parse_pins(args.lights_pins),
        active_low=not args.lights_active_high,
        enabled=not args.no_lights,
    )

    cam = None
    try:
        log.info("Loading detector from %s", args.detector)
        detector = Detector(args.detector, confidence=args.det_conf)

        log.info("Loading classifier from %s", args.classifier)
        classifier = Classifier(args.classifier)

        storage = Storage(args.out_dir)

        eink = EInk(min_s=args.display_interval)

        uploader: Uploader | None = None
        if args.server:
            log.info("Uploading to %s", args.server)
            uploader = Uploader(args.server, timeout_s=args.upload_timeout)

        # Lights on BEFORE the camera starts, so the autofocus cycle in
        # Camera() focuses under the real lighting.
        if lights.enabled:
            lights.on()
            log.info(
                "Lights on (%s) – waiting %.1fs to settle",
                "relay ok" if lights.available else "NO GPIO: lights not switched",
                args.lights_settle,
            )
            time.sleep(args.lights_settle)

        log.info("Starting camera (AF will block until settled)")
        cam = Camera(
            width=args.width,
            height=args.height,
            af=not args.no_af,
        )

        smooth_window = deque(maxlen=max(1, args.smooth))
        frame_n = 0          # frames processed since start (never reset)
        fps_frames = 0       # frames in the current ~1 s FPS window
        fps_clock = time.time()
        fps_estimate = 0.0

        log.info("Pipeline running – press Ctrl-C to stop")
        while True:
            frame_n += 1
            fps_frames += 1

            # 1. Grab frame.
            #    RGB888 capture arrays are stored BGR. Keep that representation
            #    for OpenCV and crops, and convert only for RF-DETR.
            bgr = cam.grab_frame()
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            # 2. Detect (model expects RGB)
            detections = detector.detect(rgb)

            # 3. Classify each crop.
            #    Slice from bgr; classifier.classify() does BGR→RGB internally.
            classifications = []
            counts = _empty_counts()
            for det in detections:
                x1, y1 = max(0, int(det.x1)), max(0, int(det.y1))
                x2 = min(bgr.shape[1], int(det.x2))
                y2 = min(bgr.shape[0], int(det.y2))
                if x2 <= x1 or y2 <= y1:
                    classifications.append(("other", 0.0))
                    counts["other"] += 1
                    continue

                crop = bgr[y1:y2, x1:x2]   # BGR crop
                label, conf = classifier.classify(
                    crop,
                    det.w,
                    det.h,
                    bgr.shape[1],
                    bgr.shape[0],
                )
                classifications.append((label, conf))
                counts[label] = counts.get(label, 0) + 1

            # 4. Smooth counts over recent frames
            smooth_window.append(counts)
            smooth_counts = _smooth_counts(smooth_window)
            smooth_total = sum(smooth_counts.values())

            # 5. FPS estimate (rolling over ~1 s)
            elapsed = time.time() - fps_clock
            if elapsed >= 1.0:
                fps_estimate = fps_frames / elapsed
                fps_frames = 0
                fps_clock = time.time()

            # 6. Annotate and save every N frames
            if args.save_every and (frame_n % args.save_every == 0):
                ann = draw_annotations(bgr, detections, classifications, smooth_counts)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                saved_path = storage.save(ann, smooth_counts)
                log.info(
                    "Saved %s  counts=%s",
                    saved_path.name,
                    smooth_counts,
                )

                if uploader is not None:
                    uploader.submit(saved_path, smooth_counts, timestamp)

            # 7. Update e-ink display
            eink.update({
                "counts": smooth_counts,
                "total": smooth_total,
                "updated": datetime.now().strftime("%H:%M:%S"),
                "frame": frame_n,
                "fps": f"{fps_estimate:.1f}",
            })

            # 8. Console heartbeat
            if frame_n % max(1, args.log_every) == 0:
                log.info(
                    "frame=%d  counts=%s  total=%d  detections=%d  fps=%.1f",
                    frame_n,
                    smooth_counts,
                    smooth_total,
                    len(detections),
                    fps_estimate,
                )

    except KeyboardInterrupt:
        log.info("Interrupted – shutting down")
    finally:
        lights.off()          # always, even if model loading or the camera failed
        if cam is not None:
            cam.close()
        log.info("Done (lights off)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Continuous insect-counting pipeline (RF-DETR + ONNX classifier)"
    )
    p.add_argument(
        "--detector",
        default="models/weights.pt",
        help="Path to RF-DETR Small .pt weights file",
    )
    p.add_argument(
        "--classifier",
        default="models/scale_effic.onnx",
        help="Path to ONNX classifier weights file",
    )
    p.add_argument(
        "--out-dir",
        default="captures",
        help="Directory to store annotated images and the CSV log",
    )
    p.add_argument(
        "--server",
        default="",
        help="Flask server URL, e.g. http://192.168.1.50:5000/upload  (empty = no upload)",
    )
    p.add_argument(
        "--det-conf",
        type=float,
        default=0.4,
        help="Minimum detector confidence threshold",
    )
    p.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Save one annotated frame every N frames (0 = never)",
    )
    p.add_argument(
        "--smooth",
        type=int,
        default=5,
        help="Number of recent frames used for count smoothing",
    )
    p.add_argument(
        "--display-interval",
        type=float,
        default=10.0,
        help="Minimum seconds between e-ink refreshes",
    )
    p.add_argument(
        "--upload-timeout",
        type=float,
        default=5.0,
        help="Per-upload network timeout in seconds",
    )
    p.add_argument(
        "--log-every",
        type=int,
        default=30,
        help="Print a counts line every N frames",
    )
    p.add_argument("--width", type=int, default=1920, help="Camera frame width")
    p.add_argument("--height", type=int, default=1080, help="Camera frame height")
    p.add_argument(
        "--no-af",
        action="store_true",
        help="Disable the start-up autofocus cycle (use fixed focus)",
    )

    # --- ring lights (relay board on GPIO) --------------------------------
    p.add_argument(
        "--no-lights",
        action="store_true",
        help="Do not switch the ring lights (e.g. when testing without the relay)",
    )
    p.add_argument(
        "--lights-pins",
        default="26",
        help="BCM GPIO pin(s) of the relay that switches the ring lights, e.g. 26 or 26,21",
    )
    p.add_argument(
        "--lights-active-high",
        action="store_true",
        help="Relay board switches ON when its pin is HIGH (default: ON when LOW)",
    )
    p.add_argument(
        "--lights-settle",
        type=float,
        default=1.0,
        help="Seconds to wait after the lights come on, before the camera starts",
    )
    return p.parse_args()


if __name__ == "__main__":
    run(_parse_args())
