"""Cameras: Picamera2 (Pi camera / Arducam 64MP), rpicam-still fallback, USB webcam, and a fake for testing."""
from __future__ import annotations

import random
import shutil
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

from .detect import Detection
from .util import log


class CameraError(RuntimeError):
    pass


class BaseCamera:
    name = "base"
    lens_position = None

    def set_style(self, style: str): ...
    def grab_frame(self) -> np.ndarray: raise NotImplementedError          # BGR array, stream size
    def capture_still(self, path: Path) -> dict: raise NotImplementedError  # writes a JPEG, returns metadata
    def calibrate(self): ...
    def close(self): ...


# ---------------------------------------------------------------- Picamera2
class PiCamera2Camera(BaseCamera):
    name = "picamera2"

    def __init__(self, cfg):
        from picamera2 import Picamera2
        from libcamera import Transform
        self.cfg = cfg
        self.c = cfg.camera
        self.s = cfg.stream
        self.lock = threading.RLock()
        self.picam = Picamera2()
        tr = Transform(vflip=bool(self.c.vflip), hflip=bool(self.c.hflip))
        self.still_cfg = self.picam.create_still_configuration(
            main={"size": (int(self.c.width), int(self.c.height)), "format": "RGB888"},
            transform=tr, raw=None, lores=None)
        self.stream_cfg = self.picam.create_video_configuration(
            main={"size": (int(self.s.width), int(self.s.height)), "format": "RGB888"}, transform=tr)
        self.picam.options["quality"] = int(self.c.jpeg_quality)
        self.style = None
        self.lens_position = None
        self._last_cal = 0.0
        log.info("camera: Picamera2 ready (%s)", self.picam.camera_properties.get("Model", "?"))

    def set_style(self, style):
        with self.lock:
            if style == self.style:
                return
            try:
                self.picam.stop()
            except Exception:
                pass
            self.picam.configure(self.stream_cfg if style == "stream" else self.still_cfg)
            self.picam.start()
            self._apply_fixed_controls()
            self.style = style
            time.sleep(1.0)

    def _apply_fixed_controls(self):
        ctl = {}
        if float(self.c.exposure_us) > 0:
            ctl["ExposureTime"] = int(self.c.exposure_us)
        if float(self.c.gain) > 0:
            ctl["AnalogueGain"] = float(self.c.gain)
        if self.c.focus == "manual":
            from libcamera import controls
            ctl.update({"AfMode": controls.AfModeEnum.Manual, "LensPosition": float(self.c.lens_position)})
        if ctl:
            self.picam.set_controls(ctl)

    def calibrate(self, force: bool = False):
        """Autofocus now (call with the lights on), then lock the lens, like the Mothbox's calibration."""
        if self.c.focus != "calibrate":
            return
        if not force and time.time() - self._last_cal < float(self.c.calibrate_every_s):
            return
        with self.lock:
            from libcamera import controls
            if self.style is None:
                self.set_style("still")
            self.picam.set_controls({"AfMode": controls.AfModeEnum.Auto})
            ok = self.picam.autofocus_cycle()
            md = self.picam.capture_metadata()
            pos = md.get("LensPosition")
            if ok and pos is not None and 0.5 <= float(pos) <= 15.0:
                self.lens_position = float(pos)
            elif self.lens_position is None and pos is not None:
                self.lens_position = float(pos)
            else:
                log.warning("camera: autofocus failed, keeping lens position %s", self.lens_position)
            if self.lens_position is not None:
                self.picam.set_controls({"AfMode": controls.AfModeEnum.Manual, "LensPosition": self.lens_position})
            self._last_cal = time.time()
            log.info("camera: calibrated, lens position %s", self.lens_position)

    def grab_frame(self):
        with self.lock:
            if self.style != "stream":
                self.set_style("stream")
            return self.picam.capture_array("main")   # RGB888 is stored BGR, which is what OpenCV wants

    def capture_still(self, path):
        with self.lock:
            if self.style is None:
                self.set_style("still")
            self.picam.switch_mode_and_capture_file(self.still_cfg, str(path))
            md = self.picam.capture_metadata()
        return {"exposure_us": md.get("ExposureTime"), "gain": md.get("AnalogueGain"),
                "lens_position": md.get("LensPosition")}

    def close(self):
        with self.lock:
            try:
                self.picam.stop()
                self.picam.close()
            except Exception:
                pass


# ---------------------------------------------------------------- rpicam-still fallback
class RpicamCamera(BaseCamera):
    name = "rpicam"

    def __init__(self, cfg):
        self.exe = shutil.which("rpicam-still") or shutil.which("libcamera-still")
        if not self.exe:
            raise CameraError("rpicam-still not found")
        self.cfg, self.c, self.s = cfg, cfg.camera, cfg.stream

    def _run(self, path, w, h, timeout=45):
        cmd = [self.exe, "-n", "-t", "1500", "--width", str(w), "--height", str(h), "-q", str(self.c.jpeg_quality),
               "-o", str(path)]
        if self.c.vflip:
            cmd.append("--vflip")
        if self.c.hflip:
            cmd.append("--hflip")
        if self.c.focus in ("calibrate", "auto"):
            cmd += ["--autofocus-mode", "auto", "--autofocus-on-capture"]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0:
            raise CameraError(r.stderr.strip()[-300:])

    def capture_still(self, path):
        self._run(path, self.c.width, self.c.height)
        return {}

    def grab_frame(self):
        import cv2
        tmp = Path("/tmp/bugbox_frame.jpg")
        self._run(tmp, self.s.width, self.s.height)
        return cv2.imread(str(tmp))


# ---------------------------------------------------------------- USB webcam
class OpenCVCamera(BaseCamera):
    name = "opencv"

    def __init__(self, cfg):
        import cv2
        self.cfg, self.c, self.s = cfg, cfg.camera, cfg.stream
        self.cap = cv2.VideoCapture(int(self.c.opencv_index))
        if not self.cap.isOpened():
            raise CameraError(f"cannot open camera index {self.c.opencv_index}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.c.width))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.c.height))

    def grab_frame(self):
        ok, f = self.cap.read()
        if not ok:
            raise CameraError("webcam read failed")
        return f

    def capture_still(self, path):
        import cv2
        f = self.grab_frame()
        cv2.imwrite(str(path), f, [cv2.IMWRITE_JPEG_QUALITY, int(self.c.jpeg_quality)])
        return {}

    def close(self):
        self.cap.release()


# ---------------------------------------------------------------- simulation
SPECIES_COLORS = {"stink bug": (40, 140, 60), "chinch bug": (60, 60, 160), "wireworm": (60, 150, 190),
                  "other insect": (120, 90, 90)}


class World:
    """Shared by FakeCamera (draws) and FakeBackend (reports the truth) when running --simulate."""

    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.truth: list[Detection] = []


class FakeCamera(BaseCamera):
    name = "fake"

    def __init__(self, cfg, world: World | None = None):
        self.cfg = cfg
        self.world = world or World()
        self.lens_position = 2.5

    def _scene(self, w, h):
        import cv2
        img = np.full((h, w, 3), 235, np.uint8)
        r = self.world.rng
        n = r.randint(6, 18)
        self.world.truth = []
        labels = list(SPECIES_COLORS)
        for _ in range(n):
            lab = r.choice(labels)
            bw, bh = r.randint(w // 25, w // 14), r.randint(h // 30, h // 18)
            cx, cy = r.randint(bw, w - bw), r.randint(bh, h - bh)
            cv2.ellipse(img, (cx, cy), (bw // 2, bh // 2), r.randint(0, 180), 0, 360, SPECIES_COLORS[lab], -1)
            self.world.truth.append(Detection(lab, round(r.uniform(0.7, 0.98), 2), cx, cy, bw, bh))
        return img

    def grab_frame(self):
        return self._scene(int(self.cfg.stream.width), int(self.cfg.stream.height))

    def capture_still(self, path):
        import cv2
        img = self._scene(int(self.cfg.camera.width) // 2, int(self.cfg.camera.height) // 2)
        cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return {"exposure_us": 500, "gain": 1.0, "lens_position": self.lens_position}


def make_camera(cfg, simulate=False, world=None) -> BaseCamera:
    backend = "fake" if simulate else cfg.camera.backend
    if backend == "fake":
        return FakeCamera(cfg, world)
    order = ["picamera2", "rpicam", "opencv"] if backend == "auto" else [backend]
    errors = []
    for b in order:
        try:
            if b == "picamera2":
                return PiCamera2Camera(cfg)
            if b == "rpicam":
                return RpicamCamera(cfg)
            if b == "opencv":
                return OpenCVCamera(cfg)
        except Exception as e:
            errors.append(f"{b}: {e}")
    raise CameraError("no camera backend worked: " + "; ".join(errors))
