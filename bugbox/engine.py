"""The engine ties lights, camera, storage, detection, display and schedule together."""
from __future__ import annotations

import json
import os
import statistics
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from . import __version__, diagnostics, power
from .annotate import annotate, save_jpeg
from .camera import World, make_camera
from .config import MODES
from .detect import (Detector, FakeBackend, FakeClassifier, HybridDetector, build_backend,
                     build_classifier, build_standalone_classifier)
from .display import EInk
from .lights import Lights
from .records import Result, append_csv, embed_exif, write_json
from .schedule import Schedule
from .storage import Storage, StorageUnavailable
from .util import log, stamp


class Engine:
    def __init__(self, cfg, simulate: bool = False):
        self.cfg = cfg
        self.simulate = simulate
        self.name = cfg.box_name()
        self.world = World() if simulate else None
        self.lights = Lights(cfg, simulate)
        self.storage = Storage(cfg)
        self.schedule = Schedule(cfg)
        self.eink = EInk(cfg, enabled=not simulate)
        self.camera = None
        self._detector = None
        self._detector_mode = None
        self._cap_lock = threading.RLock()
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._smooth = deque(maxlen=max(1, int(cfg.stream.smoothing_frames)))
        self._latest_jpeg: bytes | None = None
        self._usb_cache = (0.0, None)
        self._last_diag = 0.0
        self._last_counts_log = 0.0
        self._frame_n = 0
        self._slept_for_session = False
        self.recent: deque = deque(maxlen=30)
        self.photos_saved = 0
        self.errors = 0
        self.started = time.time()
        self.state = {"name": self.name, "version": __version__, "counts": {}, "total": None, "updated": "",
                      "error": "", "latency_s": None, "backend": "", "model_id": ""}

    # ------------------------------------------------------------ plumbing
    def current_mode(self) -> str:
        sw = power.read_mode_switch(int(self.cfg.power.switch_pin)) if self.cfg.power.switch_pin else None
        return sw or self.cfg.effective_mode()

    def set_mode(self, mode: str):
        if mode not in MODES:
            raise ValueError("mode must be one of: " + ", ".join(MODES))
        self.cfg.write_state(mode=mode)
        self._wake.set()
        log.info("mode set to %s", mode)

    def get_camera(self):
        if self.camera is None:
            self.camera = make_camera(self.cfg, self.simulate, self.world)
        return self.camera

    def _persist_budget(self, day: str, n: int):
        self.cfg.write_state(cloud_calls={"date": day, "n": n})

    def get_detector(self, mode: str) -> Detector:
        if self._detector is None or self._detector_mode != mode:
            backend = FakeBackend(self.world) if self.simulate else build_backend(self.cfg, mode)
            if mode == "hybrid":
                if self.simulate:
                    classifier, warn = FakeClassifier(self.world), ""
                else:
                    classifier, warn = build_classifier(self.cfg)
                saved = self.cfg.read_state().get("cloud_calls") or {}
                used = saved.get("n", 0) if saved.get("date") == time.strftime("%Y-%m-%d") else 0
                self._detector = HybridDetector(backend, classifier, self.cfg, warn, used, self._persist_budget)
                self.state["backend"] = backend.name + ("+cloud-classify" if classifier else " (cloud classifier off)")
            elif mode == "standalone" and not self.simulate:
                classifier = build_standalone_classifier(self.cfg)
                self._detector = Detector(backend, self.cfg, classifier=classifier)
                self.state["backend"] = backend.name + "+onnx-classify"
            else:
                self._detector = Detector(backend, self.cfg)
                self.state["backend"] = backend.name
            self._detector_mode = mode
            self.state["model_id"] = backend.model_id
            self._smooth.clear()
        return self._detector

    def request_stop(self):
        self._stop.set()
        self._wake.set()

    def _error(self, msg: str):
        self.errors += 1
        self.state["error"] = msg
        log.error(msg)

    def _usb(self):
        t, v = self._usb_cache
        if time.time() - t > 10:
            try:
                u = self.storage.find_usb()
                v = None if not u else {"label": u.label, "free_gb": u.free_gb, "total_gb": u.total_gb}
            except Exception:
                v = None
            self._usb_cache = (time.time(), v)
        return self._usb_cache[1]

    # ------------------------------------------------------------ one image
    def _build_result(self, dets, det, source, mode, latency, image_path="", annotated_path="") -> Result:
        c = det.count(dets)
        return Result(when=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), name=self.name, mode=mode, source=source,
                      counts=c["counts"], total=c["total"], backend=self.state.get("backend", ""),
                      model_id=self.state.get("model_id", ""), latency_s=round(latency, 3),
                      cloud_calls=det.calls_last, cache_hits=det.cache_hits_last,
                      image_path=image_path, annotated_path=annotated_path,
                      detections=[d.to_dict() for d in dets])

    def _publish(self, res: Result, img=None, dets=None, det=None, counts_override=None, total_override=None):
        counts = counts_override or res.counts
        total = res.total if total_override is None else total_override
        if img is not None and det is not None:
            try:
                ann = annotate(img, dets or [], counts, total, det.labeler, max_side=1280)
                ok, buf = cv2.imencode(".jpg", ann, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    self._latest_jpeg = buf.tobytes()
            except Exception as e:
                log.debug("annotate failed: %s", e)
        self.state.update({"counts": counts, "total": total, "updated": datetime.now().strftime("%H:%M:%S"),
                           "latency_s": res.latency_s, "error": "", "last_image": os.path.basename(res.image_path)})
        brief = {k: v for k, v in res.to_dict().items() if k != "detections"}
        self.recent.appendleft(brief)
        self.eink.update({"name": self.name, "mode": res.mode, "counts": counts,
                          "targets": list(self.cfg.detect.targets), "total": total, "usb": self._usb(),
                          "updated": self.state["updated"], "backend": self.state.get("backend", ""),
                          "photos": f"{self.photos_saved} photos", "error": ""})

    def capture_and_count(self, trigger: str = "timer") -> Result:
        """Light up, take a full photo, save it, count insects. The 'stills' pipeline."""
        with self._cap_lock:
            mode = self.current_mode()
            det = self.get_detector(mode)
            target = self.storage.pick(mode)
            day = self.storage.day_dir(target)
            base = f"{self.name}_{stamp()}"
            path = day / f"{base}.jpg"
            cam = self.get_camera()
            cam.set_style("still")
            with self.lights.lit():
                time.sleep(float(self.cfg.lights.settle_seconds))
                cam.calibrate()
                meta = cam.capture_still(path)
            img = cv2.imread(str(path))
            if img is None:
                raise RuntimeError(f"saved photo could not be read back: {path}")
            t0 = time.time()
            dets = det.detect(img)
            res = self._build_result(dets, det, "still", mode, time.time() - t0, str(path))
            ann_path = ""
            if self.cfg.storage.keep_annotated:
                ann = annotate(img, dets, res.counts, res.total, det.labeler)
                ann_path = str(day / f"{base}_annotated.jpg")
                save_jpeg(Path(ann_path), ann, 85)
            res.annotated_path = ann_path
            write_json(day / f"{base}_result.json", res)
            append_csv(self.storage.box_dir(target) / "counts.csv", res, self.cfg.detect.targets,
                       self.cfg.detect.other_label)
            embed_exif(path, {"name": self.name, "mode": mode, "counts": res.counts, "total": res.total,
                              "model": res.model_id, "backend": res.backend, **{k: v for k, v in meta.items() if v}})
            self.photos_saved += 1
            self._publish(res, img, dets, det)
            log.info("%s capture (%s): %s total=%s in %.1fs -> %s", mode, trigger, res.counts, res.total,
                     res.latency_s, target.label)
            return res

    # ------------------------------------------------------------ one video frame
    def process_frame(self) -> Result:
        with self._cap_lock:
            mode = self.current_mode()
            det = self.get_detector(mode)
            cam = self.get_camera()
            cam.set_style("stream")
            frame = cam.grab_frame()
            t0 = time.time()
            dets = det.detect(frame)
            res = self._build_result(dets, det, "frame", mode, time.time() - t0)
            self._smooth.append(res.counts)
            sm = {k: int(round(statistics.median([c.get(k, 0) for c in self._smooth]))) for k in res.counts}
            total = sum(sm.values())
            self._frame_n += 1
            n = int(self.cfg.stream.save_every_n)
            if n and self._frame_n % n == 0:
                try:
                    target = self.storage.pick(mode)
                    p = self.storage.day_dir(target) / f"{self.name}_{stamp()}.jpg"
                    save_jpeg(p, frame, int(self.cfg.camera.jpeg_quality))
                    res.image_path = str(p)
                    self.photos_saved += 1
                except StorageUnavailable as e:
                    log.debug("frame not saved: %s", e)
            if time.time() - self._last_counts_log > 10:
                self._last_counts_log = time.time()
                append_csv(self.cfg.data_dir() / "logs" / "counts.csv", res, self.cfg.detect.targets,
                           self.cfg.detect.other_label)
            self._publish(res, frame, dets, det, counts_override=sm, total_override=total)
            return res

    def detect_file(self, path, mode: str | None = None, save_annotated: str | None = None) -> Result:
        """Run the current detector on any image file (used by `bugbox detect` and `evaluate`)."""
        mode = mode or self.current_mode()
        det = self.get_detector(mode)
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(path)
        t0 = time.time()
        dets = det.detect(img)
        res = self._build_result(dets, det, "file", mode, time.time() - t0, str(path))
        if save_annotated:
            save_jpeg(Path(save_annotated), annotate(img, dets, res.counts, res.total, det.labeler), 88)
        return res

    # ------------------------------------------------------------ status
    def status(self) -> dict:
        mode = self.current_mode()
        active, end = self.schedule.status()
        s = dict(self.state)
        s.update({"mode": mode, "style": self.cfg.style_for(mode), "targets": list(self.cfg.detect.targets),
                  "other_label": self.cfg.detect.other_label, "lights_on": self.lights.is_on,
                  "usb": self._usb(), "usb_required": self.storage.usb_required(mode),
                  "photos_saved": self.photos_saved, "errors": self.errors,
                  "uptime_s": int(time.time() - self.started), "camera": getattr(self.camera, "name", "not started"),
                  "schedule_active": active, "schedule_end": end.strftime("%H:%M") if end else None,
                  "simulated": self.simulate, "wifi": "", "recent": list(self.recent)[:10],
                  "classifier": self._detector.stats() if self._detector is not None else None})
        return s

    def latest_jpeg(self):
        return self._latest_jpeg

    # ------------------------------------------------------------ main loop
    def _wait(self, seconds: float):
        self._wake.wait(timeout=max(0.0, seconds))
        self._wake.clear()

    def _idle_outside_window(self):
        self.lights.off()
        nxt = self.schedule.next_start()
        self.state["error"] = ""
        self.state["next_session"] = nxt.strftime("%Y-%m-%d %H:%M") if nxt else None
        if self.cfg.power.sleep_between_sessions and nxt and not self._slept_for_session:
            self._slept_for_session = True
            log.info("outside session window: setting wake alarm for %s and shutting down", nxt)
            if power.set_wake_alarm(int(nxt.timestamp())):
                time.sleep(2)
                power.shutdown()
        self._wait(5)

    def run(self):
        log.info("bugbox %s starting (mode=%s, simulate=%s)", __version__, self.current_mode(), self.simulate)
        next_still = 0.0
        last_mode = None
        try:
            while not self._stop.is_set():
                mode = self.current_mode()
                if mode != last_mode:
                    log.info("running in %s mode", mode)
                    last_mode = mode
                    self.state["error"] = ""
                    next_still = 0.0
                style = self.cfg.style_for(mode)
                active, _ = self.schedule.status()
                if active:
                    self._slept_for_session = False
                else:
                    self._idle_outside_window()
                    continue
                if self.cfg.lights.policy == "session" and not self.lights.is_on:
                    self.lights.on()
                try:
                    if style == "stills":
                        if time.time() >= next_still:
                            self.capture_and_count("timer")
                            next_still = time.time() + float(self.cfg.stills.interval_s)
                        self.storage.sync_pending()
                        self._wait(min(1.0, max(0.0, next_still - time.time())))
                    else:
                        t0 = time.time()
                        if self.cfg.lights.policy != "off" and not self.lights.is_on:
                            self.lights.on()
                        self.process_frame()
                        self._wait(max(0.0, 1.0 / max(0.05, float(self.cfg.stream.fps)) - (time.time() - t0)))
                except StorageUnavailable as e:
                    self._error(f"storage: {e}")
                    next_still = time.time() + 10
                    self._wait(10)
                except Exception as e:
                    self._error(f"{type(e).__name__}: {e}")
                    self._wait(min(60, 2 ** min(self.errors, 6)))
                if time.time() - self._last_diag > float(self.cfg.diagnostics.interval_s):
                    self._last_diag = time.time()
                    try:
                        diagnostics.log_snapshot(self.cfg.data_dir(), self.storage, mode)
                    except Exception as e:
                        log.debug("diagnostics failed: %s", e)
        finally:
            self.close()

    def close(self):
        try:
            self.lights.close()
        finally:
            if self.camera is not None:
                self.camera.close()
