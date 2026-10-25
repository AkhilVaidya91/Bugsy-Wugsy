"""Detection: Roboflow backends (on-Pi or cloud), tiling, de-duplication and species counting."""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

import numpy as np

from .util import log


@dataclass
class Detection:
    label: str
    confidence: float
    x: float      # centre x (pixels)
    y: float      # centre y
    w: float
    h: float
    source: str = "local"        # local | cloud | cache  (who decided the label)
    local_label: str = ""        # what the Pi's own model said, kept when the cloud changes it

    @property
    def box(self):
        return (self.x - self.w / 2, self.y - self.h / 2, self.x + self.w / 2, self.y + self.h / 2)

    @property
    def area(self):
        return max(self.w, 0) * max(self.h, 0)

    def to_dict(self):
        d = asdict(self)
        return {k: (round(v, 2) if isinstance(v, float) else v) for k, v in d.items()}


def _g(p, *names, default=None):
    for n in names:
        if isinstance(p, dict):
            if n in p and p[n] is not None:
                return p[n]
        elif hasattr(p, n):
            v = getattr(p, n)
            if v is not None:
                return v
    return default


def parse_predictions(preds) -> List[Detection]:
    """Accepts Roboflow response objects/dicts: x,y,width,height,confidence,class(_name)."""
    if preds is None:
        return []
    if isinstance(preds, dict):
        preds = preds.get("predictions", [])
    out = []
    for p in preds:
        label = _g(p, "class_name", "class", "label", default="unknown")
        out.append(Detection(str(label), float(_g(p, "confidence", default=0.0)),
                             float(_g(p, "x", default=0.0)), float(_g(p, "y", default=0.0)),
                             float(_g(p, "width", "w", default=0.0)), float(_g(p, "height", "h", default=0.0))))
    return out


# ---------------- geometry ----------------
def _inter(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def iou(a: Detection, b: Detection) -> float:
    i = _inter(a.box, b.box)
    u = a.area + b.area - i
    return i / u if u > 0 else 0.0


def ios(a: Detection, b: Detection) -> float:
    """Intersection over the smaller box: catches a partial box inside a whole one."""
    i = _inter(a.box, b.box)
    m = min(a.area, b.area)
    return i / m if m > 0 else 0.0


def nms(dets: List[Detection], thr: float = 0.6) -> List[Detection]:
    """Class-agnostic: the same insect labelled twice should count once."""
    if thr >= 1.0:
        return list(dets)
    kept: List[Detection] = []
    for d in sorted(dets, key=lambda d: -d.confidence):
        if all(iou(d, k) < thr for k in kept):
            kept.append(d)
    return kept


def merge_tiles(dets: List[Detection], thr: float = 0.6) -> List[Detection]:
    """Merge duplicates created by overlapping tiles. A box cut by a tile edge merges into the whole one."""
    kept: List[Detection] = []
    for d in sorted(dets, key=lambda d: -d.confidence):
        for i, k in enumerate(kept):
            if ios(d, k) >= thr:
                x1 = min(d.box[0], k.box[0]); y1 = min(d.box[1], k.box[1])
                x2 = max(d.box[2], k.box[2]); y2 = max(d.box[3], k.box[3])
                kept[i] = Detection(k.label, k.confidence, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)
                break
        else:
            kept.append(d)
    return kept


def tile_boxes(width: int, height: int, n: int, overlap: float):
    if n <= 1:
        return [(0, 0, width, height)]
    tw, th = width / n, height / n
    ox, oy = tw * overlap, th * overlap
    boxes = []
    for r in range(n):
        for c in range(n):
            x0 = max(0, int(c * tw - ox)); y0 = max(0, int(r * th - oy))
            x1 = min(width, int((c + 1) * tw + ox)); y1 = min(height, int((r + 1) * th + oy))
            boxes.append((x0, y0, x1, y1))
    return boxes


# ---------------- labels and counts ----------------
def _norm(s: str) -> str:
    return re.sub(r"[\s_\-]+", " ", str(s).strip().lower())


class Labeler:
    def __init__(self, targets, other_label, aliases=None):
        self.targets = list(targets)
        self.other = other_label
        self._map = {}
        for t in self.targets:
            self._map[_norm(t)] = t
            self._map[_norm(t).replace(" ", "")] = t
        for k, v in (aliases or {}).items():
            canon = next((t for t in self.targets if _norm(t) == _norm(v)), None)
            if canon:
                self._map[_norm(k)] = canon
                self._map[_norm(k).replace(" ", "")] = canon

    def canonical(self, label: str) -> str:
        n = _norm(label)
        return self._map.get(n) or self._map.get(n.replace(" ", "")) or self.other

    def count(self, dets: List[Detection], min_conf: float = 0.0) -> dict:
        counts = {t: 0 for t in self.targets}
        counts[self.other] = 0
        for d in dets:
            if d.confidence >= min_conf:
                counts[self.canonical(d.label)] += 1
        return {"counts": counts, "total": sum(counts.values())}


# ---------------- backends ----------------
def _downscale(img: np.ndarray, max_side: int):
    import cv2
    h, w = img.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1.0:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img, s


class Backend:
    name = "base"
    model_id = ""

    def infer_array(self, img: np.ndarray) -> List[Detection]:
        raise NotImplementedError


class LocalBackend(Backend):
    """Roboflow Inference running inside this Python process (weights cached after the first download)."""
    name = "local"

    def __init__(self, model_id: str, api_key: str, cache_dir: str, confidence: float):
        if not model_id:
            raise RuntimeError("roboflow.model_id is empty (use 'project/version' in config.toml)")
        os.makedirs(cache_dir, exist_ok=True)
        os.environ.setdefault("MODEL_CACHE_DIR", cache_dir)   # default is /tmp/cache, wiped on reboot
        from inference import get_model
        self.model_id = model_id
        self.conf = confidence
        log.info("local model: loading %s (first run downloads weights, needs internet)", model_id)
        self.model = get_model(model_id=model_id, api_key=api_key or None)

    def infer_array(self, img):
        try:
            res = self.model.infer(img, confidence=self.conf)
        except TypeError:
            res = self.model.infer(img)
        if isinstance(res, (list, tuple)):
            res = res[0]
        return parse_predictions(_g(res, "predictions", default=[]))


class RFDETRBackend(Backend):
    """Runs a Roboflow RF-DETR Small checkpoint directly with the rfdetr package."""
    name = "rfdetr"

    def __init__(self, weights: str, confidence: float):
        path = Path(weights).expanduser()
        if not path.is_file():
            raise RuntimeError(f"RF-DETR weights file not found: {path}")
        try:
            from rfdetr import RFDETRSmall
        except ImportError as e:
            raise RuntimeError("standalone RF-DETR requires the 'rfdetr' package (pip install rfdetr)") from e
        self.model_id = str(path)
        self.conf = confidence
        log.info("standalone RF-DETR: loading %s", path)
        self.model = RFDETRSmall(pretrain_weights=str(path))

    def infer_array(self, img):
        # RF-DETR's public API expects RGB arrays, while bugbox camera frames are BGR.
        rgb = np.ascontiguousarray(img[:, :, ::-1])
        result = self.model.predict(rgb, threshold=self.conf, include_source_image=False)
        if isinstance(result, (list, tuple)):
            result = result[0] if result else None
        if result is None:
            return []
        xyxy = np.asarray(getattr(result, "xyxy", []))
        confidence = np.asarray(getattr(result, "confidence", []))
        class_ids = np.asarray(getattr(result, "class_id", []))
        data = getattr(result, "data", {}) or {}
        names = data.get("class_name", []) if hasattr(data, "get") else []
        model_names = getattr(self.model, "class_names", None) or []
        out = []
        for i, box in enumerate(xyxy):
            x1, y1, x2, y2 = map(float, box)
            class_id = int(class_ids[i]) if i < len(class_ids) else -1
            label = names[i] if i < len(names) else (
                model_names[class_id] if 0 <= class_id < len(model_names) else str(class_id)
            )
            score = float(confidence[i]) if i < len(confidence) else 0.0
            out.append(Detection(str(label), score, (x1 + x2) / 2, (y1 + y2) / 2,
                                 max(0.0, x2 - x1), max(0.0, y2 - y1)))
        return out


class HTTPBackend(Backend):
    """Roboflow Inference HTTP API: the Serverless Cloud API, or a server on this Pi."""

    def __init__(self, api_url: str, api_key: str, model_id: str, confidence: float, max_side: int,
                 name: str = "cloud"):
        if not model_id:
            raise RuntimeError("roboflow.model_id is empty (use 'project/version' in config.toml)")
        from inference_sdk import InferenceHTTPClient, InferenceConfiguration
        self.name = name
        self.model_id = model_id
        self.max_side = int(max_side)
        self.client = InferenceHTTPClient(api_url=api_url, api_key=api_key or None)
        try:
            self.client.configure(InferenceConfiguration(confidence_threshold=confidence, api_key_transport="header"))
        except TypeError:   # older SDKs do not know api_key_transport
            self.client.configure(InferenceConfiguration(confidence_threshold=confidence))
        log.info("%s inference: %s model=%s", name, api_url, model_id)

    def infer_array(self, img):
        small, s = _downscale(img, self.max_side)
        last = None
        for attempt in range(3):
            try:
                res = self.client.infer(small, model_id=self.model_id)
                if isinstance(res, list):
                    res = res[0]
                dets = parse_predictions(res)
                if s != 1.0:
                    for d in dets:
                        d.x /= s; d.y /= s; d.w /= s; d.h /= s
                return dets
            except Exception as e:
                last = e
                time.sleep(0.5 * (2 ** attempt))
        raise RuntimeError(f"Roboflow request failed: {last}")


class FakeBackend(Backend):
    """Used by --simulate: reads the 'truth' that the fake camera drew (with a few misses)."""
    name = "fake"
    model_id = "simulated/0"

    def __init__(self, world, miss_rate: float = 0.03):
        self.world = world
        self.miss = miss_rate

    def infer_array(self, img):
        import random
        out = []
        for d in self.world.truth:
            if random.random() > self.miss:
                out.append(Detection(d.label, min(0.99, d.confidence), d.x, d.y, d.w, d.h))
        return out


class Detector:
    """Wraps a backend with tiling, NMS and counting."""
    calls_last = 0          # cloud calls made by the last detect() (hybrid only)
    cache_hits_last = 0

    def stats(self):
        return None

    def __init__(self, backend: Backend, cfg, classifier=None):
        self.backend = backend
        self.classifier = classifier
        self.cfg = cfg
        self.dcfg = cfg.detect
        self.labeler = Labeler(self.dcfg.targets, self.dcfg.other_label,
                               vars(self.dcfg.aliases) if hasattr(self.dcfg, "aliases") else {})

    def detect(self, img: np.ndarray) -> List[Detection]:
        h, w = img.shape[:2]
        n = int(self.dcfg.tiles)
        if n <= 1:
            dets = self.backend.infer_array(img)
            dets = [d for d in dets if d.confidence >= float(self.dcfg.confidence)]
            dets = nms(dets, float(self.dcfg.nms_iou))
        else:
            allp: List[Detection] = []
            for (x0, y0, x1, y1) in tile_boxes(w, h, n, float(self.dcfg.tile_overlap)):
                for d in self.backend.infer_array(np.ascontiguousarray(img[y0:y1, x0:x1])):
                    if d.confidence >= float(self.dcfg.confidence):
                        d.x += x0; d.y += y0
                        allp.append(d)
            dets = merge_tiles(allp, float(self.dcfg.nms_iou))
        if self.classifier:
            dets = self._classify(img, dets)
        return dets

    def _classify(self, img: np.ndarray, dets: List[Detection]) -> List[Detection]:
        height, width = img.shape[:2]
        classified = []
        for det in dets:
            pad_x, pad_y = det.w * 0.1, det.h * 0.1
            x1 = max(0, int(det.x - det.w / 2 - pad_x))
            y1 = max(0, int(det.y - det.h / 2 - pad_y))
            x2 = min(width, int(det.x + det.w / 2 + pad_x))
            y2 = min(height, int(det.y + det.h / 2 + pad_y))
            if x2 <= x1 or y2 <= y1:
                raise RuntimeError(f"classifier crop has invalid bounds for detection {det.to_dict()}")
            label, confidence = self.classifier.classify(np.ascontiguousarray(img[y1:y2, x1:x2]), det)
            classifier_cfg = getattr(self.cfg, "classifier", None)
            min_confidence = float(getattr(classifier_cfg, "min_confidence", 0.0))
            if confidence >= min_confidence:
                det.local_label = det.label
                det.label = label
            classified.append(det)
        return classified

    def count(self, dets):
        return self.labeler.count(dets, float(self.dcfg.confidence))


def _project_path(value: str) -> Path:
    path = Path(os.path.expanduser(value))
    return path if path.is_absolute() else Path(__file__).resolve().parents[1] / path


def build_backend(cfg, mode: str) -> Backend:
    key = cfg.api_key()
    model_id = cfg.roboflow.model_id
    if mode == "hybrid" and cfg.hybrid.detector_model_id:
        model_id = cfg.hybrid.detector_model_id
    conf = float(cfg.detect.confidence)
    if mode == "cloud":
        if not key:
            raise RuntimeError(f"cloud mode needs your Roboflow API key in ${cfg.roboflow.api_key_env}")
        return HTTPBackend(cfg.cloud.api_url, key, model_id, conf, int(cfg.cloud.max_side), name="cloud")
    if mode == "standalone" and not model_id and hasattr(cfg.standalone, "detector_weights"):
        return RFDETRBackend(str(_project_path(cfg.standalone.detector_weights)), conf)
    if cfg.standalone.server_url:
        return HTTPBackend(cfg.standalone.server_url, key, model_id, conf, 4096, name="local-server")
    cache = cfg.standalone.cache_dir or str(cfg.data_dir() / "models")
    return LocalBackend(model_id, key, os.path.expanduser(cache), conf)


# =====================================================================================
# Hybrid mode: the Pi finds and counts insects, the cloud is asked only to name single ones
# =====================================================================================
def parse_classification(res):
    """Returns (label, confidence) from a Roboflow response, or None.
    Understands single-label classification ({'top','confidence','predictions':[...]}),
    multi-label ({'predictions': {class: {'confidence'}}}) and detection responses
    (used when an object-detection model is reused on a cropped insect)."""
    if isinstance(res, (list, tuple)):
        res = res[0] if res else None
    if res is None:
        return None
    top = _g(res, "top")
    if top:
        return str(top), float(_g(res, "confidence", default=0.0))
    preds = _g(res, "predictions", default=None)
    if isinstance(preds, dict) and preds:
        best = max(preds.items(), key=lambda kv: float(_g(kv[1], "confidence", default=0.0)))
        return str(best[0]), float(_g(best[1], "confidence", default=0.0))
    if isinstance(preds, (list, tuple)) and preds:
        best = max(preds, key=lambda p: float(_g(p, "confidence", default=0.0)))
        return str(_g(best, "class_name", "class", "label", default="unknown")), float(_g(best, "confidence", default=0.0))
    return None


class RoboflowClassifier:
    """Sends one insect crop to the Roboflow Serverless Cloud API and returns (label, confidence)."""

    def __init__(self, api_url: str, api_key: str, model_id: str):
        if not api_key:
            raise RuntimeError("no Roboflow API key (run: bugbox set-key)")
        if not model_id:
            raise RuntimeError("no classifier model (set classifier.model_id or roboflow.model_id)")
        from inference_sdk import InferenceHTTPClient, InferenceConfiguration
        self.model_id = model_id
        self.client = InferenceHTTPClient(api_url=api_url, api_key=api_key)
        try:
            self.client.configure(InferenceConfiguration(confidence_threshold=0.3, api_key_transport="header"))
        except TypeError:
            self.client.configure(InferenceConfiguration(confidence_threshold=0.3))

    def classify(self, crop, det=None):
        return parse_classification(self.client.infer(crop, model_id=self.model_id))


class ONNXClassifier:
    """Runs the bundled crop classifier and supplies the box size as scale metadata."""

    labels = ("1", "2", "3", "other")

    def __init__(self, weights: str):
        path = Path(weights).expanduser()
        if not path.is_file():
            raise RuntimeError(f"ONNX classifier file not found: {path}")
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise RuntimeError("standalone classification requires 'onnxruntime'") from e
        import cv2
        from PIL import Image, ImageOps
        self._cv2 = cv2
        self._Image = Image
        self._ImageOps = ImageOps
        self.model_id = str(path)
        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.image_input, self.scale_input = [x.name for x in self.session.get_inputs()]
        self.output = self.session.get_outputs()[0].name

    def classify(self, crop, det=None):
        if det is None:
            raise ValueError("ONNXClassifier.classify requires the source detection")
        rgb = self._cv2.cvtColor(crop, self._cv2.COLOR_BGR2RGB)
        image = self._Image.fromarray(rgb)
        image = self._ImageOps.pad(image, (224, 224), color=(0, 0, 0))
        array = np.asarray(image, dtype=np.float32) / 255.0
        array = (array - np.asarray((0.485, 0.456, 0.406), dtype=np.float32)) / np.asarray(
            (0.229, 0.224, 0.225), dtype=np.float32
        )
        array = np.transpose(array, (2, 0, 1))[None, ...]
        # The exported model has two scale inputs. They match train.py's log-transformed
        # bounding-box width and height; confidence/aspect ratio are not part of this export.
        scale = np.log1p(np.asarray([[max(det.w, 0.0), max(det.h, 0.0)]], dtype=np.float32))
        logits = self.session.run([self.output], {self.image_input: array, self.scale_input: scale})[0]
        probs = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        index = int(np.argmax(probs[0]))
        return self.labels[index], float(probs[0, index])


def build_standalone_classifier(cfg):
    if not hasattr(cfg.standalone, "classifier_weights"):
        raise RuntimeError("standalone.classifier_weights is not configured")
    return ONNXClassifier(str(_project_path(cfg.standalone.classifier_weights)))


class FakeClassifier:
    """--simulate: answers with the label the fake camera drew at that spot (95% of the time)."""
    model_id = "simulated-classifier/0"

    def __init__(self, world, accuracy=0.95):
        self.world, self.acc = world, accuracy

    def classify(self, crop, det=None):
        import random
        best = min(self.world.truth, key=lambda t: (t.x - det.x) ** 2 + (t.y - det.y) ** 2, default=None)
        if best is None:
            return None
        if random.random() > self.acc:
            return "other insect", 0.7
        return best.label, 0.93


def _thumb(img: np.ndarray) -> np.ndarray:
    """16x16 colour thumbnail. Colour matters: two dark blobs of the same shape can be different species."""
    import cv2
    return cv2.resize(img, (16, 16), interpolation=cv2.INTER_AREA).astype(np.int16)


class CropCache:
    """Remembers what the cloud said about an insect so a sheet that has not changed costs no more calls.
    An insect counts as 'the same' if its box overlaps and its colour thumbnail is close."""
    MAX_DIFF = 14.0          # mean absolute difference per channel, 0-255

    def __init__(self, ttl_s: float, max_entries: int = 500):
        self.ttl, self.max = float(ttl_s), max_entries
        self.items: list = []     # (time, Detection, thumbnail, label)

    def lookup(self, det: Detection, crop: np.ndarray):
        now, t = time.time(), _thumb(crop)
        self.items = [e for e in self.items if now - e[0] < self.ttl]
        for _, d, et, label in self.items:
            if iou(det, d) > 0.5 and float(np.abs(t - et).mean()) <= self.MAX_DIFF:
                return label
        return None

    def add(self, det: Detection, crop: np.ndarray, label: str):
        self.items.append((time.time(), Detection(det.label, det.confidence, det.x, det.y, det.w, det.h),
                           _thumb(crop), label))
        self.items = self.items[-self.max:]


class CloudBudget:
    """Per-minute and per-day ceilings plus a circuit breaker, so a bad network never stalls counting."""

    def __init__(self, per_min: int, per_day: int, day_used: int = 0, today: str = "", on_change=None):
        self.per_min, self.per_day = int(per_min), int(per_day)
        self.window: list = []
        self.today = today or time.strftime("%Y-%m-%d")
        self.day_used = int(day_used)
        self.fails = 0
        self.open_until = 0.0
        self.on_change = on_change

    def _roll(self):
        d = time.strftime("%Y-%m-%d")
        if d != self.today:
            self.today, self.day_used = d, 0
        now = time.time()
        self.window = [t for t in self.window if now - t < 60]

    def allow(self, n: int) -> int:
        self._roll()
        if time.time() < self.open_until:
            return 0
        return max(0, min(n, self.per_min - len(self.window), self.per_day - self.day_used))

    def record(self, n: int):
        self._roll()
        self.window += [time.time()] * n
        self.day_used += n
        if self.on_change:
            self.on_change(self.today, self.day_used)

    def success(self):
        self.fails = 0

    def failure(self):
        self.fails += 1
        if self.fails >= 3:
            self.open_until = time.time() + 60
            self.fails = 0
            log.warning("cloud classifier failing, pausing it for 60 s (the Pi's own labels are used)")

    @property
    def left_today(self):
        self._roll()
        return max(0, self.per_day - self.day_used)


class HybridDetector(Detector):
    """Detect and count on the Pi. Ask the cloud to re-name an insect only when the Pi is unsure."""

    def __init__(self, backend: Backend, classifier, cfg, warning: str = "", day_used: int = 0, on_budget=None):
        super().__init__(backend, cfg)
        self.cls = classifier
        self.h, self.c = cfg.hybrid, cfg.classifier
        self.warning = warning
        self.cache = CropCache(float(self.c.cache_ttl_s))
        self.budget = CloudBudget(self.h.max_calls_per_minute, self.h.daily_budget, day_used, on_change=on_budget)
        self.totals = {"cloud_calls": 0, "cache_hits": 0, "relabelled": 0, "fallbacks": 0}

    def stats(self):
        return {"enabled": self.cls is not None and self.h.classify != "off", "policy": self.h.classify,
                "calls_today": self.budget.day_used, "budget_left": self.budget.left_today,
                "warning": self.warning, **self.totals}

    def _wants_cloud(self, d: Detection) -> bool:
        if self.cls is None or self.h.classify == "off":
            return False
        return True if self.h.classify == "all" else d.confidence < float(self.h.uncertain_below)

    def _crop(self, img: np.ndarray, d: Detection):
        import cv2
        H, W = img.shape[:2]
        pad = float(self.h.crop_pad)
        x1, y1, x2, y2 = d.box
        px, py = (x2 - x1) * pad, (y2 - y1) * pad
        x1, y1, x2, y2 = int(max(0, x1 - px)), int(max(0, y1 - py)), int(min(W, x2 + px)), int(min(H, y2 + py))
        if x2 - x1 < 4 or y2 - y1 < 4:
            return None
        crop = np.ascontiguousarray(img[y1:y2, x1:x2])
        target = int(self.h.crop_target_side)
        s = target / max(crop.shape[:2])
        if s > 1.0:
            crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        return crop

    def _ask(self, crop, det):
        try:
            return self.cls.classify(crop, det)
        except Exception as e:
            log.debug("classifier call failed: %s", e)
            return "__error__"

    def detect(self, img: np.ndarray) -> List[Detection]:
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FTimeout
        dets = super().detect(img)
        self.calls_last = self.cache_hits_last = 0
        todo = []
        for d in dets:
            d.local_label = d.label
            if not self._wants_cloud(d):
                continue
            crop = self._crop(img, d)
            if crop is None:
                continue
            cached = self.cache.lookup(d, crop)
            if cached:
                d.label, d.source = cached, "cache"
                self.cache_hits_last += 1
                self.totals["cache_hits"] += 1
                continue
            todo.append((d, crop))
        if not todo:
            return dets
        n = self.budget.allow(len(todo))
        self.totals["fallbacks"] += len(todo) - n          # no budget or breaker open: keep the Pi's label
        todo = todo[:n]
        if not todo:
            return dets
        self.budget.record(len(todo))
        self.calls_last = len(todo)
        self.totals["cloud_calls"] += len(todo)
        timeout = float(self.c.timeout_s)
        with ThreadPoolExecutor(max_workers=4) as ex:
            futs = [(d, crop, ex.submit(self._ask, crop, d)) for d, crop in todo]
            for d, crop, f in futs:
                try:
                    r = f.result(timeout=timeout)
                except FTimeout:
                    r = "__error__"
                if r == "__error__":
                    self.budget.failure()
                    self.totals["fallbacks"] += 1
                    continue
                self.budget.success()
                if not r or r[1] < float(self.c.min_confidence):
                    self.totals["fallbacks"] += 1
                    continue
                label, conf = r
                if self.labeler.canonical(label) != self.labeler.canonical(d.label):
                    self.totals["relabelled"] += 1
                d.label, d.source = label, "cloud"
                self.cache.add(d, crop, label)
        return dets


def build_classifier(cfg):
    """Returns (classifier or None, warning). A missing key or package must not stop local counting."""
    try:
        model = cfg.classifier.model_id or cfg.roboflow.model_id
        return RoboflowClassifier(cfg.classifier.api_url, cfg.api_key(), model), ""
    except Exception as e:
        log.warning("cloud classifier unavailable (%s). Hybrid mode will use the Pi's own labels.", e)
        return None, str(e)
