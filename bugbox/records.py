"""Result records: JSON per image, one CSV row per image, and EXIF notes inside saved photos."""
from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from .util import log


@dataclass
class Result:
    when: str
    name: str
    mode: str
    source: str                 # still | frame
    counts: dict
    total: int
    backend: str = ""
    model_id: str = ""
    latency_s: float = 0.0
    image_path: str = ""
    annotated_path: str = ""
    cloud_calls: int = 0
    cache_hits: int = 0
    detections: list = field(default_factory=list)
    note: str = ""

    def to_dict(self):
        return dict(self.__dict__)


def write_json(path: Path, res: Result):
    path.write_text(json.dumps(res.to_dict(), indent=1))


def append_csv(path: Path, res: Result, targets: list, other: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists() or path.stat().st_size == 0
    cols = ["time", "mode", "source", "image"] + list(targets) + [other, "total", "latency_s", "backend", "cloud_calls"]
    with open(path, "a", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        if new:
            w.writerow(cols)
        w.writerow([res.when, res.mode, res.source, os.path.basename(res.image_path)]
                   + [res.counts.get(t, 0) for t in targets] + [res.counts.get(other, 0), res.total,
                                                                  round(res.latency_s, 2), res.backend, res.cloud_calls])


def embed_exif(path: Path, info: dict):
    """Write the counts into the JPEG without re-encoding it (best effort)."""
    try:
        import piexif
        import piexif.helper
        ex = {"0th": {piexif.ImageIFD.Software: b"bugbox", piexif.ImageIFD.ImageDescription: str(info.get("name", "")).encode()},
              "Exif": {piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump(json.dumps(info))},
              "GPS": {}, "1st": {}, "thumbnail": None}
        piexif.insert(piexif.dump(ex), str(path))
    except Exception as e:
        log.debug("exif skipped: %s", e)
