from __future__ import annotations

import logging
import logging.handlers
import os
import tempfile
from datetime import datetime
from pathlib import Path

log = logging.getLogger("bugbox")


def setup_logging(data_dir: Path, verbose: bool = False):
    logs = Path(data_dir) / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger("bugbox")
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    if root.handlers:
        return
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = logging.handlers.RotatingFileHandler(logs / "bugbox.log", maxBytes=5_000_000, backupCount=3)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    root.addHandler(fh)
    root.addHandler(sh)


def stamp(dt: datetime | None = None) -> str:
    return (dt or datetime.now()).strftime("%Y-%m-%dT%H-%M-%S")


def atomic_write_text(path: Path, text: str):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent)
    with os.fdopen(fd, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def gb(n_bytes: float) -> float:
    return round(n_bytes / (1024 ** 3), 2)
