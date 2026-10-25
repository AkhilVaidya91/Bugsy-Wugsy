"""Periodic health snapshot (the Mothbox's Diagnostics.py): temperature, throttling, disk, Wi-Fi, voltage."""
from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path

from .util import gb, log


def _vc(*args):
    try:
        r = subprocess.run(["vcgencmd", *args], capture_output=True, text=True, timeout=5)
        return r.stdout.strip()
    except Exception:
        return ""


def snapshot(storage=None, mode="") -> dict:
    d = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "mode": mode}
    try:
        d["cpu_temp_c"] = round(int(Path("/sys/class/thermal/thermal_zone0/temp").read_text()) / 1000, 1)
    except Exception:
        pass
    thr = _vc("get_throttled")
    if thr:
        d["throttled"] = thr.split("=")[-1]
    try:
        d["load1"] = round(float(Path("/proc/loadavg").read_text().split()[0]), 2)
        mem = {l.split(":")[0]: int(l.split()[1]) for l in Path("/proc/meminfo").read_text().splitlines()[:3]}
        d["mem_free_mb"] = mem.get("MemAvailable", 0) // 1024
    except Exception:
        pass
    du = shutil.disk_usage("/")
    d["sd_free_gb"] = gb(du.free)
    if storage is not None:
        try:
            u = storage.find_usb()
            d["usb"] = None if not u else {"label": u.label, "free_gb": u.free_gb}
        except Exception:
            pass
    from . import power
    d["wifi"] = power.wifi_current()
    try:  # optional INA260 power sensor, same one the Mothbox supports
        import board, adafruit_ina260
        s = adafruit_ina260.INA260(board.I2C())
        d["voltage_v"] = round(s.voltage, 2)
        d["current_ma"] = round(s.current, 1)
    except Exception:
        pass
    return d


def log_snapshot(data_dir: Path, storage=None, mode="") -> dict:
    snap = snapshot(storage, mode)
    p = Path(data_dir) / "logs" / "diagnostics.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists() and p.stat().st_size > 5_000_000:   # keep it bounded, like the Mothbox's log trimming
        p.write_text("".join(p.read_text().splitlines(True)[-2000:]))
    with open(p, "a") as f:
        f.write(json.dumps(snap) + "\n")
    log.debug("diagnostics: %s", snap)
    return snap
