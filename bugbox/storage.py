"""Where photos go. Standalone mode writes to the USB drive, not the SD card with the OS."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .util import gb, log

_TRUE = (True, 1, "1", "true", "True")


class StorageUnavailable(RuntimeError):
    pass


@dataclass
class Target:
    root: Path          # where the "bugbox" folder lives
    kind: str           # "usb" or "sd"
    label: str
    free_gb: float
    total_gb: float


def _truthy(v) -> bool:
    return v in _TRUE


def parse_lsblk(text: str):
    """Return candidate removable partitions from `lsblk -J` output."""
    data = json.loads(text)
    found = []

    def walk(node, parent_tran, parent_rm):
        tran = node.get("tran") or parent_tran
        rm = _truthy(node.get("rm")) or _truthy(node.get("hotplug")) or parent_rm
        name = node.get("name", "")
        mps = node.get("mountpoints")
        if mps is None:
            mps = [node.get("mountpoint")]
        mps = [m for m in (mps or []) if m]
        fstype = node.get("fstype")
        if fstype and not name.startswith(("mmcblk", "loop", "zram", "ram")):
            if tran == "usb" or rm:
                found.append({
                    "path": node.get("path") or f"/dev/{name}",
                    "label": node.get("label") or name,
                    "fstype": fstype,
                    "size": int(node.get("size") or 0),
                    "mount": mps[0] if mps else None,
                })
        for ch in node.get("children") or []:
            walk(ch, tran, rm)

    for dev in data.get("blockdevices", []):
        walk(dev, None, False)
    return found


class Storage:
    def __init__(self, cfg):
        self.cfg = cfg
        self.scfg = cfg.storage
        self.name = cfg.box_name()
        self._lock = threading.Lock()
        self._last_sync = 0.0

    # ---------- USB discovery ----------
    def _lsblk(self) -> str:
        cols = "NAME,PATH,TYPE,TRAN,RM,HOTPLUG,FSTYPE,LABEL,MOUNTPOINTS,SIZE"
        for c in (cols, cols.replace("MOUNTPOINTS", "MOUNTPOINT")):
            r = subprocess.run(["lsblk", "-J", "-b", "-o", c], capture_output=True, text=True, timeout=10)
            if r.returncode == 0 and r.stdout.strip():
                return r.stdout
        raise RuntimeError("lsblk failed")

    def _try_mount(self, dev_path: str):
        try:
            r = subprocess.run(["udisksctl", "mount", "-b", dev_path], capture_output=True, text=True, timeout=30)
            out = (r.stdout + r.stderr)
            log.info("storage: udisksctl mount %s -> %s", dev_path, out.strip()[:200])
            if " at " in r.stdout:
                return r.stdout.rsplit(" at ", 1)[1].strip().rstrip(".")
        except Exception as e:
            log.warning("storage: mount failed: %s", e)
        return None

    def _writable(self, root: Path) -> bool:
        try:
            probe = root / f".bugbox_probe_{os.getpid()}"
            probe.write_text("x")
            probe.unlink()
            return True
        except Exception:
            return False

    def find_usb(self):
        try:
            parts = parse_lsblk(self._lsblk())
        except Exception as e:
            log.debug("storage: lsblk problem: %s", e)
            return None
        best = None
        for p in sorted(parts, key=lambda d: -d["size"]):
            mount = p["mount"]
            if not mount and self.scfg.auto_mount:
                mount = self._try_mount(p["path"])
            if not mount or mount in ("/", "/boot", "/boot/firmware") or mount.startswith("/boot"):
                continue
            root = Path(mount)
            if not root.is_dir() or not self._writable(root):
                continue
            du = shutil.disk_usage(root)
            best = Target(root, "usb", p["label"], gb(du.free), gb(du.total))
            break
        return best

    # ---------- SD buffer ----------
    def sd_root(self) -> Path:
        p = self.scfg.sd_folder or str(self.cfg.data_dir() / "photos")
        p = Path(os.path.expanduser(p))
        p.mkdir(parents=True, exist_ok=True)
        return p

    def sd_target(self) -> Target:
        root = self.sd_root()
        du = shutil.disk_usage(root)
        return Target(root, "sd", "SD card", gb(du.free), gb(du.total))

    # ---------- policy ----------
    def usb_required(self, mode: str) -> bool:
        return mode in list(self.scfg.usb_required_modes)

    def pick(self, mode: str) -> Target:
        usb = self.find_usb()
        if usb and usb.free_gb >= float(self.scfg.min_free_gb):
            return usb
        if usb:
            raise StorageUnavailable(f"USB drive '{usb.label}' is almost full ({usb.free_gb} GB free)")
        if self.usb_required(mode) and not self.scfg.fallback_to_sd:
            raise StorageUnavailable(f"no USB drive found ({mode} mode writes to USB only)")
        sd = self.sd_target()
        if sd.free_gb < float(self.scfg.min_free_gb):
            raise StorageUnavailable(f"SD card almost full ({sd.free_gb} GB free)")
        return sd

    def day_dir(self, target: Target, when: datetime | None = None) -> Path:
        when = when or datetime.now()
        d = target.root / self.scfg.folder / self.name / when.strftime("%Y-%m-%d")
        d.mkdir(parents=True, exist_ok=True)
        return d

    def box_dir(self, target: Target) -> Path:
        d = target.root / self.scfg.folder / self.name
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ---------- copy SD buffer to USB when a drive appears (like the Mothbox backup) ----------
    def sync_pending(self, every_s: float = 120.0) -> int:
        if time.time() - self._last_sync < every_s:
            return 0
        self._last_sync = time.time()
        src_root = self.sd_root() / self.scfg.folder
        if not src_root.is_dir():
            return 0
        usb = self.find_usb()
        if not usb or usb.free_gb < float(self.scfg.min_free_gb):
            return 0
        moved = 0
        for f in sorted(src_root.rglob("*")):
            if not f.is_file():
                continue
            rel = f.relative_to(src_root)
            dst = usb.root / self.scfg.folder / rel
            try:
                dst.parent.mkdir(parents=True, exist_ok=True)
                if f.name == "counts.csv" and dst.exists():
                    with open(dst, "a") as out, open(f) as inp:
                        next(inp, None)  # skip header
                        shutil.copyfileobj(inp, out)
                    f.unlink()
                else:
                    shutil.copy2(f, dst)
                    f.unlink()
                moved += 1
            except Exception as e:
                log.warning("storage: could not move %s: %s", f, e)
        if moved:
            log.info("storage: moved %d buffered files from SD to %s", moved, usb.label)
        return moved

    def describe(self, mode: str) -> dict:
        usb = self.find_usb()
        return {
            "usb": None if not usb else {"label": usb.label, "free_gb": usb.free_gb, "total_gb": usb.total_gb,
                                         "path": str(usb.root)},
            "sd_free_gb": self.sd_target().free_gb,
            "usb_required": self.usb_required(mode),
        }
