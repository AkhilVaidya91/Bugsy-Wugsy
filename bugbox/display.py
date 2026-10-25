"""E-ink status screen (Waveshare 2.13in V4). Drawn in a child process so GPIO libraries never clash,
and so a stuck panel can never hang the main program (same approach as the Mothbox's UpdateDisplay)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H = 250, 122     # landscape


def _font(size, bold=True):
    names = ["DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
             "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
    for n in names:
        try:
            return ImageFont.truetype(n, size)
        except Exception:
            pass
    return ImageFont.load_default()


def render(state: dict) -> Image.Image:
    img = Image.new("1", (W, H), 255)
    d = ImageDraw.Draw(img)
    d.text((2, 0), str(state.get("name", ""))[:16], font=_font(15), fill=0)
    mode = str(state.get("mode", "")).upper()
    d.text((W - 2, 2), mode, font=_font(12), fill=0, anchor="ra")
    d.line((0, 20, W, 20), fill=0)
    counts = state.get("counts") or {}
    targets = state.get("targets") or list(counts)[:3]
    x = 4
    colw = (W - 8) // max(1, len(targets) + 1)
    for t in targets:
        d.text((x, 24), str(counts.get(t, "-")), font=_font(34), fill=0)
        d.text((x, 64), t[:10], font=_font(9, False), fill=0)
        x += colw
    d.text((x, 24), str(state.get("total", "-")), font=_font(34), fill=0)
    d.text((x, 64), "TOTAL", font=_font(9, False), fill=0)
    d.line((0, 80, W, 80), fill=0)
    usb = state.get("usb")
    store = f"USB {usb['label'][:10]} {usb['free_gb']}GB" if usb else "No USB"
    d.text((2, 84), store, font=_font(10, False), fill=0)
    d.text((2, 97), f"Updated {state.get('updated', '')}", font=_font(10, False), fill=0)
    err = state.get("error")
    d.text((2, 109), ("! " + err)[:40] if err else f"{state.get('backend', '')} {state.get('photos', '')}", font=_font(9, False), fill=0)
    return img


class EInk:
    def __init__(self, cfg, enabled=True):
        self.enabled = bool(enabled and cfg.display.eink)
        self.min_s = float(cfg.display.refresh_min_s)
        self.state_file = cfg.data_dir() / "display_state.json"
        self.last = 0.0
        self._busy = threading.Lock()
        self.vendor = str(Path(__file__).resolve().parent.parent / "vendor")

    def update(self, state: dict, force=False):
        """Non-blocking: the slow e-ink refresh runs in the background, at most one at a time."""
        if not self.enabled or (not force and time.time() - self.last < self.min_s):
            return
        if self._busy.locked():
            return
        self.last = time.time()
        threading.Thread(target=self._refresh, args=(dict(state),), daemon=True, name="eink").start()

    def _refresh(self, state: dict):
        with self._busy:
            try:
                self.state_file.write_text(json.dumps(state))
                subprocess.run([sys.executable, "-m", "bugbox.display", str(self.state_file), self.vendor],
                               timeout=90, capture_output=True, cwd=str(Path(__file__).resolve().parent.parent))
            except Exception:
                pass


def _push(state_path: str, vendor: str):
    os.environ.setdefault("GPIOZERO_PIN_FACTORY", "rpigpio")    # same workaround the Mothbox uses on Pi 5
    sys.path.insert(0, vendor)
    from waveshare_epd import epd2in13_V4
    img = render(json.loads(Path(state_path).read_text()))
    epd = epd2in13_V4.EPD()
    epd.init()
    epd.display(epd.getbuffer(img))
    epd.sleep()


if __name__ == "__main__":
    try:
        _push(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "")
    except Exception as e:     # no display attached is fine
        print("display skipped:", e)
