"""Display: e-ink status panel (Waveshare 2.13in V4) via a subprocess helper.

Runs the actual GPIO/SPI calls in a child process so the e-ink driver
never blocks or crashes the main pipeline loop.
"""
from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H = 250, 122  # landscape


# ---------------------------------------------------------------------------
# Image renderer (called in the child process and for offline testing)
# ---------------------------------------------------------------------------

def _font(size: int, bold: bool = True) -> ImageFont.FreeTypeFont:
    names = [
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            if bold
            else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        ),
    ]
    for n in names:
        try:
            return ImageFont.truetype(n, size)
        except Exception:
            pass
    return ImageFont.load_default()


def render(state: dict) -> Image.Image:
    """Draw the status screen as a 1-bit PIL Image."""
    img = Image.new("1", (W, H), 255)
    d = ImageDraw.Draw(img)

    # Title row
    d.text((2, 0), "Bugsy", font=_font(15), fill=0)
    updated = str(state.get("updated", ""))
    d.text((W - 2, 2), updated, font=_font(10, False), fill=0, anchor="ra")
    d.line((0, 20, W, 20), fill=0)

    # Class counts
    counts: dict = state.get("counts") or {}
    classes = ("1", "2", "3", "other")
    x = 4
    colw = (W - 8) // len(classes)
    for cls in classes:
        d.text((x, 24), str(counts.get(cls, "-")), font=_font(28), fill=0)
        d.text((x, 58), cls[:8], font=_font(9, False), fill=0)
        x += colw

    d.line((0, 72, W, 72), fill=0)

    # Footer
    total = state.get("total", "-")
    d.text((2, 76), f"TOTAL: {total}", font=_font(11), fill=0)
    err = state.get("error", "")
    footer = ("! " + str(err))[:38] if err else f"frame {state.get('frame', '')}"
    d.text((2, 96), footer, font=_font(9, False), fill=0)
    d.text((2, 109), f"fps ~{state.get('fps', '?')}", font=_font(9, False), fill=0)

    return img


# ---------------------------------------------------------------------------
# EInk controller used by the pipeline
# ---------------------------------------------------------------------------

class EInk:
    """Non-blocking e-ink updater; refreshes at most once per `min_s` seconds."""

    def __init__(self, min_s: float = 10.0):
        self.min_s = float(min_s)
        self._state_file = Path("/tmp/bugsy_display_state.json")
        self._vendor = str(
            Path(__file__).resolve().parent.parent / "vendor"
        )
        self._lock = threading.Lock()
        self._last = 0.0

    def update(self, state: dict) -> None:
        if time.time() - self._last < self.min_s:
            return
        if self._lock.locked():
            return
        self._last = time.time()
        threading.Thread(
            target=self._push, args=(dict(state),), daemon=True, name="eink"
        ).start()

    def _push(self, state: dict) -> None:
        with self._lock:
            try:
                self._state_file.write_text(json.dumps(state))
                subprocess.run(
                    [sys.executable, __file__, str(self._state_file), self._vendor],
                    timeout=90,
                    capture_output=True,
                )
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Child-process entry point
# ---------------------------------------------------------------------------

def _push_to_hw(state_path: str, vendor: str) -> None:
    import os
    os.environ.setdefault("GPIOZERO_PIN_FACTORY", "rpigpio")
    sys.path.insert(0, vendor)
    from waveshare_epd import epd2in13_V4  # type: ignore

    img = render(json.loads(Path(state_path).read_text()))
    epd = epd2in13_V4.EPD()
    epd.init()
    epd.display(epd.getbuffer(img))
    epd.sleep()


if __name__ == "__main__":
    try:
        _push_to_hw(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "")
    except Exception as e:
        print("display skipped:", e)
