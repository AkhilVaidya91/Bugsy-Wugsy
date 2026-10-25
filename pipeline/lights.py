"""Lights: switch the ring lights through the relay board (GPIO, active low by default).

Same behaviour as the Mothbox's Attract/Flash scripts and bugbox's lights module:

  * the relay board switches ON when its input pin is driven LOW (``active_low=True``);
  * pins are set up already OFF, so the lights never flash at start-up;
  * ``GPIO.cleanup()`` is deliberately NOT called. Like the Mothbox scripts, the pins stay
    driven (OFF) when the program ends instead of floating, so a relay never turns on by itself.

If RPi.GPIO is missing or the pins cannot be opened (for example on a laptop) the module logs a
warning and does nothing, so the rest of the pipeline still runs.

Command line (handy for testing the relay and lights):

    python -m pipeline.lights blink          # 3 blinks
    python -m pipeline.lights on             # leaves the lights ON after the command exits
    python -m pipeline.lights off
"""
from __future__ import annotations

import argparse
import atexit
import contextlib
import logging
import threading
import time
from typing import Sequence

log = logging.getLogger(__name__)


class Lights:
    def __init__(
        self,
        pins: Sequence[int] = (26,),
        active_low: bool = True,
        enabled: bool = True,
        off_at_exit: bool = True,
    ):
        self.pins = [int(p) for p in pins]
        self.active_low = bool(active_low)
        self.enabled = bool(enabled) and bool(self.pins)
        self._gpio = None
        self._on = False
        self._depth = 0
        self._lock = threading.RLock()

        if not self.enabled:
            log.info("lights: disabled")
            return
        try:
            import RPi.GPIO as GPIO  # provided by python3-rpi-lgpio on the Pi 5

            GPIO.setwarnings(False)
            GPIO.setmode(GPIO.BCM)
            off_level = GPIO.HIGH if self.active_low else GPIO.LOW
            for p in self.pins:
                GPIO.setup(p, GPIO.OUT, initial=off_level)  # start OFF, no flash
            self._gpio = GPIO
            log.info(
                "lights: GPIO pins %s, active %s",
                self.pins, "low" if self.active_low else "high",
            )
            if off_at_exit:
                atexit.register(self.off)
        except Exception as exc:  # not a Pi, or no permission
            log.warning("lights: GPIO unavailable (%s) – running without lights", exc)

    # ------------------------------------------------------------------
    @property
    def is_on(self) -> bool:
        return self._on

    @property
    def available(self) -> bool:
        """True when the GPIO pins were opened successfully."""
        return self._gpio is not None

    def _write(self, on: bool) -> None:
        if self._gpio is None:
            return
        level_high = (not on) if self.active_low else on
        for p in self.pins:
            self._gpio.output(p, self._gpio.HIGH if level_high else self._gpio.LOW)

    def on(self) -> None:
        with self._lock:
            if not self.enabled:
                return
            try:
                self._write(True)
                self._on = True
            except Exception as exc:
                log.warning("lights: could not switch on (%s)", exc)

    def off(self) -> None:
        with self._lock:
            try:
                self._write(False)
            except Exception as exc:  # e.g. GPIO already torn down at interpreter exit
                log.debug("lights: off failed (%s)", exc)
            self._on = False

    def blink(self, times: int = 3, period_s: float = 0.6) -> None:
        for _ in range(times):
            self.on()
            time.sleep(period_s)
            self.off()
            time.sleep(period_s)

    @contextlib.contextmanager
    def lit(self):
        """Lights on inside the block, off after it (safe to nest)."""
        with self._lock:
            self._depth += 1
            self.on()
        try:
            yield self
        finally:
            with self._lock:
                self._depth -= 1
                if self._depth == 0:
                    self.off()


# ----------------------------------------------------------------------
def parse_pins(text: str) -> list:
    """'26' or '26,21' -> [26, 21]"""
    return [int(p) for p in str(text).replace(" ", "").split(",") if p]


def _main() -> None:
    ap = argparse.ArgumentParser(prog="python -m pipeline.lights")
    ap.add_argument("action", choices=["on", "off", "blink"])
    ap.add_argument("--pins", default="26", help="BCM GPIO pin(s) of the relay, e.g. 26 or 26,21")
    ap.add_argument("--active-high", action="store_true", help="relay board switches on when the pin is HIGH")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    lights = Lights(parse_pins(a.pins), active_low=not a.active_high, off_at_exit=False)
    if not lights.available:
        raise SystemExit("could not open the GPIO pins")
    if a.action == "on":
        lights.on()
        print("lights ON (they stay on after this command ends; run 'off' to switch them off)")
    elif a.action == "off":
        lights.off()
        print("lights OFF")
    else:
        lights.blink()
        print("blinked 3 times")


if __name__ == "__main__":
    _main()
