"""Ring-light control through the relay board (same idea as the Mothbox's Attract/Flash scripts)."""
from __future__ import annotations

import contextlib
import threading

from .util import log


class Lights:
    def __init__(self, cfg, simulate: bool = False):
        self.cfg = cfg.lights
        self.pins = [int(p) for p in self.cfg.pins]
        self.active_low = bool(self.cfg.active_low)
        self.enabled = bool(self.cfg.enabled) and self.cfg.policy != "off"
        self._gpio = None
        self._on = False
        self._lock = threading.RLock()
        self._depth = 0
        if simulate or not self.pins:
            log.info("lights: simulated (no GPIO)")
            return
        try:
            import RPi.GPIO as GPIO  # rpi-lgpio provides this on the Pi 5
            GPIO.setwarnings(False)
            GPIO.setmode(GPIO.BCM)
            idle = GPIO.HIGH if self.active_low else GPIO.LOW
            for p in self.pins:
                GPIO.setup(p, GPIO.OUT, initial=idle)
            self._gpio = GPIO
            log.info("lights: GPIO pins %s, active_%s", self.pins, "low" if self.active_low else "high")
        except Exception as e:  # not on a Pi, or no permission
            log.warning("lights: GPIO unavailable (%s), running without lights", e)

    @property
    def is_on(self) -> bool:
        return self._on

    def _write(self, on: bool):
        if self._gpio is None:
            return
        level = (not on) if self.active_low else on
        for p in self.pins:
            self._gpio.output(p, self._gpio.HIGH if level else self._gpio.LOW)

    def on(self):
        with self._lock:
            if not self.enabled:
                return
            self._write(True)
            self._on = True

    def off(self):
        with self._lock:
            self._write(False)
            self._on = False

    @contextlib.contextmanager
    def lit(self):
        """Nested-safe: the lights go off when the outermost block ends."""
        with self._lock:
            self._depth += 1
            self.on()
        try:
            yield
        finally:
            with self._lock:
                self._depth -= 1
                if self._depth == 0 and self.cfg.policy != "session":
                    self.off()

    def close(self):
        with self._lock:
            try:
                self.off()
            finally:
                if self._gpio is not None:
                    try:
                        self._gpio.cleanup(self.pins)
                    except Exception:
                        pass
