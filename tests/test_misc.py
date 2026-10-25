import os, sys, tempfile, unittest, io, contextlib
from datetime import datetime
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_engine import cfg_for
from bugbox.schedule import Schedule
from bugbox.display import render
from bugbox.config import load_config, DEFAULT_TOML
from bugbox.lights import Lights


class Misc(unittest.TestCase):
    def test_schedule_windows(self):
        s = Schedule(load_config(path="/nonexistent", overrides={"schedule": {"enabled": True, "hours": [19, 23], "runtime_min": 59}}))
        self.assertTrue(s.status(datetime(2026, 10, 24, 19, 30))[0])
        self.assertFalse(s.status(datetime(2026, 10, 24, 20, 5))[0])
        self.assertEqual(s.next_start(datetime(2026, 10, 24, 20, 5)), datetime(2026, 10, 24, 23, 0))
        s2 = Schedule(load_config(path="/nonexistent", overrides={"schedule": {"enabled": True, "hours": [23], "runtime_min": 180}}))
        self.assertTrue(s2.status(datetime(2026, 10, 25, 1, 0))[0])         # crosses midnight
        s3 = Schedule(load_config(path="/nonexistent", overrides={"schedule": {"enabled": True, "hours": [10], "weekdays": [1]}}))
        self.assertEqual(s3.next_start(datetime(2026, 10, 24, 12, 0)), datetime(2026, 10, 26, 10, 0))  # next Monday
        self.assertTrue(Schedule(load_config(path="/nonexistent")).status()[0])   # disabled = always on

    def test_display_render_size_and_content(self):
        img = render({"name": "box", "mode": "cloud", "counts": {"stink bug": 4}, "targets": ["stink bug", "chinch bug", "wireworm"],
                      "total": 7, "usb": None, "updated": "10:00:00", "error": "storage: no USB"})
        self.assertEqual(img.size, (250, 122)); self.assertEqual(img.mode, "1")
        self.assertLess(img.getextrema()[0], 255)      # something was drawn

    def test_default_config_file_parses_and_overrides_merge(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "c.toml"; p.write_text(DEFAULT_TOML.replace('mode = "standalone"', 'mode = "cloud"'))
            self.assertEqual(load_config(str(p)).mode, "cloud")
            p.write_text('[lights]\npins = [21]\n')
            c = load_config(str(p)); self.assertEqual(c.lights.pins, [21]); self.assertEqual(c.camera.width, 4624)

    def test_lights_logic_active_low(self):
        class G:
            HIGH, LOW, OUT = 1, 0, "out"; calls = []
            def setwarnings(self, *a): pass
            def setmode(self, *a): pass
            BCM = "bcm"
            def setup(self, p, m, initial=None): self.calls.append(("setup", p, initial))
            def output(self, p, v): self.calls.append(("out", p, v))
            def cleanup(self, pins=None): pass
        import types
        g = G(); mod = types.ModuleType("RPi"); gp = types.ModuleType("RPi.GPIO")
        for k in ("HIGH", "LOW", "OUT", "BCM", "setwarnings", "setmode", "setup", "output", "cleanup"):
            setattr(gp, k, getattr(g, k))
        mod.GPIO = gp; sys.modules["RPi"] = mod; sys.modules["RPi.GPIO"] = gp
        try:
            with tempfile.TemporaryDirectory() as t:
                l = Lights(cfg_for(Path(t)))
                self.assertEqual(g.calls[0], ("setup", 26, 1))              # idle = HIGH (relay off)
                with l.lit():
                    self.assertTrue(l.is_on)
                    with l.lit():
                        pass
                    self.assertTrue(l.is_on)                                 # nested block did not switch off
                self.assertFalse(l.is_on)
                self.assertIn(("out", 26, 0), g.calls)                       # on = LOW
                self.assertEqual(g.calls[-1], ("out", 26, 1))                # off = HIGH
        finally:
            sys.modules.pop("RPi", None); sys.modules.pop("RPi.GPIO", None)


if __name__ == "__main__":
    unittest.main()
