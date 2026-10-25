import csv, json, sys, tempfile, unittest, time
from pathlib import Path
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__))))
from bugbox.config import load_config
from bugbox.engine import Engine
from bugbox.storage import Target, StorageUnavailable, parse_lsblk, Storage


def cfg_for(tmp, **over):
    base = {"paths": {"data_dir": str(tmp / "data")}, "name": "testbox",
            "lights": {"settle_seconds": 0.0}, "display": {"eink": False},
            "storage": {"usb_required_modes": ["standalone", "hybrid"], "fallback_to_sd": False, "auto_mount": False},
            "stream": {"smoothing_frames": 3}}
    for k, v in over.items():
        base.setdefault(k, {}).update(v) if isinstance(v, dict) else base.__setitem__(k, v)
    return load_config(path="/nonexistent", overrides=base)


class FakeUSB:
    def __init__(self, root): self.root = root
    def __call__(self):
        return Target(Path(self.root), "usb", "SANDISK", 50.0, 64.0)


class EngineTests(unittest.TestCase):
    def test_standalone_still_goes_to_usb_not_sd(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t); usb = t / "usb"; usb.mkdir()
            e = Engine(cfg_for(t), simulate=True)
            e.storage.find_usb = FakeUSB(usb)
            res = e.capture_and_count("test")
            self.assertTrue(res.image_path.startswith(str(usb)), res.image_path)
            self.assertTrue(Path(res.image_path).exists())
            self.assertTrue(Path(res.annotated_path).exists())
            self.assertEqual(sum(res.counts.values()), res.total)
            rows = list(csv.reader(open(usb / "bugbox" / "testbox" / "counts.csv")))
            self.assertEqual(rows[0][:4], ["time", "mode", "source", "image"]); self.assertEqual(len(rows), 2)
            sd = t / "data" / "photos"
            self.assertFalse(any(sd.rglob("*.jpg")) if sd.exists() else False)
            self.assertIn("testbox", e.status()["name"])
            self.assertIsNotNone(e.latest_jpeg())

    def test_standalone_without_usb_refuses(self):
        with tempfile.TemporaryDirectory() as t:
            e = Engine(cfg_for(Path(t)), simulate=True)
            e.storage.find_usb = lambda: None
            with self.assertRaises(StorageUnavailable):
                e.capture_and_count("test")

    def test_sd_fallback_then_sync_to_usb(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t); usb = t / "usb"; usb.mkdir()
            e = Engine(cfg_for(t, storage={"fallback_to_sd": True, "usb_required_modes": []}), simulate=True)
            e.storage.find_usb = lambda: None
            r1 = e.capture_and_count("a")
            self.assertTrue(r1.image_path.startswith(str(t / "data" / "photos")))
            e.storage.find_usb = FakeUSB(usb)
            moved = e.storage.sync_pending(every_s=0)
            self.assertGreaterEqual(moved, 3)
            self.assertTrue(list(usb.rglob("*.jpg")))
            self.assertFalse(list((t / "data" / "photos").rglob("*.jpg")))

    def test_cloud_stream_smooths_and_logs(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            e = Engine(cfg_for(t, mode="cloud"), simulate=True)
            e.cfg.write_state(mode="cloud")
            outs = [e.process_frame() for _ in range(4)]
            self.assertEqual(outs[0].mode, "cloud"); self.assertEqual(outs[0].source, "frame")
            st = e.status()
            self.assertEqual(st["style"], "stream"); self.assertIsInstance(st["total"], int)
            self.assertTrue((t / "data" / "logs" / "counts.csv").exists())
            self.assertFalse(list(t.rglob("*.jpg")))     # nothing saved to disk in plain stream mode

    def test_mode_switch_persists(self):
        with tempfile.TemporaryDirectory() as t:
            e = Engine(cfg_for(Path(t)), simulate=True)
            self.assertEqual(e.current_mode(), "standalone")
            e.set_mode("cloud")
            self.assertEqual(e.current_mode(), "cloud")
            e2 = Engine(cfg_for(Path(t)), simulate=True)
            self.assertEqual(e2.current_mode(), "cloud")
            with self.assertRaises(ValueError):
                e.set_mode("bogus")

    def test_run_loop_takes_timed_stills(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t); usb = t / "usb"; usb.mkdir()
            e = Engine(cfg_for(t, stills={"interval_s": 1}), simulate=True)
            e.storage.find_usb = FakeUSB(usb)
            import threading
            th = threading.Thread(target=e.run); th.start()
            time.sleep(3.5); e.request_stop(); th.join(10)
            self.assertGreaterEqual(len(list(usb.rglob("*_result.json"))), 2)


class LsblkTests(unittest.TestCase):
    SAMPLE = json.dumps({"blockdevices": [
        {"name": "mmcblk0", "path": "/dev/mmcblk0", "type": "disk", "tran": None, "rm": False, "hotplug": False,
         "fstype": None, "label": None, "mountpoints": [None], "size": 31914983424, "children": [
            {"name": "mmcblk0p1", "path": "/dev/mmcblk0p1", "type": "part", "fstype": "vfat", "label": "bootfs",
             "mountpoints": ["/boot/firmware"], "size": 536870912, "rm": False, "hotplug": False},
            {"name": "mmcblk0p2", "path": "/dev/mmcblk0p2", "type": "part", "fstype": "ext4", "label": "rootfs",
             "mountpoints": ["/"], "size": 31000000000, "rm": False, "hotplug": False}]},
        {"name": "sda", "path": "/dev/sda", "type": "disk", "tran": "usb", "rm": True, "hotplug": True,
         "fstype": None, "label": None, "mountpoints": [None], "size": 64000000000, "children": [
            {"name": "sda1", "path": "/dev/sda1", "type": "part", "fstype": "exfat", "label": "SANDISK",
             "mountpoints": ["/media/bugs/SANDISK"], "size": 63999000000, "rm": True, "hotplug": True}]}]})

    def test_only_usb_partition_found(self):
        parts = parse_lsblk(self.SAMPLE)
        self.assertEqual([p["label"] for p in parts], ["SANDISK"])
        self.assertEqual(parts[0]["mount"], "/media/bugs/SANDISK")

    def test_find_usb_uses_mount_and_checks_writable(self):
        with tempfile.TemporaryDirectory() as t:
            mnt = Path(t) / "usb"; mnt.mkdir()
            sample = self.SAMPLE.replace("/media/bugs/SANDISK", str(mnt))
            s = Storage(cfg_for(Path(t)))
            s._lsblk = lambda: sample
            u = s.find_usb()
            self.assertIsNotNone(u); self.assertEqual(u.kind, "usb"); self.assertEqual(str(u.root), str(mnt))
            s._lsblk = lambda: self.SAMPLE.replace("/media/bugs/SANDISK", "/nonexistent/path")
            self.assertIsNone(s.find_usb())


if __name__ == "__main__":
    unittest.main()
