import os, sys, tempfile, time, unittest
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from test_engine import cfg_for, FakeUSB
from bugbox.detect import (Backend, Detection, HybridDetector, parse_classification, CloudBudget, CropCache)
from bugbox.engine import Engine


class Scripted(Backend):
    def __init__(self, dets): self.dets = dets
    def infer_array(self, img): return [Detection(*d) for d in self.dets]


class Cls:
    """Counts calls; answers from a table keyed by approximate x position."""
    model_id = "cls/1"
    def __init__(self, table, fail=False): self.table, self.fail, self.calls = table, fail, 0
    def classify(self, crop, det=None):
        self.calls += 1
        if self.fail: raise RuntimeError("network down")
        return self.table[int(det.x // 100)]


def sheet():
    """Three insects drawn as distinct blobs so the crop hash differs."""
    img = np.full((400, 600, 3), 230, np.uint8)
    import cv2
    cv2.circle(img, (100, 100), 20, (20, 120, 20), -1)
    cv2.rectangle(img, (190, 80), (230, 130), (30, 30, 160), -1)
    cv2.ellipse(img, (310, 100), (30, 10), 30, 0, 360, (60, 150, 190), -1)
    return img


DETS = [("stink bug", 0.95, 100, 100, 50, 50),      # confident -> no cloud call
        ("other insect", 0.55, 210, 105, 60, 60),   # unsure -> cloud says chinch bug
        ("other insect", 0.60, 310, 100, 80, 40)]   # unsure -> cloud says wireworm


class HybridTests(unittest.TestCase):
    def make(self, cls, **hyb):
        cfg = cfg_for(Path(tempfile.mkdtemp()), hybrid=hyb)
        return HybridDetector(Scripted(DETS), cls, cfg), cfg

    def test_parse_classification_shapes(self):
        self.assertEqual(parse_classification({"top": "wireworm", "confidence": .9, "predictions": []}), ("wireworm", .9))
        self.assertEqual(parse_classification({"predictions": [{"class": "a", "confidence": .2}, {"class": "b", "confidence": .7}]}), ("b", .7))
        self.assertEqual(parse_classification({"predictions": {"x": {"confidence": .4}, "y": {"confidence": .8}}}), ("y", .8))
        det = {"predictions": [{"x": 1, "y": 1, "width": 5, "height": 5, "confidence": .5, "class": "stink bug"},
                               {"x": 1, "y": 1, "width": 5, "height": 5, "confidence": .9, "class": "wireworm"}]}
        self.assertEqual(parse_classification([det]), ("wireworm", .9))
        self.assertIsNone(parse_classification({"predictions": []}))

    def test_only_unsure_insects_go_to_the_cloud_then_cache_covers_repeats(self):
        cls = Cls({2: ("chinch bug", .9), 3: ("wireworm", .8)})
        det, _ = self.make(cls)
        img = sheet()
        dets = det.detect(img)
        self.assertEqual(cls.calls, 2)                              # the 0.95 stink bug was not sent
        c = det.count(dets)["counts"]
        self.assertEqual((c["stink bug"], c["chinch bug"], c["wireworm"], c["other insect"]), (1, 1, 1, 0))
        self.assertEqual([d.source for d in dets], ["local", "cloud", "cloud"])
        self.assertEqual(dets[1].local_label, "other insect")
        self.assertEqual(det.calls_last, 2)
        det.detect(img)                                             # same sheet again, as in a live stream
        self.assertEqual(cls.calls, 2)                              # no new cloud calls at all
        self.assertEqual(det.cache_hits_last, 2)
        self.assertEqual(det.stats()["cache_hits"], 2)

    def test_changed_insect_in_same_spot_is_asked_again(self):
        cls = Cls({2: ("chinch bug", .9), 3: ("wireworm", .8)})
        det, _ = self.make(cls)
        img = sheet(); det.detect(img)
        import cv2
        cv2.rectangle(img, (190, 80), (230, 130), (200, 200, 30), -1)   # different insect at that spot
        cv2.circle(img, (205, 100), 8, (0, 0, 0), -1)
        det.detect(img)
        self.assertGreater(cls.calls, 2)

    def test_budget_caps_calls_and_falls_back_to_local_labels(self):
        cls = Cls({2: ("chinch bug", .9), 3: ("wireworm", .8)})
        det, _ = self.make(cls, daily_budget=1)
        dets = det.detect(sheet())
        self.assertEqual(cls.calls, 1)
        self.assertEqual(det.stats()["budget_left"], 0)
        self.assertEqual(det.count(dets)["total"], 3)                  # still counts all three
        self.assertEqual(sum(1 for d in dets if d.source == "cloud"), 1)
        self.assertEqual(det.stats()["fallbacks"], 1)

    def test_network_failure_keeps_local_labels_and_trips_breaker(self):
        cls = Cls({}, fail=True)
        det, _ = self.make(cls, classify="all")
        for _ in range(3):
            dets = det.detect(sheet())
        self.assertEqual(det.count(dets)["total"], 3)
        self.assertEqual(det.count(dets)["counts"]["stink bug"], 1)    # local label survived
        calls_before = cls.calls
        det.detect(sheet())
        self.assertEqual(cls.calls, calls_before)                      # breaker open: no more calls for a while

    def test_low_confidence_cloud_answer_is_ignored(self):
        cls = Cls({2: ("chinch bug", .2), 3: ("wireworm", .3)})
        det, _ = self.make(cls)
        dets = det.detect(sheet())
        self.assertEqual([d.source for d in dets], ["local"] * 3)
        self.assertEqual(det.count(dets)["counts"]["other insect"], 2)

    def test_classify_off_never_calls(self):
        cls = Cls({})
        det, _ = self.make(cls, classify="off")
        det.detect(sheet()); self.assertEqual(cls.calls, 0)

    def test_budget_rollover_and_rate_limit(self):
        b = CloudBudget(per_min=2, per_day=10)
        self.assertEqual(b.allow(5), 2); b.record(2); self.assertEqual(b.allow(5), 0)
        b2 = CloudBudget(per_min=100, per_day=3, day_used=3, today="2000-01-01")   # old date resets
        self.assertEqual(b2.allow(5), 3)


class HybridEngineTests(unittest.TestCase):
    def test_hybrid_still_on_usb_with_stats(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t); usb = t / "usb"; usb.mkdir()
            e = Engine(cfg_for(t, hybrid={"classify": "all"}), simulate=True)
            e.storage.find_usb = FakeUSB(usb)
            e.set_mode("hybrid")
            res = e.capture_and_count("test")
            self.assertEqual(res.mode, "hybrid"); self.assertTrue(res.image_path.startswith(str(usb)))
            self.assertEqual(res.cloud_calls, sum(res.counts.values()) if res.cloud_calls else res.cloud_calls)
            st = e.status()
            self.assertEqual(st["mode"], "hybrid"); self.assertTrue(st["classifier"]["enabled"])
            self.assertIn("cloud-classify", st["backend"])
            self.assertGreaterEqual(st["classifier"]["calls_today"], 1)
            row = open(usb / "bugbox" / "testbox" / "counts.csv").read().splitlines()
            self.assertTrue(row[0].endswith("cloud_calls"))
            # state file keeps the daily budget across restarts
            e2 = Engine(cfg_for(t, hybrid={"classify": "all"}), simulate=True)
            e2.set_mode("hybrid"); e2.get_detector("hybrid")
            self.assertEqual(e2.status()["classifier"]["calls_today"], st["classifier"]["calls_today"])

    def test_hybrid_requires_usb_like_standalone(self):
        from bugbox.storage import StorageUnavailable
        with tempfile.TemporaryDirectory() as t:
            e = Engine(cfg_for(Path(t)), simulate=True); e.set_mode("hybrid"); e.storage.find_usb = lambda: None
            with self.assertRaises(StorageUnavailable):
                e.capture_and_count("x")

    def test_hybrid_without_key_degrades_to_local_labels(self):
        os.environ.pop("ROBOFLOW_API_KEY", None)
        from bugbox.detect import build_classifier
        cls, warn = build_classifier(cfg_for(Path(tempfile.mkdtemp()), roboflow={"model_id": "bugs/1"}))
        self.assertIsNone(cls); self.assertIn("API key", warn)


if __name__ == "__main__":
    unittest.main()
