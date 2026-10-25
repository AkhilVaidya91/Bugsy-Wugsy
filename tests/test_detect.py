import sys, unittest
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__))))
import numpy as np
from bugbox.config import load_config
from bugbox.detect import (Detection, Labeler, nms, merge_tiles, tile_boxes, parse_predictions,
                           Detector, Backend)


class Scripted(Backend):
    """Returns different boxes per call, to mimic tiles."""
    def __init__(self, per_call): self.per_call = list(per_call); self.calls = 0
    def infer_array(self, img):
        r = self.per_call[self.calls % len(self.per_call)]; self.calls += 1
        return [Detection(*d) for d in r]


class T(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config()

    def test_parse_dict_and_object(self):
        class P:  # like a pydantic prediction
            class_name = "Stink_Bug"; confidence = 0.9; x = 10; y = 10; width = 4; height = 6
        a = parse_predictions({"predictions": [{"x": 1, "y": 2, "width": 3, "height": 4, "confidence": .8, "class": "wireworm"}]})
        b = parse_predictions([P()])
        self.assertEqual(a[0].label, "wireworm"); self.assertEqual(b[0].label, "Stink_Bug"); self.assertEqual(b[0].h, 6)

    def test_labeler(self):
        lab = Labeler(["stink bug", "chinch bug", "wireworm"], "other insect", {"bean beetle": "wireworm"})
        for raw, want in [("Stink Bug", "stink bug"), ("stink_bug", "stink bug"), ("StinkBug", "stink bug"),
                          ("chinch-bug", "chinch bug"), ("bean beetle", "wireworm"), ("ladybug", "other insect")]:
            self.assertEqual(lab.canonical(raw), want, raw)
        r = lab.count([Detection("stink bug", .9, 0, 0, 1, 1), Detection("moth", .9, 0, 0, 1, 1),
                       Detection("wireworm", .2, 0, 0, 1, 1)], min_conf=0.5)
        self.assertEqual(r["counts"], {"stink bug": 1, "chinch bug": 0, "wireworm": 0, "other insect": 1})
        self.assertEqual(r["total"], 2)

    def test_nms_dedupes_double_label(self):
        d = [Detection("stink bug", .9, 50, 50, 20, 20), Detection("other", .6, 51, 50, 20, 20),
             Detection("wireworm", .8, 200, 200, 20, 20)]
        self.assertEqual(len(nms(d, 0.6)), 2)
        self.assertEqual(len(nms(d, 1.0)), 3)

    def test_tiles_cover_image_and_merge_cut_insect(self):
        boxes = tile_boxes(1000, 1000, 2, 0.15)
        self.assertEqual(len(boxes), 4)
        self.assertEqual(min(b[0] for b in boxes), 0); self.assertEqual(max(b[2] for b in boxes), 1000)
        # one insect straddling x=500: whole in tile0, cut in tile1
        whole = Detection("stink bug", .9, 500, 200, 60, 40)
        cut = Detection("stink bug", .7, 520, 200, 20, 40)
        out = merge_tiles([whole, cut], 0.6)
        self.assertEqual(len(out), 1)
        far = Detection("wireworm", .9, 800, 800, 40, 40)
        self.assertEqual(len(merge_tiles([whole, cut, far], 0.6)), 2)

    def test_detector_tiling_counts_once(self):
        cfg = load_config(overrides={"detect": {"tiles": 2, "confidence": 0.5}})
        # tile 0 (top-left) sees an insect at its right edge; tile 1 (top-right) sees the same insect partially
        b = Scripted([[("stink bug", .9, 490, 100, 60, 40)], [("stink bug", .8, 55, 100, 20, 40)], [], []])
        det = Detector(b, cfg)
        img = np.zeros((1000, 1000, 3), np.uint8)
        dets = det.detect(img)
        self.assertEqual(det.count(dets)["total"], 1)

    def test_confidence_filter(self):
        b = Scripted([[("wireworm", .3, 10, 10, 5, 5), ("wireworm", .8, 50, 50, 5, 5)]])
        det = Detector(b, self.cfg)
        self.assertEqual(det.count(det.detect(np.zeros((100, 100, 3), np.uint8)))["counts"]["wireworm"], 1)


if __name__ == "__main__":
    unittest.main()
