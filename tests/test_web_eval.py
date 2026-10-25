import csv, json, sys, tempfile, unittest, urllib.request
from pathlib import Path
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__))))
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from test_engine import cfg_for, FakeUSB
from bugbox.engine import Engine
from bugbox import webui
from bugbox.evaluate import evaluate


def call(url, data=None):
    req = urllib.request.Request(url, data=None if data is None else json.dumps(data).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status, r.read(), r.headers.get("Content-Type")
    except urllib.error.HTTPError as e:
        return e.code, e.read(), e.headers.get("Content-Type")


class WebTests(unittest.TestCase):
    def test_dashboard_api(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t); usb = t / "usb"; usb.mkdir()
            e = Engine(cfg_for(t), simulate=True); e.storage.find_usb = FakeUSB(usb)
            srv = webui.start(e, "127.0.0.1", 0); port = srv.server_address[1]; base = f"http://127.0.0.1:{port}"
            try:
                code, body, ct = call(base + "/"); self.assertEqual(code, 200); self.assertIn(b"bugbox", body)
                code, body, _ = call(base + "/api/status"); st = json.loads(body)
                self.assertEqual(st["mode"], "standalone"); self.assertEqual(st["targets"], ["stink bug", "chinch bug", "wireworm"])
                code, body, _ = call(base + "/api/capture", {}); res = json.loads(body)
                self.assertEqual(code, 200); self.assertIn("counts", res)
                code, body, ct = call(base + "/latest.jpg"); self.assertEqual(code, 200); self.assertEqual(ct, "image/jpeg")
                self.assertTrue(body[:2] == b"\xff\xd8")
                code, body, _ = call(base + "/api/mode", {"mode": "cloud"}); self.assertEqual(json.loads(body)["mode"], "cloud")
                code, body, _ = call(base + "/api/mode", {"mode": "nope"}); self.assertEqual(code, 500)
                code, body, _ = call(base + "/api/status"); self.assertEqual(json.loads(body)["mode"], "cloud")
                self.assertEqual(call(base + "/missing")[0], 404)
            finally:
                srv.shutdown(); e.close()

    def test_no_usb_reports_error_over_http(self):
        with tempfile.TemporaryDirectory() as t:
            e = Engine(cfg_for(Path(t)), simulate=True); e.storage.find_usb = lambda: None
            srv = webui.start(e, "127.0.0.1", 0); base = f"http://127.0.0.1:{srv.server_address[1]}"
            try:
                code, body, _ = call(base + "/api/capture", {})
                self.assertEqual(code, 500); self.assertIn("no USB", json.loads(body)["error"])
            finally:
                srv.shutdown(); e.close()


class EvalTests(unittest.TestCase):
    def test_evaluate_perfect_detector(self):
        import random
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            e = Engine(cfg_for(t), simulate=True)
            e.world.rng = random.Random(1)
            import bugbox.detect as D
            e._detector = None
            # make the fake backend lossless for this test
            e.get_detector("standalone").backend.miss = 0.0
            rows = []
            for i in range(5):
                p = t / f"img{i}.jpg"
                e.get_camera().capture_still(p)
                truth = {}
                for d in e.world.truth:
                    truth[d.label] = truth.get(d.label, 0) + 1
                rows.append({"image": p.name, "stink bug": truth.get("stink bug", 0), "chinch bug": truth.get("chinch bug", 0),
                             "wireworm": truth.get("wireworm", 0), "total": len(e.world.truth)})
                # evaluation re-reads the file, but the fake backend reports e.world.truth of the LAST scene
                # so score each image right after drawing it
                with open(t / "truth.csv", "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows[-1:])
                out = evaluate(e, str(t), str(t / "truth.csv"))
                self.assertEqual(out["summary"]["total"]["exact_match"], 1.0)
                self.assertEqual(out["summary"]["stink bug"]["count_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
