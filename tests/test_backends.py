import os, sys, tempfile, types, unittest
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__)))); sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import numpy as np
from test_engine import cfg_for
from bugbox.detect import build_backend, Detector


def install_fake_inference():
    """Stand-ins that mimic the documented Roboflow APIs, so the glue code can be tested without the packages."""
    seen = {}

    class Pred:      # like inference's pydantic prediction
        def __init__(self, c, x, y, w, h, conf):
            self.class_name, self.x, self.y, self.width, self.height, self.confidence = c, x, y, w, h, conf

    class Resp:
        def __init__(self, preds): self.predictions = preds

    class FakeModel:
        def infer(self, img, confidence=None):
            seen["local_shape"] = img.shape; seen["local_conf"] = confidence
            return [Resp([Pred("Stink_Bug", 100, 100, 40, 30, 0.9), Pred("beetle", 300, 200, 20, 20, 0.8),
                          Pred("wireworm", 50, 50, 10, 10, 0.2)])]

    inf = types.ModuleType("inference")
    def get_model(model_id, api_key=None):
        seen["model_id"] = model_id; seen["api_key"] = api_key; seen["cache"] = os.environ.get("MODEL_CACHE_DIR")
        return FakeModel()
    inf.get_model = get_model

    class Cfg:
        def __init__(self, confidence_threshold=None, api_key_transport=None):
            seen["http_cfg"] = (confidence_threshold, api_key_transport)

    class Client:
        def __init__(self, api_url, api_key=None): seen["api_url"] = api_url; seen["http_key"] = api_key
        def configure(self, c): return self
        def infer(self, img, model_id):
            seen["http_shape"] = img.shape; seen["http_model"] = model_id
            # coordinates are in the shrunken image's pixels
            return {"predictions": [{"x": 64, "y": 36, "width": 20, "height": 10, "confidence": 0.95, "class": "stink bug"}]}

    sdk = types.ModuleType("inference_sdk")
    sdk.InferenceHTTPClient = Client; sdk.InferenceConfiguration = Cfg
    sys.modules["inference"] = inf; sys.modules["inference_sdk"] = sdk
    return seen


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.seen = install_fake_inference()
    def tearDown(self):
        sys.modules.pop("inference", None); sys.modules.pop("inference_sdk", None)

    def test_local_backend_uses_persistent_cache_and_counts(self):
        with tempfile.TemporaryDirectory() as t:
            os.environ.pop("MODEL_CACHE_DIR", None)
            cfg = cfg_for(__import__("pathlib").Path(t), roboflow={"model_id": "bugs/3"})
            b = build_backend(cfg, "standalone")
            det = Detector(b, cfg)
            dets = det.detect(np.zeros((480, 640, 3), np.uint8))
            c = det.count(dets)
            self.assertEqual(self.seen["model_id"], "bugs/3")
            self.assertTrue(self.seen["cache"].endswith("models"), self.seen["cache"])
            self.assertEqual(c["counts"]["stink bug"], 1); self.assertEqual(c["counts"]["other insect"], 1)
            self.assertEqual(c["total"], 2)   # the 0.2-confidence wireworm is below 0.5

    def test_cloud_backend_downscales_and_rescales_boxes(self):
        os.environ["ROBOFLOW_API_KEY"] = "secret"
        with tempfile.TemporaryDirectory() as t:
            cfg = cfg_for(__import__("pathlib").Path(t), roboflow={"model_id": "bugs/3"}, mode="cloud")
            b = build_backend(cfg, "cloud")
            dets = b.infer_array(np.zeros((1440, 2560, 3), np.uint8))   # 2x the 1280 max side
            self.assertEqual(self.seen["http_shape"][1], 1280)
            self.assertEqual(self.seen["api_url"], "https://serverless.roboflow.com")
            self.assertEqual(self.seen["http_cfg"][1], "header")
            self.assertAlmostEqual(dets[0].x, 128.0); self.assertAlmostEqual(dets[0].w, 40.0)
        os.environ.pop("ROBOFLOW_API_KEY")

    def test_cloud_without_key_explains(self):
        os.environ.pop("ROBOFLOW_API_KEY", None)
        with tempfile.TemporaryDirectory() as t:
            cfg = cfg_for(__import__("pathlib").Path(t), roboflow={"model_id": "bugs/3"})
            with self.assertRaises(RuntimeError) as cm:
                build_backend(cfg, "cloud")
            self.assertIn("ROBOFLOW_API_KEY", str(cm.exception))

    def test_local_server_option_uses_http_client(self):
        with tempfile.TemporaryDirectory() as t:
            cfg = cfg_for(__import__("pathlib").Path(t), roboflow={"model_id": "bugs/3"},
                          standalone={"server_url": "http://localhost:9001"})
            b = build_backend(cfg, "standalone")
            self.assertEqual(self.seen["api_url"], "http://localhost:9001"); self.assertEqual(b.name, "local-server")


if __name__ == "__main__":
    unittest.main()
