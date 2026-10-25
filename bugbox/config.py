"""Configuration: one commented TOML file, merged over built-in defaults."""
from __future__ import annotations

import copy
import json
import os
import tomllib
from pathlib import Path
from types import SimpleNamespace

DEFAULT_TOML = r'''# bugbox configuration. Edit this file, then restart bugbox.
name = ""                 # box name used in file names; blank = this Pi's hostname
mode = "standalone"       # "standalone" (everything on the Pi)
                          # "hybrid"     (detect on the Pi, ask the cloud only to classify single insects)
                          # "cloud"      (every frame is analysed by Roboflow, uses the most credits)

[paths]
data_dir = "~/bugbox_data"        # logs, state, local photo buffer, model cache

[lights]
enabled = true
pins = [26]               # BCM GPIO pins of the relays that switch the ring lights
active_low = true         # relay boards used by the Mothbox switch ON when the pin is LOW
policy = "capture"        # "capture" = on only around each photo (always on while streaming)
                          # "session" = on for the whole run, "off" = never switch lights
settle_seconds = 1.0      # let the lights come up before the shutter

[camera]
backend = "auto"          # auto | picamera2 | rpicam | opencv | fake
width = 4624              # still size (the 64MP sensor also offers 9248x6944, 3840x2160, 1920x1080)
height = 3472
vflip = false             # set true if your photos come out upside down
hflip = false
focus = "calibrate"       # calibrate = autofocus every calibrate_every_s (like the Mothbox)
                          # manual = fixed lens_position, auto = leave libcamera autofocus alone
lens_position = 2.6       # diopters, only used when focus = "manual"
calibrate_every_s = 600
exposure_us = 0           # 0 = automatic. A short fixed value (e.g. 1000) reduces motion blur
gain = 0.0                # 0 = automatic
jpeg_quality = 95
opencv_index = 0          # only for backend = "opencv" (USB webcam)

[storage]
usb_required_modes = ["standalone", "hybrid"]   # these modes refuse to write photos to the SD card
fallback_to_sd = false              # true = buffer on the SD card when no USB, copy over later
auto_mount = true                   # try to mount an unmounted USB drive (udisksctl)
min_free_gb = 1.0                   # stop writing when less than this is free
folder = "bugbox"                   # folder created on the drive
sd_folder = ""                      # blank = <data_dir>/photos
keep_annotated = true               # save a copy with boxes drawn on it

[detect]
confidence = 0.5
nms_iou = 0.6             # merges duplicate boxes (1.0 = off)
tiles = 1                 # 1 = whole image, 2 = 2x2 overlapping tiles (helps small insects)
tile_overlap = 0.15
targets = ["stink bug", "chinch bug", "wireworm"]   # species reported separately
other_label = "other insect"                        # everything else is counted here
# [detect.aliases]        # map model class names to a target, e.g.
# stinkbug = "stink bug"

[roboflow]
api_key_env = "ROBOFLOW_API_KEY"   # the key is read from this environment variable
model_id = ""                      # "your-project/1"

[standalone]
style = "stills"          # "stills" = timed photos saved to the USB drive, "stream" = live frames
server_url = ""           # optional: a local Roboflow Inference server, e.g. http://localhost:9001
cache_dir = ""            # model weight cache; blank = <data_dir>/models

[hybrid]
style = "stills"          # "stills" = timed photos saved to USB, "stream" = live frames
detector_model_id = ""    # the model that runs on the Pi; blank = roboflow.model_id
classify = "uncertain"    # "uncertain" = ask the cloud only when the Pi is unsure, "all" = every insect, "off"
uncertain_below = 0.85    # Pi confidence under this triggers a cloud check
max_calls_per_minute = 30
daily_budget = 500        # hard stop per day; after that the Pi's own labels are used
crop_pad = 0.25           # extra context around each insect crop
crop_target_side = 320    # small crops are enlarged to this before upload

[classifier]
api_url = "https://serverless.roboflow.com"
model_id = ""             # classification model "project/version". Blank = reuse roboflow.model_id on the crop
min_confidence = 0.6      # cloud answers below this are ignored (the Pi's label stays)
cache_ttl_s = 900         # an insect that has not changed is not asked about again
timeout_s = 10

[cloud]
style = "stream"          # full cloud: "stream" = frames sent at [stream].fps, or "stills"
api_url = "https://serverless.roboflow.com"
max_side = 1280           # frames are shrunk to this before upload

[stills]
interval_s = 60

[stream]
fps = 1.0
width = 1280
height = 720
smoothing_frames = 5      # displayed counts are the median of the last N frames
save_every_n = 0          # save every Nth frame to storage (0 = never)

[schedule]
enabled = false           # false = run all the time
weekdays = [1, 2, 3, 4, 5, 6, 7]   # 1 = Monday
hours = [19, 21, 23, 2, 4]
minutes = [0]
runtime_min = 59

[power]
sleep_between_sessions = false     # shut down and wake by RTC alarm (run `bugbox power-setup` once)
switch_pin = 0                     # BCM pin; grounded = cloud mode, open = standalone (0 = disabled)

[display]
eink = true               # Waveshare 2.13in e-paper, if installed
refresh_min_s = 30

[diagnostics]
interval_s = 300

[web]
enabled = true
host = "127.0.0.1"        # use "0.0.0.0" to open the dashboard from another computer
port = 8080
'''


MODES = ("standalone", "hybrid", "cloud")


def _deep_merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def defaults() -> dict:
    return tomllib.loads(DEFAULT_TOML)


def _ns(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _ns(v) for k, v in d.items()})
    return d


def config_search_paths():
    env = os.environ.get("BUGBOX_CONFIG")
    if env:
        yield Path(env).expanduser()
    yield Path("~/.config/bugbox/config.toml").expanduser()
    yield Path.cwd() / "config" / "config.toml"


def find_config_path():
    for p in config_search_paths():
        if p.is_file():
            return p
    return None


class Config(SimpleNamespace):
    """Attribute access (cfg.lights.pins) plus helpers."""

    def data_dir(self) -> Path:
        p = Path(os.path.expanduser(self.paths.data_dir))
        p.mkdir(parents=True, exist_ok=True)
        return p

    def box_name(self) -> str:
        if self.name:
            return str(self.name)
        import socket
        return socket.gethostname()

    def api_key(self) -> str:
        return os.environ.get(self.roboflow.api_key_env, "")

    # ---- persisted runtime state (mode chosen from the CLI or dashboard) ----
    def _state_path(self) -> Path:
        return self.data_dir() / "state.json"

    def read_state(self) -> dict:
        try:
            return json.loads(self._state_path().read_text())
        except Exception:
            return {}

    def write_state(self, **kv):
        st = self.read_state()
        st.update(kv)
        tmp = self._state_path().with_suffix(".tmp")
        tmp.write_text(json.dumps(st))
        os.replace(tmp, self._state_path())

    def effective_mode(self) -> str:
        m = os.environ.get("BUGBOX_MODE") or self.read_state().get("mode") or self.mode
        return m if m in MODES else "standalone"

    def style_for(self, mode: str) -> str:
        s = {"standalone": self.standalone, "hybrid": self.hybrid, "cloud": self.cloud}.get(mode, self.standalone).style
        return s if s in ("stills", "stream") else "stills"

    def as_dict(self) -> dict:
        def un(o):
            if isinstance(o, SimpleNamespace):
                return {k: un(v) for k, v in vars(o).items()}
            return o
        return un(self)


def load_config(path=None, overrides: dict | None = None) -> Config:
    base = defaults()
    p = Path(path).expanduser() if path else find_config_path()
    if p and p.is_file():
        with open(p, "rb") as f:
            base = _deep_merge(base, tomllib.load(f))
    if overrides:
        base = _deep_merge(base, overrides)
    cfg = Config(**{k: _ns(v) for k, v in base.items()})
    cfg.config_path = str(p) if p else "(built-in defaults)"
    return cfg
