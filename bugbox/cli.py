from __future__ import annotations

import argparse
import getpass
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from . import __version__, power
from .config import DEFAULT_TOML, MODES, find_config_path, load_config
from .util import log, setup_logging

ENV_FILE = Path("~/.config/bugbox/env").expanduser()


def load_env_file():
    try:
        for line in ENV_FILE.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    except FileNotFoundError:
        pass


def _engine(args):
    from .engine import Engine
    cfg = load_config(args.config)
    if getattr(args, "mode", None) in MODES:
        os.environ["BUGBOX_MODE"] = args.mode
    setup_logging(cfg.data_dir(), args.verbose)
    return Engine(cfg, simulate=args.simulate)


def _print_counts(res):
    print(json.dumps({"time": res.when, "mode": res.mode, "counts": res.counts, "total": res.total,
                      "latency_s": res.latency_s, "image": res.image_path}, indent=2))


# ----------------------------------------------------------------- commands
def cmd_run(args):
    e = _engine(args)
    from . import webui
    if e.cfg.web.enabled and not args.no_web:
        host, port = (args.host or e.cfg.web.host), int(args.port or e.cfg.web.port)
        try:
            webui.start(e, host, port)
        except OSError as ex:
            log.error("dashboard could not start on %s:%s (%s)", host, port, ex)
    signal.signal(signal.SIGTERM, lambda *a: e.request_stop())
    signal.signal(signal.SIGINT, lambda *a: e.request_stop())
    e.run()


def cmd_once(args):
    e = _engine(args)
    try:
        _print_counts(e.capture_and_count("cli"))
    finally:
        e.close()


def cmd_detect(args):
    e = _engine(args)
    try:
        res = e.detect_file(args.image, save_annotated=args.out)
        _print_counts(res)
        if args.out:
            print("annotated copy:", args.out)
    finally:
        e.close()


def cmd_evaluate(args):
    from .evaluate import evaluate
    e = _engine(args)
    try:
        out = evaluate(e, args.folder, args.truth, args.out)
        print(f"{out['images']} images scored")
        for k, v in out["summary"].items():
            print(f"  {k:14s} exact {v['exact_match']*100:5.1f}%   count accuracy {v['count_accuracy']*100:5.1f}%   "
                  f"mean abs error {v['mean_abs_error']}")
    finally:
        e.close()


def cmd_mode(args):
    cfg = load_config(args.config)
    if args.which:
        cfg.write_state(mode=args.which)
        print(f"mode set to {args.which} (running service picks it up within a few seconds)")
    else:
        print(cfg.effective_mode())


def cmd_lights(args):
    from .lights import Lights
    cfg = load_config(args.config)
    cfg.lights.enabled = True
    l = Lights(cfg, simulate=args.simulate)
    try:
        if args.action == "on":
            l.on(); print("lights on (Ctrl+C to switch off)")
            while True:
                time.sleep(1)
        elif args.action == "off":
            l.off(); print("lights off")
        else:
            for _ in range(3):
                l.on(); time.sleep(0.6); l.off(); time.sleep(0.6)
            print("blinked 3 times")
    except KeyboardInterrupt:
        pass
    finally:
        l.close()


def cmd_camera_test(args):
    from .camera import make_camera
    from .lights import Lights
    cfg = load_config(args.config)
    cfg.lights.enabled = True
    cam = make_camera(cfg, args.simulate)
    lights = Lights(cfg, args.simulate)
    out = Path(args.out).expanduser()
    try:
        cam.set_style("still")
        with lights.lit():
            time.sleep(float(cfg.lights.settle_seconds))
            cam.calibrate(force=True) if hasattr(cam, "calibrate") and cam.name == "picamera2" else None
            meta = cam.capture_still(out)
        print("saved", out, meta)
    finally:
        lights.close(); cam.close()


def cmd_status(args):
    cfg = load_config(args.config)
    from .storage import Storage
    print(f"bugbox {__version__}")
    print("config      :", cfg.config_path)
    print("mode        :", cfg.effective_mode(), f"(style: {cfg.style_for(cfg.effective_mode())})")
    print("model id    :", cfg.roboflow.model_id or "NOT SET (roboflow.model_id)")
    print("api key     :", "set" if cfg.api_key() else f"missing (run: bugbox set-key)")
    if cfg.effective_mode() == "hybrid":
        print("classifier  :", cfg.classifier.model_id or "(reuses roboflow.model_id on each crop)",
              f"| policy {cfg.hybrid.classify} | budget {cfg.hybrid.daily_budget}/day")
    print("lights      : pins", list(cfg.lights.pins), "active_low" if cfg.lights.active_low else "active_high")
    s = Storage(cfg)
    d = s.describe(cfg.effective_mode())
    print("usb drive   :", d["usb"] or ("NOT FOUND" + (" (required in this mode)" if d["usb_required"] else "")))
    print("sd free     :", d["sd_free_gb"], "GB")
    print("wifi        :", power.wifi_current() or "-")
    try:
        from .camera import make_camera
        cam = make_camera(cfg, args.simulate); print("camera      :", cam.name); cam.close()
    except Exception as e:
        print("camera      : NOT AVAILABLE -", e)
    try:
        import inference  # noqa
        print("local model : 'inference' package installed")
    except Exception:
        print("local model : 'inference' package NOT installed (standalone mode needs it)")
    try:
        import inference_sdk  # noqa
        print("cloud sdk   : 'inference-sdk' installed")
    except Exception:
        print("cloud sdk   : 'inference-sdk' NOT installed (cloud mode needs it)")


def cmd_warmup(args):
    """Download the model weights once, while online, so standalone mode can run offline."""
    from .detect import build_backend
    cfg = load_config(args.config)
    setup_logging(cfg.data_dir(), args.verbose)
    mode = "hybrid" if cfg.effective_mode() == "hybrid" else "standalone"
    os.environ["BUGBOX_MODE"] = mode
    if cfg.standalone.server_url:
        print("standalone.server_url is set, so weights live in that server, nothing to download here")
        return
    backend = build_backend(cfg, mode)
    import numpy as np
    backend.infer_array(np.zeros((640, 640, 3), np.uint8))
    print("model ready; cached in", os.environ.get("MODEL_CACHE_DIR"))


def cmd_wifi(args):
    ok, msg = power.wifi_connect(args.ssid, args.password)
    print(("connected: " if ok else "failed: ") + msg)


def cmd_time(args):
    if args.timezone:
        print(power.set_timezone(args.timezone))
    if args.set:
        print(power.set_clock(args.set))
    if args.ntp:
        power.ntp_on(); print("network time on")
    print(time.strftime("now: %Y-%m-%d %H:%M:%S %Z"))


def cmd_power_setup(args):
    ok, msg = power.eeprom_setup()
    print(("ok: " if ok else "failed: ") + msg)
    print("Then set [power] sleep_between_sessions = true and [schedule] enabled = true in config.toml.")


def cmd_set_key(args):
    key = getpass.getpass("Roboflow API key (hidden): ").strip()
    if not key:
        print("nothing entered"); return
    ENV_FILE.parent.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    lines = [l for l in (ENV_FILE.read_text().splitlines() if ENV_FILE.exists() else [])
             if not l.startswith(cfg.roboflow.api_key_env + "=")]
    lines.append(f"{cfg.roboflow.api_key_env}={key}")
    ENV_FILE.write_text("\n".join(lines) + "\n")
    ENV_FILE.chmod(0o600)
    print("saved to", ENV_FILE)


def cmd_init_config(args):
    p = Path("~/.config/bugbox/config.toml").expanduser()
    if p.exists() and not args.force:
        print(p, "already exists (use --force to overwrite)"); return
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(DEFAULT_TOML)
    print("wrote", p)


def cmd_install_service(args):
    user = args.user or os.environ.get("USER") or getpass.getuser()
    home = Path(os.path.expanduser(f"~{user}"))
    root = Path(__file__).resolve().parent.parent
    unit = f"""[Unit]
Description=bugbox insect imager
After=network-online.target
Wants=network-online.target

[Service]
User={user}
WorkingDirectory={root}
EnvironmentFile=-{home}/.config/bugbox/env
ExecStart={sys.executable} -m bugbox run
Restart=on-failure
RestartSec=10
SupplementaryGroups=gpio video spi i2c

[Install]
WantedBy=multi-user.target
"""
    tmp = Path("/tmp/bugbox.service"); tmp.write_text(unit)
    for cmd in (["sudo", "cp", str(tmp), "/etc/systemd/system/bugbox.service"],
                ["sudo", "systemctl", "daemon-reload"], ["sudo", "systemctl", "enable", "--now", "bugbox"]):
        subprocess.run(cmd, check=True)
    print("service installed. Logs: journalctl -u bugbox -f")


# ----------------------------------------------------------------- parser
def build_parser():
    p = argparse.ArgumentParser(prog="bugbox", description="Mothbox-style insect imager with standalone and cloud modes")
    p.add_argument("--config", help="path to config.toml")
    p.add_argument("--simulate", action="store_true", help="fake camera, lights and detector (no hardware)")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(name, fn, help_, **kw):
        sp = sub.add_parser(name, help=help_)
        sp.set_defaults(fn=fn)
        return sp

    s = add("run", cmd_run, "run the imager and dashboard"); s.add_argument("--mode", choices=list(MODES))
    s.add_argument("--no-web", action="store_true"); s.add_argument("--host"); s.add_argument("--port")
    s = add("once", cmd_once, "take one photo and count the insects"); s.add_argument("--mode", choices=list(MODES))
    s = add("detect", cmd_detect, "count insects in an existing image file"); s.add_argument("image")
    s.add_argument("--mode", choices=list(MODES)); s.add_argument("--out", help="save an annotated copy here")
    s = add("evaluate", cmd_evaluate, "score the model against hand counts"); s.add_argument("folder"); s.add_argument("truth")
    s.add_argument("--out"); s.add_argument("--mode", choices=list(MODES))
    s = add("mode", cmd_mode, "show or set the mode"); s.add_argument("which", nargs="?", choices=list(MODES))
    s = add("lights", cmd_lights, "switch the ring lights"); s.add_argument("action", choices=["on", "off", "blink"])
    s = add("camera-test", cmd_camera_test, "light up and take one test photo"); s.add_argument("--out", default="~/camera_test.jpg")
    add("status", cmd_status, "check config, USB drive, camera and packages")
    add("warmup", cmd_warmup, "download the model for offline standalone use")
    s = add("wifi", cmd_wifi, "join a Wi-Fi network"); s.add_argument("ssid"); s.add_argument("password")
    s = add("time", cmd_time, "set time zone or clock"); s.add_argument("--timezone"); s.add_argument("--set", help="'YYYY-MM-DD HH:MM:SS'")
    s.add_argument("--ntp", action="store_true")
    add("power-setup", cmd_power_setup, "one-time bootloader setup for RTC wake-up")
    add("set-key", cmd_set_key, "store your Roboflow API key (hidden prompt)")
    s = add("init-config", cmd_init_config, "write the default config file"); s.add_argument("--force", action="store_true")
    s = add("install-service", cmd_install_service, "start bugbox automatically at boot"); s.add_argument("--user")
    return p


def main(argv=None):
    load_env_file()
    args = build_parser().parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
