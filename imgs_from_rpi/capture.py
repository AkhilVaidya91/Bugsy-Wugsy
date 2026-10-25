import argparse
import subprocess
import time
from datetime import datetime
from pathlib import Path

from libcamera import controls
from picamera2 import Picamera2


def send(path: Path, dest: str) -> None:
    subprocess.run(["scp", "-q", str(path), dest], check=True)


def focus(cam: Picamera2) -> float | None:
    """Run one AF cycle in the current (fast) mode. Return lens position or None."""
    cam.set_controls({"AfMode": controls.AfModeEnum.Auto,
                      "AfRange": controls.AfRangeEnum.Normal})
    ok = cam.autofocus_cycle()  # blocks until the cycle finishes
    lens = cam.capture_metadata().get("LensPosition")
    print(f"AF {'ok' if ok else 'FAILED'}, lens position {lens}")
    return lens if ok else None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="captures", help="local output folder")
    p.add_argument("--count", type=int, default=1, help="number of images")
    p.add_argument("--interval", type=float, default=5.0, help="seconds between images")
    p.add_argument("--dest", help="scp target, e.g. user@192.168.1.50:C:/Users/user/pics/")
    p.add_argument("--delete-after-send", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cam = Picamera2()
    # Small main stream makes libcamera pick a fast binned sensor mode for AF
    preview_cfg = cam.create_preview_configuration(main={"size": (2312, 1736)})
    cam.configure(preview_cfg)
    cam.start()
    time.sleep(2.0)  # let AE/AWB settle

    try:
        for i in range(args.count):
            lens = focus(cam)

            # Lock the focus found above for the full-res capture
            still_controls = {}
            if lens is not None:
                still_controls = {"AfMode": controls.AfModeEnum.Manual,
                                  "LensPosition": lens}
            still_cfg = cam.create_still_configuration(controls=still_controls)

            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            path = out_dir / f"img_{stamp}.jpg"
            # switches to full res, captures, then returns to preview mode
            cam.switch_mode_and_capture_file(still_cfg, str(path))
            print(f"saved {path}")

            if args.dest:
                try:
                    send(path, args.dest)
                    print(f"sent  {path.name}")
                    if args.delete_after_send:
                        path.unlink()
                except subprocess.CalledProcessError:
                    print(f"send failed, keeping {path}")

            if i < args.count - 1:
                time.sleep(args.interval)
    finally:
        cam.stop()
        cam.close()


if __name__ == "__main__":
    main()