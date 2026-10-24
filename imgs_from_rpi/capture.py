import argparse
import subprocess
import time
from datetime import datetime
from pathlib import Path

from picamera2 import Picamera2


def send(path: Path, dest: str) -> None:
    subprocess.run(["scp", "-q", str(path), dest], check=True)


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
    cam.configure(cam.create_still_configuration())
    cam.start()
    time.sleep(5.0)  # let auto exposure / white balance settle

    try:
        for i in range(args.count):
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            path = out_dir / f"img_{stamp}.jpg"
            cam.capture_file(str(path))
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