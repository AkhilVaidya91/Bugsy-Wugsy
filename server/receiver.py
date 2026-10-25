"""
server/receiver.py – simple Flask server that receives images and metadata
from the Raspberry Pi pipeline and stores them locally.

Usage
-----
pip install flask
python server/receiver.py --out-dir received --port 5000

The Pi POSTs to  http://<your-PC-ip>:5000/upload
"""
from __future__ import annotations

import argparse
import csv
import logging
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, request

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("receiver")

app = Flask(__name__)
_OUT_DIR: Path = Path("received")

_CSV_HEADER = [
    "timestamp", "image_filename",
    "count_1", "count_2", "count_3", "count_other", "total",
]


def _csv_path() -> Path:
    return _OUT_DIR / "counts.csv"


def _ensure_csv_header() -> None:
    p = _csv_path()
    if not p.exists() or p.stat().st_size == 0:
        with open(p, "w", newline="") as f:
            csv.writer(f, lineterminator="\n").writerow(_CSV_HEADER)


# ---------------------------------------------------------------------------
# Upload endpoint
# ---------------------------------------------------------------------------

@app.route("/upload", methods=["POST"])
def upload():
    if "image" not in request.files:
        return jsonify({"error": "no image field"}), 400

    file = request.files["image"]
    timestamp = request.form.get("timestamp", datetime.now().strftime("%Y%m%d_%H%M%S"))

    # Save image
    img_name = file.filename or f"frame_{timestamp}.jpg"
    img_path = _OUT_DIR / img_name
    file.save(str(img_path))

    # Parse counts
    def _int(key: str) -> int:
        try:
            return int(request.form.get(key, 0))
        except ValueError:
            return 0

    counts = {
        "count_1": _int("count_1"),
        "count_2": _int("count_2"),
        "count_3": _int("count_3"),
        "count_other": _int("count_other"),
        "total": _int("total"),
    }

    # Append CSV row
    _ensure_csv_header()
    with open(_csv_path(), "a", newline="") as f:
        csv.writer(f, lineterminator="\n").writerow([
            timestamp,
            img_name,
            counts["count_1"],
            counts["count_2"],
            counts["count_3"],
            counts["count_other"],
            counts["total"],
        ])

    log.info("Received %s  counts=%s", img_name, counts)
    return jsonify({"status": "ok", "saved": img_name}), 200


# ---------------------------------------------------------------------------
# Health-check endpoint (useful to verify Pi → PC connectivity)
# ---------------------------------------------------------------------------

@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"status": "alive"}), 200


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Flask receiver for the Bugsy-Wugsy pipeline"
    )
    p.add_argument("--out-dir", default="received", help="Directory to store received files")
    p.add_argument("--port", type=int, default=5000, help="Port to listen on")
    p.add_argument(
        "--host",
        default="0.0.0.0",
        help="Interface to bind to (0.0.0.0 = all interfaces on the LAN)",
    )
    args = p.parse_args()

    global _OUT_DIR
    _OUT_DIR = Path(args.out_dir)
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    _ensure_csv_header()

    log.info("Receiver starting on %s:%d  →  saving to %s", args.host, args.port, _OUT_DIR)
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
