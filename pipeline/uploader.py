"""Uploader: POST annotated images + metadata to a Flask server on the LAN."""
from __future__ import annotations

import logging
import queue
import threading
from pathlib import Path
from typing import Dict

log = logging.getLogger(__name__)


class Uploader:
    """
    Background worker that POSTs saved frames to an HTTP endpoint.

    Items are queued so the main pipeline loop is never blocked by
    network latency. If the server is unreachable a warning is logged
    and the upload is dropped.
    """

    def __init__(self, server_url: str, timeout_s: float = 5.0):
        """
        Parameters
        ----------
        server_url : str
            Full URL of the Flask receiver, e.g.
            ``http://192.168.1.50:5000/upload``
        timeout_s : float
            Per-request network timeout.
        """
        self._url = server_url.rstrip("/")
        self._timeout = float(timeout_s)
        self._q: queue.Queue = queue.Queue(maxsize=32)
        self._thread = threading.Thread(
            target=self._worker, daemon=True, name="uploader"
        )
        self._thread.start()

    # ------------------------------------------------------------------
    def submit(
        self,
        image_path: Path,
        counts: Dict[str, int],
        timestamp: str,
    ) -> None:
        """Enqueue a frame for upload (non-blocking; drops if queue is full)."""
        try:
            self._q.put_nowait((image_path, counts, timestamp))
        except queue.Full:
            log.warning("upload queue full – frame dropped")

    # ------------------------------------------------------------------
    def _worker(self) -> None:
        import requests

        while True:
            image_path, counts, timestamp = self._q.get()
            try:
                with open(image_path, "rb") as fh:
                    resp = requests.post(
                        self._url,
                        files={"image": (image_path.name, fh, "image/jpeg")},
                        data={
                            "timestamp": timestamp,
                            "count_1": counts.get("1", 0),
                            "count_2": counts.get("2", 0),
                            "count_3": counts.get("3", 0),
                            "count_other": counts.get("other", 0),
                            "total": sum(counts.values()),
                        },
                        timeout=self._timeout,
                    )
                if not resp.ok:
                    log.warning("server returned %s", resp.status_code)
            except Exception as exc:
                log.warning("upload failed: %s", exc)
            finally:
                self._q.task_done()
