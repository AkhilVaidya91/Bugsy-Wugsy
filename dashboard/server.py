"""NC insect map dashboard: serves one Leaflet page plus /api/counts (standard library only).

Live counts come from the MothBox's existing bugbox output; nothing on the Pi is changed.
Built with help from Claude Code (Anthropic), cited per hackathon rules.

Run:  python3 dashboard/server.py            then open http://localhost:8090
"""
from __future__ import annotations

import csv
import json
import os
import random
import sys
import threading
import time
import urllib.parse
import urllib.request
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# ---- the one setting: where the real counts come from ------------------------------------------
# http(s) URL -> the Pi's bugbox dashboard API (needs [web] host = "0.0.0.0" in the Pi's config.toml)
# file path   -> a bugbox counts.csv (e.g. copied with: scp bugs@bugsbunny:<usb>/bugbox/<box>/counts.csv .)
SOURCE = os.environ.get("BUGBOX_SOURCE", "http://bugsbunny.local:8080/api/status")
PORT = int(os.environ.get("DASH_PORT", "8090"))

SPECIES = ["stink_bug", "chinch_bug", "wireworm", "other_insect"]
STATIC = Path(__file__).parent / "static"

# ---- analytics over time: every reading is appended here and survives restarts --------------------
HISTORY_FILE = Path(__file__).parent / "data" / "history.csv"
SAMPLE_SECONDS = 150        # background sampling, so history grows even with no browser open
MIN_RECORD_SECONDS = 60     # extra page loads / "Refresh now" clicks closer than this are not recorded
HISTORY_COLS = ["time", "trap_id"] + SPECIES + ["total"]

LIVE_TRAP = {"id": "mothbox", "name": "LIVE - MothBox (NCSU Plant Science Bldg)", "county": "Wake",
             "fips": "37183", "lat": 35.786, "lon": -78.680, "simulated": False}

# (name, county, fips, lat, lon) for the simulated farm traps
SIM_SITES = [
    ("Smithfield farm", "Johnston", "37101", 35.508, -78.339),
    ("Clinton farm", "Sampson", "37163", 34.998, -78.323),
    ("Kenansville farm", "Duplin", "37061", 34.962, -77.962),
    ("Greenville farm", "Pitt", "37147", 35.613, -77.366),
    ("Wilson farm", "Wilson", "37195", 35.721, -77.916),
    ("Nashville farm", "Nash", "37127", 35.974, -77.966),
    ("Lumberton farm", "Robeson", "37155", 34.618, -79.009),
    ("Monroe farm", "Union", "37179", 34.985, -80.549),
    ("Goldsboro farm", "Wayne", "37191", 35.385, -77.993),
    ("Tarboro farm", "Edgecombe", "37065", 35.897, -77.536),
]

_lock = threading.Lock()
_last_live = {"counts": {s: 0 for s in SPECIES}, "updated": None, "ok": False, "error": "no reading yet"}
_sim_rng = random.Random(37)
_sim_state = {site[2]: {s: _sim_rng.randint(0, 12) for s in SPECIES} for site in SIM_SITES}
_history: list = []         # rows of HISTORY_COLS, oldest first
_last_record = 0.0


def load_history():
    if not HISTORY_FILE.exists():
        return
    with open(HISTORY_FILE, newline="") as f:
        for r in csv.DictReader(f):
            _history.append({"time": r["time"], "trap_id": r["trap_id"],
                             **{k: int(r[k]) for k in SPECIES + ["total"]}})


def record(snap: dict):
    """Append one row per trap; the live trap only when the Pi actually answered."""
    global _last_record
    if time.time() - _last_record < MIN_RECORD_SECONDS:
        return
    _last_record = time.time()
    rows = [{"time": snap["generated"], "trap_id": t["id"], **t["counts"], "total": t["total"]}
            for t in snap["traps"] if t["simulated"] or snap["live_ok"]]
    _history.extend(rows)
    HISTORY_FILE.parent.mkdir(exist_ok=True)
    new = not HISTORY_FILE.exists()
    with open(HISTORY_FILE, "a", newline="") as f:
        w = csv.DictWriter(f, HISTORY_COLS, lineterminator="\n")
        if new:
            w.writeheader()
        w.writerows(rows)


def history(trap_id: str, hours: float) -> dict:
    cutoff = datetime.fromtimestamp(time.time() - hours * 3600).strftime("%Y-%m-%d %H:%M:%S") if hours else ""
    with _lock:
        rows = [r for r in _history if r["trap_id"] == trap_id and r["time"] >= cutoff]
    return {"trap_id": trap_id, "hours": hours, "rows": rows}


def sampler():
    while True:
        time.sleep(SAMPLE_SECONDS)
        try:
            snapshot()
        except Exception as e:
            print("sampler:", e)


def _key(name: str) -> str:
    """bugbox labels ("stink bug", "other insect") -> our keys (stink_bug, other_insect)."""
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def read_live() -> dict:
    """Return {counts, updated} from SOURCE; raises on failure."""
    if SOURCE.startswith(("http://", "https://")):
        with urllib.request.urlopen(SOURCE, timeout=4) as r:
            st = json.load(r)
        counts = {_key(k): int(v or 0) for k, v in (st.get("counts") or {}).items()}
        updated = st.get("updated") or ""
        if updated and len(updated) <= 8:  # bugbox gives HH:MM:SS
            updated = datetime.now().strftime("%Y-%m-%d ") + updated
    else:
        with open(os.path.expanduser(SOURCE), newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            raise ValueError("counts.csv is empty")
        last = rows[-1]
        counts = {_key(k): int(float(v or 0)) for k, v in last.items() if _key(k) in SPECIES}
        updated = last.get("time", "")
    return {"counts": {s: counts.get(s, 0) for s in SPECIES}, "updated": updated}


def step_sim():
    """Random walk so simulated traps drift a little on every refresh."""
    for c in _sim_state.values():
        for s in SPECIES:
            c[s] = max(0, min(40, c[s] + _sim_rng.choice([-2, -1, 0, 0, 1, 1, 2, 3])))


def snapshot() -> dict:
    with _lock:
        try:
            live = read_live()
            _last_live.update(live, ok=True, error="")
        except Exception as e:
            _last_live.update(ok=False, error=f"{type(e).__name__}: {e}")
        step_sim()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        traps = [dict(LIVE_TRAP, counts=dict(_last_live["counts"]),
                      total=sum(_last_live["counts"].values()), updated=_last_live["updated"] or "never")]
        for name, county, fips, lat, lon in SIM_SITES:
            c = dict(_sim_state[fips])
            traps.append({"id": fips, "name": f"{name} (Simulated)", "county": county, "fips": fips,
                          "lat": lat, "lon": lon, "simulated": True, "counts": c, "total": sum(c.values()),
                          "updated": now})
        snap = {"generated": now, "source": SOURCE, "live_ok": _last_live["ok"],
                "live_error": _last_live["error"], "species": SPECIES, "traps": traps}
        record(snap)
        return snap


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/api/counts":
            self._send(200, json.dumps(snapshot()).encode(), "application/json")
        elif path == "/api/history":
            q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            body = history(q.get("trap", [LIVE_TRAP["id"]])[0], float(q.get("hours", ["0"])[0] or 0))
            self._send(200, json.dumps(body).encode(), "application/json")
        elif path in ("/", "/index.html"):
            self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif path == "/nc_counties.geojson":
            self._send(200, (STATIC / "nc_counties.geojson").read_bytes(), "application/geo+json")
        else:
            self._send(404, b'{"error":"not found"}', "application/json")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        SOURCE = sys.argv[1]
    load_history()
    threading.Thread(target=sampler, daemon=True, name="sampler").start()
    print(f"NC insect map on http://localhost:{PORT}  (live source: {SOURCE}, history: {HISTORY_FILE})")
    ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()
