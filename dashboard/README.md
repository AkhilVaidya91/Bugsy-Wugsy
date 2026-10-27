# NC Insect Map (live dashboard)

A map of North Carolina counties coloured by insect counts, styled like a rainfall map. It shows trap markers, one dot per detected insect, an optional heat layer, a species toggle and auto-refresh every 2.5 minutes. It uses only the Python standard library and loads Leaflet from a CDN.

## Run

```bash
python3 dashboard/server.py                                      # reads the Pi at http://bugsbunny.local:8080/api/status
python3 dashboard/server.py ~/counts.csv                         # or read a bugbox counts.csv
```

Open http://localhost:8090. During the demo, press **Refresh now** so judges don't have to wait for the next auto-refresh.

## Analytics over time

The panel under the map shows how counts change over time for any trap: a line per species (hover for the exact numbers at each reading), plus the latest total, peak, average, change over the range, the most counted species and the number of readings. Pick the trap and time range (1 h, 6 h, 24 h, all) at the top of the panel. **Table view** lists every reading.

The server saves a reading every 2.5 minutes in the background (`SAMPLE_SECONDS`), even with no browser open. Extra refreshes closer than 60 s apart aren't saved. Readings go to `dashboard/data/history.csv`, which survives restarts and is ignored by git. The live trap is only saved while the Pi is answering, so offline periods are left out instead of being filled with repeated stale numbers. History is served at `GET /api/history?trap=mothbox&hours=24` (`hours=0` returns everything).

## Where the real data comes from

The only real trap is the **MothBox in Wake County** (NCSU Plant Science Building), marked with a red star labelled "LIVE - MothBox". Its counts come from the existing bugbox output; nothing on the Pi is changed. Set the source with `SOURCE` at the top of `server.py`, the `BUGBOX_SOURCE` environment variable, or the first command-line argument:

- **HTTP (default):** the Pi's own dashboard API, `GET /api/status` (fields `counts["stink bug" | "chinch bug" | "wireworm" | "other insect"]`, `total`, `updated`). On the Pi, set `[web] host = "0.0.0.0"` in `~/.config/bugbox/config.toml` so the laptop can reach it.
- **File:** the last row of the box's `counts.csv`, found at `<usb>/bugbox/<box>/counts.csv`. If the hackathon Wi-Fi blocks device-to-device traffic, copy it with `scp bugs@bugsbunny:<that path> ~/counts.csv`.

If the source can't be reached, the page keeps the last reading and shows a red "Pi offline - showing last reading" badge.

## Simulated traps

The 10 other traps are **simulated**: Johnston, Sampson, Duplin, Pitt, Wilson, Nash, Robeson, Union, Wayne and Edgecombe counties. Their counts take a small random walk on every refresh. Each one is labelled "Simulated" in its name, popup and the legend.

County boundaries are saved in `static/nc_counties.geojson`, filtered to NC (state FIPS 37) from plotly's public US counties GeoJSON (US Census data).

## AI tools

This dashboard was built with help from **Claude Code** (Anthropic).
