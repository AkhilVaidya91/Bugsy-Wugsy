# bugbox

A Mothbox-style insect imager that you install as a normal app on your current Raspberry Pi OS.
No firmware, no re-flashing the SD card, any user name.

It drives the same hardware as the Mothbox (relay board + ring lights on GPIO, Arducam 64MP, USB drive,
Waveshare e-ink) and counts insects by species.

## The modes

| Mode | Detection and counting | Cloud use | Photos |
|---|---|---|---|
| **standalone** | On the Pi (Roboflow Inference, free, works offline after `bugbox warmup`) | none | USB drive only |
| **hybrid** | On the Pi | The cloud is asked to **name single insects** the Pi is unsure about | USB drive only |
| cloud (optional) | Every frame analysed by Roboflow | most credits, not recommended on a budget | USB if present |

Switch any time: dashboard buttons, `bugbox mode hybrid`, or a jumper on `power.switch_pin`.

### How hybrid mode keeps cloud use tiny

1. The Pi detects and counts every insect locally, so the **total count never depends on the network**.
2. Only insects the Pi is unsure about (`hybrid.uncertain_below`, default 0.85) are cut out and sent to the cloud.
   Set `classify = "all"` to check every insect, or `"off"` to never use the cloud.
3. **A cache** remembers each answer. If the insect has not moved or changed (box overlap and colour thumbnail),
   it is never sent again, so a live stream of a static sheet costs nothing after the first pass.
4. **Hard limits:** `max_calls_per_minute` and `daily_budget` (remembered across restarts). When they are used up,
   or the network fails (three failures pause the cloud for a minute), the Pi's own labels are used and counting continues.
5. Each crop is padded and enlarged (`crop_pad`, `crop_target_side`), which also helps with small insects on a large sheet.
   The dashboard and annotated photos mark cloud-named insects with `*`.

**Which cloud model?** `[classifier] model_id` takes a Roboflow classification model (`project/version`).
Leave it blank to reuse your detection model on the cropped insect, which needs no extra training. Single-label classifiers,
multi-label classifiers and detection models are all understood. Tell me if you meant a different cloud service.

**Cost.** Per Roboflow's September 18, 2026 announcement, running models locally with Roboflow Inference no longer uses
credits, and the Serverless API is billed per image by model. Their pricing page works out to roughly
`max(processing ms, 100 ms) / 500,000` credits per image, about 0.2 credits per 1,000 images for a small model.
Check your own numbers under Plan and Billing in Roboflow; plans and prices there change.

## Install

```bash
unzip bugbox.zip && cd bugbox
bash install.sh              # add --cloud-only to skip the big local-model package (not for standalone or hybrid)
bugbox set-key               # paste your Roboflow API key (hidden prompt, stored in ~/.config/bugbox/env)
nano ~/.config/bugbox/config.toml    # set  roboflow.model_id = "your-project/1"   and   mode = "hybrid"
bugbox status
bugbox lights blink          # relay board + ring lights
bugbox camera-test           # one lit test photo, ~/camera_test.jpg
bugbox warmup                # download the Pi's model once, while online
bugbox run                   # dashboard at http://localhost:8080
bugbox install-service       # optional: start at boot
```

Try everything without hardware or a Roboflow account: `bugbox --simulate run`.

## What it does (Mothbox functions, and where they live)

| Mothbox function | bugbox |
|---|---|
| Lights switched by relay, active low, GPIO pin(s) | `[lights]` pins, active_low, policy (`capture`, `session`, `off`), settle time |
| Timed full-resolution photos, fixed exposure option | `[camera]`, `[stills] interval_s` |
| Autofocus calibration every 10 min, lens locked between | `focus = "calibrate"`, `calibrate_every_s` |
| Vertical flip | `vflip`, `hflip` |
| Photo folders named after the box, timestamps, EXIF notes | `<usb>/bugbox/<name>/<date>/<name>_<time>.jpg`, counts written into EXIF |
| Back up to USB, copy over later | Writes to USB directly. With `fallback_to_sd = true` it buffers on the SD card and moves files over when a drive appears |
| Free-space guard | `min_free_gb` |
| Session schedule (weekdays, hours, minutes, runtime) | `[schedule]` |
| Sleep between sessions with RTC wake-up | `[power] sleep_between_sessions` + `bugbox power-setup` (same bootloader keys the Mothbox sets) |
| Mode jumper | `[power] switch_pin` (grounded = hybrid) |
| E-ink status screen | `[display]`, now shows species counts, USB, last update |
| Diagnostics log (temperature, throttling, disk, Wi-Fi, voltage) | `logs/diagnostics.jsonl` every `[diagnostics] interval_s` |
| Time zone and clock | `bugbox time --timezone America/New_York`, `--set`, `--ntp` |
| Wi-Fi provisioning | `bugbox wifi BugBytes hackathon` |
| Settings editor | `config.toml` (commented), plus mode, capture and lights controls in the dashboard |

**Not copied:** GPS, battery percentage, Party mode, HDR bracketing, PNG/BMP output, LED blink codes, Pro-board detection.

## New in bugbox

- Per-species counts for `[detect] targets` (default stink bug, chinch bug, wireworm) plus "other insect" and a total.
  Class names are matched loosely (`Stink_Bug`, `stinkbug` and `stink bug` are the same); add `[detect.aliases]` for odd names.
- Class-agnostic de-duplication (`nms_iou`), and optional tiling (`tiles = 2`) with merging for small insects on a large photo.
- **Dashboard** (`http://localhost:8080`): live counts, annotated image, mode switch, "Capture and count now", lights test,
  cloud calls used today. Set `[web] host = "0.0.0.0"` to open it from another computer (the hackathon Wi-Fi may block
  device-to-device traffic; Pi Connect screen sharing on the Pi always works).
- `bugbox evaluate photos/ truth.csv` scores the model against hand counts
  (columns: `image, stink bug, chinch bug, wireworm, total`): exact-match, count accuracy and error per species.
- Annotated copy of every photo, one JSON per photo, and a `counts.csv` per box (with a `cloud_calls` column).

## Storage rules

`standalone` and `hybrid` refuse to write photos to the SD card (`usb_required_modes`). If no drive is found you get a clear
error on the dashboard and e-ink, and nothing is lost silently. The drive is mounted automatically with `udisksctl` if the
desktop has not done it. Full-cloud stream mode saves nothing unless `stream.save_every_n` is set.

## Things to know

- **Speed:** the Pi 5 runs the model on its CPU. Pick a small model and watch `latency_s` on the dashboard.
  The first `bugbox warmup` needs internet. Weights are cached under `~/bugbox_data/models`
  (Roboflow's default `/tmp/cache` is wiped on reboot).
- **Python:** Roboflow's `inference` package supports Python up to 3.11, which Raspberry Pi OS Bookworm ships.
- **Wi-Fi:** unlike the Mothbox firmware, bugbox does not switch Wi-Fi off, so hybrid mode and the dashboard work all the time.
- **Crops from a detection model:** the "reuse the detection model on the crop" default is a pragmatic shortcut. A model trained on whole
  sheets can behave differently on a single close-up insect; if cloud answers look wrong, train a small classifier on cropped insects
  and put it in `classifier.model_id`. Compare using `bugbox evaluate` with `classify = "off"` and with `"all"`.

## What has and has not been tested

Tested (simulation, 36 automated tests): configuration, schedule windows, detection merging and counting, storage rules
(USB only, SD fallback, USB parsing), both capture pipelines, mode switching, the dashboard API, the evaluation scorer, the
hybrid classifier (only unsure insects sent, cache, daily budget, failure fallback and circuit breaker, ignoring low-confidence
answers), and the Roboflow glue code against stand-ins that follow Roboflow's documented API.

**Not tested on real hardware:** the Picamera2 capture and autofocus code, the relay GPIO, the e-ink push, the
`inference` and `inference-sdk` packages themselves, `udisksctl` mounting, and the installer. Expect to fix small things on first run;
`bugbox status`, `bugbox lights blink` and `bugbox camera-test` are there to find them one at a time.
