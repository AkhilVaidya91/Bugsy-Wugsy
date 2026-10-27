# Bugsy-Wugsy

Bugsy-Wugsy is a real-time insect detection, identification, counting, and
alert system for Raspberry Pi. It is designed for Mothbox-style deployments
using a relay-controlled light ring, a Raspberry Pi camera, optional USB
storage, and an optional Waveshare e-ink display.

The main application is called **bugbox**. It runs as a normal application on
an existing Raspberry Pi OS installation; it does not require custom firmware
or re-flashing the SD card.

## Features

- Detects insects in still images or a live camera stream.
- Counts insects by species, including configured target species and an
  `other insect` category.
- Supports local, hybrid, and cloud processing modes.
- Saves annotated images, per-image JSON records, and CSV count summaries.
- Provides a local web dashboard with live counts and capture controls.
- Controls relay-connected ring lights and optional e-ink status hardware.
- Includes scheduling, diagnostics, USB storage protection, and optional
  automatic startup.
- Supports simulated operation without camera, GPIO, display, or Roboflow
  hardware.

## Processing modes

| Mode | Detection and classification | Network use | Recommended for |
| --- | --- | --- | --- |
| `standalone` | Runs on the Raspberry Pi with Roboflow Inference | None after model warmup | Offline deployments and lowest recurring cost |
| `hybrid` | Detects locally; sends selected insect crops to the cloud for classification | Only uncertain crops by default | Better labels while keeping cloud usage low |
| `cloud` | Sends frames to Roboflow for analysis | Highest usage | Experiments and deployments with reliable internet |

Change modes from the dashboard, with `bugbox mode <mode>`, or with the
configured hardware mode switch. The default mode is `standalone`.

### How hybrid mode limits cloud usage

1. The Pi detects and counts every insect locally, so the total count does not
   depend on network availability.
2. By default, only detections below `hybrid.uncertain_below` are sent for
   cloud classification. Set `hybrid.classify = "all"` to check every insect,
   or `"off"` to disable cloud classification.
3. A cache avoids sending an unchanged insect crop repeatedly.
4. `max_calls_per_minute` and `daily_budget` provide hard usage limits. If a
   limit is reached, the local label is retained and counting continues.
5. Network failures temporarily pause cloud classification; local detection
   continues.

Cloud-classified insects are marked with `*` in the dashboard and annotated
images. Set `[classifier] model_id` to a Roboflow classification model
(`project/version`). If it is blank, the configured detection model is reused
on the cropped insect.

## Hardware and software requirements

### Recommended hardware

- Raspberry Pi 5 running Raspberry Pi OS Bookworm.
- Arducam or compatible Picamera2 camera.
- Relay board and ring lights connected to GPIO.
- USB drive for photo storage in `standalone` and `hybrid` modes.
- Optional Waveshare 2.13-inch e-ink display.
- Internet access for installation, model warmup, and cloud mode.

The application can also run in simulation mode on a regular computer.

### Software prerequisites

- Raspberry Pi OS with `sudo` access.
- Python 3.11 or the version supported by the installed Roboflow Inference
  package.
- Internet access during installation.

## Installation

From the project directory on the Raspberry Pi:

```bash
unzip bugbox.zip
cd bugbox
bash install.sh
```

The installer:

1. Installs Raspberry Pi camera, GPIO, SPI, image, and USB-mounting packages.
2. Creates a virtual environment at `.venv`.
3. Installs the Roboflow cloud package and, unless disabled, the local
   inference package.
4. Installs the Waveshare e-ink library.
5. Creates a default configuration and the `bugbox` launcher.

Installation options:

```bash
bash install.sh --cloud-only  # Skip the large local inference package
bash install.sh --no-eink     # Skip the Waveshare e-ink library
```

`--cloud-only` is not suitable for `standalone` mode and is generally not
appropriate for `hybrid` mode.

If the `bugbox` command is not found after installation, open a new shell or
add the launcher directory to `PATH`:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

## Quick start

Configure the Roboflow key and model:

```bash
bugbox set-key
nano ~/.config/bugbox/config.toml
```

Set at least:

```toml
[roboflow]
model_id = "your-project/1"

mode = "hybrid"
```

Then check the installation and hardware one component at a time:

```bash
bugbox status
bugbox lights blink
bugbox camera-test
bugbox warmup
bugbox run
```

The dashboard is available at <http://localhost:8080>. To open it from
another computer on the same network, set:

```toml
[web]
host = "0.0.0.0"
```

Device-to-device traffic may be blocked by some Wi-Fi networks. Pi Connect
screen sharing is an alternative when the dashboard cannot be reached
remotely.

To run at boot:

```bash
bugbox install-service
```

To try the application without hardware or a Roboflow account:

```bash
bugbox --simulate run
```

## Common commands

| Command | Purpose |
| --- | --- |
| `bugbox status` | Show configuration, hardware, storage, and model status |
| `bugbox run` | Start the dashboard and capture service |
| `bugbox mode standalone` | Select local processing |
| `bugbox mode hybrid` | Select local detection with optional cloud classification |
| `bugbox mode cloud` | Select full cloud processing |
| `bugbox lights blink` | Test the relay and ring lights |
| `bugbox camera-test` | Capture one lit test image at `~/camera_test.jpg` |
| `bugbox warmup` | Download and cache the local model |
| `bugbox evaluate photos/ truth.csv` | Score predictions against hand-counted images |
| `bugbox time --timezone America/New_York` | Configure the time zone |
| `bugbox wifi <ssid> <password>` | Configure Wi-Fi provisioning |
| `bugbox power-setup` | Configure RTC wake-up for scheduled sleeping |

Run `bugbox --help` for the complete command and option list.

## Configuration

The default configuration is created at:

```text
~/.config/bugbox/config.toml
```

The `BUGBOX_CONFIG` environment variable can point to another file. Important
settings include:

| Section | Settings |
| --- | --- |
| `[camera]` | Camera backend, resolution, orientation, focus, exposure, and JPEG quality |
| `[lights]` | GPIO pins, active-low behavior, light policy, and settle time |
| `[storage]` | USB requirement, SD fallback, automatic mounting, free-space guard, and annotated images |
| `[detect]` | Confidence threshold, duplicate-box merging, tiling, targets, and aliases |
| `[roboflow]` | API-key environment variable and detection model |
| `[standalone]` | Local detector/classifier weights and still/stream behavior |
| `[hybrid]` | Cloud classification policy, confidence threshold, budgets, and crop settings |
| `[classifier]` | Cloud classifier endpoint, model, cache, timeout, and minimum confidence |
| `[cloud]` | Full-cloud stream settings and maximum upload size |
| `[stills]` / `[stream]` | Capture interval, frame rate, smoothing, and save frequency |
| `[schedule]` | Weekdays, hours, minutes, and session duration |
| `[power]` | RTC sleep behavior and optional physical mode switch |
| `[display]` / `[diagnostics]` | E-ink refresh and diagnostics intervals |
| `[web]` | Dashboard host, port, and enabled state |

The checked-in defaults in `bugbox/config.py` are also a complete reference
for supported settings and their comments.

### Species and aliases

Target species are configured with `[detect] targets`:

```toml
[detect]
targets = ["stink bug", "chinch bug", "wireworm"]
other_label = "other insect"

[detect.aliases]
stinkbug = "stink bug"
```

Class names are normalized, so names such as `Stink_Bug`, `stinkbug`, and
`stink bug` can be treated as the same target.

## Storage and output

By default, `standalone` and `hybrid` modes refuse to save photos to the SD
card. This protects the Pi from filling its boot disk. Connect a USB drive
before starting capture.

Storage behavior can be changed with:

```toml
[storage]
usb_required_modes = ["standalone", "hybrid"]
fallback_to_sd = false
auto_mount = true
min_free_gb = 1.0
keep_annotated = true
```

Photos are stored under a structure similar to:

```text
<usb>/bugbox/<box-name>/<date>/<box-name>_<time>.jpg
```

Depending on the selected mode and configuration, the application also
produces:

- Annotated copies with detection boxes and labels.
- One JSON record per captured image.
- `counts.csv` with per-species counts and cloud-call information.
- `logs/diagnostics.jsonl` with temperature, throttling, disk, Wi-Fi, and
  voltage information.
- Cached local model files under `~/bugbox_data/models`.

Full-cloud stream mode does not save images unless
`stream.save_every_n` is greater than zero.

## Evaluation

Use the evaluation command to compare predictions with hand-counted images:

```bash
bugbox evaluate photos/ truth.csv
```

The truth CSV uses these columns:

```text
image, stink bug, chinch bug, wireworm, total
```

The report includes exact matches, count accuracy, and per-species error.
For comparing local and cloud classification, run the evaluation once with
`classify = "off"` and again with `classify = "all"`.

## Standalone pipeline

The repository also contains a separate continuous pipeline in
[pipeline/](C:/Users/Lenovo/Documents/projects/hackathon/PSI/Bugsy-Wugsy/Bugsy-Wugsy/pipeline/).
It is a lower-level workflow that captures frames with Picamera2, runs an
RF-DETR detector and ONNX classifier, saves annotated frames, and can upload
results to a PC receiver.

See [pipeline/README.md](C:/Users/Lenovo/Documents/projects/hackathon/PSI/Bugsy-Wugsy/Bugsy-Wugsy/pipeline/README.md)
for its dependencies, command-line flags, receiver setup, connectivity
checks, and CSV schema. It is separate from the `bugbox` installer and
configuration described above.

## Troubleshooting

Run these checks in order:

```bash
bugbox status
bugbox lights blink
bugbox camera-test
bugbox warmup
```

Common causes of startup problems:

- The camera is not detected by Picamera2 or the selected backend.
- The relay GPIO pin or active-low setting is incorrect.
- The USB drive is not mounted or has less than `storage.min_free_gb`
  available.
- The Roboflow API key or `roboflow.model_id` is missing.
- The local model has not been downloaded with `bugbox warmup`.
- The selected Python version is unsupported by the installed Roboflow
  package.
- SPI has not been enabled for the e-ink display.

The installer attempts to enable SPI automatically. If that fails, enable it
manually with `sudo raspi-config`, then reboot if requested.

## Development and testing

The test suite can run without Raspberry Pi hardware by using simulation and
stand-in model interfaces:

```bash
python -m pytest tests
```

The test suite covers configuration, scheduling, detection merging and
counting, storage rules, capture pipelines, mode switching, dashboard APIs,
evaluation, hybrid classification, caching, budgets, failure fallback, and
Roboflow integration glue.

Real hardware paths are not covered by the automated tests, including
Picamera2 autofocus, relay GPIO, e-ink output, `udisksctl` mounting, the
installer, and the third-party `inference` packages. Test these separately
with `status`, `lights blink`, and `camera-test` before deploying.

## Project layout

readme.md             Main documentation
bugbox/                 Main application and command-line interface
pipeline/               Separate continuous Pi-to-PC pipeline
server/                 Flask receiver for the separate pipeline
tests/                  Automated tests
models/                 Local detector and classifier weights
classification_training/Training utilities for crop classification
annotated_detection/    Detection training utilities
scale_aware/            Scale-aware model training utilities
```

## License

See [LICENSE](C:/Users/Lenovo/Documents/projects/hackathon/PSI/Bugsy-Wugsy/Bugsy-Wugsy/LICENSE).
