# Bugsy-Wugsy Local Pipeline

A self-contained, continuously-running insect detection + classification pipeline
for Raspberry Pi 5 with Picamera2.

## Folder layout

```
pipeline/
├── __init__.py
├── camera.py       # Picamera2 continuous streaming
├── classifier.py   # ONNX EfficientNetV2-M crop classifier
├── detector.py     # RF-DETR Small .pt detection
├── display.py      # Waveshare 2.13 in e-ink status panel
├── main.py         # Pipeline entry point (argparse CLI)
├── storage.py      # Annotated JPEG + CSV saving (on Pi)
└── uploader.py     # Background HTTP POST to the PC Flask server

server/
└── receiver.py     # Flask server that runs on your PC
```

The bundled `models/scale_effic.onnx` is the scale-aware model exported by
`scale_aware/Bugs Detection.yolo26/train.py`. Its image input is RGB float32
in `[0, 1]`; ImageNet normalization is inside the model. Its two scale inputs
are `log1p(original_bbox_area_pixels) / 20` and
`original_bbox_area_pixels / original_frame_area`. The runtime must use the
original detector box dimensions, not the padded crop dimensions.

## Pi dependencies

```bash
# System
sudo apt install python3-picamera2 libcamera-apps

# Python
pip install rfdetr onnxruntime opencv-python-headless pillow requests
```

## PC dependencies (for the receiver)

```bash
pip install flask
```

---

## Running on the Pi

```bash
# from the repo root
python -m pipeline.main --detector models/weights.pt --classifier models/scale_effic.onnx --out-dir captures --server http://192.168.1.138:5000/upload --det-conf 0.4 --save-every 10
```

### Key flags

| Flag | Default | Meaning |
|------|---------|---------|
| `--detector` | `models/weights.pt` | RF-DETR Small checkpoint |
| `--classifier` | `models/scale_effic.onnx` | ONNX classifier |
| `--out-dir` | `captures` | Local storage directory |
| `--server` | _(empty)_ | Flask server URL (omit to disable upload) |
| `--det-conf` | `0.4` | Minimum detection confidence |
| `--save-every` | `10` | Save 1 annotated frame every N frames |
| `--smooth` | `5` | Frame-count smoothing window |
| `--display-interval` | `10.0` | Min seconds between e-ink refreshes |
| `--no-af` | off | Disable continuous autofocus |

---

## Running the PC receiver

```bash
# find your PC's LAN IP first
ipconfig          # Windows
ip a              # Linux / macOS

python server/receiver.py --out-dir received --port 5000
```

Received images and a `counts.csv` are stored in `received/`.

---

## Connectivity checklist (Pi → PC)

Both devices must be on the **same Wi-Fi or wired LAN**.

1. **Find your PC's IP** – run `ipconfig` (Windows) or `ip a` (Linux).
   Look for an address like `192.168.1.x` on the interface connected
   to the same router as the Pi.

2. **Open the firewall port on the PC**

   *Windows* – allow inbound TCP 5000 in Windows Defender Firewall, or run:
   ```powershell
   New-NetFirewallRule -DisplayName "Bugsy receiver" `
       -Direction Inbound -Protocol TCP -LocalPort 5000 -Action Allow
   ```

   *Linux* – `sudo ufw allow 5000/tcp`

3. **Test from the Pi** before starting the full pipeline:
   ```bash
   curl http://<PC-IP>:5000/ping
   # Expected: {"status": "alive"}
   ```

4. **Pass the correct URL** to `--server`:
   ```
   --server http://192.168.1.138:5000/upload
   ```

If the upload fails the pipeline keeps running – the frame is just logged
as a warning and dropped from the upload queue.

---

## Output files (on the Pi)

```
captures/
├── counts.csv              # one row per saved frame
├── frame_20241025_123456_789.jpg
└── ...
```

### CSV schema

| column | description |
|--------|-------------|
| `timestamp` | `YYYYMMDD_HHMMSS_mmm` |
| `image_filename` | basename of the JPEG |
| `count_1` | insects classified as class 1 |
| `count_2` | class 2 |
| `count_3` | class 3 |
| `count_other` | class "other" |
| `total` | sum of all counts |

The PC receiver creates an identical `received/counts.csv`.
