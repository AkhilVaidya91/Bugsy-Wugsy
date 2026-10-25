#!/bin/bash
# bugbox installer: a normal app on your current Raspberry Pi OS. No firmware, no re-flashing.
#
#   bash install.sh                 full install (local model + cloud)
#   bash install.sh --cloud-only    skip the heavy local-model package
#   bash install.sh --no-eink       skip the e-ink library
#
# Works for any user name (it does not need a user called "pi").
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_MODEL=1; EINK=1
for a in "$@"; do
  case "$a" in
    --cloud-only) LOCAL_MODEL=0 ;;
    --no-eink) EINK=0 ;;
    -h|--help) sed -n 2,9p "$0"; exit 0 ;;
  esac
done
say() { printf '\n==> %s\n' "$1"; }

say "1/6  System packages"
sudo apt update
sudo apt install -y python3-venv python3-pip python3-picamera2 python3-opencv python3-numpy python3-pil \
  python3-piexif python3-rpi-lgpio python3-gpiozero python3-spidev udisks2 git fonts-dejavu-core

say "2/6  Python environment (inside $HERE/.venv, sees the system camera packages)"
python3 -m venv --system-site-packages "$HERE/.venv"
PIP="$HERE/.venv/bin/pip"
"$PIP" install --upgrade pip
"$PIP" install inference-sdk                       # cloud mode
if [ "$LOCAL_MODEL" = 1 ]; then                    # standalone mode
  NUMPY_VER="$("$HERE/.venv/bin/python" -c 'import numpy; print(numpy.__version__)')"
  echo "numpy==$NUMPY_VER" > "$HERE/.venv/constraints.txt"   # keep the numpy the camera stack was built with
  if ! "$PIP" install -c "$HERE/.venv/constraints.txt" inference; then
    echo "WARNING: 'inference' did not install with the system numpy pinned. Retrying without the pin."
    "$PIP" install inference || echo "WARNING: 'inference' failed. Cloud mode still works; fix this before using standalone mode."
  fi
fi

say "3/6  Interfaces (SPI for the e-ink display)"
sudo raspi-config nonint do_spi 0 || echo "(could not enable SPI automatically; do it in raspi-config)"

if [ "$EINK" = 1 ]; then
  say "4/6  E-ink library (Waveshare, taken from the Mothbox firmware repo)"
  TMP="$(mktemp -d)"
  git clone --depth 1 --filter=blob:none --sparse https://github.com/Digital-Naturalism-Laboratories/Mothbox_Firmware "$TMP/mf"
  (cd "$TMP/mf" && git sparse-checkout set Mothbox_DIY/scripts/RaspberryPi_JetsonNano_Epaper/lib)
  mkdir -p "$HERE/vendor"
  cp -r "$TMP/mf/Mothbox_DIY/scripts/RaspberryPi_JetsonNano_Epaper/lib/waveshare_epd" "$HERE/vendor/"
  rm -rf "$TMP"
else
  say "4/6  E-ink library skipped"
fi

say "5/6  Config and launcher"
PYTHONPATH="$HERE" "$HERE/.venv/bin/python" -m bugbox init-config
mkdir -p "$HOME/.local/bin" "$HOME/bugbox_data"
cat > "$HOME/.local/bin/bugbox" <<LAUNCH
#!/bin/bash
export PYTHONPATH="$HERE"
exec "$HERE/.venv/bin/python" -m bugbox "\$@"
LAUNCH
chmod +x "$HOME/.local/bin/bugbox"

say "6/6  Check"
"$HOME/.local/bin/bugbox" status || true

cat <<'DONE'

Installed. Next:
  1. bugbox set-key                      # paste your Roboflow API key (hidden)
  2. nano ~/.config/bugbox/config.toml   # set roboflow.model_id = "your-project/1"
  3. bugbox lights blink                 # relay + ring lights (GPIO 26 by default)
  4. bugbox camera-test                  # one lit test photo at ~/camera_test.jpg
  5. bugbox warmup                       # standalone and hybrid: download the Pi's model once while online
  6. bugbox run                          # dashboard at http://localhost:8080
  7. bugbox install-service             # optional: start automatically at boot

If 'bugbox' is not found, open a new terminal or run: export PATH="$HOME/.local/bin:$PATH"
DONE
