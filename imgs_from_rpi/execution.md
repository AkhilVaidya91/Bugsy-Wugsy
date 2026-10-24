Capture stills from an ArduCam on a Pi 5, then keep them on the Pi or get them to your PC. The unknowns are the camera type and the transfer direction.

## Usage

```bash
# one image, kept on the Pi
python3 capture.py

# 10 images, 5 s apart, sent to the PC
python3 capture.py --count 10 --interval 5 --dest user@192.168.1.50:/home/user/pics/
```

## One-time setup for sending to PC

On the Pi:

```bash
ssh-keygen -t ed25519          # press Enter through the prompts
ssh-copy-id user@192.168.1.50  # on Windows, append ~/.ssh/id_ed25519.pub to the PC user's authorized_keys manually
```

Then confirm `ssh user@192.168.1.50` works without a password.

## Pull option from the PC (no PC SSH server needed)

```bash
scp "pi@raspberrypi.local:~/captures/*.jpg" ./pics/
```