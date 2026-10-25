"""System helpers: Wi-Fi, time zone, clock, and the RTC wake alarm (what the Mothbox's Scheduler does)."""
from __future__ import annotations

import re
import subprocess
from datetime import datetime

from .util import log


def run(cmd, timeout=60, check=False):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=check)


# ---------------- Wi-Fi ----------------
def wifi_connect(ssid: str, password: str) -> tuple[bool, str]:
    r = run(["sudo", "-n", "nmcli", "dev", "wifi", "connect", ssid, "password", password], timeout=90)
    return r.returncode == 0, (r.stdout + r.stderr).strip()


def wifi_current() -> str:
    try:
        r = run(["nmcli", "-t", "-f", "ACTIVE,SSID", "dev", "wifi"], timeout=10)
        for line in r.stdout.splitlines():
            if line.startswith("yes:"):
                return line.split(":", 1)[1]
    except Exception:
        pass
    return ""


# ---------------- time ----------------
def set_timezone(tz: str) -> tuple[bool, str]:
    r = run(["sudo", "-n", "timedatectl", "set-timezone", tz])
    return r.returncode == 0, (r.stdout + r.stderr).strip()


def set_clock(when: str) -> tuple[bool, str]:
    """when = 'YYYY-MM-DD HH:MM:SS' (turns off network time, sets the clock, writes the RTC)."""
    dt = datetime.strptime(when, "%Y-%m-%d %H:%M:%S")
    run(["sudo", "-n", "timedatectl", "set-ntp", "false"])
    r = run(["sudo", "-n", "date", dt.strftime("%m%d%H%M%Y.%S")])
    if r.returncode == 0:
        run(["sudo", "-n", "hwclock", "-w"])
    return r.returncode == 0, (r.stdout + r.stderr).strip()


def ntp_on():
    run(["sudo", "-n", "timedatectl", "set-ntp", "true"])


# ---------------- RTC wake / shutdown ----------------
def eeprom_setup() -> tuple[bool, str]:
    """Same keys the Mothbox Scheduler sets so the Pi can wake itself from halt."""
    want = {"POWER_OFF_ON_HALT": "1", "WAKE_ON_GPIO": "0"}
    r = run(["sudo", "-n", "rpi-eeprom-config"])
    if r.returncode != 0:
        return False, r.stderr.strip()
    cur = {}
    for line in r.stdout.splitlines():
        m = re.match(r"(\w+)=(\d+)", line)
        if m:
            cur[m.group(1)] = m.group(2)
    if all(cur.get(k) == v for k, v in want.items()):
        return True, "EEPROM already set"
    cur.update(want)
    open("/tmp/bugbox_eeprom.txt", "w").write("\n".join(f"{k}={v}" for k, v in cur.items()) + "\n")
    r = run(["sudo", "-n", "rpi-eeprom-config", "--apply", "/tmp/bugbox_eeprom.txt"], timeout=120)
    return r.returncode == 0, "EEPROM updated, reboot to apply" if r.returncode == 0 else r.stderr.strip()


def set_wake_alarm(epoch: int):
    run(["sudo", "-n", "sh", "-c", "echo 0 > /sys/class/rtc/rtc0/wakealarm"])
    r = run(["sudo", "-n", "sh", "-c", f"echo {int(epoch)} > /sys/class/rtc/rtc0/wakealarm"])
    log.info("power: wake alarm set for %s (rc=%s)", datetime.fromtimestamp(epoch), r.returncode)
    return r.returncode == 0


def shutdown():
    log.info("power: shutting down")
    subprocess.Popen(["sudo", "-n", "shutdown", "-h", "now"])


# ---------------- mode jumper ----------------
def read_mode_switch(pin: int):
    """Returns 'cloud' if the pin is grounded, 'standalone' if open, None if not readable."""
    if not pin:
        return None
    try:
        import RPi.GPIO as GPIO
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        low = GPIO.input(pin) == GPIO.LOW
        return "cloud" if low else "standalone"
    except Exception:
        return None
