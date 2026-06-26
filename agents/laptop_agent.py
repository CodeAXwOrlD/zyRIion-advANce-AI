"""
ZYRION Laptop Control Agent
Executes system actions on Linux Mint / GNOME desktop.
Called by voice_capture.py after Commander routes to 'laptop_control'.
"""

import os
import sys
import subprocess
import time
from datetime import datetime

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.logger import log

SCREENSHOT_DIR = os.path.expanduser("~/Pictures/zyrion_screenshots")
os.makedirs(SCREENSHOT_DIR, exist_ok=True)


def _run(cmd, shell=False):
    """Run a shell command silently. Returns (success, output)."""
    try:
        result = subprocess.run(
            cmd if shell else cmd.split(),
            shell=shell,
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.returncode == 0, result.stdout.strip()
    except Exception as e:
        log.error(f"[LAPTOP] command failed: {e}")
        return False, str(e)


def open_app(target: str) -> str:
    """Open any application by name."""
    ok, _ = _run(f"gtk-launch {target}", shell=True)
    if not ok:
        ok, _ = _run(f"xdg-open {target}", shell=True)
    if not ok:
        ok, _ = _run(target, shell=True)
    if ok:
        log.info(f"[LAPTOP] opened app: {target}")
        return f"Opened {target}."
    log.warning(f"[LAPTOP] could not open app: {target}")
    return f"Couldn't open {target}."


def open_url(target: str) -> str:
    """Open a URL in the default browser."""
    if not target.startswith("http"):
        target = "https://" + target
    ok, _ = _run(f"xdg-open {target}", shell=True)
    if ok:
        log.info(f"[LAPTOP] opened URL: {target}")
        return f"Opening {target}."
    return f"Couldn't open that URL."


def open_folder(target: str) -> str:
    """Open a folder in the file manager."""
    path = os.path.expanduser(target)
    ok, _ = _run(f"xdg-open {path}", shell=True)
    if ok:
        log.info(f"[LAPTOP] opened folder: {path}")
        return f"Opened {path}."
    return f"Couldn't open that folder."


def open_file(target: str) -> str:
    """Open any file with its default application."""
    path = os.path.expanduser(target)
    ok, _ = _run(f"xdg-open {path}", shell=True)
    if ok:
        log.info(f"[LAPTOP] opened file: {path}")
        return f"Opened {path}."
    return f"Couldn't open that file."


def take_screenshot(target: str = None) -> str:
    """Take a screenshot and save it."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"screenshot_{timestamp}.png"
    path = os.path.join(SCREENSHOT_DIR, filename)
    time.sleep(0.5)  # brief delay so ZYRION window doesn't interfere
    ok, _ = _run(f"scrot {path}", shell=True)
    if ok:
        log.info(f"[LAPTOP] screenshot saved: {path}")
        return f"Screenshot saved."
    return "Screenshot failed."


def set_volume(target: str) -> str:
    """Set volume to a percentage, or mute/unmute."""
    t = target.lower().strip()
    if t in ("mute", "off", "silent"):
        ok, _ = _run("pactl set-sink-mute @DEFAULT_SINK@ 1", shell=True)
        return "Muted." if ok else "Couldn't mute."
    if t in ("unmute", "on"):
        ok, _ = _run("pactl set-sink-mute @DEFAULT_SINK@ 0", shell=True)
        return "Unmuted." if ok else "Couldn't unmute."
    if t in ("up", "louder"):
        ok, _ = _run("pactl set-sink-volume @DEFAULT_SINK@ +10%", shell=True)
        return "Volume up." if ok else "Couldn't change volume."
    if t in ("down", "lower", "quieter"):
        ok, _ = _run("pactl set-sink-volume @DEFAULT_SINK@ -10%", shell=True)
        return "Volume down." if ok else "Couldn't change volume."
    # numeric percentage
    digits = ''.join(c for c in t if c.isdigit())
    if digits:
        ok, _ = _run(f"pactl set-sink-volume @DEFAULT_SINK@ {digits}%", shell=True)
        return f"Volume set to {digits}%." if ok else "Couldn't set volume."
    return "Didn't understand volume command."


def set_brightness(target: str) -> str:
    """Set brightness to a percentage."""
    t = target.lower().strip()
    if t in ("up", "brighter"):
        ok, _ = _run("brightnessctl set +10%", shell=True)
        return "Brightness up." if ok else "Couldn't change brightness."
    if t in ("down", "dimmer", "lower"):
        ok, _ = _run("brightnessctl set 10%-", shell=True)
        return "Brightness down." if ok else "Couldn't change brightness."
    digits = ''.join(c for c in t if c.isdigit())
    if digits:
        ok, _ = _run(f"brightnessctl set {digits}%", shell=True)
        return f"Brightness set to {digits}%." if ok else "Couldn't set brightness."
    return "Didn't understand brightness command."


def lock_screen(target: str = None) -> str:
    ok, _ = _run("loginctl lock-session", shell=True)
    return "Screen locked." if ok else "Couldn't lock screen."


def sleep_system(target: str = None) -> str:
    ok, _ = _run("systemctl suspend", shell=True)
    return "Going to sleep." if ok else "Couldn't suspend."


def shutdown(target: str = None) -> str:
    ok, _ = _run("shutdown now", shell=True)
    return "Shutting down." if ok else "Couldn't shut down."


def restart(target: str = None) -> str:
    ok, _ = _run("reboot", shell=True)
    return "Restarting." if ok else "Couldn't restart."


def run_command(target: str) -> str:
    """Run any arbitrary shell command."""
    log.info(f"[LAPTOP] running shell command: {target}")
    ok, output = _run(target, shell=True)
    if ok:
        return output[:200] if output else "Done."
    return f"Command failed."


def search_web(target: str) -> str:
    """Open browser with a search query."""
    query = target.replace(" ", "+")
    url = f"https://www.google.com/search?q={query}"
    return open_url(url)


# Action dispatch table
ACTIONS = {
    "open_app":        open_app,
    "open_url":        open_url,
    "open_folder":     open_folder,
    "open_file":       open_file,
    "screenshot":      take_screenshot,
    "set_volume":      set_volume,
    "set_brightness":  set_brightness,
    "lock_screen":     lock_screen,
    "sleep":           sleep_system,
    "shutdown":        shutdown,
    "restart":         restart,
    "run_command":     run_command,
    "search_web":      search_web,
}


def execute(action: str, target: str = "") -> str:
    """Main entry point. Called from voice_capture.py."""
    fn = ACTIONS.get(action)
    if not fn:
        log.warning(f"[LAPTOP] unknown action: {action}")
        return f"I don't know how to {action} yet."
    return fn(target)
