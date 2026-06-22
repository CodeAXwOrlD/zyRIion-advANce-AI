"""
ZYRION Security Monitor
-----------------------
Tracks failed voice verification attempts across clap events.
Enforces lockout after repeated failures to prevent brute-force access.

- 5 failures within 2 minutes -> full lockout for 5 minutes
- Every failure/success logged to security_log.txt with timestamp + score
"""

import os
import time
from datetime import datetime

MAX_FAILURES = 5
FAILURE_WINDOW = 120       # seconds — sliding window for failure tracking
LOCKOUT_DURATION = 300     # seconds — clap detection ignored during lockout

LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "security_log.txt")

# In-memory state (resets on process restart)
_failure_timestamps = []
_lockout_until = 0.0


def _log(entry: str):
    """Append a timestamped line to the security log."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"{timestamp} | {entry}\n"
    try:
        with open(LOG_PATH, "a") as f:
            f.write(line)
    except Exception as e:
        print(f"[SECURITY] log write failed: {e}")


def _prune_old_failures():
    """Drop failures outside the sliding window."""
    cutoff = time.time() - FAILURE_WINDOW
    while _failure_timestamps and _failure_timestamps[0] < cutoff:
        _failure_timestamps.pop(0)


def record_failure(score: float, attempt: int):
    """Record a failed verification attempt. Triggers lockout if threshold exceeded."""
    global _lockout_until

    now = time.time()
    _failure_timestamps.append(now)
    _prune_old_failures()

    _log(f"FAILED  | score={score:.4f} | attempt={attempt}")
    print(f"[SECURITY] failure logged (score={score:.4f}, attempt={attempt})")

    if len(_failure_timestamps) >= MAX_FAILURES:
        _lockout_until = now + LOCKOUT_DURATION
        _failure_timestamps.clear()
        _log(f"LOCKOUT | {MAX_FAILURES} failures in {FAILURE_WINDOW}s — locked for {LOCKOUT_DURATION}s")
        print(f"[SECURITY] LOCKOUT ACTIVE — all access blocked for {LOCKOUT_DURATION // 60} minutes")


def record_success():
    """Record a successful verification. Clears failure streak."""
    _failure_timestamps.clear()
    _log("SUCCESS | access granted")


def is_locked_out() -> bool:
    """True if currently in lockout mode."""
    return time.time() < _lockout_until


def lockout_remaining() -> int:
    """Seconds remaining in lockout (0 if not locked)."""
    return max(0, int(_lockout_until - time.time()))
