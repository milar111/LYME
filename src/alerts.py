import os
import time
import datetime
import threading
import cv2
from src import config

_last_alert_times: dict = {}

def _cooldown_ok(event_type: str) -> bool:
    """Returns True if enough time has passed since the last alert of this type."""
    now = time.time()
    last = _last_alert_times.get(event_type, 0)
    if now - last >= getattr(config, 'ALERT_COOLDOWN_SECONDS', 30):
        _last_alert_times[event_type] = now
        return True
    return False

def save_snapshot(frame, event_type: str) -> str:
    """Save a JPEG snapshot locally and return its file path."""
    snapshot_dir = getattr(config, 'SNAPSHOT_DIR', "data/snapshots")
    os.makedirs(snapshot_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(snapshot_dir, f"{event_type}_{timestamp}.jpg")
    cv2.imwrite(path, frame)
    return path

def log_event(message: str):
    """Append a timestamped line to a local log file."""
    log_file = getattr(config, 'LOG_FILE', "data/logs.txt")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}\n"
    print(line, end="")
    with open(log_file, "a") as f:
        f.write(line)

def fire_alert(frame, event_type: str, description: str, ai_answer: str = ""):
    """
    Main alert function - now strictly local logging and snapshots.
    """
    if not _cooldown_ok(event_type):
        return 

    frame_copy = frame.copy()

    def _local_process():
        log_event(f"{event_type.upper()}: {description}" + (f" | AI: {ai_answer}" if ai_answer else ""))
        # Only save a local picture for record-keeping
        save_snapshot(frame_copy, event_type)

    threading.Thread(target=_local_process, daemon=True).start()