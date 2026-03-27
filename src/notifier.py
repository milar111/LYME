"""
notifier.py
───────────
Sends push notifications via ntfy.sh — completely free, no account required.

Setup (one-time, ~2 minutes):
  1. Install the ntfy app on your phone (iOS / Android — search "ntfy").
  2. Open the app → tap "+" → Subscribe to topic.
  3. Enter the topic string you set in config.NTFY_TOPIC  (e.g. "lyme-cam-a8f3x")
     — keep it hard to guess, it acts as your private channel.
  4. That's it.  Notifications from this script will appear instantly.

No login, no API key, no server to run.
"""

import urllib.request
import urllib.error
import json
import threading
from src import config


def _do_send(title: str, message: str, priority: str, snapshot_path: str | None):
    """Blocking send — always called from a background thread."""
    topic = getattr(config, 'NTFY_TOPIC', '')
    if not topic:
        print("[Notifier] NTFY_TOPIC not set in config — skipping notification.")
        return

    url = f"https://ntfy.sh/{topic}"
    headers = {
        "Title": title.encode(),
        "Priority": priority.encode(),
        "Tags": b"warning,camera",
        "Content-Type": b"text/plain",
    }

    body = message.encode()
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")

    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            if resp.status == 200:
                print(f"[Notifier] Push sent: {title}")
            else:
                print(f"[Notifier] Unexpected status {resp.status}")
    except urllib.error.URLError as e:
        print(f"[Notifier] Failed to send push notification: {e}")


def send(title: str, message: str,
         priority: str = "high",
         snapshot_path: str | None = None):
    """
    Fire-and-forget push notification.
    priority: "min" | "low" | "default" | "high" | "urgent"
    """
    threading.Thread(
        target=_do_send,
        args=(title, message, priority, snapshot_path),
        daemon=True
    ).start()


# ── Convenience wrappers ──────────────────────────────────────────────────────

def intrusion_alert(ai_description: str = ""):
    msg = "Person detected in forbidden zone."
    if ai_description:
        msg += f"\n\nAI: {ai_description}"
    send("🚨 INTRUSION DETECTED", msg, priority="urgent")


def intrusion_cleared():
    send("✅ Zone cleared", "The forbidden zone is now empty.", priority="default")