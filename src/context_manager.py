"""
context_manager.py
──────────────────
Handles the AI monitoring context — the list of things Qwen should watch for.

Persistence:
  • Saved to data/context.json on first run.
  • Loaded silently on every subsequent run — never asks again.
  • To reconfigure: delete data/context.json and restart.
"""

import json
import os
import re

_CONTEXT_FILE = "data/context.json"
_things_to_watch: list[str] = []


# ── Persistence ───────────────────────────────────────────────────────────────

def _save(things: list[str]) -> None:
    os.makedirs("data", exist_ok=True)
    with open(_CONTEXT_FILE, "w") as f:
        json.dump(things, f, indent=2)
    print(f"[Context] Saved to {_CONTEXT_FILE}  (delete to reconfigure).")


def _load() -> list[str] | None:
    if not os.path.exists(_CONTEXT_FILE):
        return None
    with open(_CONTEXT_FILE) as f:
        data = json.load(f)
    if isinstance(data, list) and data:
        return data
    return None


# ── Setup prompt ──────────────────────────────────────────────────────────────

def _ask_user() -> list[str]:
    print()
    print("─" * 62)
    print("  WHAT SHOULD THE AI MONITOR FOR?")
    print("  Enter one item per line. Empty line when done.")
    print()
    print("  Examples:")
    print("    bottle")
    print("    cap")
    print("    person carrying a bag")
    print("    weapon")
    print("─" * 62)

    things = []
    idx = 1
    while True:
        raw = input(f"  Item {idx}: ").strip()
        if not raw:
            if not things:
                print("  (No items entered — defaulting to 'a person'.)")
                things = ["a person"]
            break
        things.append(raw)
        idx += 1

    return things


# ── Public API ────────────────────────────────────────────────────────────────

def init_context() -> None:
    global _things_to_watch
    loaded = _load()
    if loaded:
        _things_to_watch = loaded
        print(f"[Context] Loaded {len(_things_to_watch)} monitored item(s):")
        for i, t in enumerate(_things_to_watch, 1):
            print(f"  {i}. {t}")
        return
    print("[Context] No saved context — asking once.")
    _things_to_watch = _ask_user()
    _save(_things_to_watch)


def build_blip_questions() -> list[str]:
    """
    Strict, minimal prompt — tells Qwen exactly what format to use.
    Keeping it simple is key for reliable parsing on a small model.
    """
    return [
        (
            f"Is there a {thing} in this image? "
            f"Reply with ONLY this format, no extra text: "
            f"CONFIDENCE: <number 0-100>% | <yes or no> | <one short sentence>"
        )
        for thing in _things_to_watch
    ]


def get_summary() -> str:
    return ", ".join(_things_to_watch) if _things_to_watch else "nothing"


def get_items() -> list[str]:
    return list(_things_to_watch)


# ── Answer parsing ────────────────────────────────────────────────────────────

def parse_confidence(answer: str) -> int:
    """
    Extract confidence % from answer.
    Expected format: "CONFIDENCE: 87% | yes | A bottle is on the table."
    Returns 0-100, or -1 if parsing fails.
    """
    match = re.search(r'CONFIDENCE[:\s]+(\d{1,3})\s*%', answer, re.IGNORECASE)
    if match:
        return min(int(match.group(1)), 100)
    # Fallback: any number followed by %
    match = re.search(r'(\d{1,3})\s*%', answer)
    if match:
        return min(int(match.group(1)), 100)
    return -1


def parse_detected(answer: str) -> bool:
    """
    Returns True if Qwen answered 'yes' in the structured response.
    Used to decide whether to fire a push notification.
    """
    # Look for | yes | pattern
    match = re.search(r'\|\s*(yes|no)\s*\|', answer, re.IGNORECASE)
    if match:
        return match.group(1).lower() == "yes"
    # Fallback: confidence >= 60 counts as detected
    conf = parse_confidence(answer)
    if conf >= 0:
        return conf >= 60
    return False