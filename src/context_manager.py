"""
context_manager.py
──────────────────
Handles the AI monitoring context — the list of things BLIP should watch for.

Persistence:
  • Saved to data/context.json on first run.
  • Loaded silently on every subsequent run — never asks again.
  • To reconfigure: delete data/context.json and restart.

Usage:
  from src.context_manager import init_context, build_blip_questions, get_summary

  init_context()                    # call once at startup
  questions = build_blip_questions() # list of BLIP yes/no questions to cycle through
  summary   = get_summary()          # short string for the HUD / notifications
"""

import json
import os

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
    print("    a person carrying a bag or backpack")
    print("    someone touching the equipment")
    print("    a person wearing a high-visibility vest")
    print("    any person present")
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
    """Load from disk or ask the user. Call once at startup."""
    global _things_to_watch

    loaded = _load()
    if loaded:
        _things_to_watch = loaded
        print(f"[Context] Loaded {len(_things_to_watch)} monitored item(s) "
              f"from {_CONTEXT_FILE}:")
        for i, t in enumerate(_things_to_watch, 1):
            print(f"  {i}. {t}")
        return

    print("[Context] No saved context — asking once (won't ask again after this).")
    _things_to_watch = _ask_user()
    _save(_things_to_watch)
    print(f"[Context] Monitoring {len(_things_to_watch)} item(s).")


def build_blip_questions() -> list[str]:
    """
    Returns one BLIP yes/no question per monitored item.
    BLIP works best with short, direct, factual questions.
    The caller cycles through these questions across successive 3-second ticks.
    """
    return [
        f"Is there {thing} visible in the image? Answer yes or no."
        for thing in _things_to_watch
    ]


def get_summary() -> str:
    """Comma-joined list of monitored items — for HUD display."""
    return ", ".join(_things_to_watch) if _things_to_watch else "nothing"


def get_items() -> list[str]:
    return list(_things_to_watch)