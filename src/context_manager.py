import json
import os
import re

_CONTEXT_FILE = "data/context.json"
_things_to_watch: list[str] = []


_RESERVED = {"person", "human", "people", "face", "man", "woman"}


def _save(things: list[str]) -> None:
    os.makedirs("data", exist_ok=True)
    with open(_CONTEXT_FILE, "w") as f:
        json.dump(things, f, indent=2)


def _load() -> list[str] | None:
    if not os.path.exists(_CONTEXT_FILE):
        return None
    with open(_CONTEXT_FILE) as f:
        data = json.load(f)
    if isinstance(data, list) and data:
        cleaned = [x for x in data if x.lower() not in _RESERVED]
        return cleaned if cleaned else None
    return None



def init_context() -> None:
    global _things_to_watch
    loaded = _load()
    if loaded:
        _things_to_watch = loaded
        _save(_things_to_watch)
        print(f"[Context] Loaded {len(_things_to_watch)} item(s): {', '.join(_things_to_watch)}")
    else:
        _things_to_watch = []
        print("[Context] No saved context — add items via the web UI.")



def add_item(item: str) -> None:
    global _things_to_watch
    item = item.strip()
    if item.lower() in _RESERVED:
        print(f"[Context] Skipped reserved word: {item} (MediaPipe handles person detection)")
        return
    if item and item not in _things_to_watch:
        _things_to_watch.append(item)
        _save(_things_to_watch)
        print(f"[Context] Added: {item}")



def remove_item(idx: int) -> None:
    global _things_to_watch
    if 0 <= idx < len(_things_to_watch):
        removed = _things_to_watch.pop(idx)
        _save(_things_to_watch)
        print(f"[Context] Removed: {removed}")


def get_all_items() -> list[str]:
    return list(_things_to_watch)



def build_blip_questions() -> list[str]:
    """
    Strict format prompt — Qwen must reply:
    CONFIDENCE: X% | yes or no | one short sentence
    """
    return [
        (
            f"Is there a {thing} in this image? "
            f"Reply with ONLY this format, no extra text: "
            f"CONFIDENCE: <0-100>% | <yes or no> | <one short sentence>"
        )
        for thing in _things_to_watch
    ]


def get_summary() -> str:
    return ", ".join(_things_to_watch) if _things_to_watch else "nothing"


def get_items() -> list[str]:
    return list(_things_to_watch)



def parse_confidence(answer: str) -> int:
    match = re.search(r'CONFIDENCE[:\s]+(\d{1,3})\s*%', answer, re.IGNORECASE)
    if match:
        return min(int(match.group(1)), 100)
    match = re.search(r'(\d{1,3})\s*%', answer)
    if match:
        return min(int(match.group(1)), 100)
    return -1


def parse_detected(answer: str) -> bool:
    match = re.search(r'\|\s*(yes|no)\s*\|', answer, re.IGNORECASE)
    if match:
        return match.group(1).lower() == "yes"
    conf = parse_confidence(answer)
    return conf >= 60 if conf >= 0 else False