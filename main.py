"""
main.py  —  LYME Security Camera
─────────────────────────────────
Runs the full detection pipeline + Flask web UI.

Flask routes:
  GET  /               → index.html
  GET  /video_feed     → MJPEG stream
  GET  /status         → person / intrusion / frame count
  GET  /ai_status      → Qwen answers + confidence per item
  GET  /zone           → current zone polygon (normalised)
  POST /zone           → save new zone polygon
  DELETE /zone         → clear zone
  GET  /context        → current monitored items
  POST /context        → add item
  DELETE /context/<i>  → remove item by index
  POST /shutdown       → stop the loop
"""

import warnings
import logging
import time
import threading
import cv2
import numpy as np
import json
import os
from flask import Flask, Response, jsonify, render_template, request
from src import actions, config
from src.tracker import VisionTracker
from src.zone_manager import (init_zone, draw_zones, body_centre_in_zone,
                               get_zone_pts, load_zone_from_points, clear_zone)
from src.intrusion_guard import IntrusionGuard, GuardState
from src.context_manager import (init_context, build_blip_questions,
                                  get_summary, get_items, parse_confidence,
                                  parse_detected, add_item, remove_item,
                                  get_all_items)
from src import notifier
from src.alerts import fire_alert, log_event

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)


class WebUIState:
    def __init__(self):
        self.latest_frame: bytes | None = None
        self.person_in_sight = False
        self.is_intruding = False
        self.guard_state = GuardState.CLEAR
        self.dwell_progress = 0.0
        self.last_seen: float | None = None
        self.frame_count = 0
        self.lock = threading.Lock()

    def update(self, frame, person_detected: bool, is_intruding: bool,
               guard_state: GuardState, dwell_progress: float):
        with self.lock:
            self.person_in_sight = person_detected
            self.is_intruding = is_intruding
            self.guard_state = guard_state
            self.dwell_progress = dwell_progress
            if person_detected:
                self.last_seen = time.time()
            self.frame_count += 1
            ret, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if ret:
                self.latest_frame = jpeg.tobytes()

ui_state = WebUIState()
_running = True


app = Flask(__name__, template_folder="demo_app/templates")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    def generate():
        while True:
            with ui_state.lock:
                frame = ui_state.latest_frame
            if frame:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            time.sleep(0.03)
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/status")
def status():
    with ui_state.lock:
        last_seen_ago = round(time.time() - ui_state.last_seen, 1) if ui_state.last_seen else None
        return jsonify(
            person_in_sight=ui_state.person_in_sight,
            is_intruding=ui_state.is_intruding,
            guard_state=ui_state.guard_state.name,
            dwell_progress=ui_state.dwell_progress,
            last_seen_ago=last_seen_ago,
            frame_count=ui_state.frame_count,
            object_alert=time.monotonic() < _object_alert_until,
        )

@app.route("/ai_status")
def ai_status():
    with _ai_lock:
        items = get_all_items()
        answers = list(_answers)
    result = []
    for item, ans in zip(items, answers):
        conf = parse_confidence(ans)
        detected = parse_detected(ans)
        desc = ans.split("|")[-1].strip() if "|" in ans else ans
        result.append({
            "item": item,
            "answer": ans,
            "description": desc[:80],
            "confidence": conf,
            "detected": detected,
        })
    any_detected = any(r["detected"] for r in result)
    return jsonify(items=result, any_detected=any_detected)


@app.route("/zone", methods=["GET"])
def get_zone():
    zone_file = "data/zone.json"
    if not os.path.exists(zone_file):
        return jsonify(points=None)
    with open(zone_file) as f:
        pts = json.load(f)
    return jsonify(points=pts)

@app.route("/zone", methods=["POST"])
def set_zone():
    """
    Receives normalised points [{x, y}, ...] from the browser.
    Saves to zone.json and hot-reloads the zone without restart.
    """
    data = request.get_json()
    pts = data.get("points", [])
    if len(pts) < 3:
        return jsonify(error="Need at least 3 points"), 400

    norm = [[p["x"], p["y"]] for p in pts]
    os.makedirs("data", exist_ok=True)
    with open("data/zone.json", "w") as f:
        json.dump(norm, f)

    global zone_active
    zone_active = load_zone_from_points(norm)
    log_event(f"Zone updated via UI ({len(norm)} vertices).")
    return jsonify(ok=True)

@app.route("/zone", methods=["DELETE"])
def delete_zone():
    global zone_active
    clear_zone()
    if os.path.exists("data/zone.json"):
        os.remove("data/zone.json")
    zone_active = False
    log_event("Zone cleared via UI.")
    return jsonify(ok=True)


@app.route("/context", methods=["GET"])
def get_context():
    return jsonify(items=get_all_items())

@app.route("/context", methods=["POST"])
def post_context():
    data = request.get_json()
    item = data.get("item", "").strip()
    if not item:
        return jsonify(error="Empty item"), 400
    add_item(item)
    _rebuild_questions()
    return jsonify(ok=True, items=get_all_items())

@app.route("/context/<int:idx>", methods=["DELETE"])
def delete_context(idx: int):
    items = get_all_items()
    if idx < 0 or idx >= len(items):
        return jsonify(error="Index out of range"), 400
    remove_item(idx)
    _rebuild_questions()
    return jsonify(ok=True, items=get_all_items())

@app.route("/shutdown", methods=["POST"])
def shutdown():
    global _running
    _running = False
    return jsonify(status="shutting_down")

@app.route("/debug/ai_crop")
def debug_ai_crop():
    """Shows exactly what Qwen sees — open http://localhost:4000/debug/ai_crop in a browser."""
    with ui_state.lock:
        frame_bytes = ui_state.latest_frame
    if not frame_bytes:
        return "No frame yet", 503

    arr = np.frombuffer(frame_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    zone_pts = get_zone_pts()
    if zone_pts is not None and len(zone_pts) >= 3:
        x, y, cw, ch = cv2.boundingRect(np.array(zone_pts, dtype=np.int32))
        x, y = max(0, x), max(0, y)
        cropped = frame[y:y + ch, x:x + cw]
        if cropped.size > 0:
            frame = cropped

    ret, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ret:
        return "Encode failed", 500
    return Response(jpeg.tobytes(), mimetype="image/jpeg")

def run_flask():
    app.run(host="0.0.0.0", port=4000, debug=False, use_reloader=False)


print("Loading Qwen2.5-VL model...")
actions.init_model()

print("Initialising MediaPipe tracker...")
tracker = VisionTracker()

cap = cv2.VideoCapture(config.CAMERA_SOURCE)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera.")

ok, first_frame = cap.read()
if not ok:
    raise RuntimeError("Cannot read from camera.")
first_frame = cv2.flip(first_frame, 1)

zone_active = init_zone(first_frame)
init_context()


_questions: list[str] = []
_answers: list[str] = []
_items: list[str] = []
_ai_lock = threading.Lock()
_ai_busy = False
_blip_idx = 0
last_ai_check: float = time.time()
_AI_INTERVAL = 5.0

_last_item_push: dict[str, float] = {}
_ITEM_PUSH_COOLDOWN = 60.0

_item_first_seen: dict[str, float] = {}
_ITEM_DWELL_SECONDS = 2.0

def _rebuild_questions():
    """Called on startup and whenever context items change."""
    global _questions, _answers, _items
    with _ai_lock:
        _items = get_all_items()
        _questions = build_blip_questions()
        _answers = ["..." for _ in _questions]
    print(f"[main] Monitoring for: {get_summary()}")

_rebuild_questions()


threading.Thread(target=run_flask, daemon=True).start()
print(f"\n[main] Web UI → http://localhost:4000")
print("[main] Running — press Ctrl+C to quit.\n")


guard = IntrusionGuard()
_last_cleared_push: float = 0.0
_CLEARED_PUSH_COOLDOWN = 60.0
_flash_until: float = 0.0
_FLASH_DURATION = 0.4
_object_alert_until: float = 0.0
_OBJECT_ALERT_DURATION = 15.0


def _run_qwen_async(frame_copy, question: str, answer_idx: int,
                    item_name: str, crop_pts=None):
    global _ai_busy
    try:
        print(f"[Qwen] Scanning for '{item_name}'...")
        answer = actions.query_frame(frame_copy, question, crop_pts=crop_pts)
        print(f"[Qwen] Result for '{item_name}': {answer}")
        with _ai_lock:
            if answer_idx < len(_answers):
                _answers[answer_idx] = answer

        if parse_detected(answer):
            global _object_alert_until
            _object_alert_until = time.monotonic() + _OBJECT_ALERT_DURATION

            now = time.monotonic()

            if item_name not in _item_first_seen:
                _item_first_seen[item_name] = now
                print(f"[Qwen] First detection of '{item_name}' — confirming...")
            
            dwell = now - _item_first_seen[item_name]

            if dwell >= _ITEM_DWELL_SECONDS:
                last = _last_item_push.get(item_name, 0.0)
                if now - last >= _ITEM_PUSH_COOLDOWN:
                    _last_item_push[item_name] = now
                    conf = parse_confidence(answer)
                    conf_str = f"{conf}%" if conf >= 0 else "unknown"
                    desc = answer.split("|")[-1].strip() if "|" in answer else answer
                    print(f"[Qwen] ⚠️  ALERT for '{item_name}' — seen for {dwell:.1f}s, sending push!")
                    notifier.send(
                        title=f"🔍 OBJECT DETECTED: {item_name}",
                        message=(
                            f"{item_name} spotted in the monitored zone ({conf_str} confidence).\n\n"
                            f"AI: {desc}"
                        ),
                        priority="urgent"
                    )
                    log_event(f"OBJECT DETECTED [{item_name}] ({conf_str}): {desc}")
        else:
            if item_name in _item_first_seen:
                del _item_first_seen[item_name]
    except Exception as e:
        print(f"[Qwen] ERROR for '{item_name}': {e}")
        with _ai_lock:
            if answer_idx < len(_answers):
                _answers[answer_idx] = f"err: {e}"
    finally:
        _ai_busy = False


def _draw_hud(frame, guard_state: GuardState,
              dwell_progress: float, alert_active: bool) -> None:
    h, w = frame.shape[:2]
    draw_zones(frame, alert_active=alert_active)

    if alert_active:
        if int((time.monotonic() - _flash_until) / _FLASH_DURATION) % 2 == 0:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 8)

    if guard_state == GuardState.DWELLING and dwell_progress > 0:
        bar_w = int((w - 20) * dwell_progress)
        cv2.rectangle(frame, (10, h - 22), (w - 10, h - 8), (40, 40, 40), -1)
        cv2.rectangle(frame, (10, h - 22), (10 + bar_w, h - 8), (0, 165, 255), -1)
        cv2.putText(frame, "Confirming intrusion...", (10, h - 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 1)

    if alert_active:
        banner = "!! INTRUSION DETECTED !!"
        font, scale = cv2.FONT_HERSHEY_DUPLEX, 1.2
        (bw, bh), _ = cv2.getTextSize(banner, font, scale, 2)
        bx, by = (w - bw) // 2, h - 18
        overlay = frame.copy()
        cv2.rectangle(overlay, (bx - 8, by - bh - 8),
                      (bx + bw + 8, by + 8), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        cv2.putText(frame, banner, (bx, by), font, scale, (0, 0, 255), 2)


try:
    while _running and cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        person, gesture, objects, pose_landmarks = tracker.process_frame(frame)
        body_in = body_centre_in_zone(pose_landmarks, w, h) if zone_active else False

        prev_guard_state = guard.state
        guard_state, dwell_progress = guard.update(body_in)
        alert_active = guard.is_intruding()

        if guard.just_triggered(prev_guard_state):
            _flash_until = time.monotonic()
            with _ai_lock:
                ai_ctx = " | ".join(
                    f"{it}: {an}" for it, an in zip(_items, _answers)
                )
            fire_alert(frame, "intrusion", "Person in zone", ai_answer=ai_ctx)
            notifier.intrusion_alert(ai_description=ai_ctx)

        if prev_guard_state == GuardState.INTRUDING and guard_state == GuardState.CLEAR:
            now = time.monotonic()
            if now - _last_cleared_push >= _CLEARED_PUSH_COOLDOWN:
                _last_cleared_push = now
                notifier.intrusion_cleared()

        if alert_active:
            _flash_until = time.monotonic()

        with _ai_lock:
            has_questions = len(_questions) > 0

        if not _ai_busy and has_questions and zone_active and time.time() - last_ai_check >= _AI_INTERVAL:
            _ai_busy = True
            last_ai_check = time.time()
            with _ai_lock:
                q_count = len(_questions)
            q_idx = _blip_idx % q_count
            _blip_idx += 1
            with _ai_lock:
                q = _questions[q_idx]
                it = _items[q_idx] if q_idx < len(_items) else ""
            threading.Thread(
                target=_run_qwen_async,
                args=(frame.copy(), q, q_idx, it, get_zone_pts()),
                daemon=True
            ).start()


        _draw_hud(frame, guard_state, dwell_progress, alert_active)
        ui_state.update(frame, person, alert_active, guard_state, dwell_progress)

finally:
    cap.release()
    tracker.close()
    print("Shutdown complete.")