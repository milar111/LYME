import warnings
import logging
import time
import threading
import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request
from src import actions, config
from src.tracker import VisionTracker
from src.zone_manager import init_zone, draw_zones, body_centre_in_zone, get_zone_pts
from src.intrusion_guard import IntrusionGuard, GuardState
from src.context_manager import (init_context, build_blip_questions,
                                  get_summary, get_items,
                                  parse_confidence, parse_detected)
from src import notifier
from src.alerts import fire_alert, log_event

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

class WebUIState:
    def __init__(self):
        self.latest_frame: bytes | None = None
        self.person_in_sight = False
        self.last_seen: float | None = None
        self.frame_count = 0
        self.lock = threading.Lock()

    def update(self, frame, person_detected: bool):
        with self.lock:
            self.person_in_sight = person_detected
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
            last_seen_ago=last_seen_ago,
            frame_count=ui_state.frame_count
        )

@app.route("/shutdown", methods=['POST'])
def shutdown():
    global _running
    _running = False
    return jsonify(status="shutting_down")

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
print(f"[main] Zone: {'active' if zone_active else 'none — monitor-only mode'}")

init_context()
_questions   = build_blip_questions()
_items       = get_items()
_blip_idx    = 0

threading.Thread(target=run_flask, daemon=True).start()
print("\n[main] Web UI active at http://localhost:4000")
print(f"[main] Monitoring for: {get_summary()}")
print("[main] Running — use the Web UI to monitor or shut down.\n")

guard = IntrusionGuard()
prev_guard_state: GuardState = GuardState.CLEAR

_answers: list[str] = ["..." for _ in _questions]
_ai_lock   = threading.Lock()
_ai_busy   = False
last_ai_check: float = 0.0
_AI_INTERVAL = 10.0

_last_item_push: list[float] = [0.0 for _ in _questions]
_ITEM_PUSH_COOLDOWN = 60.0

_last_cleared_push: float = 0.0
_CLEARED_PUSH_COOLDOWN = 60.0

_flash_until: float = 0.0
_FLASH_DURATION = 0.4

def _run_qwen_async(frame_copy, question: str, answer_idx: int,
                    item_name: str, crop_pts=None):
    global _ai_busy
    try:
        answer = actions.query_frame(frame_copy, question, crop_pts=crop_pts)
        with _ai_lock:
            _answers[answer_idx] = answer

        if parse_detected(answer):
            now = time.monotonic()
            if now - _last_item_push[answer_idx] >= _ITEM_PUSH_COOLDOWN:
                _last_item_push[answer_idx] = now
                conf = parse_confidence(answer)
                conf_str = f"{conf}%" if conf >= 0 else "unknown confidence"
                notifier.send(
                    title=f"⚠️ DETECTED: {item_name}",
                    message=(
                        f"{item_name} detected in forbidden zone "
                        f"({conf_str}).\n\nAI: {answer}"
                    ),
                    priority="high"
                )
                log_event(f"OBJECT DETECTED [{item_name}] ({conf_str}): {answer}")

    except Exception as e:
        with _ai_lock:
            _answers[answer_idx] = f"err: {e}"
    finally:
        _ai_busy = False

def _put_label(frame, text: str, x: int, y: int,
               colour=(0, 255, 0), scale: float = 0.78, thickness: int = 2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    pad = 5
    overlay = frame.copy()
    cv2.rectangle(overlay,
                  (x - pad, y - th - pad),
                  (x + tw + pad, y + baseline + pad),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.putText(frame, text, (x, y), font, scale, colour, thickness)

def _draw_hud(frame, person: bool, gesture: str, objects: list,
              guard_state: GuardState, dwell_progress: float,
              alert_active: bool) -> None:
    h, w = frame.shape[:2]
    draw_zones(frame, alert_active=alert_active)

    if alert_active:
        if int((time.monotonic() - _flash_until) / _FLASH_DURATION) % 2 == 0:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 8)

    SCALE, LINE_H, x0, y0 = 0.65, 28, 10, 36

    person_colour = (0, 0, 255) if (person and zone_active and guard_state != GuardState.CLEAR) else (0, 255, 0)
    _put_label(frame,
               f"Person: {'IN ZONE' if (person and guard_state != GuardState.CLEAR) else ('detected' if person else 'none')}",
               x0, y0, person_colour, SCALE)

    _put_label(frame, f"Gesture: {gesture}", x0, y0 + LINE_H, (0, 255, 0), SCALE)
    obj_text = ", ".join(objects) if objects else "none"
    _put_label(frame, f"Objects: {obj_text}", x0, y0 + LINE_H * 2, (0, 255, 0), SCALE)

    with _ai_lock:
        answers = list(_answers)

    for i, (item, ans) in enumerate(zip(_items, answers)):
        short_item = item if len(item) <= 20 else item[:17] + "..."
        conf = parse_confidence(ans)
        detected = parse_detected(ans)

        if ans == "...": colour = (180, 180, 180)
        elif detected and conf >= 70: colour = (0, 80, 255)
        elif detected and conf >= 40: colour = (0, 165, 255)
        else: colour = (0, 220, 100)

        if conf >= 0:
            display = f"{conf}% | {ans.split('|')[-1].strip()[:35]}" if '|' in ans else ans[:40]
        else:
            display = ans[:40]
        
        if len(display) > 40:
            display = display[:37] + "..."
        
        _put_label(frame, f"AI [{short_item}]: {display}", x0, y0 + LINE_H * (3 + i), colour, SCALE)

    status_y = y0 + LINE_H * (3 + len(_items))
    _put_label(frame, f"Zone: {guard_state.name}", x0, status_y, (0, 0, 255) if alert_active else (180, 180, 180), SCALE)

    if guard_state == GuardState.DWELLING and dwell_progress > 0:
        bar_w = int((w - 20) * dwell_progress)
        cv2.rectangle(frame, (10, h - 22), (w - 10, h - 8), (40, 40, 40), -1)
        cv2.rectangle(frame, (10, h - 22), (10 + bar_w, h - 8), (0, 165, 255), -1)

try:
    while _running and cap.isOpened():
        ok, frame = cap.read()
        if not ok: break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        person, gesture, objects, pose_landmarks = tracker.process_frame(frame)
        body_in = body_centre_in_zone(pose_landmarks, w, h) if zone_active else False

        prev_guard_state = guard.state
        guard_state, dwell_progress = guard.update(body_in)
        alert_active = guard.is_intruding()

        with _ai_lock:
            ai_context = " | ".join(f"{item}: {ans}" for item, ans in zip(_items, _answers))

        if guard.just_triggered(prev_guard_state):
            _flash_until = time.monotonic()
            log_event(f"INTRUSION detected. AI context: {ai_context}")
            fire_alert(frame, "intrusion", "Person in forbidden zone", ai_answer=ai_context)
            notifier.intrusion_alert(ai_description=ai_context)

        if prev_guard_state == GuardState.INTRUDING and guard_state == GuardState.CLEAR:
            log_event("Zone cleared.")
            now = time.monotonic()
            if now - _last_cleared_push >= _CLEARED_PUSH_COOLDOWN:
                _last_cleared_push = now
                notifier.intrusion_cleared()

        if alert_active:
            _flash_until = time.monotonic()

        if not _ai_busy and _questions and time.time() - last_ai_check >= _AI_INTERVAL:
            _ai_busy, last_ai_check = True, time.time()
            q_idx = _blip_idx % len(_questions)
            _blip_idx += 1
            threading.Thread(
                target=_run_qwen_async,
                args=(frame.copy(), _questions[q_idx], q_idx, _items[q_idx], get_zone_pts()),
                daemon=True
            ).start()

        _draw_hud(frame, person, gesture, objects, guard_state, dwell_progress, alert_active)
        ui_state.update(frame, person)

finally:
    cap.release()
    cv2.destroyAllWindows()
    tracker.close()
    print("Shutdown complete.")