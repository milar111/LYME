import cv2
import json
import os
import numpy as np
from typing import Optional
from src import config

_ZONE_FILE = "data/zone.json"

_forbidden_pts: Optional[np.ndarray] = None
_free_label_pos: Optional[tuple[int, int]] = None
_frame_size: tuple[int, int] = (640, 480)



def _save_zone(norm_pts: list) -> None:
    os.makedirs("data", exist_ok=True)
    with open(_ZONE_FILE, "w") as f:
        json.dump(norm_pts, f)
    print(f"[ZoneManager] Zone saved ({len(norm_pts)} vertices).")


def _load_zone() -> Optional[list]:
    if not os.path.exists(_ZONE_FILE):
        return None
    with open(_ZONE_FILE) as f:
        data = json.load(f)
    return data if isinstance(data, list) and len(data) >= 3 else None



def _compute_free_label_pos(w: int, h: int) -> tuple[int, int]:
    pts = _forbidden_pts
    if pts is None:
        return (10, 28)
    flat = pts.reshape(-1, 2)
    step = max(w, h) // 20
    free_xs, free_ys = [], []
    for gy in range(0, h, step):
        for gx in range(0, w, step):
            if cv2.pointPolygonTest(flat, (float(gx), float(gy)), False) < 0:
                free_xs.append(gx)
                free_ys.append(gy)
    if not free_xs:
        return (10, 28)
    return (int(sum(free_xs) / len(free_xs)), int(sum(free_ys) / len(free_ys)))



def _apply_norm_pts(norm_pts: list) -> bool:
    """Convert normalised pts → pixel array and store globally.
    Points are sorted counter-clockwise around their centroid
    so they always form a clean polygon regardless of click order.
    """
    global _forbidden_pts, _free_label_pos
    w, h = _frame_size
    pixel_pts = [(int(p[0] * w), int(p[1] * h)) for p in norm_pts]

    import math
    cx = sum(x for x, y in pixel_pts) / len(pixel_pts)
    cy = sum(y for x, y in pixel_pts) / len(pixel_pts)
    pixel_pts.sort(key=lambda p: math.atan2(p[1] - cy, p[0] - cx))

    _forbidden_pts = np.array(pixel_pts, dtype=np.int32).reshape(-1, 1, 2)
    _free_label_pos = _compute_free_label_pos(w, h)
    return True



def init_zone(frame: np.ndarray) -> bool:
    """Call once at startup with the first camera frame."""
    global _frame_size
    h, w = frame.shape[:2]
    _frame_size = (w, h)

    norm_pts = _load_zone()
    if norm_pts:
        print(f"[ZoneManager] Loaded zone ({len(norm_pts)} vertices).")
        return _apply_norm_pts(norm_pts)
    print("[ZoneManager] No zone saved — draw one in the web UI.")
    return False


def load_zone_from_points(norm_pts: list) -> bool:
    """
    Hot-reload zone from normalised [{x,y},...] or [[x,y],...] points.
    Called by the POST /zone Flask route — no restart needed.
    """
    converted = []
    for p in norm_pts:
        if isinstance(p, dict):
            converted.append([p["x"], p["y"]])
        else:
            converted.append([p[0], p[1]])

    if len(converted) < 3:
        return False

    print(f"[ZoneManager] Hot-reloaded zone ({len(converted)} vertices).")
    return _apply_norm_pts(converted)


def clear_zone() -> None:
    """Remove zone from memory. Called by DELETE /zone."""
    global _forbidden_pts, _free_label_pos
    _forbidden_pts = None
    _free_label_pos = None
    print("[ZoneManager] Zone cleared.")


def has_zone() -> bool:
    return _forbidden_pts is not None


def get_zone_pts() -> Optional[np.ndarray]:
    """Returns polygon as int32 numpy array (N,2) or None."""
    pts = _forbidden_pts
    return pts.reshape(-1, 2).copy() if pts is not None else None



def point_in_zone(x: int, y: int) -> bool:
    pts = _forbidden_pts
    if pts is None:
        return False
    flat = pts.reshape(-1, 2)
    return cv2.pointPolygonTest(flat, (float(x), float(y)), False) >= 0


def body_centre_in_zone(pose_landmarks, frame_width: int, frame_height: int) -> bool:
    """
    Torso midpoint (avg of shoulders + hips).
    MediaPipe indices: 11=L-shoulder 12=R-shoulder 23=L-hip 24=R-hip
    """
    if _forbidden_pts is None or not pose_landmarks:
        return False
    lm = pose_landmarks
    try:
        ls, rs, lh, rh = lm[11], lm[12], lm[23], lm[24]
    except IndexError:
        return False
    cx = ((ls.x + rs.x + lh.x + rh.x) / 4.0) * frame_width
    cy = ((ls.y + rs.y + lh.y + rh.y) / 4.0) * frame_height
    return point_in_zone(int(cx), int(cy))



def _put_text_with_bg(frame, text, pos, scale, colour, thickness=2):
    (tw, th), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x, y = pos
    pad = 4
    overlay = frame.copy()
    cv2.rectangle(overlay,
                  (x - pad, y - th - pad),
                  (x + tw + pad, y + baseline + pad),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.putText(frame, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thickness)


def draw_zones(frame: np.ndarray, alert_active: bool = False) -> np.ndarray:
    h, w = frame.shape[:2]

    pts = _forbidden_pts
    free_pos = _free_label_pos

    if pts is not None:
        flat = pts.reshape(-1, 2)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts],
                     (0, 0, 200) if alert_active else (0, 0, 255))
        alpha = 0.45 if alert_active else 0.25
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        cv2.polylines(frame, [pts], isClosed=True,
                      color=(0, 50, 255) if alert_active else (0, 0, 255),
                      thickness=3 if alert_active else 2)

        M = cv2.moments(flat)
        if M["m00"] != 0:
            fcx = int(M["m10"] / M["m00"])
            fcy = int(M["m01"] / M["m00"])
        else:
            fcx, fcy = int(flat[0][0]), int(flat[0][1])

        label = "FORBIDDEN ZONE"
        (lw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)
        lx = max(4, min(fcx - lw // 2, w - lw - 4))
        _put_text_with_bg(frame, label, (lx, fcy), 0.85,
                          (0, 80, 255) if alert_active else (0, 0, 255), 2)

    if free_pos is not None:
        flx, fly = free_pos
        label = "FREE ZONE"
        (lw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)
        lx = max(4, min(flx - lw // 2, w - lw - 4))
        _put_text_with_bg(frame, label, (lx, fly), 0.85, (0, 220, 0), 2)
    elif pts is None:
        _put_text_with_bg(frame, "NO ZONE SET", (10, 30), 0.85, (100, 100, 100), 2)

    return frame