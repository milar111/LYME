"""
zone_manager.py
───────────────
Manages the forbidden zone polygon.

Fixes vs previous version:
  • Backspace works on macOS — key 127 (Delete) AND key 8 both undo last point.
  • 4+ point polygon draws correctly — uses cv2.convexHull so the fill never
    self-intersects regardless of click order.
  • Editor window gets WINDOW_NORMAL flag + explicit focus pump so Enter/Esc
    are reliably received on macOS.
  • Points are numbered so you know which one backspace will remove.

Persistence:
  • Saved to data/zone.json on first draw.
  • Loaded silently on every subsequent run — editor never opens again.
  • To redraw: delete data/zone.json and restart.
"""

import cv2
import json
import os
import numpy as np
from typing import Optional
from src import config

_ZONE_FILE = "data/zone.json"

_forbidden_pts: Optional[np.ndarray] = None
_free_label_pos: Optional[tuple[int, int]] = None
_frame_size: tuple[int, int] = (0, 0)


# ── Persistence ───────────────────────────────────────────────────────────────

def _save_zone(norm_pts: list[tuple[float, float]]) -> None:
    os.makedirs("data", exist_ok=True)
    with open(_ZONE_FILE, "w") as f:
        json.dump(norm_pts, f)
    print(f"[ZoneManager] Zone saved to {_ZONE_FILE}  (delete to redraw).")


def _load_zone() -> Optional[list[tuple[float, float]]]:
    if not os.path.exists(_ZONE_FILE):
        return None
    with open(_ZONE_FILE) as f:
        data = json.load(f)
    return [tuple(p) for p in data]


# ── Free-area centroid ────────────────────────────────────────────────────────

def _compute_free_label_pos(w: int, h: int) -> tuple[int, int]:
    if _forbidden_pts is None:
        return (10, 28)
    step = max(w, h) // 20
    free_xs, free_ys = [], []
    for gy in range(0, h, step):
        for gx in range(0, w, step):
            if cv2.pointPolygonTest(_forbidden_pts, (float(gx), float(gy)), False) < 0:
                free_xs.append(gx)
                free_ys.append(gy)
    if not free_xs:
        return (10, 28)
    return (int(sum(free_xs) / len(free_xs)), int(sum(free_ys) / len(free_ys)))


# ── Interactive editor ────────────────────────────────────────────────────────

# Shared list mutated by mouse callback
_editor_points: list[tuple[int, int]] = []

def _mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        _editor_points.append((x, y))


def _run_editor(frame: np.ndarray) -> list[tuple[int, int]]:
    global _editor_points
    _editor_points = []

    win = "Zone Editor"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, frame.shape[1], frame.shape[0])
    cv2.setMouseCallback(win, _mouse_callback)

    # Pump a few frames so macOS gives the window keyboard focus
    for _ in range(5):
        cv2.imshow(win, frame)
        cv2.waitKey(30)

    while True:
        display = frame.copy()
        h, w = display.shape[:2]
        pts = _editor_points

        # ── Draw polygon preview ──────────────────────────────────────────────
        if len(pts) >= 3:
            pts_arr = np.array(pts, dtype=np.int32)

            # convexHull keeps the fill correct regardless of click order
            hull = cv2.convexHull(pts_arr)

            overlay = display.copy()
            cv2.fillPoly(overlay, [hull], (0, 0, 180))
            cv2.addWeighted(overlay, 0.35, display, 0.65, 0, display)

            # Draw actual click-order outline in dim colour
            cv2.polylines(display, [pts_arr], isClosed=True,
                          color=(80, 80, 255), thickness=1)
            # Draw convex hull outline prominently
            cv2.polylines(display, [hull], isClosed=True,
                          color=(0, 0, 255), thickness=2)

        elif len(pts) == 2:
            cv2.line(display, pts[0], pts[1], (0, 0, 255), 2)

        # ── Draw numbered points ──────────────────────────────────────────────
        for i, pt in enumerate(pts):
            cv2.circle(display, pt, 7, (0, 0, 255), -1)
            cv2.circle(display, pt, 7, (255, 255, 255), 1)
            cv2.putText(display, str(i + 1), (pt[0] + 9, pt[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

        # ── Instructions ─────────────────────────────────────────────────────
        instructions = [
            "Left-click: place point",
            "Backspace / Delete: undo last point",
            "Enter: confirm (need >= 3 pts)",
            "Esc: cancel",
        ]
        for j, line in enumerate(instructions):
            cv2.putText(display, line,
                        (10, h - 12 - (len(instructions) - 1 - j) * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1,
                        cv2.LINE_AA)

        status_col = (0, 220, 100) if len(pts) >= 3 else (0, 200, 255)
        cv2.putText(display,
                    f"Points: {len(pts)}  {'— ready, press Enter' if len(pts) >= 3 else '(need >= 3)'}",
                    (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_col, 2, cv2.LINE_AA)

        cv2.imshow(win, display)
        key = cv2.waitKey(20) & 0xFF

        if key in (13, 10) and len(pts) >= 3:   # Enter / Return
            break
        elif key == 27:                           # Esc — cancel
            _editor_points = []
            break
        elif key in (8, 127) and pts:             # Backspace (8) or Delete (127)
            _editor_points.pop()

    cv2.destroyWindow(win)
    # Small pause so the window teardown doesn't swallow the next imshow
    cv2.waitKey(100)
    return list(_editor_points)


# ── Public init ───────────────────────────────────────────────────────────────

def init_zone(frame: np.ndarray) -> bool:
    global _forbidden_pts, _free_label_pos, _frame_size
    h, w = frame.shape[:2]
    _frame_size = (w, h)

    norm_pts = _load_zone()

    if norm_pts:
        print(f"[ZoneManager] Loaded zone from {_ZONE_FILE}  ({len(norm_pts)} vertices).")
    else:
        print("[ZoneManager] No saved zone — opening editor (only happens once).")
        pixel_pts = _run_editor(frame)
        if len(pixel_pts) < 3:
            print("[ZoneManager] Editor cancelled — no forbidden zone.")
            return False

        # Store the convex hull point order so the saved polygon is always valid
        hull_pts = cv2.convexHull(np.array(pixel_pts, dtype=np.int32))
        pixel_pts = [tuple(p[0]) for p in hull_pts]

        norm_pts = [(px / w, py / h) for px, py in pixel_pts]
        _save_zone(norm_pts)

    pixel_pts_list = [(int(x * w), int(y * h)) for x, y in norm_pts]
    _forbidden_pts = np.array(pixel_pts_list, dtype=np.int32)
    _free_label_pos = _compute_free_label_pos(w, h)
    return True


def has_zone() -> bool:
    return _forbidden_pts is not None


def get_zone_pts() -> Optional[np.ndarray]:
    """Returns the forbidden zone polygon (int32 numpy array) or None."""
    return _forbidden_pts.copy() if _forbidden_pts is not None else None


# ── Geometry ──────────────────────────────────────────────────────────────────

def point_in_zone(x: int, y: int) -> bool:
    if _forbidden_pts is None:
        return False
    return cv2.pointPolygonTest(_forbidden_pts, (float(x), float(y)), False) >= 0


def body_centre_in_zone(pose_landmarks, frame_width: int, frame_height: int) -> bool:
    """
    Uses torso midpoint (avg of shoulders + hips).
    A hand crossing the line alone will NOT trigger this.
    MediaPipe indices: 11=L-shoulder, 12=R-shoulder, 23=L-hip, 24=R-hip
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


# ── Drawing ───────────────────────────────────────────────────────────────────

def _put_text_with_bg(frame, text, pos, scale, colour, thickness=2):
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x, y = pos
    pad = 4
    overlay = frame.copy()
    cv2.rectangle(overlay,
                  (x - pad, y - th - pad),
                  (x + tw + pad, y + baseline + pad),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thickness)


def draw_zones(frame: np.ndarray, alert_active: bool = False) -> np.ndarray:
    h, w = frame.shape[:2]

    if _forbidden_pts is not None:
        overlay = frame.copy()
        cv2.fillPoly(overlay, [_forbidden_pts],
                     (0, 0, 200) if alert_active else (0, 0, 255))
        alpha = 0.45 if alert_active else 0.25
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        cv2.polylines(frame, [_forbidden_pts], isClosed=True,
                      color=(0, 50, 255) if alert_active else (0, 0, 255),
                      thickness=3 if alert_active else 2)

        # FORBIDDEN ZONE label at centroid
        M = cv2.moments(_forbidden_pts)
        if M["m00"] != 0:
            fcx = int(M["m10"] / M["m00"])
            fcy = int(M["m01"] / M["m00"])
        else:
            fcx, fcy = int(_forbidden_pts[0][0]), int(_forbidden_pts[0][1])
        label = "FORBIDDEN ZONE"
        (lw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)
        lx = max(4, min(fcx - lw // 2, w - lw - 4))
        _put_text_with_bg(frame, label, (lx, fcy), 0.85,
                          (0, 80, 255) if alert_active else (0, 0, 255), 2)

    # FREE ZONE label at free-area centroid
    if _free_label_pos is not None:
        flx, fly = _free_label_pos
        label = "FREE ZONE"
        (lw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2)
        lx = max(4, min(flx - lw // 2, w - lw - 4))
        _put_text_with_bg(frame, label, (lx, fly), 0.85, (0, 220, 0), 2)
    else:
        _put_text_with_bg(frame, "FREE ZONE", (10, 30), 0.85, (0, 220, 0), 2)

    return frame