import cv2
import numpy as np
from src import config


class MotionDetector:
    def __init__(self):
        self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=50,
            detectShadows=False
        )

    def detect(self, frame: np.ndarray) -> tuple:
        fg_mask = self._bg_subtractor.apply(frame)

        fg_mask = cv2.GaussianBlur(fg_mask, (config.MOTION_BLUR_SIZE, config.MOTION_BLUR_SIZE), 0)
        _, fg_mask = cv2.threshold(fg_mask, 25, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.dilate(fg_mask, kernel, iterations=config.MOTION_DILATE_ITERATIONS)

        changed_pixels = cv2.countNonZero(fg_mask)
        motion = bool(changed_pixels > config.MOTION_THRESHOLD)

        return motion, fg_mask

    def draw_overlay(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Draw motion contours on the frame for debugging."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if cv2.contourArea(c) > 500:
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 165, 255), 2)
        return frame


def is_in_alert_zone(frame: np.ndarray, mask: np.ndarray) -> bool:
    if config.ALERT_ZONE is None:
        return True

    h, w = frame.shape[:2]
    pts = np.array(
        [(int(x * w), int(y * h)) for x, y in config.ALERT_ZONE],
        dtype=np.int32
    )

    zone_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(zone_mask, [pts], (255,))

    overlap = cv2.bitwise_and(mask, zone_mask)
    return bool(cv2.countNonZero(overlap) > 0)


def draw_alert_zone(frame: np.ndarray) -> np.ndarray:
    """Draw the alert zone polygon on the frame."""
    if config.ALERT_ZONE is None:
        return frame

    h, w = frame.shape[:2]
    pts = np.array(
        [(int(x * w), int(y * h)) for x, y in config.ALERT_ZONE],
        dtype=np.int32
    )
    cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 255), thickness=2)
    cv2.putText(frame, "ALERT ZONE", tuple(pts[0].tolist()),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return frame