"""
tracker.py
──────────
Wraps three MediaPipe LIVE_STREAM models:
  • PoseLandmarker    → person detection + raw landmarks for zone geometry
  • ObjectDetector    → object class labels
  • GestureRecognizer → hand gesture labels

Key fixes vs original:
  • BGR → RGB conversion before wrapping in mp.Image
  • Each model gets its own mp.Image instance (no shared buffer)
  • Monotonic timestamp counter (MediaPipe rejects non-increasing timestamps)
  • Exposes raw_pose_landmarks for body-centre-in-zone calculations
  • close() method for clean shutdown
"""

import mediapipe as mp
import urllib.request
import numpy as np
import os
import time


class VisionTracker:
    def __init__(self):
        self.person_detected = False
        self.current_gesture = "NO HAND"
        self.current_objects: list[str] = []

        # Raw pose landmarks from the most recent callback — used by zone_manager
        # to compute the torso midpoint rather than checking any single landmark.
        self.raw_pose_landmarks = None

        self._timestamp_ms: int = 0

        self._init_models()

        BaseOptions = mp.tasks.BaseOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        pose_options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path="pose_landmarker_full.task"),
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=self._pose_callback
        )
        self.pose_landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(pose_options)

        obj_options = mp.tasks.vision.ObjectDetectorOptions(
            base_options=BaseOptions(model_asset_path="efficientdet_lite0.tflite"),
            running_mode=VisionRunningMode.LIVE_STREAM,
            score_threshold=0.5,
            result_callback=self._obj_callback
        )
        self.object_detector = mp.tasks.vision.ObjectDetector.create_from_options(obj_options)

        gest_options = mp.tasks.vision.GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path="gesture_recognizer.task"),
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=self._gest_callback
        )
        self.gesture_recognizer = mp.tasks.vision.GestureRecognizer.create_from_options(gest_options)

    # ── Model download ────────────────────────────────────────────────────────

    def _init_models(self):
        models = {
            "pose_landmarker_full.task": (
                "https://storage.googleapis.com/mediapipe-models/"
                "pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
            ),
            "efficientdet_lite0.tflite": (
                "https://storage.googleapis.com/mediapipe-models/"
                "object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite"
            ),
            "gesture_recognizer.task": (
                "https://storage.googleapis.com/mediapipe-models/"
                "gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"
            ),
        }
        for path, url in models.items():
            if not os.path.exists(path):
                print(f"Downloading {path}...")
                urllib.request.urlretrieve(url, path)
                print(f"  ✓ {path} ready.")

    # ── Async callbacks ───────────────────────────────────────────────────────

    def _pose_callback(self, result, output_image, timestamp_ms):
        if result.pose_landmarks:
            self.person_detected = True
            self.raw_pose_landmarks = result.pose_landmarks[0]
        else:
            self.person_detected = False
            self.raw_pose_landmarks = None

    def _obj_callback(self, result, output_image, timestamp_ms):
        self.current_objects = [
            detection.categories[0].category_name
            for detection in result.detections
        ]

    def _gest_callback(self, result, output_image, timestamp_ms):
        if result.gestures and len(result.gestures) > 0:
            gest = result.gestures[0][0].category_name
            self.current_gesture = "HAND DETECTED" if gest == "None" else gest
        else:
            self.current_gesture = "NO HAND"

    # ── Timestamp helper ──────────────────────────────────────────────────────

    def _next_timestamp(self) -> int:
        """Strictly increasing millisecond timestamp — MediaPipe requirement."""
        now_ms = int(time.monotonic() * 1000)
        if now_ms <= self._timestamp_ms:
            now_ms = self._timestamp_ms + 1
        self._timestamp_ms = now_ms
        return now_ms

    # ── Main entry point ──────────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray):
        """
        frame: BGR numpy array from OpenCV.
        Returns (person_detected, gesture, objects, raw_pose_landmarks).
        raw_pose_landmarks is a list of NormalizedLandmark or None.
        """
        rgb = frame[:, :, ::-1].copy()   # BGR → RGB

        self.pose_landmarker.detect_async(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
            self._next_timestamp()
        )
        self.object_detector.detect_async(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
            self._next_timestamp()
        )
        self.gesture_recognizer.recognize_async(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
            self._next_timestamp()
        )

        return (
            self.person_detected,
            self.current_gesture,
            list(set(self.current_objects)),
            self.raw_pose_landmarks,
        )

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def close(self):
        self.pose_landmarker.close()
        self.object_detector.close()
        self.gesture_recognizer.close()