import cv2

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import time
import os
import urllib.request
import threading

MODEL_PATH = "pose_landmarker_full.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"


class DetectionState:
    """Shared mutable state between detection thread and Flask."""
    def __init__(self):
        self.person_in_sight = False
        self.last_seen: float | None = None
        self.frame_count = 0
        self.lock = threading.Lock()

    def set_detected(self, detected: bool):
        with self.lock:
            self.person_in_sight = detected
            if detected:
                self.last_seen = time.time()
            self.frame_count += 1

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "person_in_sight": self.person_in_sight,
                "last_seen": self.last_seen,
                "frame_count": self.frame_count,
            }


class Detector:
    def __init__(self, state: DetectionState, camera_index: int = 0):
        self.state = state
        self.camera_index = camera_index
        self._running = False
        self._thread: threading.Thread | None = None

        self._frame_lock = threading.Lock()
        self._latest_frame: bytes | None = None

        self._ensure_model()

    @staticmethod
    def _ensure_model():
        if not os.path.exists(MODEL_PATH):
            print(f"Downloading pose model to {MODEL_PATH} …")
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            print("Download complete.")

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

    def get_frame(self) -> bytes | None:
        with self._frame_lock:
            return self._latest_frame

    def _run(self):
        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        def result_callback(result, _output_image, _timestamp_ms):
            self.state.set_detected(len(result.pose_landmarks) > 0)

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=result_callback,
            min_pose_detection_confidence=0.2,
            min_pose_presence_confidence=0.2,
            min_tracking_confidence=0.2,
        )

        with PoseLandmarker.create_from_options(options) as landmarker:
            cap = cv2.VideoCapture(self.camera_index)
            if not cap.isOpened():
                print(f"[Detector] Cannot open camera {self.camera_index}")
                self._running = False
                return

            while self._running and cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break

                frame = cv2.flip(frame, 1)

                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                landmarker.detect_async(mp_image, int(time.time() * 1000))

                ret, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if ret:
                    with self._frame_lock:
                        self._latest_frame = jpeg.tobytes()

            cap.release()