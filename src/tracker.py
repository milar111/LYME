import mediapipe as mp
import urllib.request
import os
import time

class VisionTracker:
    def __init__(self):
        self.person_detected = False
        self.current_gesture = "NO HAND"
        self.current_objects = []

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

    def _init_models(self):
        models = {
            "pose_landmarker_full.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
            "efficientdet_lite0.tflite": "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite",
            "gesture_recognizer.task": "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"
        }
        for path, url in models.items():
            if not os.path.exists(path):
                urllib.request.urlretrieve(url, path)

    def _pose_callback(self, result, output_image, timestamp_ms):
        self.person_detected = len(result.pose_landmarks) > 0

    def _obj_callback(self, result, output_image, timestamp_ms):
        self.current_objects = [detection.categories[0].category_name for detection in result.detections]

    def _gest_callback(self, result, output_image, timestamp_ms):
        if result.gestures and len(result.gestures) > 0:
            gest = result.gestures[0][0].category_name
            self.current_gesture = "HAND DETECTED" if gest == "None" else gest
        else:
            self.current_gesture = "NO HAND"

    def process_frame(self, frame):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        timestamp = int(time.time() * 1000)
        
        self.pose_landmarker.detect_async(mp_image, timestamp)
        self.object_detector.detect_async(mp_image, timestamp)
        self.gesture_recognizer.recognize_async(mp_image, timestamp)

        return self.person_detected, self.current_gesture, list(set(self.current_objects))