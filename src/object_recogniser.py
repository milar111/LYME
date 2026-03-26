import cv2
import mediapipe as mp
import time
import os
import urllib.request

MODEL_PATH = "efficientdet_lite0.tflite"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/int8/1/efficientdet_lite0.tflite"

if not os.path.exists(MODEL_PATH):
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

BaseOptions = mp.tasks.BaseOptions
ObjectDetector = mp.tasks.vision.ObjectDetector
ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

current_objects = []

def result_callback(result, output_image, timestamp_ms):
    global current_objects
    current_objects = [detection.categories[0].category_name for detection in result.detections]

options = ObjectDetectorOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.LIVE_STREAM,
    score_threshold=0.5,
    result_callback=result_callback
)

with ObjectDetector.create_from_options(options) as detector:
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        
        frame = cv2.flip(frame, 1)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        
        timestamp = int(time.time() * 1000)
        detector.detect_async(mp_image, timestamp)
        
        unique_objects = list(set(current_objects))
        display_text = ", ".join(unique_objects) if unique_objects else "NO OBJECTS"
        color = (0, 255, 0) if unique_objects else (0, 0, 255)
        
        cv2.putText(frame, f"Detecting: {display_text}", (30, 50), 
                    cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)
        
        cv2.imshow('Object Demo', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()