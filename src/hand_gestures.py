import cv2
import mediapipe as mp
import time
import os
import urllib.request

MODEL_PATH = "gesture_recognizer.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"

if not os.path.exists(MODEL_PATH):
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

current_gesture = "NO HAND"

def result_callback(result, output_image, timestamp_ms):
    global current_gesture
    if result.gestures and len(result.gestures) > 0:
        current_gesture = result.gestures[0][0].category_name
        if current_gesture == "None":
            current_gesture = "HAND DETECTED"
    else:
        current_gesture = "NO HAND"

options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=result_callback
)

with GestureRecognizer.create_from_options(options) as recognizer:
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        
        frame = cv2.flip(frame, 1)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        
        timestamp = int(time.time() * 1000)
        recognizer.recognize_async(mp_image, timestamp)
        
        color = (0, 255, 0) if current_gesture != "NO HAND" else (0, 0, 255)
        cv2.putText(frame, f"Gesture: {current_gesture}", (30, 50), 
                    cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)
        
        cv2.imshow('Gesture Demo', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()