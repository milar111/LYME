import cv2
import mediapipe as mp
import time
import os
import urllib.request


MODEL_PATH = "pose_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"

if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Download complete.")

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

person_in_sight = False

def result_callback(result, output_image, timestamp_ms):
    global person_in_sight
    person_in_sight = len(result.pose_landmarks) > 0

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=result_callback
)


with PoseLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        
        timestamp = int(time.time() * 1000)
        landmarker.detect_async(mp_image, timestamp)

        
        msg = "PERSON IN SIGHT" if person_in_sight else "NO ONE YET..."
        color = (0, 255, 0) if person_in_sight else (0, 0, 255)
        cv2.putText(frame, msg, (30, 50), cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)
        
        cv2.imshow('Demo', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()