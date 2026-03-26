import os
import cv2
import subprocess
import numpy as np
import time
from src import config
from src import actions
import warnings
import logging

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

user_question = input("What would you like to ask the AI about the mountain? (e.g. 'Is it snowing?'): ")

print("--- INITIALIZING AI ---")
actions.init_model()
print("--- AI READY ---")

ffmpeg_cmd = [
    "ffmpeg",
    "-protocol_whitelist", "file,http,https,tcp,tls,https",
    "-i", config.CAMERA_SOURCE,
    "-f", "rawvideo",
    "-pix_fmt", "bgr24",
    "-vf", "scale=1280:720",
    "-r", "1",
    "-loglevel", "error",
    "-"
]

process = subprocess.Popen(
    ffmpeg_cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    bufsize=10**8
)

width, height = 1280, 720
frame_size = width * height * 3
last_ai_check = 0
current_answer = "Waiting for AI..."

print("--- LYME AI ASSISTANT ACTIVE ---")

try:
    while True:
        raw_frame = process.stdout.read(frame_size)
        if not raw_frame or len(raw_frame) != frame_size:
            break
        
        frame = np.frombuffer(raw_frame, np.uint8).reshape((height, width, 3)).copy()

        if time.time() - last_ai_check > 10:
            try:
                current_answer = actions.query_mountain(frame, user_question)
                actions.log_incident(f"Q: {user_question} | A: {current_answer}")
            except Exception as e:
                current_answer = f"Error: {e}"
            last_ai_check = time.time()

        cv2.putText(frame, f"Question: {user_question}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"AI Answer: {current_answer}", (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        cv2.imshow("LYME Mountain Intelligence", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    process.terminate()
    cv2.destroyAllWindows()