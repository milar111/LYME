import os
import cv2
import subprocess
import numpy as np
from src import config
from src import actions

# Use FFmpeg to handle HLS/M3U8 stream
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

print("Starting FFmpeg stream process...")
process = subprocess.Popen(
    ffmpeg_cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    bufsize=10**8
)

print("--- LYME SYSTEM ACTIVE ---")
print("Brown object detection mode")
print("Connecting to Pamporo camera stream...")
print("Press 'q' to quit")

frame_count = 0
detection_interval = 5  # Run detection every 5 frames to reduce lag
status_text = "WAITING"
color = (255, 255, 255)

# Frame dimensions (must match FFmpeg output: 1280x720)
width, height = 1280, 720
frame_size = width * height * 3

print("Ready. Stream starting...\n")

import sys
try:
    while True:
        try:
            raw_frame = process.stdout.read(frame_size)
            if not raw_frame or len(raw_frame) != frame_size:
                print(f"EOF or incomplete frame ({len(raw_frame)}/{frame_size} bytes)")
                break
            
            frame = np.frombuffer(raw_frame, np.uint8).reshape((height, width, 3))
            frame = frame.copy()  # Make a writable copy
            frame_count += 1
            print(f"✓ Frame {frame_count} received", end="\r")
            sys.stdout.flush()

            # Run brown detection every N frames (reduces lag)
            if frame_count % detection_interval == 0:
                try:
                    # Save current frame and analyze for brown objects
                    actions.save_frame(frame, "data/capture.jpg")
                    answer = actions.query_avalanche("data/capture.jpg", provider="local")
                    status_text = answer.split(" - ")[0].upper()  # Extract "yes" or "no"
                    color = (0, 255, 0) if status_text == "YES" else (0, 0, 255)
                    print(f"\nFrame {frame_count}: {answer}")
                except Exception as ex:
                    status_text = "ERROR"
                    color = (0, 165, 255)
                    print(f"\nDetection failed: {ex}")
            
            # Display status overlay
            cv2.putText(frame, f"Brown Detection: {status_text}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(frame, f"Frame: {frame_count}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Single window display
            cv2.imshow("LYME Brown Detector", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        except Exception as e:
            print(f"Frame processing error: {e}")
            break

finally:
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    cv2.destroyAllWindows()
    print("\nStream closed.")