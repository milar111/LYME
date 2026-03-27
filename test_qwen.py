"""
test_qwen.py — Quick test to verify Qwen AI object detection.

Usage:
  ./.venv/bin/python test_qwen.py

This will:
  1. Open your camera and warm it up (discard first 30 frames)
  2. Capture a good frame
  3. Ask Qwen: "Is there a bottle in this image?"
  4. Print the answer
  5. Save the frame to data/test_frame.jpg so you can see what Qwen saw
"""

import cv2
import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))

from src import actions, config

ITEM = "bottle"

print(f"Loading Qwen model...")
actions.init_model()

print(f"Opening camera (source={config.CAMERA_SOURCE})...")
cap = cv2.VideoCapture(config.CAMERA_SOURCE)
if not cap.isOpened():
    print("ERROR: Cannot open camera")
    sys.exit(1)

print("Warming up camera (2 seconds)...")
warmup_start = time.time()
while time.time() - warmup_start < 2.0:
    cap.read()

ok, frame = cap.read()
cap.release()

if not ok:
    print("ERROR: Cannot read frame")
    sys.exit(1)

frame = cv2.flip(frame, 1)

os.makedirs("data", exist_ok=True)
cv2.imwrite("data/test_frame.jpg", frame)
print(f"Frame saved to data/test_frame.jpg ({frame.shape[1]}x{frame.shape[0]})")

avg = frame.mean()
if avg < 10:
    print(f"⚠️  WARNING: Frame is very dark (avg brightness={avg:.1f}/255)")
    print("   The camera may not be working properly.")
else:
    print(f"Frame brightness: {avg:.1f}/255 — looks good!")

question = (
    f"Is there a {ITEM} in this image? "
    f"Reply with ONLY this format, no extra text: "
    f"CONFIDENCE: <0-100>% | <yes or no> | <one short sentence>"
)

print(f"\nAsking Qwen: '{question}'")
print("(This takes ~10-20 seconds on CPU...)\n")

answer = actions.query_frame(frame, question)
print(f"═══════════════════════════════════════")
print(f"  Qwen answer: {answer}")
print(f"═══════════════════════════════════════")
print(f"\nCheck data/test_frame.jpg to see what Qwen was looking at.")
