import os
import cv2
from src import config
from src import actions

cap = cv2.VideoCapture(config.CAMERA_INDEX)

print("--- LYME SYSTEM ACTIVE ---")
print("Brown object detection mode")
print("Press 'q' to quit")

frame_count = 0
detection_interval = 5  # Run detection every 5 frames to reduce lag
status_text = "WAITING"
color = (255, 255, 255)

while True:
    ret, frame = cap.read()
    if not ret: 
        break

    frame = cv2.flip(frame, 1)
    frame_count += 1

    # Run brown detection every N frames (reduces lag)
    if frame_count % detection_interval == 0:
        try:
            # Save current frame and analyze for brown objects
            actions.save_frame(frame, "data/capture.jpg")
            answer = actions.query_avalanche("data/capture.jpg", provider="local")
            status_text = answer.split(" - ")[0].upper()  # Extract "yes" or "no"
            color = (0, 255, 0) if status_text == "YES" else (0, 0, 255)
            print(f"Frame {frame_count}: {answer}")
        except Exception as ex:
            status_text = "ERROR"
            color = (0, 165, 255)
            print(f"Detection failed: {ex}")
    
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

cap.release()
cv2.destroyAllWindows()