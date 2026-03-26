import warnings
import logging
import cv2
import time
import numpy as np
from src import actions
from src.tracker import VisionTracker

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

user_question = input("What should the LLM analyze?: ")

print("Loading model... please wait.")
actions.init_model()
tracker = VisionTracker()

cap = cv2.VideoCapture(0)
last_ai_check = 0
current_llm_answer = "Waiting for LLM..."

try:
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        
        frame = cv2.flip(frame, 1)
        
        person, gesture, objects = tracker.process_frame(frame)
        
        if time.time() - last_ai_check > 10:
            try:
                current_llm_answer = actions.query_mountain(frame, user_question)
            except Exception as e:
                current_llm_answer = f"Error: {str(e)}"
            last_ai_check = time.time()

        obj_text = ", ".join(objects) if objects else "NONE"
        
        cv2.putText(frame, f"Person: {person}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Gesture: {gesture}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Objects: {obj_text}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.putText(frame, f"Question: {user_question}", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Answer: {current_llm_answer}", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Unified Vision AI", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()