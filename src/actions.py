import os
import datetime 
import subprocess
import torch
from PIL import Image
from transformers import pipeline

vqa_pipeline = None

def init_model():
    global vqa_pipeline
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    vqa_pipeline = pipeline("visual-question-answering", model="Salesforce/blip-vqa-base", device=device)

def query_mountain(frame, question):
    global vqa_pipeline
    if vqa_pipeline is None:
        init_model()
    
    import cv2
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_frame)
    
    result = vqa_pipeline(pil_img, question, top_k=1)
    
    if isinstance(result, list) and len(result) > 0:
        return result[0].get('answer', str(result))
    return str(result)

def log_incident(message):
    if not os.path.exists("data"):
        os.makedirs("data")
    
    with open("data/logs.txt", "a") as f:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] {message}\n")