import os
import datetime 
import subprocess
import torch
from typing import Any
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering

processor: Any = None
model: Any = None
device: Any = None

def init_model():
    global processor, model, device
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(device) # type: ignore
    
def query_mountain(frame, question):
    global processor, model, device
    if processor is None or model is None:
        init_model()
        
    assert processor is not None
    assert model is not None
    assert device is not None
    
    import cv2
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_frame)
    
    inputs = processor(pil_img, question, return_tensors="pt").to(device)
    
    out = model.generate(**inputs, max_new_tokens=30)
    answer = processor.decode(out[0], skip_special_tokens=True)
    
    return answer

def log_incident(message):
    if not os.path.exists("data"):
        os.makedirs("data")
    
    with open("data/logs.txt", "a") as f:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] {message}\n")