import os
import datetime
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering
from typing import Any

model: Any = None
processor: Any = None
device: Any = None

def init_model():
    global model, processor, device
    if model is not None:
        return model, processor

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model_id = "Salesforce/blip-vqa-base"
    print(f"Loading {model_id} on {device}...")

    processor = BlipProcessor.from_pretrained(model_id)
    model = BlipForQuestionAnswering.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if str(device) == "mps" else torch.float32
    ).to(device)

    model.eval()
    print("Model loaded.")
    return model, processor


def query_mountain(frame, question):
    global model, processor, device
    if model is None:
        init_model()

    import cv2
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_frame)

    inputs = processor(pil_img, question, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=50)

    answer = processor.decode(output[0], skip_special_tokens=True)
    return answer


def log_incident(message):
    if not os.path.exists("data"):
        os.makedirs("data")

    with open("data/logs.txt", "a") as f:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] {message}\n")