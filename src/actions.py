import os
import datetime
import torch
import numpy as np
from PIL import Image
from typing import Any
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import cv2

model: Any = None
processor: Any = None
device: Any = None

def init_model():
    global model, processor, device

    
    model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    device = "cpu"

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float32,   
        local_files_only=True,      
    ).to("cpu")

    processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)

def query_frame(frame, question, crop_pts=None):
    global model, processor, device
    if model is None or processor is None:
        init_model()

    img = frame.copy()

    
    if crop_pts is not None and len(crop_pts) >= 3:
        x, y, cw, ch = cv2.boundingRect(np.array(crop_pts, dtype=np.int32))
        x, y = max(0, x), max(0, y)
        cropped = img[y:y + ch, x:x + cw]
        if cropped.size > 0:
            img = cropped

    rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_frame)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_img},
                {"type": "text",  "text": question},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)

    generated_ids = model.generate(**inputs, max_new_tokens=64)
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    return output_text[0]


query_mountain = query_frame

def log_incident(message):
    if not os.path.exists("data"):
        os.makedirs("data")
    
    with open("data/logs.txt", "a") as f:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] {message}\n")