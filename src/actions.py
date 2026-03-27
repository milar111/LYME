"""
actions.py
──────────
AI frame analysis using Qwen2.5-VL-3B-Instruct.

  • query_frame() crops the frame to the forbidden zone bounding box
    before sending to Qwen — focuses the model and improves accuracy.
  • Prompts ask for strict structured output: CONFIDENCE: X% | yes/no | sentence
"""

import os
import datetime
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from typing import Any, Optional

model: Any = None
processor: Any = None
device: Any = None

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
MIN_PIXELS = 64  * 28 * 28
MAX_PIXELS = 256 * 28 * 28


# ── Init ──────────────────────────────────────────────────────────────────────

def init_model():
    global model, processor, device
    if model is not None:
        return model, processor

    device = torch.device("cpu")
    print(f"[Actions] Loading {MODEL_ID} on {device} ...")
    print("[Actions] First run downloads ~6 GB — subsequent runs load from cache.")

    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        device_map="cpu",
    )
    model.eval()
    print("[Actions] Model ready.")
    return model, processor


# ── Zone crop ─────────────────────────────────────────────────────────────────

def _crop_to_zone(frame: np.ndarray,
                  crop_pts: Optional[np.ndarray]) -> np.ndarray:
    """Crop frame to forbidden zone bounding box. Falls back to full frame."""
    if crop_pts is None or len(crop_pts) < 3:
        return frame
    x, y, w, h = cv2.boundingRect(crop_pts)
    fh, fw = frame.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(fw, x + w), min(fh, y + h)
    if x2 <= x1 or y2 <= y1:
        return frame
    return frame[y1:y2, x1:x2]


# ── Inference ─────────────────────────────────────────────────────────────────

def query_frame(frame: np.ndarray,
                question: str,
                crop_pts: Optional[np.ndarray] = None) -> str:
    """
    Run Qwen2.5-VL on a BGR frame cropped to the forbidden zone.
    Returns structured answer: "CONFIDENCE: X% | yes/no | description"
    """
    global model, processor, device
    if model is None:
        init_model()

    region = _crop_to_zone(frame, crop_pts)
    rgb = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_img},
                {"type": "text",  "text": question},
            ],
        }
    ]

    text_prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text_prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=60)

    trimmed = [
        out[len(inp):]
        for inp, out in zip(inputs.input_ids, generated_ids)
    ]
    return processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


# ── Logging ───────────────────────────────────────────────────────────────────

def log_incident(message: str):
    os.makedirs("data", exist_ok=True)
    with open("data/logs.txt", "a") as f:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] {message}\n")