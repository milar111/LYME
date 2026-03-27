"""
actions.py
──────────
AI frame analysis using Qwen2.5-VL-3B-Instruct (replaces BLIP VQA).

Qwen2.5-VL is a proper vision-language model — it gives natural language
answers rather than BLIP's single yes/no tokens, making alert notifications
much more descriptive.

Hardware note (MacBook Pro with Intel GPU / CPU):
  • Runs on CPU via torch — no MPS/CUDA needed.
  • Image pixels are capped at 256*28*28 to keep inference fast on CPU.
    Increase max_pixels if you have more headroom.
  • Inference will take ~5-10 s per query on CPU — the caller (_AI_INTERVAL
    in main.py) should be set to >= 8.0 s to avoid queuing up requests.

Dependencies (add to requirements.txt / pip install):
    pip install git+https://github.com/huggingface/transformers accelerate
    pip install qwen-vl-utils
"""

import os
import datetime
import cv2
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from typing import Any

# ── Globals ───────────────────────────────────────────────────────────────────

model: Any = None
processor: Any = None
device: Any = None

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

# Keep image small on CPU — reduces tokens and speeds up inference significantly.
# 256 * 28 * 28 = ~200k pixels (~450x450). Fine for security camera frames.
MIN_PIXELS = 64 * 28 * 28
MAX_PIXELS = 256 * 28 * 28


# ── Init ──────────────────────────────────────────────────────────────────────

def init_model():
    global model, processor, device

    if model is not None:
        return model, processor

    # Intel MacBook / no CUDA → CPU. MPS targets Apple Silicon only.
    device = torch.device("cpu")
    print(f"[Actions] Loading {MODEL_ID} on {device} ...")
    print("[Actions] First run will download ~6 GB — subsequent runs load from cache.")

    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,   # float32 on CPU — bfloat16 not well supported
        device_map="cpu",
    )
    model.eval()
    print("[Actions] Model ready.")
    return model, processor


# ── Inference ─────────────────────────────────────────────────────────────────

def query_frame(frame, question: str) -> str:
    """
    Run Qwen2.5-VL on a BGR OpenCV frame with the given question.
    Returns a natural-language answer string.
    """
    global model, processor, device

    if model is None:
        init_model()

    # Convert BGR → RGB PIL image
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)

    # Build the chat message Qwen2.5-VL expects
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_img},
                {"type": "text",  "text": question},
            ],
        }
    ]

    # Prepare inputs
    text_prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text_prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    # Generate — keep max_new_tokens short for speed
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=64)

    # Trim prompt tokens from output
    trimmed = [
        out[len(inp):]
        for inp, out in zip(inputs.input_ids, generated_ids)
    ]
    answer = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    return answer


# ── Local logging ─────────────────────────────────────────────────────────────

def log_incident(message: str):
    if not os.path.exists("data"):
        os.makedirs("data")
    with open("data/logs.txt", "a") as f:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] {message}\n")