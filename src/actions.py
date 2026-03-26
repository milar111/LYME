import os
import datetime
import subprocess
import base64

try:
    import openai
except ImportError:
    openai = None


def trigger_alarm():
    # This plays the "Ping" sound found on every Mac
    subprocess.run(["afplay", "/System/Library/Sounds/Ping.aiff"], check=True)
    
    # Log the breach to a file
    with open("data/logs.txt", "a") as f:
        now = datetime.datetime.now().strftime("%H:%M:%S")
        f.write(f"[{now}] WARNING: Optical Link Interrupted!\n")


def signal_clear():
    # Logic for when the laser is hitting correctly
    pass


def save_frame(frame, path="data/capture.jpg"):
    """Save a frame to disk and return the path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    import cv2
    cv2.imwrite(path, frame)
    return path


def _get_local_avalanche_score(image_path):
    import cv2
    import numpy as np

    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Could not load image at {image_path}")

    # Convert BGR to HSV for color detection
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define brown color range in HSV
    # Brown is typically: Hue 10-25 (reddish-brown), Saturation 50-255, Value 50-200
    lower_brown = np.array([8, 50, 50])
    upper_brown = np.array([25, 255, 200])
    
    # Create mask for brown pixels
    brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
    
    # Calculate brown pixel ratio
    brown_ratio = np.sum(brown_mask > 0) / (brown_mask.shape[0] * brown_mask.shape[1])
    
    # If more than 5% of image is brown, consider it detected
    is_detected = brown_ratio > 0.05

    return is_detected, brown_ratio



def query_avalanche(image_path, provider="local", model="gemini-pro-1.0"):
    """Send an image to an LLM and ask if an avalanche is present."""
    if provider == "local":
        # Test mode: override with env var LYME_TEST_RESPONSE=yes or LYME_TEST_RESPONSE=no
        test_response = os.getenv("LYME_TEST_RESPONSE")
        if test_response in ("yes", "no"):
            return f"{test_response} - TEST MODE (from LYME_TEST_RESPONSE env var)"
        
        # Otherwise use brown detection heuristic
        is_detected, brown_ratio = _get_local_avalanche_score(image_path)
        status = "yes" if is_detected else "no"
        return f"{status} - brown_ratio={brown_ratio:.3f}"

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    prompt = (
        "You are an avalanche risk assistant. Review the following image (base64-encoded) "
        "and answer with 'yes' or 'no' plus a short rationale if it appears to show an avalanche.\n\n"
        "Image (base64):\n" + image_b64
    )

    if provider == "openai":
        if openai is None:
            raise RuntimeError("openai package is not installed. Run: pip install openai")

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable is required")

        openai.api_key = api_key
        completion = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert in snow and avalanche detection."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=150,
        )

        return completion["choices"][0]["message"]["content"].strip()

    elif provider == "vertex":
        try:
            from google.cloud import aiplatform
            from google.cloud.aiplatform.gapic import PredictionServiceClient
            from google.cloud.aiplatform.gapic.types import PredictRequest
        except ImportError:
            raise RuntimeError("google-cloud-aiplatform not installed. Run: pip install google-cloud-aiplatform")

        project = os.getenv("GCP_PROJECT")
        location = os.getenv("GCP_LOCATION", "us-central1")
        if not project:
            raise RuntimeError("GCP_PROJECT environment variable is required")

        client = PredictionServiceClient()
        endpoint = f"projects/{project}/locations/{location}/endpoints/{model}"

        instance = {
            "content": prompt,
            "mime_type": "text/plain",
        }

        response = client.predict(
            endpoint=endpoint,
            instances=[instance],
            parameters={},
        )

        if response.predictions:
            return str(response.predictions[0])
        return "No prediction returned"

    else:
        raise ValueError("provider must be 'local', 'openai' or 'vertex'")



def capture_and_check_avalanche(frame, image_path="data/capture.jpg", provider=None, model=None):
    path = save_frame(frame, image_path)
    if provider is None:
        provider = os.getenv("LYME_AI_PROVIDER", "local")
    if model is None:
        model = os.getenv("LYME_AI_MODEL", "gemini-pro-1.0")
    return query_avalanche(path, provider=provider, model=model)
