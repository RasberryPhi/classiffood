
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os
import requests

def load_model():
    model_path = "app/model.pt"
    if not os.path.exists(model_path):
        print("Downloading model...")
        url = "https://huggingface.co/Kamilatr/freshwinsmodel/resolve/main/best.pt"  # Replace with actual link
        r = requests.get(url)
        with open(model_path, "wb") as f:
            f.write(r.content)
    return YOLO(model_path)

async def run_inference(file, model):
    image = Image.open(file.file)
    results = model(image)
    probs = results[0].probs.data.cpu().numpy()
    names = results[0].names
    top3 = sorted(zip(names, probs), key=lambda x: x[1], reverse=True)[:3]
    return {name: float(f"{prob:.4f}") for name, prob in top3}
