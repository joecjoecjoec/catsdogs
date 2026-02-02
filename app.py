# app.py
import io
import json
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from torchvision import models, transforms


# -------------------------------------------------
# Paths
# -------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "model.pt"
META_PATH = MODELS_DIR / "meta.json"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Missing model file: {MODEL_PATH}")
if not META_PATH.exists():
    raise FileNotFoundError(f"Missing meta file: {META_PATH}")

# -------------------------------------------------
# Load meta
# -------------------------------------------------
meta = json.loads(META_PATH.read_text(encoding="utf-8"))
ARCH = meta["arch"]                      # "mobilenet_v3_small"
IMG_SIZE = int(meta["img_size"])         # 224
CLASS_NAMES: List[str] = meta["class_names"]  # ["cats", "dogs"]
NORM_MEAN = meta["normalize_mean"]
NORM_STD = meta["normalize_std"]

# -------------------------------------------------
# Build model (must match training)
# -------------------------------------------------
def build_model(arch: str, num_classes: int) -> nn.Module:
    if arch == "mobilenet_v3_small":
        m = models.mobilenet_v3_small(weights=None)  # weights=None because we load our own weights
        in_features = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_features, num_classes)
        return m
    raise ValueError(f"Unknown arch in meta.json: {arch}")


DEVICE = torch.device("cpu")

model = build_model(ARCH, num_classes=len(CLASS_NAMES)).to(DEVICE)   

ckpt = torch.load(MODEL_PATH, map_location=DEVICE)                  

if isinstance(ckpt, dict) and "state_dict" in ckpt:
    state_dict = ckpt["state_dict"]
else:
    state_dict = ckpt

model.load_state_dict(state_dict, strict=False)
model.eval()


preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
])

# -------------------------------------------------
# FastAPI app (THIS MUST BE NAMED `app`)
# -------------------------------------------------
app = FastAPI(title="Cats vs Dogs Classifier")

@app.get("/health")
def health():
    return {"status": "ok", "arch": ARCH, "classes": CLASS_NAMES}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # read image
    content = await file.read()
    img = Image.open(io.BytesIO(content)).convert("RGB")

    x = preprocess(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().tolist()

     
    probs_t = torch.tensor(probs)

    max_prob = float(probs_t.max().item())
    pred_idx = int(probs_t.argmax().item())
    pred_class = CLASS_NAMES[pred_idx]

    
    if max_prob < 0.60:
        pred_class = "uncertain"

    return {
        "class": pred_class,
        "prob": max_prob,
        "probs": {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
    }