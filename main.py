from contextlib import asynccontextmanager
import io
import os
import tempfile

from deepface import DeepFace
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
from PIL import Image as PILImage
import torch
import torch.nn as nn
from torchvision import transforms


# MesoNet architecture (must match training/train_model_torch.py exactly)
class MesoNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(8), nn.MaxPool2d(2, 2),
            nn.Conv2d(8, 8, 5, padding=2), nn.ReLU(), nn.BatchNorm2d(8), nn.MaxPool2d(2, 2),
            nn.Conv2d(8, 16, 5, padding=2), nn.ReLU(), nn.BatchNorm2d(16), nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 16, 5, padding=2), nn.ReLU(), nn.BatchNorm2d(16), nn.MaxPool2d(4, 4),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(16 * 8 * 8, 16),
            nn.LeakyReLU(0.1),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.classifier(self.features(x)).squeeze(1)


# Model weights and inference transforms
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_infer_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3),
])

try:
    deepfake_model = MesoNet().to(_device)
    deepfake_model.load_state_dict(
        torch.load("custom_deepfake_model_torch.pth", map_location=_device, weights_only=True)
    )
    deepfake_model.eval()
    print(f"[OK] PyTorch MesoNet loaded on {_device}")
except Exception as e:
    print(f"[ERROR] Could not load deepfake model: {e}")
    deepfake_model = None


@asynccontextmanager
async def save_temp_image(file: UploadFile):
    """Save an uploaded image to a temporary file on disk and ensure its deletion on exit."""
    suffix = os.path.splitext(file.filename or "")[1] or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        path = tmp.name
    try:
        yield path
    finally:
        if os.path.exists(path):
            os.remove(path)


def convert_numpy(obj):
    """Recursively convert NumPy data types to Python native types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert_numpy(v) for v in obj]
    return obj


app = FastAPI(title="DeepFace UI")

os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("static/index.html", "r") as f:
        return f.read()


@app.post("/api/analyze")
async def analyze_face(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")
    try:
        async with save_temp_image(file) as temp_path:
            results = DeepFace.analyze(
                img_path=temp_path,
                actions=("gender", "emotion", "race"),
                enforce_detection=False,
                detector_backend="mtcnn",
            )
            return {"success": True, "results": convert_numpy(results)}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/verify")
async def verify_faces(img1: UploadFile = File(...), img2: UploadFile = File(...)):
    if not img1.content_type.startswith("image/") or not img2.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Files provided are not images.")
    try:
        async with save_temp_image(img1) as temp_path1, save_temp_image(img2) as temp_path2:
            result = DeepFace.verify(
                img1_path=temp_path1,
                img2_path=temp_path2,
                enforce_detection=False,
                detector_backend="mtcnn",
            )
            return {"success": True, "result": convert_numpy(result)}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/deepfake")
async def detect_deepfake(file: UploadFile = File(...)):
    if deepfake_model is None:
        raise HTTPException(status_code=500, detail="Deepfake model failed to load.")
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    try:
        content = await file.read()
        img = PILImage.open(io.BytesIO(content)).convert("RGB")
        tensor = _infer_transform(img).unsqueeze(0).to(_device)

        with torch.no_grad():
            pred = deepfake_model(tensor).item()

        # Training labels: 0 = real, 1 = fake
        label = "fake" if pred >= 0.5 else "real"
        confidence = float(pred) if label == "fake" else float(1.0 - pred)

        return {
            "success": True,
            "result": {
                "label": label,
                "confidence": confidence,
            },
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
