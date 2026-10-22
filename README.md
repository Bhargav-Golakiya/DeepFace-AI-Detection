# DeepFace AI & DeepFake Detection

A local web application for facial attribute analysis, facial identity verification, and deepfake image detection.

## Project Structure

```text
├── main.py                          # FastAPI backend and MesoNet inference service
├── requirements.txt                 # Python dependencies
├── README.md                        # Documentation
├── custom_deepfake_model_torch.pth  # PyTorch model weights (loaded by main.py)
├── custom_deepfake_model.h5         # Legacy TensorFlow model weights (reference only)
├── training_metrics.png             # Training loss and accuracy plots
├── static/
│   ├── index.html                   # Login and registration landing page
│   ├── app.html                     # Main dashboard interface
│   ├── app.js                       # Frontend interaction and API client
│   └── styles.css                   # Application styles
└── training/
    ├── train_model_torch.py         # PyTorch GPU training pipeline (active)
    └── train_model.py               # TensorFlow/Keras training pipeline (legacy reference)
```

## Quick Start

1. **Activate your virtual environment and install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the FastAPI server:**
   ```bash
   python -m uvicorn main:app --host 127.0.0.1 --port 8000
   ```

3. **Open the application:**
   Navigate to [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

## API Endpoints

- `GET /` — Serves the login landing page.
- `GET /static/*` — Serves static dashboard assets (`app.html`, `app.js`, `styles.css`).
- `POST /api/analyze` — Analyzes gender, emotion, and ethnicity using DeepFace (`multipart/form-data` with an image `file`).
- `POST /api/verify` — Compares two faces to verify identity using DeepFace (`multipart/form-data` with `img1` and `img2`).
- `POST /api/deepfake` — Classifies image authenticity (real vs. deepfake) via PyTorch MesoNet (`multipart/form-data` with an image `file`).

## Models and Training

- **Active Runtime Model:** `custom_deepfake_model_torch.pth` is loaded at startup by `main.py` onto CUDA (or CPU fallback). It implements a 4-block MesoNet architecture trained with binary cross-entropy.
- **Active Training Script:** `training/train_model_torch.py` trains the PyTorch MesoNet across three datasets (Kaggle RealVsFake, folder dataset, and Hugging Face Parquet shards).
- **Legacy Artifact:** `custom_deepfake_model.h5` and `training/train_model.py` are the original TensorFlow implementation, retained for reference and reproducibility.
