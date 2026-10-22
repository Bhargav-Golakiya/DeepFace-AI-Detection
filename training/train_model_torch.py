"""
train_model_torch.py — GPU-Accelerated DeepFake Training Pipeline (PyTorch)
ACTIVE PIPELINE: Produces 'custom_deepfake_model_torch.pth' used by the live FastAPI application (main.py).
Merges 3 datasets and trains MesoNet on RTX 3050 GPU:
  - Dataset 1 : Kaggle RealVsFake-81k  (CSV + image files)
  - Dataset 2 : real-vs-fake           (folder structure)
  - Dataset 3 : Tiny GenImage          (Hugging Face Parquet, images embedded as bytes)
"""

import os
import io
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from PIL import Image

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[INFO] GPU detected: {gpu_name} ({vram:.1f} GB VRAM)")
    torch.backends.cudnn.benchmark = True
else:
    print("[WARNING] No GPU found — running on CPU (will be slow).")

print(f"[INFO] Using device: {device}\n")

# Configuration
DS1_BASE = r"C:\Users\bgola\Desktop\DeepFake Dataset\Dataset 1"
DS2_BASE = r"C:\Users\bgola\Desktop\DeepFake Dataset\Dataset 2\real_vs_fake\real-vs-fake"
DS3_BASE = r"C:\Users\bgola\Desktop\DeepFake Dataset\Dataset 3\data"

IMG_SIZE = 256
BATCH_SIZE = 64 if torch.cuda.is_available() else 32
EPOCHS = 10
LR = 1e-3
NUM_WORKERS = 0  # Set to 0 on Windows to avoid spawn errors

# Image transforms and data augmentation
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3),
])

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3),
])

# Dataset classes

class FilepathDataset(Dataset):
    """Loads images from disk given a filepath + label DataFrame."""
    def __init__(self, df, transform=None):
        self.df        = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        label = 1.0 if str(row["label"]).lower() == "fake" else 0.0
        try:
            img = Image.open(row["filepath"]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (0, 0, 0))
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32)


class ParquetDataset(Dataset):
    """Loads images embedded as raw bytes in Hugging Face Parquet shards."""
    def __init__(self, shard_paths, transform=None):
        self.transform = transform
        self.samples   = []   # list of (bytes, label)

        total = len(shard_paths)
        for i, fpath in enumerate(shard_paths):
            print(f"   Loading parquet shard {i+1}/{total}: {os.path.basename(fpath)}", end="\r")
            df = pd.read_parquet(fpath, columns=["image", "label"])
            for _, row in df.iterrows():
                self.samples.append((row["image"]["bytes"], int(row["label"])))
        print(f"\n   Loaded {len(self.samples):,} images from {total} parquet shards.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        raw, label = self.samples[idx]
        try:
            img = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception:
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (0, 0, 0))
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(float(label), dtype=torch.float32)


# Load datasets

# Dataset 1 — CSV
print("[1/4] Loading Dataset 1 (Kaggle RealVsFake CSV)...")

def fix_ds1_path(p):
    parts = str(p).split("RealVsFake/RealVsFake/")
    if len(parts) > 1:
        return os.path.join(DS1_BASE, "RealVsFake", "RealVsFake", parts[1])
    return p

ds1_train_df = pd.read_csv(os.path.join(DS1_BASE, "train_labels.csv"), dtype=str)
ds1_val_df   = pd.read_csv(os.path.join(DS1_BASE, "val_labels.csv"),   dtype=str)
for df in [ds1_train_df, ds1_val_df]:
    df["filepath"] = df["filepath"].apply(fix_ds1_path)
    df["label"]    = df["label"].map({"0": "real", "1": "fake"})
ds1_train_df = ds1_train_df[ds1_train_df["filepath"].apply(os.path.exists)]
ds1_val_df   = ds1_val_df[ds1_val_df["filepath"].apply(os.path.exists)]
print(f"   Train: {len(ds1_train_df):,}  |  Val: {len(ds1_val_df):,}")

# Dataset 2 — Folders
print("[2/4] Loading Dataset 2 (real-vs-fake folders)...")

def scan_folder(split):
    paths, labels = [], []
    for cls in ["real", "fake"]:
        folder = os.path.join(DS2_BASE, split, cls)
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            for p in glob.glob(os.path.join(folder, ext)):
                paths.append(p)
                labels.append(cls)
    return pd.DataFrame({"filepath": paths, "label": labels})

ds2_train_df = scan_folder("train")
ds2_val_df   = scan_folder("valid")
print(f"   Train: {len(ds2_train_df):,}  |  Val: {len(ds2_val_df):,}")

# Merge DS1 + DS2 into FilepathDataset
combined_train_df = pd.concat([ds1_train_df, ds2_train_df], ignore_index=True).sample(frac=1, random_state=42)
combined_val_df   = pd.concat([ds1_val_df,   ds2_val_df],   ignore_index=True).sample(frac=1, random_state=42)

ds12_train = FilepathDataset(combined_train_df, transform=train_tf)
ds12_val   = FilepathDataset(combined_val_df,   transform=val_tf)
print(f"   Combined DS1+DS2  Train: {len(ds12_train):,}  |  Val: {len(ds12_val):,}")

# Dataset 3 — Parquet
print("\n[3/4] Loading Dataset 3 (Tiny GenImage Parquet)...")
ds3_train_shards = sorted(glob.glob(os.path.join(DS3_BASE, "train-*.parquet")))
ds3_val_shards   = sorted(glob.glob(os.path.join(DS3_BASE, "validation-*.parquet")))

ds3_train = ParquetDataset(ds3_train_shards, transform=train_tf)
ds3_val   = ParquetDataset(ds3_val_shards,   transform=val_tf)

# Merge all
print("\n[4/4] Merging all datasets...")
full_train = ConcatDataset([ds12_train, ds3_train])
full_val   = ConcatDataset([ds12_val,   ds3_val])
print(f"   Total Train: {len(full_train):,}  |  Total Val: {len(full_val):,}")

train_loader = DataLoader(full_train, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())
val_loader   = DataLoader(full_val,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())

# MesoNet Architecture (PyTorch)
class MesoNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3,  8,  3, padding=1), nn.ReLU(), nn.BatchNorm2d(8),  nn.MaxPool2d(2, 2),
            # Block 2
            nn.Conv2d(8,  8,  5, padding=2), nn.ReLU(), nn.BatchNorm2d(8),  nn.MaxPool2d(2, 2),
            # Block 3
            nn.Conv2d(8,  16, 5, padding=2), nn.ReLU(), nn.BatchNorm2d(16), nn.MaxPool2d(2, 2),
            # Block 4
            nn.Conv2d(16, 16, 5, padding=2), nn.ReLU(), nn.BatchNorm2d(16), nn.MaxPool2d(4, 4),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(16 * 8 * 8, 16),  # 256x256 → /2 /2 /2 /4 = 8x8, 16ch → 1024
            nn.LeakyReLU(0.1),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.classifier(self.features(x)).squeeze(1)


model = MesoNet().to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"\nMesoNet loaded: {total_params:,} parameters | Device: {device}")

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=2, verbose=True)

# Training Loop
history = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}
best_val_acc = 0.0
patience_counter = 0
PATIENCE = 3

print(f"\nTraining started on {device} — {EPOCHS} epochs, batch size {BATCH_SIZE}\n")

for epoch in range(1, EPOCHS + 1):
    # --- Train ---
    model.train()
    train_loss, train_correct, train_total = 0.0, 0, 0

    for batch_idx, (imgs, labels) in enumerate(train_loader):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        preds = model(imgs)
        loss  = criterion(preds, labels)
        loss.backward()
        optimizer.step()

        train_loss    += loss.item() * imgs.size(0)
        predicted      = (preds >= 0.5).float()
        train_correct += (predicted == labels).sum().item()
        train_total   += imgs.size(0)

        if (batch_idx + 1) % 100 == 0:
            running_acc = 100.0 * train_correct / train_total
            print(f"  Epoch {epoch}/{EPOCHS} | Batch {batch_idx+1}/{len(train_loader)} "
                  f"| Loss: {loss.item():.4f} | Acc: {running_acc:.1f}%", end="\r")

    avg_train_loss = train_loss / train_total
    avg_train_acc  = 100.0 * train_correct / train_total

    # --- Validate ---
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds      = model(imgs)
            loss       = criterion(preds, labels)
            val_loss  += loss.item() * imgs.size(0)
            predicted  = (preds >= 0.5).float()
            val_correct += (predicted == labels).sum().item()
            val_total   += imgs.size(0)

    avg_val_loss = val_loss / val_total
    avg_val_acc  = 100.0 * val_correct / val_total

    history["train_loss"].append(avg_train_loss)
    history["train_acc"].append(avg_train_acc)
    history["val_loss"].append(avg_val_loss)
    history["val_acc"].append(avg_val_acc)

    scheduler.step(avg_val_loss)

    print(f"\nEpoch {epoch}/{EPOCHS}  "
          f"Train Loss: {avg_train_loss:.4f}  Train Acc: {avg_train_acc:.2f}%  "
          f"| Val Loss: {avg_val_loss:.4f}  Val Acc: {avg_val_acc:.2f}%")

    # Save best model
    if avg_val_acc > best_val_acc:
        best_val_acc = avg_val_acc
        torch.save(model.state_dict(), "custom_deepfake_model_torch.pth")
        print(f"  Best model saved (Val Acc: {best_val_acc:.2f}%)")
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping triggered after {epoch} epochs.")
            break

# Plot Metrics
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(history["train_acc"], label="Train Acc")
ax1.plot(history["val_acc"],   label="Val Acc")
ax1.set_title("Accuracy")
ax1.legend()
ax1.set_xlabel("Epoch")

ax2.plot(history["train_loss"], label="Train Loss")
ax2.plot(history["val_loss"],   label="Val Loss")
ax2.set_title("Loss")
ax2.legend()
ax2.set_xlabel("Epoch")

plt.tight_layout()
plt.savefig("training_metrics.png", dpi=150)

print("\nTraining complete.")
print(f"   Best Val Accuracy : {best_val_acc:.2f}%")
print("   Model saved       : custom_deepfake_model_torch.pth")
print("   Chart saved       : training_metrics.png")
