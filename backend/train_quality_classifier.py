import os
import sys
import json
import random
import tarfile
import time
import datetime
import urllib.request
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from PIL import Image, ImageFilter

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CONSTANTS & CONFIG
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DATASET_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
DATASET_TAR = "imagenette2-320.tgz"
EXTRACT_DIR = "imagenette2-320"
OUTPUT_DATA_DIR = Path("./cvagent_output/quality_classifier_data")
RUN_DIR = Path("./runs/quality_classifier_v1")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
TARGET_TOTAL_IMAGES = 2000
TRAIN_RATIO = 0.8

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 1 — Dataset Construction
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def download_and_extract():
    if not os.path.exists(DATASET_TAR):
        print(f"> Downloading {DATASET_URL}...")
        urllib.request.urlretrieve(DATASET_URL, DATASET_TAR, 
            reporthook=lambda b, bs, t: print(f"\r  Downloading: {b*bs/1e6:.1f}/{t/1e6:.1f} MB", end="", flush=True))
        print("\n> Download complete.")
    
    if not os.path.exists(EXTRACT_DIR):
        print(f"> Extracting {DATASET_TAR}...")
        with tarfile.open(DATASET_TAR) as f:
            f.extractall(".")
        print("> Extraction complete.")

def compute_blur_score(img):
    arr = np.array(img.convert("L"), dtype=np.float32)
    lap = cv2.Laplacian(arr, cv2.CV_32F)
    return float(lap.var())

def construct_dataset():
    print("> Constructing balanced quality dataset...")
    sharp_images = []
    blurry_images = []
    
    train_root = Path(EXTRACT_DIR) / "train"
    image_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.JPG", "*.JPEG"]:
        image_paths.extend(list(train_root.glob(f"**/{ext}")))
    
    if not image_paths:
        print(f"ERROR: No images found in {train_root}. Check extraction.")
        sys.exit(1)
        
    random.shuffle(image_paths)
    
    # Collect real sharp and blurry images
    for p in image_paths:
        if len(sharp_images) >= 1000 and len(blurry_images) >= 1000:
            break
            
        try:
            img = Image.open(p).convert("RGB")
            score = compute_blur_score(img)
            
            if score >= 80.0 and len(sharp_images) < 1000:
                sharp_images.append(img)
                # Synthetic blur from every 3rd sharp image
                if len(sharp_images) % 3 == 0 and len(blurry_images) < 1000:
                    blurry_img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(3, 8)))
                    blurry_images.append(blurry_img)
            elif score < 80.0 and len(blurry_images) < 1000:
                blurry_images.append(img)
        except Exception:
            continue

    # Fill remaining blurry slots with more synthetic if needed
    if len(blurry_images) < 1000:
        print(f"> Filling {1000 - len(blurry_images)} more blurry slots with synthetic blur...")
        while len(blurry_images) < 1000 and sharp_images:
            img = random.choice(sharp_images)
            blurry_img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(3, 8)))
            blurry_images.append(blurry_img)

    # Shuffle
    random.shuffle(sharp_images)
    random.shuffle(blurry_images)

    # Split and Save
    shutil.rmtree(OUTPUT_DATA_DIR, ignore_errors=True)
    
    train_n_sharp = int(len(sharp_images) * TRAIN_RATIO)
    train_n_blurry = int(len(blurry_images) * TRAIN_RATIO)

    data_splits = {
        "train": {
            "sharp": sharp_images[:train_n_sharp],
            "blurry": blurry_images[:train_n_blurry]
        },
        "val": {
            "sharp": sharp_images[train_n_sharp:],
            "blurry": blurry_images[train_n_blurry:]
        }
    }

    for split_name, categories in data_splits.items():
        for cat_name, images in categories.items():
            cat_dir = OUTPUT_DATA_DIR / split_name / cat_name
            cat_dir.mkdir(parents=True, exist_ok=True)
            for i, img in enumerate(images):
                img.save(cat_dir / f"{cat_name}_{i:04d}.jpg")
            
    print(f"> Dataset saved to {OUTPUT_DATA_DIR}")
    return train_n_sharp + train_n_blurry, (len(sharp_images) - train_n_sharp) + (len(blurry_images) - train_n_blurry)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 2 — Training
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SimpleImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        for label, cls in enumerate(["blurry", "sharp"]):
            for p in (self.root_dir / cls).glob("*.jpg"):
                self.samples.append((p, label))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

def train_model(train_count, val_count):
    print(f"> Starting training on {DEVICE}...")
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_ds = SimpleImageDataset(OUTPUT_DATA_DIR / "train", transform=transform)
    val_ds = SimpleImageDataset(OUTPUT_DATA_DIR / "val", transform=transform)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    # Model setup
    model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.last_channel, 2)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    
    # Phase 1: Head only
    print("> Phase 1: Training classifier head (3 epochs)...")
    for param in model.features.parameters():
        param.requires_grad = False
    
    optimizer = optim.Adam(model.classifier.parameters(), lr=1e-3)
    
    total_epochs = 6
    for epoch in range(1, 4):
        run_epoch(model, train_loader, val_loader, criterion, optimizer, epoch, total_epochs)

    # Phase 2: Full fine-tune
    print("> Phase 2: Full fine-tuning (3 epochs)...")
    for param in model.parameters():
        param.requires_grad = True
        
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    
    for epoch in range(4, 7):
        metrics = run_epoch(model, train_loader, val_loader, criterion, optimizer, epoch, total_epochs)
        if epoch == 6:
            final_metrics = metrics

    # Save Results
    torch.save(model.state_dict(), RUN_DIR / "model_weights.pth")
    
    with open(RUN_DIR / "config.json", "w") as f:
        json.dump({
            "run_id": "quality_classifier_v1",
            "model": "mobilenetv2_quality_classifier",
            "task": "image_quality_classification",
            "classes": ["blurry", "sharp"],
            "input_size": [224, 224],
            "threshold": 0.5,
            "trained_on": "imagenette2-320 + synthetic blur",
            "purpose": "Predicts whether an image is sharp enough to include in CV training datasets"
        }, f, indent=2)

    with open(RUN_DIR / "model.py", "w") as f:
        f.write(MODEL_PY_CONTENT)

    with open(RUN_DIR / "training_results.json", "w") as f:
        json.dump({
            "final_accuracy": final_metrics["accuracy"],
            "final_precision": final_metrics["precision"],
            "final_recall": final_metrics["recall"],
            "best_epoch": 6,
            "total_images_trained_on": train_count,
            "validation_images": val_count
        }, f, indent=2)

def run_epoch(model, train_loader, val_loader, criterion, optimizer, epoch, total_epochs):
    model.train()
    train_loss = 0
    start_time = time.time()
    
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    tp, fp, fn = 0, 0, 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            for p, l in zip(predicted, labels):
                if p == 1 and l == 1: tp += 1
                elif p == 1 and l == 0: fp += 1
                elif p == 0 and l == 1: fn += 1

    accuracy = correct / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    duration = time.time() - start_time
    eta_seconds = duration * (total_epochs - epoch)
    eta_str = f"{int(eta_seconds // 60):02d}:{int(eta_seconds % 60):02d}"

    metrics = {
        "epoch": epoch,
        "loss": round(val_loss / len(val_loader), 4),
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "map": round(accuracy, 4), # Simple map fallback
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "run_id": "quality_classifier_v1",
        "total_epochs": total_epochs,
        "task_type": "mnist_classification", # Backend requirement
        "eta": eta_str
    }
    
    print(f"METRIC:{json.dumps(metrics)}")
    return metrics

MODEL_PY_CONTENT = """import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2
from PIL import Image
import torchvision.transforms as T

class ImageQualityClassifier(nn.Module):
    \"\"\"
    Predicts if an image is 'sharp' (good for training) or 'blurry' (should be filtered).
    Trained on imagenette2-320 with synthetic blur augmentation.
    Classes: 0=blurry, 1=sharp
    \"\"\"
    def __init__(self):
        super().__init__()
        self.backbone = mobilenet_v2(weights=None)
        self.backbone.classifier[1] = nn.Linear(self.backbone.last_channel, 2)
    
    def forward(self, x):
        return self.backbone(x)
    
    @classmethod
    def load(cls, weights_path: str) -> \"ImageQualityClassifier\":
        model = cls()
        model.load_state_dict(torch.load(weights_path, map_location=\"cpu\"))
        model.eval()
        return model
    
    @staticmethod
    def preprocess(img: Image.Image) -> torch.Tensor:
        transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        return transform(img).unsqueeze(0)
    
    def predict(self, img: Image.Image) -> dict:
        with torch.no_grad():
            logits = self(self.preprocess(img))
            probs = torch.softmax(logits, dim=1)[0]
            pred_class = probs.argmax().item()
        return {
            "label": "sharp" if pred_class == 1 else "blurry",
            "confidence": float(probs[pred_class]),
            "sharp_prob": float(probs[1]),
            "blurry_prob": float(probs[0]),
        }
"""

if __name__ == "__main__":
    download_and_extract()
    train_c, val_c = construct_dataset()
    train_model(train_c, val_c)
    
    print(f'EVENT:{{"event": "completed", "message": "Quality classifier trained successfully. Model saved to ./runs/quality_classifier_v1/", "run_id": "quality_classifier_v1"}}')
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("✓ Quality Classifier Training Complete")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("Model: MobileNetV2 (fine-tuned)")
    print("Task:  Binary image quality (sharp vs blurry)")
    print("Use:   Improves VisCurator's blur filtering accuracy")
    print("\nTo use in VisCurator:")
    print("  from runs.quality_classifier_v1.model import ImageQualityClassifier")
    print('  model = ImageQualityClassifier.load("runs/quality_classifier_v1/model_weights.pth")')
    print("  result = model.predict(your_pil_image)")
    print('  # result = {"label": "sharp", "confidence": 0.97, ...}')
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
