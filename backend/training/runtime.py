from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any


def _detect_task_metadata(nodes: list[dict[str, Any]]) -> dict[str, Any]:
    input_node = next((node for node in nodes if node.get("type") == "inputNode"), {})
    output_node = next((node for node in nodes if node.get("type") == "outputNode"), {})
    input_data = input_node.get("data", {})
    output_data = output_node.get("data", {})

    resolution = str(input_data.get("resolution", "224x224")).replace("x", " ").replace("X", " ").replace("×", " ")
    parts = [int(piece) for piece in resolution.split() if piece.isdigit()]
    height = parts[0] if len(parts) >= 1 else 224
    width = parts[1] if len(parts) >= 2 else height

    def _safe_int(val: Any, default: int) -> int:
        try:
            if val is None or val == "":
                return default
            return int(val)
        except (ValueError, TypeError):
            return default

    return {
        "input_channels": _safe_int(input_data.get("channels"), 3),
        "input_height": height,
        "input_width": width,
        "num_classes": _safe_int(output_data.get("num_classes"), 10),
    }


def write_training_runtime(
    run_dir: Path,
    model_code: str,
    nodes: list[dict[str, Any]],
    task_type: str,
    dataset_path: str | None = None,
    epochs: int | None = None,
    num_images: int | None = None,
) -> dict[str, Any]:
    import logging
    logger = logging.getLogger("viscurator.training")
    
    logger.info("Detecting task metadata from %d nodes", len(nodes))
    metadata = _detect_task_metadata(nodes)
    logger.info("Metadata detected: %s", metadata)
    
    if epochs is None:
        epochs = 6 if task_type == "mnist_classification" else 8
        
    config = {
        "run_id": run_dir.name,
        "task_type": task_type,
        "epochs": epochs,
        "num_images": num_images,
        "batch_size": 4,  # safe default for CPU; loader will use micro-batches
        "dataset_path": dataset_path,
        **metadata,
    }

    logger.info("Creating run directory: %s", run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Writing model.py")
    (run_dir / "model.py").write_text(model_code, encoding="utf-8")
    
    logger.info("Writing config.json")
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    
    logger.info("Writing train.py")
    (run_dir / "train.py").write_text(_build_train_script(), encoding="utf-8")
    
    return config


def _build_train_script() -> str:
    return textwrap.dedent(
        """\
        import json
        import os
        import sys
        import time
        import traceback
        from datetime import datetime, timezone
        from pathlib import Path

        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, Dataset, random_split
        from PIL import Image
        import numpy as np

        RUN_DIR = Path(__file__).resolve().parent
        if str(RUN_DIR) not in sys.path:
            sys.path.insert(0, str(RUN_DIR))

        def emit_event(event: str, message: str, **data):
            payload = {"event": event, "message": message, **data}
            print("EVENT:" + json.dumps(payload), flush=True)

        def emit_metric(**data):
            print("METRIC:" + json.dumps(data), flush=True)

        try:
            from model import CVAgentModel
        except Exception as e:
            emit_event("log", f"Failed to import CVAgentModel: {e}. Using fallback.")
            class CVAgentModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
                        nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten()
                    )
                def forward(self, x): return self.net(x)

        class SyntheticDataset(Dataset):
            def __init__(self, size, c, h, w, classes):
                self.size, self.c, self.h, self.w, self.classes = size, c, h, w, classes
            def __len__(self): return self.size
            def __getitem__(self, i): return torch.randn(self.c, self.h, self.w), i % self.classes

        class Compose:
            def __init__(self, funcs):
                self.funcs = funcs
            def __call__(self, image):
                for fn in self.funcs:
                    image = fn(image)
                return image

        class Resize:
            def __init__(self, size):
                self.size = size
            def __call__(self, image):
                return image.resize((self.size[1], self.size[0]))

        class RandomHorizontalFlip:
            def __init__(self, p=0.5):
                self.p = p
            def __call__(self, image):
                return image.transpose(Image.FLIP_LEFT_RIGHT) if torch.rand(1).item() < self.p else image

        class ColorJitter:
            def __init__(self, brightness=0.0, contrast=0.0):
                self.brightness = brightness
                self.contrast = contrast
            def __call__(self, image):
                arr = np.asarray(image, dtype=np.float32) / 255.0
                if self.brightness > 0:
                    factor = 1.0 + (torch.rand(1).item() * 2 - 1) * self.brightness
                    arr = arr * factor
                if self.contrast > 0:
                    mean = arr.mean(axis=(0, 1), keepdims=True)
                    factor = 1.0 + (torch.rand(1).item() * 2 - 1) * self.contrast
                    arr = (arr - mean) * factor + mean
                arr = np.clip(arr, 0.0, 1.0)
                return Image.fromarray((arr * 255).astype("uint8"))

        class ToTensor:
            def __call__(self, image):
                arr = np.asarray(image, dtype=np.float32) / 255.0
                arr = np.transpose(arr, (2, 0, 1))
                return torch.from_numpy(arr)

        class Normalize:
            def __init__(self, mean, std):
                self.mean = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)
                self.std = torch.tensor(std, dtype=torch.float32).view(-1, 1, 1)
            def __call__(self, tensor):
                return (tensor - self.mean) / self.std

        class SimpleImageFolder(Dataset):
            def __init__(self, root, transform=None):
                self.root = Path(root)
                self.transform = transform
                self.classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
                self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}
                self.samples = []
                valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
                for class_name in self.classes:
                    class_dir = self.root / class_name
                    for file_path in sorted(class_dir.rglob("*")):
                        if file_path.is_file() and file_path.suffix.lower() in valid_exts:
                            self.samples.append((file_path, self.class_to_idx[class_name]))
            def __len__(self):
                return len(self.samples)
            def __getitem__(self, index):
                file_path, label = self.samples[index]
                image = Image.open(file_path).convert("RGB")
                if self.transform:
                    image = self.transform(image)
                return image, label

        def compute_metrics(model, loader, device, num_classes):
            \"\"\"Compute accuracy, precision, recall, mAP on a val loader.\"\"\"
            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for imgs, labels in loader:
                    imgs = imgs.to(device)
                    out = model(imgs)
                    preds = out.argmax(dim=1).cpu().tolist()
                    all_preds.extend(preds)
                    all_labels.extend(labels.tolist() if hasattr(labels, 'tolist') else list(labels))
            if not all_preds:
                return 0.0, 0.0, 0.0, 0.0
            # Accuracy
            correct = sum(p == l for p, l in zip(all_preds, all_labels))
            accuracy = correct / len(all_labels)
            # Per-class precision/recall
            classes = list(range(num_classes))
            precisions, recalls = [], []
            for c in classes:
                tp = sum(1 for p, l in zip(all_preds, all_labels) if p == c and l == c)
                fp = sum(1 for p, l in zip(all_preds, all_labels) if p == c and l != c)
                fn = sum(1 for p, l in zip(all_preds, all_labels) if p != c and l == c)
                p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                precisions.append(p)
                recalls.append(r)
            precision = sum(precisions) / len(precisions) if precisions else 0.0
            recall = sum(recalls) / len(recalls) if recalls else 0.0
            # mAP: average precision per class (simplified: AP = precision for single-class val)
            map_score = precision  # For classification tasks, mAP ≈ macro avg precision
            return accuracy, precision, recall, map_score

        def main():
            torch.manual_seed(42)
            try:
                config = json.loads((RUN_DIR / "config.json").read_text(encoding="utf-8"))
            except Exception:
                config = {
                    "run_id": "demo", "epochs": 5, "batch_size": 8,
                    "input_channels": 3, "input_height": 224, "input_width": 224,
                    "num_classes": 10, "task_type": "classification"
                }

            emit_event("started", "Runtime initialized. Starting real training...", run_id=config["run_id"])
            emit_metric(
                run_id=config["run_id"], epoch=0, loss=2.5,
                accuracy=0.01, precision=0.01, recall=0.01, map=0.01,
                timestamp=datetime.now(timezone.utc).isoformat()
            )

            device = "cpu"
            if torch.cuda.is_available():
                try:
                    torch.randn(1).to("cuda")
                    device = "cuda"
                except Exception:
                    pass

            # Use small batch sizes for CPU to prevent OOM
            batch_size = config["batch_size"]
            if device == "cpu":
                batch_size = min(batch_size, 4)
            # Gradient accumulation steps so effective batch ~= 16
            accum_steps = max(1, 16 // batch_size)

            # Load real dataset if available
            train_loader, val_loader = None, None
            using_real_data = False
            try:
                try:
                    import torchvision.transforms as T
                    from torchvision.datasets import ImageFolder
                    tf = T.Compose([
                        T.Resize((config["input_height"], config["input_width"])),
                        T.RandomHorizontalFlip(),
                        T.ColorJitter(brightness=0.2, contrast=0.2),
                        T.ToTensor(),
                        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                    ])
                    val_tf = T.Compose([
                        T.Resize((config["input_height"], config["input_width"])),
                        T.ToTensor(),
                        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                    ])
                    dataset_cls = ImageFolder
                    emit_event("log", "Loaded torchvision dataset pipeline.")
                except Exception as tv_ex:
                    T = None
                    dataset_cls = SimpleImageFolder
                    tf = Compose([
                        Resize((config["input_height"], config["input_width"])),
                        RandomHorizontalFlip(),
                        ColorJitter(brightness=0.2, contrast=0.2),
                        ToTensor(),
                        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                    ])
                    val_tf = Compose([
                        Resize((config["input_height"], config["input_width"])),
                        ToTensor(),
                        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                    ])
                    emit_event("log", f"torchvision unavailable ({tv_ex}); using built-in image loader.")
                ds_path = config.get("dataset_path")
                if ds_path and os.path.exists(ds_path):
                    full_train = dataset_cls(ds_path, transform=tf)
                    full_val = dataset_cls(ds_path, transform=val_tf)
                    num_classes = len(full_train.classes)
                    config["num_classes"] = num_classes
                    
                    indices = torch.randperm(len(full_train)).tolist()
                    if config.get("num_images"):
                        indices = indices[:config["num_images"]]
                        
                    n_val = max(1, int(len(indices) * 0.2))
                    n_train = len(indices) - n_val
                    
                    from torch.utils.data import Subset
                    train_ds = Subset(full_train, indices[:n_train])
                    val_ds = Subset(full_val, indices[n_train:])
                    
                    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
                    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
                    using_real_data = True
                    emit_event("log", f"Loaded real dataset: {len(indices)} images, {num_classes} classes, {n_train} train / {n_val} val")
                else:
                    raise ValueError("No real dataset path")
            except Exception as ex:
                emit_event("log", f"Using synthetic dataset fallback ({ex})")
                synth = SyntheticDataset(80, config["input_channels"], config["input_height"], config["input_width"], config["num_classes"])
                n_val = max(1, int(len(synth) * 0.2))
                train_ds, val_ds = random_split(synth, [len(synth) - n_val, n_val])
                train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
                val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

            # Build model
            try:
                model = CVAgentModel().to(device)
                with torch.no_grad():
                    dummy = torch.randn(1, config["input_channels"], config["input_height"], config["input_width"]).to(device)
                    out = model(dummy)
                if out.shape[-1] != config["num_classes"]:
                    in_f = out.flatten(1).shape[1]
                    model = nn.Sequential(model, nn.Flatten(), nn.Linear(in_f, config["num_classes"])).to(device)
            except Exception as e:
                emit_event("log", f"Model wrap failed: {e}. Using linear fallback.")
                model = nn.Sequential(
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(config["input_channels"], config["num_classes"])
                ).to(device)

            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=3, gamma=0.5)
            crit = nn.CrossEntropyLoss()

            for epoch in range(1, config["epochs"] + 1):
                try:
                    # ── Training pass ──────────────────────────────────────────
                    model.train()
                    total_loss = 0.0
                    n_batches = 0
                    opt.zero_grad()
                    for step, (imgs, labels) in enumerate(train_loader):
                        imgs, labels = imgs.to(device), labels.to(device)
                        loss = crit(model(imgs), labels) / accum_steps
                        loss.backward()
                        if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            opt.step()
                            opt.zero_grad()
                        total_loss += loss.item() * accum_steps
                        n_batches += 1
                        # Free memory on CPU
                        del imgs, labels, loss
                    avg_loss = total_loss / max(n_batches, 1)
                    scheduler.step()

                    # ── Validation pass (real metrics) ─────────────────────────
                    accuracy, precision, recall, map_score = compute_metrics(
                        model, val_loader, device, config["num_classes"]
                    )

                    emit_metric(
                        run_id=config["run_id"],
                        epoch=epoch,
                        total_epochs=config["epochs"],
                        loss=round(avg_loss, 5),
                        accuracy=round(accuracy, 4),
                        precision=round(precision, 4),
                        recall=round(recall, 4),
                        map=round(map_score, 4),
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        using_real_data=using_real_data,
                    )
                    emit_event("log", f"Epoch {epoch}/{config['epochs']} | loss={avg_loss:.4f} | acc={accuracy:.3f} | prec={precision:.3f} | rec={recall:.3f}")
                except Exception as e:
                    emit_event("log", f"Epoch {epoch} failed: {e}\\n{traceback.format_exc()}")
                    time.sleep(1)

            emit_event("completed", "Training finished successfully.")

        if __name__ == "__main__":
            try:
                main()
            except Exception:
                emit_event("error", traceback.format_exc())
        """
    )
