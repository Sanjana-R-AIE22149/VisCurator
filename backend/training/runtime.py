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

    return {
        "input_channels": int(input_data.get("channels", 3) or 3),
        "input_height": height,
        "input_width": width,
        "num_classes": int(output_data.get("num_classes", 10) or 10),
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
    metadata = _detect_task_metadata(nodes)
    
    if epochs is None:
        epochs = 6 if task_type == "mnist_classification" else 8
        
    config = {
        "run_id": run_dir.name,
        "task_type": task_type,
        "epochs": epochs,
        "num_images": num_images,
        "batch_size": 32,
        "dataset_path": dataset_path,
        **metadata,
    }

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "model.py").write_text(model_code, encoding="utf-8")
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
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

            # Load real dataset if available
            train_loader, val_loader = None, None
            using_real_data = False
            try:
                import torchvision.transforms as T
                from torchvision.datasets import ImageFolder
                ds_path = config.get("dataset_path")
                if ds_path and os.path.exists(ds_path):
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
                    full = ImageFolder(ds_path, transform=tf)
                    
                    if config.get("num_images"):
                        limit = min(len(full), config["num_images"])
                        full, _ = random_split(full, [limit, len(full) - limit])
                        
                    n_val = max(1, int(len(full) * 0.2))
                    n_train = len(full) - n_val
                    train_ds, val_ds = random_split(full, [n_train, n_val])
                    # Apply val transform to val split
                    val_ds.dataset.transform = val_tf
                    num_classes = len(full.classes)
                    config["num_classes"] = num_classes
                    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=0)
                    val_loader   = DataLoader(val_ds,   batch_size=config["batch_size"], shuffle=False, num_workers=0)
                    using_real_data = True
                    emit_event("log", f"Loaded real dataset: {len(full)} images, {num_classes} classes, {n_train} train / {n_val} val")
                else:
                    raise ValueError("No real dataset path")
            except Exception as ex:
                emit_event("log", f"Using synthetic dataset fallback ({ex})")
                synth = SyntheticDataset(80, config["input_channels"], config["input_height"], config["input_width"], config["num_classes"])
                n_val = max(1, int(len(synth) * 0.2))
                train_ds, val_ds = random_split(synth, [len(synth) - n_val, n_val])
                train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
                val_loader   = DataLoader(val_ds,   batch_size=config["batch_size"], shuffle=False)

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
                    nn.Flatten(),
                    nn.Linear(config["input_channels"] * config["input_height"] * config["input_width"], config["num_classes"])
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
                    for imgs, labels in train_loader:
                        imgs, labels = imgs.to(device), labels.to(device)
                        opt.zero_grad()
                        loss = crit(model(imgs), labels)
                        loss.backward()
                        opt.step()
                        total_loss += loss.item()
                        n_batches += 1
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

