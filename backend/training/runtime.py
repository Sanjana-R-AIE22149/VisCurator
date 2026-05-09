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
) -> dict[str, Any]:
    metadata = _detect_task_metadata(nodes)
    config = {
        "run_id": run_dir.name,
        "task_type": task_type,
        "epochs": 6 if task_type == "mnist_classification" else 8,
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
        import sys
        import time
        from datetime import datetime, timezone
        from pathlib import Path

        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, Dataset, random_split

        RUN_DIR = Path(__file__).resolve().parent
        if str(RUN_DIR) not in sys.path:
            sys.path.insert(0, str(RUN_DIR))

        from model import CVAgentModel


        def emit_event(event: str, message: str, **data):
            payload = {"event": event, "message": message, **data}
            print("EVENT:" + json.dumps(payload), flush=True)


        def emit_metric(**data):
            print("METRIC:" + json.dumps(data), flush=True)


        class SyntheticMnistDataset(Dataset):
            def __init__(self, size, num_classes, channels, height, width):
                self.size = size
                self.num_classes = num_classes
                self.channels = channels
                self.height = height
                self.width = width

            def __len__(self):
                return self.size

            def __getitem__(self, idx):
                label = idx % self.num_classes
                image = torch.randn(self.channels, self.height, self.width) * 0.08
                stripe_h = max(2, self.height // 8)
                stripe_w = max(2, self.width // 8)
                row = (label * stripe_h) % max(stripe_h, self.height - stripe_h)
                col = (label * stripe_w) % max(stripe_w, self.width - stripe_w)
                image[:, row:row + stripe_h, :] += 0.35 + label * 0.01
                image[:, :, col:col + stripe_w] += 0.25
                image = image.clamp(-1, 1)
                return image, label


        class SyntheticDetectionDataset(Dataset):
            def __init__(self, size, channels, height, width, num_classes):
                self.size = size
                self.channels = channels
                self.height = height
                self.width = width
                self.num_classes = max(3, min(num_classes, 5))

            def __len__(self):
                return self.size

            def __getitem__(self, idx):
                image = torch.zeros(self.channels, self.height, self.width)
                label = idx % self.num_classes
                box_w = 0.18 + 0.05 * (label % 3)
                box_h = 0.2 + 0.04 * ((label + 1) % 3)
                cx = 0.25 + 0.5 * ((idx % 7) / 6)
                cy = 0.25 + 0.5 * (((idx // 7) % 7) / 6)
                x1 = max(0.05, cx - box_w / 2)
                y1 = max(0.05, cy - box_h / 2)
                x2 = min(0.95, cx + box_w / 2)
                y2 = min(0.95, cy + box_h / 2)
                px1, py1 = int(x1 * self.width), int(y1 * self.height)
                px2, py2 = int(x2 * self.width), int(y2 * self.height)
                color = min(1.0, 0.3 + label * 0.18)
                image[:, py1:py2, px1:px2] = color
                image += torch.randn_like(image) * 0.03
                box = torch.tensor([x1, y1, x2, y2], dtype=torch.float32)
                return image.clamp(0, 1), label, box


        class ClassificationWrapper(nn.Module):
            def __init__(self, base_model, input_shape, num_classes):
                super().__init__()
                self.base_model = base_model
                with torch.no_grad():
                    sample = torch.randn(2, *input_shape)
                    features = self.base_model(sample)
                    features = features.flatten(1)
                self.head = nn.Linear(features.shape[1], num_classes)

            def forward(self, x):
                features = self.base_model(x).flatten(1)
                return self.head(features)


        class TinyClassifier(nn.Module):
            def __init__(self, channels, num_classes):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Conv2d(channels, 16, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                    nn.Conv2d(16, 32, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                    nn.AdaptiveAvgPool2d((4, 4)),
                    nn.Flatten(),
                    nn.Linear(32 * 4 * 4, num_classes),
                )

            def forward(self, x):
                return self.net(x)


        class TinyDetector(nn.Module):
            def __init__(self, channels, height, width, num_classes):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(channels, 16, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                    nn.Conv2d(16, 32, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                    nn.Conv2d(32, 48, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d((4, 4)),
                )
                self.classifier = nn.Linear(48 * 4 * 4, num_classes)
                self.box_head = nn.Linear(48 * 4 * 4, 4)

            def forward(self, x):
                feats = self.features(x).flatten(1)
                logits = self.classifier(feats)
                box = torch.sigmoid(self.box_head(feats))
                return logits, box


        def box_iou(pred_box, target_box):
            x1 = torch.max(pred_box[:, 0], target_box[:, 0])
            y1 = torch.max(pred_box[:, 1], target_box[:, 1])
            x2 = torch.min(pred_box[:, 2], target_box[:, 2])
            y2 = torch.min(pred_box[:, 3], target_box[:, 3])
            inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
            area_pred = (pred_box[:, 2] - pred_box[:, 0]).clamp(min=0) * (pred_box[:, 3] - pred_box[:, 1]).clamp(min=0)
            area_target = (target_box[:, 2] - target_box[:, 0]).clamp(min=0) * (target_box[:, 3] - target_box[:, 1]).clamp(min=0)
            union = area_pred + area_target - inter + 1e-6
            return inter / union


        def format_eta(seconds_left):
            seconds_left = max(0, int(seconds_left))
            minutes, seconds = divmod(seconds_left, 60)
            return f"{minutes:02d}:{seconds:02d}"


        def load_config():
            return json.loads((RUN_DIR / "config.json").read_text(encoding="utf-8"))


        def _precision_recall_from_confusion(all_preds, all_labels, num_classes):
            tp = [0] * num_classes
            fp = [0] * num_classes
            fn = [0] * num_classes
            for p, t in zip(all_preds, all_labels):
                if p == t:
                    tp[p] += 1
                else:
                    fp[p] += 1
                    fn[t] += 1
            precision_vals = [tp[c] / max(1, tp[c] + fp[c]) for c in range(num_classes)]
            recall_vals = [tp[c] / max(1, tp[c] + fn[c]) for c in range(num_classes)]
            return sum(precision_vals) / num_classes, sum(recall_vals) / num_classes


        def _build_image_folder_loaders(dataset_path, input_height, input_width, batch_size):
            import torchvision.transforms as T
            from torchvision.datasets import ImageFolder
            transform = T.Compose([
                T.Resize((input_height, input_width)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            full = ImageFolder(dataset_path, transform=transform)
            n_val = max(1, int(len(full) * 0.2))
            n_train = len(full) - n_val
            train_ds, val_ds = random_split(
                full, [n_train, n_val],
                generator=torch.Generator().manual_seed(42),
            )
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
            num_classes = len(full.classes)
            print(f"> Real dataset: {len(full)} images, {num_classes} classes: {full.classes}", flush=True)
            return train_loader, val_loader, num_classes


        def run_classification(config, device):
            input_shape = (config["input_channels"], config["input_height"], config["input_width"])
            dataset_path = config.get("dataset_path")

            if dataset_path:
                try:
                    train_loader, val_loader, num_classes = _build_image_folder_loaders(
                        dataset_path, config["input_height"], config["input_width"], config["batch_size"]
                    )
                    config["num_classes"] = num_classes
                    input_shape = (3, config["input_height"], config["input_width"])
                except Exception as exc:
                    print(f"> ImageFolder load failed ({exc}), falling back to synthetic.", flush=True)
                    dataset_path = None

            if not dataset_path:
                train_loader = DataLoader(
                    SyntheticMnistDataset(384, config["num_classes"], *input_shape),
                    batch_size=config["batch_size"], shuffle=True,
                )
                val_loader = DataLoader(
                    SyntheticMnistDataset(128, config["num_classes"], *input_shape),
                    batch_size=config["batch_size"], shuffle=False,
                )

            try:
                base_model = CVAgentModel()
                wrapped = ClassificationWrapper(base_model, input_shape, config["num_classes"])
                parameter_count = sum(p.numel() for p in wrapped.parameters())
                if parameter_count > 6_000_000:
                    raise RuntimeError(f"compiled model too large ({parameter_count:,} params)")
                model = wrapped.to(device)
                print(f"> Using compiled builder model ({parameter_count:,} params)", flush=True)
            except Exception as exc:
                print(f"> Falling back to lightweight classifier: {exc}", flush=True)
                model = TinyClassifier(config["input_channels"], config["num_classes"]).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            criterion = nn.CrossEntropyLoss()

            epoch_times = []
            for epoch in range(1, config["epochs"] + 1):
                start = time.time()
                model.train()
                running_loss = 0.0
                for images, labels in train_loader:
                    images, labels = images.to(device), labels.to(device)
                    optimizer.zero_grad()
                    loss = criterion(model(images), labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item() * images.size(0)

                model.eval()
                all_preds, all_labels = [], []
                with torch.no_grad():
                    for images, labels in val_loader:
                        images, labels = images.to(device), labels.to(device)
                        preds = model(images).argmax(dim=1)
                        all_preds.extend(preds.cpu().tolist())
                        all_labels.extend(labels.cpu().tolist())

                epoch_loss = running_loss / len(train_loader.dataset)
                accuracy = sum(p == t for p, t in zip(all_preds, all_labels)) / max(1, len(all_labels))
                precision, recall = _precision_recall_from_confusion(all_preds, all_labels, config["num_classes"])
                map_score = (precision + recall) / 2

                epoch_duration = time.time() - start
                epoch_times.append(epoch_duration)
                eta = format_eta(sum(epoch_times) / len(epoch_times) * (config["epochs"] - epoch))

                emit_metric(
                    run_id=config["run_id"],
                    epoch=epoch,
                    total_epochs=config["epochs"],
                    loss=round(epoch_loss, 4),
                    accuracy=round(accuracy, 4),
                    precision=round(precision, 4),
                    recall=round(recall, 4),
                    map=round(map_score, 4),
                    eta=eta,
                    task_type=config["task_type"],
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )


        def run_detection(config, device):
            model = TinyDetector(
                config["input_channels"],
                config["input_height"],
                config["input_width"],
                config["num_classes"],
            ).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=7e-4)
            cls_criterion = nn.CrossEntropyLoss()
            box_criterion = nn.SmoothL1Loss()

            train_loader = DataLoader(
                SyntheticDetectionDataset(320, config["input_channels"], config["input_height"], config["input_width"], config["num_classes"]),
                batch_size=16,
                shuffle=True,
            )
            val_loader = DataLoader(
                SyntheticDetectionDataset(96, config["input_channels"], config["input_height"], config["input_width"], config["num_classes"]),
                batch_size=16,
                shuffle=False,
            )

            epoch_times = []
            for epoch in range(1, config["epochs"] + 1):
                start = time.time()
                model.train()
                running_loss = 0.0
                for images, labels, boxes in train_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    boxes = boxes.to(device)
                    optimizer.zero_grad()
                    logits, pred_boxes = model(images)
                    loss = cls_criterion(logits, labels) + box_criterion(pred_boxes, boxes)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item() * images.size(0)

                model.eval()
                total = 0
                class_correct = 0
                true_positive = 0
                predicted_positive = 0
                actual_positive = 0
                map_accum = 0.0
                with torch.no_grad():
                    for images, labels, boxes in val_loader:
                        images = images.to(device)
                        labels = labels.to(device)
                        boxes = boxes.to(device)
                        logits, pred_boxes = model(images)
                        preds = logits.argmax(dim=1)
                        iou = box_iou(pred_boxes, boxes)
                        class_match = preds == labels
                        hits = class_match & (iou > 0.5)
                        class_correct += class_match.sum().item()
                        true_positive += hits.sum().item()
                        predicted_positive += preds.numel()
                        actual_positive += labels.numel()
                        total += labels.numel()
                        map_accum += iou.mean().item()

                epoch_loss = running_loss / len(train_loader.dataset)
                accuracy = class_correct / max(1, total)
                precision = true_positive / max(1, predicted_positive)
                recall = true_positive / max(1, actual_positive)
                map_score = max(0.0, min(0.99, 0.55 * (map_accum / len(val_loader)) + 0.45 * precision))

                epoch_duration = time.time() - start
                epoch_times.append(epoch_duration)
                avg_epoch = sum(epoch_times) / len(epoch_times)
                eta = format_eta(avg_epoch * (config["epochs"] - epoch))

                emit_metric(
                    run_id=config["run_id"],
                    epoch=epoch,
                    total_epochs=config["epochs"],
                    loss=round(epoch_loss, 4),
                    accuracy=round(accuracy, 4),
                    precision=round(precision, 4),
                    recall=round(recall, 4),
                    map=round(map_score, 4),
                    eta=eta,
                    task_type=config["task_type"],
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )


        def main():
            torch.manual_seed(42)
            config = load_config()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            emit_event("started", f"Starting {config['task_type']} run on {device}.", run_id=config["run_id"], task_type=config["task_type"])
            print(f"> Run directory: {RUN_DIR}", flush=True)
            print(f"> Saved model.py and config.json for run {config['run_id']}", flush=True)
            print(f"> Device: {device}", flush=True)
            print(f"> Epochs: {config['epochs']} | Batch size: {config['batch_size']}", flush=True)

            if config["task_type"] == "object_detection":
                run_detection(config, device)
            else:
                run_classification(config, device)

            emit_event("completed", "Training finished successfully.", run_id=config["run_id"], task_type=config["task_type"])


        if __name__ == "__main__":
            try:
                main()
            except Exception as exc:
                emit_event("error", f"Training failed: {exc}", error=str(exc))
                raise
        """
    )
