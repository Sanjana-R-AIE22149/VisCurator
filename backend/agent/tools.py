from __future__ import annotations

import asyncio
import json
import logging
import random
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiohttp
import numpy as np
import torch
from datasets import load_dataset, load_dataset_builder
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

logger = logging.getLogger(__name__)

HF_API_URL = "https://huggingface.co/api/datasets"
HF_TIMEOUT = aiohttp.ClientTimeout(total=30)
IMAGE_TAG_HINTS = {"image-classification", "object-detection", "image"}


async def _safe_emit(emit_callback: Any, message: str) -> None:
    try:
        if emit_callback is not None:
            await emit_callback(message)
    except Exception:
        logger.exception("emit_callback failed")


def _infer_task_type(tags: list[str]) -> str:
    lowered = [tag.lower() for tag in tags]
    if any("object-detection" in tag or "detection" in tag for tag in lowered):
        return "detection"
    if any("image-classification" in tag or "classification" in tag for tag in lowered):
        return "classification"
    return "unknown"


def _matches_query_words(dataset_id: str, query_words: list[str]) -> bool:
    lowered = dataset_id.lower()
    return any(word in lowered for word in query_words)


def _filter_hf_results(raw_items: list[dict[str, Any]], query: str) -> list[dict[str, Any]]:
    query_words = [word for word in query.lower().split() if len(word) > 1]
    filtered: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw_items:
        dataset_id = str(item.get("id") or item.get("datasetId") or "").strip()
        if not dataset_id or dataset_id in seen:
            continue
        seen.add(dataset_id)
        tags = item.get("tags") or []
        if not isinstance(tags, list):
            tags = []
        lowered_tags = [str(tag).lower() for tag in tags]
        if not (
            any(any(hint in tag for hint in IMAGE_TAG_HINTS) for tag in lowered_tags)
            or _matches_query_words(dataset_id, query_words)
        ):
            continue
        description = (
            item.get("description")
            or item.get("cardData", {}).get("summary")
            or item.get("cardData", {}).get("description")
            or ""
        )
        filtered.append(
            {
                "dataset_id": dataset_id,
                "description": str(description)[:200],
                "downloads": int(item.get("downloads") or 0),
                "tags": tags,
                "task_type": _infer_task_type(tags),
            }
        )
    return filtered


async def _search_once(query: str, max_results: int) -> list[dict[str, Any]]:
    params = {
        "search": query,
        "limit": max_results,
        "sort": "downloads",
        "full": "false",
    }
    async with aiohttp.ClientSession(timeout=HF_TIMEOUT) as session:
        async with session.get(HF_API_URL, params=params) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise RuntimeError(f"HuggingFace search failed ({resp.status}): {text[:200]}")
            payload = await resp.json()
            if not isinstance(payload, list):
                raise RuntimeError("Unexpected HuggingFace API response.")
            return payload


async def search_huggingface(query: str, max_results: int = 8) -> dict[str, Any]:
    try:
        raw_primary = await _search_once(query, max_results)
        filtered = _filter_hf_results(raw_primary, query)
        if len(filtered) < 3:
            shorter = query.strip().split()[0] if query.strip().split() else query.strip()
            if shorter and shorter.lower() != query.strip().lower():
                raw_retry = await _search_once(shorter, max_results)
                merged = raw_primary + raw_retry
                filtered = _filter_hf_results(merged, query)
        if not filtered:
            return {"status": "error", "error": f"No HuggingFace image datasets found for '{query}'."}
        return {"status": "success", "datasets": filtered[:max_results]}
    except Exception as e:
        logger.exception("search_huggingface failed")
        return {"status": "error", "error": str(e)}


def _feature_summary(features: Any) -> dict[str, str]:
    summary: dict[str, str] = {}
    try:
        for name, feature in dict(features).items():
            summary[str(name)] = type(feature).__name__
    except Exception:
        pass
    return summary


def _inspect_from_builder_sync(dataset_id: str) -> dict[str, Any]:
    builder = load_dataset_builder(dataset_id)
    info = builder.info
    splits = {name: split.num_examples for name, split in (info.splits or {}).items()}
    features = info.features or {}

    image_column = None
    label_column = None
    num_classes = None
    class_names = None

    for name, feature in dict(features).items():
        feature_type = type(feature).__name__.lower()
        if image_column is None and (feature_type == "image" or str(name).lower() in {"image", "img"}):
            image_column = str(name)
        if label_column is None and (
            feature_type == "classlabel" or str(name).lower() in {"label", "class"}
        ):
            label_column = str(name)
            if hasattr(feature, "names") and getattr(feature, "names", None):
                class_names = list(feature.names)
                num_classes = len(class_names)

    return {
        "status": "success",
        "dataset_id": dataset_id,
        "splits": splits,
        "features": _feature_summary(features),
        "description": (info.description or "")[:500],
        "homepage": info.homepage or "",
        "license": info.license or "",
        "image_column": image_column,
        "label_column": label_column,
        "num_classes": num_classes,
        "class_names": class_names,
    }


def _inspect_from_row_sync(dataset_id: str) -> dict[str, Any]:
    sample = load_dataset(dataset_id, split="train[:1]", trust_remote_code=True)
    if len(sample) == 0:
        raise RuntimeError("Dataset sample is empty.")
    row = sample[0]
    image_column = None
    label_column = None
    for key, value in row.items():
        lowered = str(key).lower()
        if image_column is None and (lowered in {"image", "img"} or isinstance(value, Image.Image)):
            image_column = str(key)
        if label_column is None and lowered in {"label", "class"}:
            label_column = str(key)
    return {
        "status": "success",
        "dataset_id": dataset_id,
        "splits": {"train": len(sample)},
        "features": {str(key): type(value).__name__ for key, value in row.items()},
        "description": "",
        "homepage": "",
        "license": "",
        "image_column": image_column,
        "label_column": label_column,
        "num_classes": None,
        "class_names": None,
    }


async def inspect_dataset(dataset_id: str) -> dict[str, Any]:
    try:
        try:
            return await asyncio.wait_for(asyncio.to_thread(_inspect_from_builder_sync, dataset_id), timeout=30.0)
        except asyncio.TimeoutError:
            return {"status": "error", "error": "timeout"}
        except Exception:
            return await asyncio.wait_for(asyncio.to_thread(_inspect_from_row_sync, dataset_id), timeout=30.0)
    except asyncio.TimeoutError:
        return {"status": "error", "error": "timeout"}
    except Exception as e:
        logger.exception("inspect_dataset failed")
        return {"status": "error", "error": str(e)}


def _resolve_label_names(dataset: Any, label_column: str) -> dict[int, str]:
    try:
        features = getattr(dataset, "features", {})
        feature = features.get(label_column)
        if feature is not None and hasattr(feature, "names") and getattr(feature, "names", None):
            return {idx: str(name) for idx, name in enumerate(feature.names)}
    except Exception:
        pass
    return {}


async def download_and_clean(
    dataset_id: str,
    image_column: str,
    label_column: str,
    output_dir: str,
    max_images: int = 500,
    emit_callback: Any = None,
) -> dict[str, Any]:
    try:
        await _safe_emit(emit_callback, f"> Loading dataset {dataset_id}...")
        try:
            dataset = await asyncio.to_thread(load_dataset, dataset_id, split="train", trust_remote_code=True)
            split_name = "train"
        except Exception:
            dataset_dict = await asyncio.to_thread(load_dataset, dataset_id, trust_remote_code=True)
            if not dataset_dict:
                return {"status": "error", "error": "Dataset has no splits."}
            split_name = next(iter(dataset_dict.keys()))
            dataset = dataset_dict[split_name]

        label_names = _resolve_label_names(dataset, label_column)
        out_path = Path(output_dir)
        if out_path.exists():
            shutil.rmtree(out_path)
        out_path.mkdir(parents=True, exist_ok=True)

        accepted_by_label: dict[str, list[Image.Image]] = defaultdict(list)
        rejected_small = 0
        rejected_blur = 0
        rejected_dup = 0
        dedup_seen: set[str] = set()
        processed_rows = 0
        total_rows = len(dataset)

        for idx, row in enumerate(dataset):
            if sum(len(items) for items in accepted_by_label.values()) >= max_images:
                break
            processed_rows += 1
            if processed_rows % 50 == 0:
                await _safe_emit(emit_callback, f"> Processing image {processed_rows}/{total_rows}...")

            img = row.get(image_column)
            if img is None or not isinstance(img, Image.Image):
                continue
            if img.size[0] < 32 or img.size[1] < 32:
                rejected_small += 1
                continue

            gray_var = float(np.var(np.array(img.convert("L"), dtype=float)))
            if gray_var < 50:
                rejected_blur += 1
                continue

            try:
                pixel_0 = img.getpixel((0, 0))
            except Exception:
                pixel_0 = (0, 0, 0)
            dedup_hash = f"{img.size}-{pixel_0}"
            if dedup_hash in dedup_seen:
                rejected_dup += 1
                continue
            dedup_seen.add(dedup_hash)

            raw_label = row.get(label_column)
            if isinstance(raw_label, int) and raw_label in label_names:
                label = label_names[raw_label]
            else:
                label = str(raw_label)
            accepted_by_label[label].append(img.copy().convert("RGB"))

        total_downloaded = sum(len(items) for items in accepted_by_label.values())
        if total_downloaded == 0:
            return {"status": "error", "error": "All images were rejected during cleaning."}

        rng = random.Random(42)
        split_counts = {"train": 0, "val": 0, "test": 0}
        class_distribution: dict[str, int] = {}

        for label, images in accepted_by_label.items():
            rng.shuffle(images)
            count = len(images)
            class_distribution[label] = count
            train_cut = int(count * 0.70)
            val_cut = train_cut + int(count * 0.15)
            split_map = {
                "train": images[:train_cut],
                "val": images[train_cut:val_cut],
                "test": images[val_cut:],
            }
            if count == 1:
                split_map = {"train": images, "val": [], "test": []}
            elif count == 2:
                split_map = {"train": images[:1], "val": images[1:], "test": []}

            for split, split_images in split_map.items():
                class_dir = out_path / split / label
                class_dir.mkdir(parents=True, exist_ok=True)
                for image_index, image in enumerate(split_images):
                    file_path = class_dir / f"{label}_{image_index:05d}.jpg"
                    image.save(file_path, "JPEG", quality=90)
                    split_counts[split] += 1

        await _safe_emit(emit_callback, f"> Finished cleaning. Accepted {total_downloaded} images.")
        return {
            "status": "success",
            "output_dir": str(out_path),
            "source_split": split_name,
            "total_downloaded": total_downloaded,
            "rejected_small": rejected_small,
            "rejected_blur": rejected_blur,
            "rejected_dup": rejected_dup,
            "class_distribution": class_distribution,
            "splits": split_counts,
        }
    except Exception as e:
        logger.exception("download_and_clean failed")
        return {"status": "error", "error": str(e)}


async def annotate_with_clip(
    image_dir: str,
    class_names: list[str],
    output_dir: str,
    emit_callback: Any = None,
) -> dict[str, Any]:
    try:
        import cv2

        def _as_feature_tensor(output: Any) -> torch.Tensor:
            if isinstance(output, torch.Tensor):
                return output
            if hasattr(output, "text_embeds") and output.text_embeds is not None:
                return output.text_embeds
            if hasattr(output, "image_embeds") and output.image_embeds is not None:
                return output.image_embeds
            if hasattr(output, "pooler_output") and output.pooler_output is not None:
                return output.pooler_output
            if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
                hidden = output.last_hidden_state
                return hidden[:, 0, :] if hidden.ndim == 3 else hidden
            raise TypeError(f"Unsupported CLIP feature output type: {type(output).__name__}")

        def fallback_bbox(img: Image.Image) -> tuple[int, int, int, int]:
            arr = np.array(img.convert("RGB"))
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            rect = (max(1, img.width // 20), max(1, img.height // 20), max(2, img.width - img.width // 10), max(2, img.height - img.height // 10))
            try:
                mask = np.zeros(gray.shape[:2], np.uint8)
                bgd = np.zeros((1, 65), np.float64)
                fgd = np.zeros((1, 65), np.float64)
                cv2.grabCut(arr, mask, rect, bgd, fgd, 3, cv2.GC_INIT_WITH_RECT)
                fg = np.where((mask == 1) | (mask == 3), 255, 0).astype("uint8")
            except Exception:
                _, fg = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return (0, 0, img.width, img.height)
            largest = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)
            if w <= 1 or h <= 1:
                return (0, 0, img.width, img.height)
            return int(x), int(y), int(w), int(h)

        def bbox_to_yolo(x: int, y: int, w: int, h: int, img_w: int, img_h: int) -> tuple[float, float, float, float]:
            cx = (x + w / 2) / img_w
            cy = (y + h / 2) / img_h
            nw = w / img_w
            nh = h / img_h
            return round(cx, 6), round(cy, 6), round(nw, 6), round(nh, 6)

        image_root = Path(image_dir)
        out_root = Path(output_dir)
        out_root.mkdir(parents=True, exist_ok=True)
        labels_root = out_root / "labels"
        labels_root.mkdir(parents=True, exist_ok=True)
        anno_dir = out_root / "annotations"
        anno_dir.mkdir(parents=True, exist_ok=True)

        image_paths = sorted([p for p in image_root.rglob("*") if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
        if not image_paths:
            return {"status": "error", "error": "No images found for annotation."}

        await _safe_emit(emit_callback, "> Downloading CLIP model (first run only, ~350MB)...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        model.eval()

        prompts = [f"a photo of a {name}" for name in class_names]
        with torch.no_grad():
            text_inputs = processor(text=prompts, return_tensors="pt", padding=True)
            text_inputs = {key: value.to(device) for key, value in text_inputs.items()}
            text_features = _as_feature_tensor(model.get_text_features(**text_inputs))
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        annotations: list[dict[str, Any]] = []
        coco_records: list[dict[str, Any]] = []
        class_confidence: dict[str, list[float]] = defaultdict(list)
        verified_count = 0
        low_confidence_count = 0

        for batch_start in range(0, len(image_paths), 16):
            batch_paths = image_paths[batch_start:batch_start + 16]
            images = [Image.open(path).convert("RGB") for path in batch_paths]
            with torch.no_grad():
                image_inputs = processor(images=images, return_tensors="pt")
                image_inputs = {key: value.to(device) for key, value in image_inputs.items()}
                image_features = _as_feature_tensor(model.get_image_features(**image_inputs))
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                logits = image_features @ text_features.T
                probs = logits.softmax(dim=-1).cpu().numpy()

            for local_index, image_path in enumerate(batch_paths):
                rel_path = image_path.relative_to(image_root)
                split = rel_path.parts[0] if len(rel_path.parts) > 1 else "train"
                original_label = image_path.parent.name
                prob_vector = probs[local_index]
                predicted_index = int(prob_vector.argmax())
                predicted_class = class_names[predicted_index]
                confidence = float(prob_vector[predicted_index])
                verified = original_label == predicted_class
                if verified:
                    verified_count += 1
                if confidence < 0.6:
                    low_confidence_count += 1
                class_confidence[predicted_class].append(confidence)

                # Bounding box
                img_obj = images[local_index]
                img_w, img_h = img_obj.size
                x_abs, y_abs, w_abs, h_abs = fallback_bbox(img_obj)
                cx, cy, nw, nh = bbox_to_yolo(x_abs, y_abs, w_abs, h_abs, img_w, img_h)

                labels_dir = labels_root / split
                labels_dir.mkdir(parents=True, exist_ok=True)
                (labels_dir / f"{image_path.stem}.txt").write_text(f"{predicted_index} {cx} {cy} {nw} {nh}\n", encoding="utf-8")

                coco_records.append({
                    "file_name": f"images/{rel_path}".replace("\\", "/"),
                    "split": split,
                    "width": img_w,
                    "height": img_h,
                    "class_id": predicted_index,
                    "bbox_abs": [x_abs, y_abs, w_abs, h_abs]
                })

                annotations.append(
                    {
                        "image_path": str(rel_path).replace("\\", "/"),
                        "original_label": original_label,
                        "clip_predicted_class": predicted_class,
                        "clip_confidence": confidence,
                        "verified": verified,
                    }
                )

            processed = batch_start + len(batch_paths)
            if processed % 50 == 0 or processed == len(image_paths):
                await _safe_emit(emit_callback, f"> Annotated {processed}/{len(image_paths)} images with CLIP + Bboxes...")

        # Write YOLO data.yaml
        yaml_path = out_root / "data.yaml"
        yaml_path.write_text(
            f"path: .\n"
            f"train: images/train\n"
            f"val: images/val\n"
            f"test: images/test\n"
            f"nc: {len(class_names)}\n"
            f"names: {json.dumps(class_names)}\n",
            encoding="utf-8",
        )

        # Write COCO JSONs
        coco_categories = [{"id": i, "name": c, "supercategory": "object"} for i, c in enumerate(class_names)]
        for split_name in ("train", "val", "test"):
            split_records = [r for r in coco_records if r["split"] == split_name]
            if not split_records:
                continue
            coco_images, coco_annos = [], []
            for img_id, rec in enumerate(split_records):
                x, y, w, h = rec["bbox_abs"]
                coco_images.append({
                    "id": img_id,
                    "file_name": rec["file_name"],
                    "width": rec["width"],
                    "height": rec["height"],
                })
                coco_annos.append({
                    "id": img_id,
                    "image_id": img_id,
                    "category_id": rec["class_id"],
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                })
            
            coco_out = {
                "info": {"description": "VisCurator CLIP Dataset", "version": "1.0"},
                "categories": coco_categories,
                "images": coco_images,
                "annotations": coco_annos,
            }
            (anno_dir / f"instances_{split_name}.json").write_text(
                json.dumps(coco_out, indent=2), encoding="utf-8"
            )

        class_breakdown = {
            class_name: {
                "count": len(scores),
                "avg_confidence": (sum(scores) / len(scores)) if scores else 0.0,
            }
            for class_name, scores in class_confidence.items()
        }
        summary = {
            "verified_count": verified_count,
            "verification_rate": verified_count / len(annotations),
            "low_confidence_count": low_confidence_count,
            "class_breakdown": class_breakdown,
        }
        payload = {
            "metadata": {
                "tool": "CLIP zero-shot + OpenCV GrabCut",
                "model": "openai/clip-vit-base-patch32",
                "class_names": class_names,
                "total_images": len(annotations),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            "annotations": annotations,
            "summary": summary,
        }
        (anno_dir / "annotations.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return {"status": "success", **summary, "summary": summary, "output_dir": str(out_root)}
    except Exception as e:
        logger.exception("annotate_with_clip failed")
        return {"status": "error", "error": str(e)}
