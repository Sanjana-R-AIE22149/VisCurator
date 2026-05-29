from __future__ import annotations

import asyncio
import json
import logging
import random
import shutil
from collections import defaultdict
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

# ── CLIP singleton — loaded once in a background thread on first use ──────────
_clip_model: CLIPModel | None = None
_clip_processor: CLIPProcessor | None = None
_clip_lock = asyncio.Lock()


def _load_clip_sync() -> tuple[CLIPModel, CLIPProcessor]:
    """Load CLIP from cache (or download). Runs in a thread."""
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    model.eval()
    return model, processor


async def _get_clip() -> tuple[CLIPModel, CLIPProcessor]:
    global _clip_model, _clip_processor
    async with _clip_lock:
        if _clip_model is None:
            _clip_model, _clip_processor = await asyncio.to_thread(_load_clip_sync)
    return _clip_model, _clip_processor  # type: ignore


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
    params = {"search": query, "limit": max_results, "sort": "downloads", "full": "false"}
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
                filtered = _filter_hf_results(raw_primary + raw_retry, query)
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
    image_column = label_column = num_classes = class_names = None
    for name, feature in dict(features).items():
        feature_type = type(feature).__name__.lower()
        if image_column is None and (feature_type == "image" or str(name).lower() in {"image", "img"}):
            image_column = str(name)
        if label_column is None and (feature_type == "classlabel" or str(name).lower() in {"label", "class"}):
            label_column = str(name)
            if hasattr(feature, "names") and getattr(feature, "names", None):
                class_names = list(feature.names)
                num_classes = len(class_names)
    return {
        "status": "success", "dataset_id": dataset_id, "splits": splits,
        "features": _feature_summary(features), "description": (info.description or "")[:500],
        "homepage": info.homepage or "", "license": info.license or "",
        "image_column": image_column, "label_column": label_column,
        "num_classes": num_classes, "class_names": class_names,
    }


def _inspect_from_row_sync(dataset_id: str) -> dict[str, Any]:
    sample = load_dataset(dataset_id, split="train[:1]", trust_remote_code=True)
    if len(sample) == 0:
        raise RuntimeError("Dataset sample is empty.")
    row = sample[0]
    image_column = label_column = None
    for key, value in row.items():
        lowered = str(key).lower()
        if image_column is None and (lowered in {"image", "img"} or isinstance(value, Image.Image)):
            image_column = str(key)
        if label_column is None and lowered in {"label", "class"}:
            label_column = str(key)
    return {
        "status": "success", "dataset_id": dataset_id, "splits": {"train": len(sample)},
        "features": {str(key): type(value).__name__ for key, value in row.items()},
        "description": "", "homepage": "", "license": "",
        "image_column": image_column, "label_column": label_column,
        "num_classes": None, "class_names": None,
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


# ── Sync worker for download_and_clean ───────────────────────────────────────

def _download_and_clean_sync(
    dataset_id: str,
    image_column: str,
    label_column: str,
    output_dir: str,
    max_images: int,
    progress_queue: Any,  # queue to push progress strings back to async side
) -> dict[str, Any]:
    dataset = None
    split_name = "train"
    try:
        dataset = load_dataset(dataset_id, split="train", trust_remote_code=True)
    except Exception:
        dataset_dict = load_dataset(dataset_id, trust_remote_code=True)
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
    rejected_small = rejected_blur = rejected_dup = 0
    dedup_seen: set[str] = set()
    processed_rows = 0
    total_rows = len(dataset)
    blurry_samples: list[Image.Image] = []
    duplicate_samples: list[Image.Image] = []
    accepted_samples: list[Image.Image] = []  # for processed preview

    for idx, row in enumerate(dataset):
        if sum(len(v) for v in accepted_by_label.values()) >= max_images:
            break
        processed_rows += 1
        if processed_rows % 50 == 0:
            progress_queue.put_nowait(f"> Processing image {processed_rows}/{total_rows}...")

        img = row.get(image_column)
        if img is None or not isinstance(img, Image.Image):
            continue
        if img.size[0] < 32 or img.size[1] < 32:
            rejected_small += 1
            continue

        # Laplacian variance blur detection (same metric as annotator)
        try:
            import cv2 as _cv2
            gray_arr = np.array(img.convert("L"), dtype=np.float32)
            lap_var = float(_cv2.Laplacian(gray_arr, _cv2.CV_32F).var())
        except Exception:
            lap_var = float(np.var(np.array(img.convert("L"), dtype=float)))
        if lap_var < 20.0:
            rejected_blur += 1
            if len(blurry_samples) < 12:
                blurry_samples.append(img.copy().convert("RGB"))
            continue

        try:
            pixel_0 = img.getpixel((0, 0))
        except Exception:
            pixel_0 = (0, 0, 0)
        dedup_hash = f"{img.size}-{pixel_0}"
        if dedup_hash in dedup_seen:
            rejected_dup += 1
            if len(duplicate_samples) < 12:
                duplicate_samples.append(img.copy().convert("RGB"))
            continue
        dedup_seen.add(dedup_hash)

        raw_label = row.get(label_column)
        label = label_names.get(raw_label, str(raw_label)) if isinstance(raw_label, int) else str(raw_label)
        accepted_by_label[label].append(img.copy().convert("RGB"))
        if len(accepted_samples) < 12:
            accepted_samples.append(img.copy().convert("RGB"))

    total_downloaded = sum(len(v) for v in accepted_by_label.values())
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
        split_map = {"train": images[:train_cut], "val": images[train_cut:val_cut], "test": images[val_cut:]}
        if count == 1:
            split_map = {"train": images, "val": [], "test": []}
        elif count == 2:
            split_map = {"train": images[:1], "val": images[1:], "test": []}
        for split, split_images in split_map.items():
            class_dir = out_path / split / label
            class_dir.mkdir(parents=True, exist_ok=True)
            for i, image in enumerate(split_images):
                image.save(class_dir / f"{label}_{i:05d}.jpg", "JPEG", quality=90)
                split_counts[split] += 1

    # Save blur/dup/processed previews
    previews_dir = out_path / "_previews"
    blurry_preview_paths: list[str] = []
    dup_preview_paths: list[str] = []
    processed_preview_paths: list[str] = []
    try:
        for subdir in ("blurry", "duplicates", "processed"):
            (previews_dir / subdir).mkdir(parents=True, exist_ok=True)
        for i, img in enumerate(blurry_samples):
            p = previews_dir / "blurry" / f"blur_{i:04d}.jpg"
            img.save(p, "JPEG", quality=80)
            blurry_preview_paths.append(str(p.relative_to(out_path.parent)).replace("\\", "/"))
        for i, img in enumerate(duplicate_samples):
            p = previews_dir / "duplicates" / f"dup_{i:04d}.jpg"
            img.save(p, "JPEG", quality=80)
            dup_preview_paths.append(str(p.relative_to(out_path.parent)).replace("\\", "/"))
        for i, img in enumerate(accepted_samples):
            p = previews_dir / "processed" / f"proc_{i:04d}.jpg"
            img.save(p, "JPEG", quality=80)
            processed_preview_paths.append(str(p.relative_to(out_path.parent)).replace("\\", "/"))
    except Exception:
        pass

    return {
        "status": "success", "output_dir": str(out_path), "source_split": split_name,
        "total_downloaded": total_downloaded, "rejected_small": rejected_small,
        "rejected_blur": rejected_blur, "rejected_dup": rejected_dup,
        "class_distribution": class_distribution, "splits": split_counts,
        "blurry_preview_paths": blurry_preview_paths, "dup_preview_paths": dup_preview_paths,
        "processed_preview_paths": processed_preview_paths,
    }


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

        loop = asyncio.get_running_loop()
        progress_queue: asyncio.Queue = asyncio.Queue()

        # Drain progress messages while the sync worker runs in a thread
        async def _drain():
            while True:
                msg = await progress_queue.get()
                if msg is None:
                    break
                await _safe_emit(emit_callback, msg)

        drain_task = asyncio.create_task(_drain())

        result = await asyncio.to_thread(
            _download_and_clean_sync,
            dataset_id, image_column, label_column, output_dir, max_images, progress_queue,
        )

        await progress_queue.put(None)  # signal drain to stop
        await drain_task

        if result.get("status") == "success":
            await _safe_emit(emit_callback, f"> Finished cleaning. Accepted {result['total_downloaded']} images.")
        return result
    except Exception as e:
        logger.exception("download_and_clean failed")
        return {"status": "error", "error": str(e)}


# ── Sync worker for annotate_with_clip ───────────────────────────────────────

def _annotate_sync(
    image_dir: str,
    class_names: list[str],
    output_dir: str,
    progress_queue: Any,
    clip_model: CLIPModel,
    clip_processor: CLIPProcessor,
) -> dict[str, Any]:
    import cv2

    def _as_feature_tensor(output: Any) -> torch.Tensor:
        if isinstance(output, torch.Tensor):
            return output
        for attr in ("text_embeds", "image_embeds", "pooler_output"):
            val = getattr(output, attr, None)
            if val is not None:
                return val
        if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
            h = output.last_hidden_state
            return h[:, 0, :] if h.ndim == 3 else h
        raise TypeError(f"Unsupported CLIP output type: {type(output).__name__}")

    def fallback_bbox(img: Image.Image) -> tuple[int, int, int, int]:
        """Fast bbox: Otsu threshold + largest contour. No GrabCut."""
        arr = np.array(img.convert("RGB"))
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        _, fg = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return (0, 0, img.width, img.height)
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        return (0, 0, img.width, img.height) if w <= 1 or h <= 1 else (int(x), int(y), int(w), int(h))

    def bbox_to_yolo(x: int, y: int, w: int, h: int, iw: int, ih: int):
        return round((x + w/2)/iw, 6), round((y + h/2)/ih, 6), round(w/iw, 6), round(h/ih, 6)

    image_root = Path(image_dir)
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    labels_root = out_root / "labels"
    labels_root.mkdir(parents=True, exist_ok=True)
    anno_dir = out_root / "annotations"
    anno_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted([p for p in image_root.rglob("*")
                          if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if not image_paths:
        return {"status": "error", "error": "No images found for annotation."}

    progress_queue.put_nowait("> CLIP model ready. Starting annotation...")
    device = next(clip_model.parameters()).device
    model = clip_model
    processor = clip_processor

    prompts = [f"a photo of a {name}" for name in class_names]
    with torch.no_grad():
        text_inputs = processor(text=prompts, return_tensors="pt", padding=True)
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
        text_features = _as_feature_tensor(model.get_text_features(**text_inputs))
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    annotations: list[dict] = []
    coco_records: list[dict] = []
    class_confidence: dict[str, list[float]] = defaultdict(list)
    verified_count = low_confidence_count = 0

    for batch_start in range(0, len(image_paths), 16):
        batch_paths = image_paths[batch_start:batch_start + 16]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        with torch.no_grad():
            image_inputs = processor(images=images, return_tensors="pt")
            image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
            image_features = _as_feature_tensor(model.get_image_features(**image_inputs))
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            probs = (image_features @ text_features.T).softmax(dim=-1).cpu().numpy()

        for li, image_path in enumerate(batch_paths):
            rel_path = image_path.relative_to(image_root)
            split = rel_path.parts[0] if len(rel_path.parts) > 1 else "train"
            prob_vector = probs[li]
            predicted_index = int(prob_vector.argmax())
            predicted_class = class_names[predicted_index]
            confidence = float(prob_vector[predicted_index])
            if image_path.parent.name == predicted_class:
                verified_count += 1
            if confidence < 0.6:
                low_confidence_count += 1
            class_confidence[predicted_class].append(confidence)

            img_obj = images[li]
            iw, ih = img_obj.size
            x_abs, y_abs, w_abs, h_abs = fallback_bbox(img_obj)
            cx, cy, nw, nh = bbox_to_yolo(x_abs, y_abs, w_abs, h_abs, iw, ih)

            lbl_dir = labels_root / split
            lbl_dir.mkdir(parents=True, exist_ok=True)
            (lbl_dir / f"{image_path.stem}.txt").write_text(
                f"{predicted_index} {cx} {cy} {nw} {nh}\n", encoding="utf-8")

            coco_records.append({
                "file_name": f"images/{rel_path}".replace("\\", "/"),
                "split": split, "width": iw, "height": ih,
                "class_id": predicted_index, "bbox_abs": [x_abs, y_abs, w_abs, h_abs],
            })
            annotations.append({
                "image_path": str(rel_path).replace("\\", "/"),
                "original_label": image_path.parent.name,
                "clip_predicted_class": predicted_class,
                "clip_confidence": confidence,
                "verified": image_path.parent.name == predicted_class,
            })

        processed = batch_start + len(batch_paths)
        if processed % 50 == 0 or processed == len(image_paths):
            progress_queue.put_nowait(f"> Annotated {processed}/{len(image_paths)} images with CLIP + Bboxes...")

    # Write YOLO data.yaml
    (out_root / "data.yaml").write_text(
        f"path: .\ntrain: images/train\nval: images/val\ntest: images/test\n"
        f"nc: {len(class_names)}\nnames: {json.dumps(class_names)}\n", encoding="utf-8")

    # Write COCO JSONs
    coco_categories = [{"id": i, "name": c, "supercategory": "object"} for i, c in enumerate(class_names)]
    for split_name in ("train", "val", "test"):
        split_records = [r for r in coco_records if r["split"] == split_name]
        if not split_records:
            continue
        coco_images, coco_annos = [], []
        for img_id, rec in enumerate(split_records):
            x, y, w, h = rec["bbox_abs"]
            coco_images.append({"id": img_id, "file_name": rec["file_name"], "width": rec["width"], "height": rec["height"]})
            coco_annos.append({"id": img_id, "image_id": img_id, "category_id": rec["class_id"],
                               "bbox": [x, y, w, h], "area": w * h, "iscrowd": 0})
        (anno_dir / f"instances_{split_name}.json").write_text(
            json.dumps({"info": {"description": "VisCurator CLIP Dataset", "version": "1.0"},
                        "categories": coco_categories, "images": coco_images, "annotations": coco_annos}, indent=2),
            encoding="utf-8")

    class_breakdown = {cn: {"count": len(s), "avg_confidence": sum(s)/len(s) if s else 0.0}
                       for cn, s in class_confidence.items()}
    summary = {
        "verified_count": verified_count,
        "verification_rate": verified_count / max(len(annotations), 1),
        "low_confidence_count": low_confidence_count,
        "class_breakdown": class_breakdown,
    }
    (anno_dir / "annotations.json").write_text(
        json.dumps({"metadata": {"tool": "CLIP zero-shot + OpenCV GrabCut",
                                 "model": "openai/clip-vit-base-patch32",
                                 "class_names": class_names, "total_images": len(annotations),
                                 "timestamp": datetime.now(timezone.utc).isoformat()},
                    "annotations": annotations, "summary": summary}, indent=2), encoding="utf-8")

    # Deblur previews
    deblur_preview: list[dict] = []
    previews_dir = image_root.parent / "_previews"
    blurry_dir = previews_dir / "blurry"
    if blurry_dir.exists():
        deblur_out = previews_dir / "deblurred"
        deblur_out.mkdir(parents=True, exist_ok=True)
        for bp in sorted(blurry_dir.glob("*.jpg"))[:8]:
            try:
                orig = Image.open(bp).convert("RGB")
                arr = np.array(orig, dtype=np.float32)
                blurred = cv2.GaussianBlur(arr, (0, 0), 3)
                sharp_arr = np.clip(arr + 1.5 * (arr - blurred), 0, 255).astype(np.uint8)
                sharp_img = Image.fromarray(sharp_arr)
                out_p = deblur_out / bp.name
                sharp_img.save(out_p, "JPEG", quality=85)
                before_blur = round(float(cv2.Laplacian(np.array(orig.convert("L"), dtype=np.float32), cv2.CV_32F).var()), 1)
                after_blur = round(float(cv2.Laplacian(np.array(sharp_img.convert("L"), dtype=np.float32), cv2.CV_32F).var()), 1)
                deblur_preview.append({
                    "id": len(deblur_preview), "label": "blurry",
                    "before_url": str(bp.relative_to(image_root.parent.parent)).replace("\\", "/"),
                    "after_url": str(out_p.relative_to(image_root.parent.parent)).replace("\\", "/"),
                    "before_blur": before_blur, "after_blur": after_blur,
                })
            except Exception:
                pass

    return {"status": "success", **summary, "summary": summary,
            "output_dir": str(out_root), "deblur_preview": deblur_preview}


async def annotate_with_clip(
    image_dir: str,
    class_names: list[str],
    output_dir: str,
    emit_callback: Any = None,
) -> dict[str, Any]:
    try:
        await _safe_emit(emit_callback, "> Loading CLIP model (cached after first run)...")
        model, processor = await _get_clip()
        await _safe_emit(emit_callback, "> CLIP ready. Starting annotation in background thread...")

        progress_queue: asyncio.Queue = asyncio.Queue()

        async def _drain():
            while True:
                msg = await progress_queue.get()
                if msg is None:
                    break
                await _safe_emit(emit_callback, msg)

        drain_task = asyncio.create_task(_drain())

        result = await asyncio.to_thread(
            _annotate_sync, image_dir, class_names, output_dir, progress_queue, model, processor
        )

        await progress_queue.put(None)
        await drain_task
        return result
    except Exception as e:
        logger.exception("annotate_with_clip failed")
        return {"status": "error", "error": str(e)}
