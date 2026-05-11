"""
VisCurator / CVAgent — Heavyweight Annotation Engine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Utilizes SAM (Segment Anything Model) and CLIP to auto-annotate and 
segment a raw dataset based on user-provided few-shot seeds.

Runs as an isolated subprocess to manage heavy memory/GPU requirements.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any
from collections import Counter

import numpy as np
import torch
import cv2
import imagehash
from PIL import Image

try:
    from transformers import CLIPProcessor, CLIPModel, SamModel, SamProcessor
except ImportError:
    print("EVENT:" + json.dumps({"event": "error", "message": "transformers library is missing. Install with: pip install transformers>=4.30.0"}))
    sys.exit(1)

def emit_event(event: str, message: str, **data: Any) -> None:
    payload = {"event": event, "message": message, **data}
    print("EVENT:" + json.dumps(payload), flush=True)

def laplacian_var(img: Image.Image) -> float:
    arr = np.array(img.convert("L"), dtype=np.float32)
    lap = cv2.Laplacian(arr, cv2.CV_32F)
    return float(lap.var())

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    seed_dir = out_dir / "seeds"
    
    samples_dir = out_dir / "samples"
    (samples_dir / "raw").mkdir(parents=True, exist_ok=True)
    (samples_dir / "filtered").mkdir(parents=True, exist_ok=True)
    (samples_dir / "processed").mkdir(parents=True, exist_ok=True)

    if not seed_dir.exists() or not any(seed_dir.iterdir()):
        emit_event("error", "No seeds uploaded. Please define classes and upload seeds first.")
        sys.exit(1)

    emit_event("started", "Initializing Foundation Models. This may take a moment to download weights on first run...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        # Load CLIP for Classification
        emit_event("log", "Loading CLIP (openai/clip-vit-base-patch32)...")
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Load SAM for Segmentation
        emit_event("log", "Loading SAM (facebook/sam-vit-base)...")
        sam_model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
        sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    except Exception as e:
        emit_event("error", f"Failed to load models. Ensure sufficient RAM/VRAM. Error: {e}")
        sys.exit(1)

    emit_event("log", "Models loaded successfully. Preparing seed embeddings...")

    # 1. Compute Seed Embeddings from uploaded directories
    seed_embeddings = {}
    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
    for class_dir in seed_dir.iterdir():
        if not class_dir.is_dir(): continue
        class_name = class_dir.name
        embeddings = []
        for img_path in class_dir.rglob("*"):
            if not img_path.is_file() or img_path.suffix.lower() not in valid_exts: continue
            try:
                img = Image.open(img_path).convert("RGB")
                inputs = clip_processor(images=img, return_tensors="pt").to(device)
                with torch.no_grad():
                    embed = clip_model.get_image_features(**inputs)
                    embed = embed / embed.norm(p=2, dim=-1, keepdim=True)
                    embeddings.append(embed)
            except Exception as e:
                emit_event("log", f"Failed to process seed {img_path.name}: {e}")
        
        if embeddings:
            # Average the embeddings for this class to create a class prototype
            avg_embed = torch.stack(embeddings).mean(dim=0)
            avg_embed = avg_embed / avg_embed.norm(p=2, dim=-1, keepdim=True)
            seed_embeddings[class_name] = avg_embed

    if not seed_embeddings:
        emit_event("error", "Failed to compute any valid seed embeddings. Cannot proceed.")
        sys.exit(1)

    classes = list(seed_embeddings.keys())
    class_tensors = torch.cat([seed_embeddings[c] for c in classes])

    # Ensure output directories exist
    train_dir = out_dir / "train"
    for c in classes:
        (train_dir / c).mkdir(parents=True, exist_ok=True)

    # Gather all images to process
    all_images = [f for f in raw_dir.rglob("*") if f.is_file() and f.suffix.lower() in valid_exts]
    total_imgs = len(all_images)
    
    emit_event("log", f"Processing {total_imgs} images...")

    report = {
        "job_id": args.job_id,
        "classes": classes,
        "class_counts": {c: 0 for c in classes},
        "confidence_distribution": {"high": 0, "medium": 0, "low": 0},
        "annotated_samples": []
    }
    
    preprocess_report = {
        "dataset_id": args.job_id,
        "output_dir": str(out_dir),
        "plan": {"reasoning": "Auto-Annotation Pipeline"},
        "stage_samples": {"raw": [], "filtered": [], "processed": []},
        "before_stats": {"images": total_imgs, "class_distribution": {}},
        "after_stats": {
            "images": 0,
            "blur_filtered": 0,
            "duplicates_removed": 0,
            "class_distribution": {}
        },
        "blur_scatter": []
    }

    seen_hashes = set()
    BLUR_THRESHOLD_LARGE = 80.0
    BLUR_THRESHOLD_SMALL = 8.0

    # 2. Process, Clean, Classify, and Segment
    for idx, img_path in enumerate(all_images):
        try:
            raw_img = Image.open(img_path).convert("RGB")
            
            # --- CLEANING (Blur / Dup) ---
            blur_value = laplacian_var(raw_img)
            img_hash = str(imagehash.dhash(raw_img))
            _thresh = BLUR_THRESHOLD_SMALL if (raw_img.width < 128 or raw_img.height < 128) else BLUR_THRESHOLD_LARGE
            is_blurry = blur_value < _thresh
            is_dup = img_hash in seen_hashes
            
            if len(preprocess_report["stage_samples"]["raw"]) < 5:
                fname = f"raw_{idx:05d}.jpg"
                raw_img.save(samples_dir / "raw" / fname, "JPEG", quality=85)
                preprocess_report["stage_samples"]["raw"].append({"url": f"samples/raw/{fname}", "label": "unlabeled", "id": idx})

            preprocess_report["blur_scatter"].append({
                "id": idx,
                "laplacian": round(blur_value, 1),
                "resolution": int((raw_img.width * raw_img.height) / 1000),
                "accepted": not is_blurry and not is_dup,
            })
            
            if is_blurry or is_dup:
                if is_blurry: preprocess_report["after_stats"]["blur_filtered"] += 1
                if is_dup: preprocess_report["after_stats"]["duplicates_removed"] += 1
                
                if len(preprocess_report["stage_samples"]["filtered"]) < 5:
                    reason = "blurry" if is_blurry else "duplicate"
                    fname = f"filtered_{idx:05d}.jpg"
                    raw_img.save(samples_dir / "filtered" / fname, "JPEG", quality=85)
                    preprocess_report["stage_samples"]["filtered"].append({"url": f"samples/filtered/{fname}", "label": "unlabeled", "reason": reason, "id": idx})
                continue
                
            seen_hashes.add(img_hash)
            
            # --- CLASSIFICATION (CLIP) ---
            inputs = clip_processor(images=raw_img, return_tensors="pt").to(device)
            with torch.no_grad():
                img_embed = clip_model.get_image_features(**inputs)
                img_embed = img_embed / img_embed.norm(p=2, dim=-1, keepdim=True)
                
                similarities = (img_embed @ class_tensors.T).squeeze(0)
                best_idx = similarities.argmax().item()
                best_score = similarities[best_idx].item()
                assigned_class = classes[best_idx]
            
            # --- SEGMENTATION (SAM) ---
            w, h = raw_img.size
            input_points = [[[w // 2, h // 2]]] 
            sam_inputs = sam_processor(raw_img, input_points=input_points, return_tensors="pt").to(device)
            
            with torch.no_grad():
                sam_outputs = sam_model(**sam_inputs)
                
            masks = sam_processor.image_processor.post_process_masks(
                sam_outputs.pred_masks.cpu(), sam_inputs["original_sizes"].cpu(), sam_inputs["reshaped_input_sizes"].cpu()
            )
            
            best_mask = masks[0][0][0].numpy()
            
            masked_np = np.array(raw_img)
            white_bg = np.ones_like(masked_np) * 255
            masked_img_np = np.where(best_mask[:, :, None], masked_np, white_bg)
            final_img = Image.fromarray(masked_img_np.astype(np.uint8))
            
            save_name = f"anno_{idx:05d}.jpg"
            save_path = train_dir / assigned_class / save_name
            final_img.save(save_path, "JPEG", quality=90)
            
            # Update Stats
            preprocess_report["after_stats"]["images"] += 1
            preprocess_report["after_stats"]["class_distribution"][assigned_class] = preprocess_report["after_stats"]["class_distribution"].get(assigned_class, 0) + 1
            
            if len(preprocess_report["stage_samples"]["processed"]) < 5:
                fname = f"proc_{idx:05d}.jpg"
                final_img.save(samples_dir / "processed" / fname, "JPEG", quality=85)
                preprocess_report["stage_samples"]["processed"].append({"url": f"samples/processed/{fname}", "label": assigned_class, "id": idx})
            
            report["class_counts"][assigned_class] += 1
            if best_score > 0.8: report["confidence_distribution"]["high"] += 1
            elif best_score > 0.5: report["confidence_distribution"]["medium"] += 1
            else: report["confidence_distribution"]["low"] += 1
            
            if len(report["annotated_samples"]) < 12:
                report["annotated_samples"].append({
                    "url": f"train/{assigned_class}/{save_name}",
                    "label": assigned_class,
                    "confidence": round(best_score, 2)
                })

            if idx % 10 == 0 or idx == total_imgs - 1:
                progress = int((idx + 1) / total_imgs * 100)
                emit_event("progress", f"Processed {idx+1}/{total_imgs} images", progress=progress)

        except Exception as e:
            emit_event("log", f"Failed to process {img_path.name}: {e}")

    # Write reports
    anno_report_path = out_dir / "annotation_report.json"
    anno_report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    
    prep_report_path = out_dir / "preprocessing_report.json"
    prep_report_path.write_text(json.dumps(preprocess_report, indent=2), encoding="utf-8")
    
    # Emit the preprocess report event so the UI can catch it
    emit_event("report", "Pipeline finished", **preprocess_report)
    emit_event("completed", "Auto-annotation finished successfully.", report_path=str(anno_report_path))

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        emit_event("error", f"Fatal annotator error: {exc}")
