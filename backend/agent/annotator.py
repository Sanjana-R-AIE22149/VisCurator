"""
VisCurator / CVAgent — Heavyweight Annotation Engine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Pipeline:
  1. CLIP few-shot classification   — assigns each image to a class
  2. SAM segmentation               — produces a binary mask for the main object
  3. Mask → tight bounding box      — derives (x, y, w, h) from mask contour
  4. Dual export:
       YOLO  → labels/<split>/<image>.txt   (class_idx cx cy w h, normalised 0-1)
       COCO  → annotations/instances_<split>.json

Runs as an isolated subprocess (heavy RAM/GPU usage).
"""

import argparse
import json
import logging
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import imagehash
import numpy as np
import torch
from PIL import Image

try:
    from transformers import CLIPModel, CLIPProcessor, SamModel, SamProcessor
except ImportError:
    print("EVENT:" + json.dumps({"event": "error", "message": "transformers library is missing. pip install transformers>=4.30.0"}))
    sys.exit(1)


# ── Helpers ───────────────────────────────────────────────────────────────────

def emit_event(event: str, message: str, **data: Any) -> None:
    payload = {"event": event, "message": message, **data}
    print("EVENT:" + json.dumps(payload), flush=True)


def laplacian_var(img: Image.Image) -> float:
    arr = np.array(img.convert("L"), dtype=np.float32)
    return float(cv2.Laplacian(arr, cv2.CV_32F).var())


def mask_to_tight_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Convert a binary boolean/uint8 mask to a tight (x, y, w, h) bounding box.

    Returns None if the mask is empty (no foreground pixels).
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return x0, y0, x1 - x0 + 1, y1 - y0 + 1   # x, y, w, h  (pixel absolute)


def bbox_to_yolo(x: int, y: int, w: int, h: int, img_w: int, img_h: int) -> tuple[float, float, float, float]:
    """Convert absolute (x, y, w, h) to normalised YOLO (cx, cy, w, h)."""
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    return round(cx, 6), round(cy, 6), round(nw, 6), round(nh, 6)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id",         required=True)
    parser.add_argument("--raw-dir",        required=True)
    parser.add_argument("--out-dir",        required=True)
    parser.add_argument("--min-confidence", type=float, default=0.30,
                        help="CLIP cosine-similarity threshold. Below → low_confidence/")
    parser.add_argument("--blur-threshold", type=float, default=80.0,
                        help="Laplacian-variance threshold. Below → blur rejected.")
    args = parser.parse_args()

    raw_dir     = Path(args.raw_dir)
    out_dir     = Path(args.out_dir)
    seed_dir    = out_dir / "seeds"
    min_conf    = args.min_confidence

    # ── Output directories ────────────────────────────────────────────────────
    samples_dir        = out_dir / "samples"
    blur_rejected_dir  = out_dir / "blur_rejected"
    low_conf_dir       = out_dir / "low_confidence"
    images_train_dir   = out_dir / "dataset" / "images" / "train"
    images_val_dir     = out_dir / "dataset" / "images" / "val"
    labels_train_dir   = out_dir / "dataset" / "labels" / "train"
    labels_val_dir     = out_dir / "dataset" / "labels" / "val"
    anno_dir           = out_dir / "dataset" / "annotations"

    for d in [
        samples_dir / "raw", samples_dir / "filtered", samples_dir / "processed",
        blur_rejected_dir, low_conf_dir,
        images_train_dir, images_val_dir,
        labels_train_dir, labels_val_dir,
        anno_dir,
    ]:
        d.mkdir(parents=True, exist_ok=True)

    # ── Guard: seeds must exist ───────────────────────────────────────────────
    if not seed_dir.exists() or not any(seed_dir.iterdir()):
        emit_event("error", "No seeds uploaded. Define classes and upload seed images first.")
        sys.exit(1)

    emit_event("started", "Initialising Foundation Models (CLIP + SAM). First run downloads weights…")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        emit_event("log", "Loading CLIP (openai/clip-vit-base-patch32)…")
        clip_model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        emit_event("log", "Loading SAM (facebook/sam-vit-base)…")
        sam_model     = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
        sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    except Exception as e:
        emit_event("error", f"Failed to load models: {e}")
        sys.exit(1)

    emit_event("log", "Models loaded. Computing seed class embeddings…")

    # ── Seed embeddings ───────────────────────────────────────────────────────
    seed_embeddings: dict[str, torch.Tensor] = {}
    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}

    for class_dir in seed_dir.iterdir():
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        embeddings: list[torch.Tensor] = []
        for img_path in class_dir.rglob("*"):
            if not img_path.is_file() or img_path.suffix.lower() not in valid_exts:
                continue
            try:
                img    = Image.open(img_path).convert("RGB")
                inputs = clip_processor(images=img, return_tensors="pt").to(device)
                with torch.no_grad():
                    embed = clip_model.get_image_features(**inputs)
                    embed = embed / embed.norm(p=2, dim=-1, keepdim=True)
                    embeddings.append(embed)
            except Exception as exc:
                emit_event("log", f"Skipping seed {img_path.name}: {exc}")

        if embeddings:
            avg = torch.stack(embeddings).mean(dim=0)
            seed_embeddings[class_name] = avg / avg.norm(p=2, dim=-1, keepdim=True)

    if not seed_embeddings:
        emit_event("error", "Could not compute any seed embeddings. Check seed images.")
        sys.exit(1)

    classes        = sorted(seed_embeddings.keys())
    class_idx_map  = {c: i for i, c in enumerate(classes)}
    class_tensors  = torch.cat([seed_embeddings[c] for c in classes])   # (N_cls, D)

    emit_event("log", f"Classes ({len(classes)}): {', '.join(classes)}")

    # ── Collect raw images ────────────────────────────────────────────────────
    all_images = [f for f in raw_dir.rglob("*") if f.is_file() and f.suffix.lower() in valid_exts]
    total_imgs = len(all_images)
    emit_event("log", f"Found {total_imgs} images to process.")

    # ── Per-image state ───────────────────────────────────────────────────────
    seen_hashes: set[str] = set()
    BLUR_LARGE: float     = args.blur_threshold
    BLUR_SMALL: float     = args.blur_threshold / 10.0

    # COCO accumulators (train + val split 80/20)
    coco_records: list[dict] = []   # will be split later

    # Stats for the frontend report
    stats = Counter()
    class_counts: dict[str, int] = {c: 0 for c in classes}
    conf_dist: dict[str, int]    = {"high": 0, "medium": 0, "low": 0}
    low_conf_count = 0
    blur_filtered  = 0
    dup_filtered   = 0
    annotated_samples: list[dict] = []
    low_conf_samples:  list[dict] = []
    blur_scatter:      list[dict] = []
    stage_samples:     dict       = {"raw": [], "filtered": [], "processed": []}

    # ── Main processing loop ──────────────────────────────────────────────────
    for idx, img_path in enumerate(all_images):
        try:
            raw_img = Image.open(img_path).convert("RGB")
            img_w, img_h = raw_img.size

            # ── Blur / duplicate filter ───────────────────────────────────────
            blur_val = laplacian_var(raw_img)
            img_hash = str(imagehash.dhash(raw_img))
            thresh   = BLUR_SMALL if (img_w < 128 or img_h < 128) else BLUR_LARGE
            is_blurry = blur_val < thresh
            is_dup    = img_hash in seen_hashes

            if len(stage_samples["raw"]) < 5:
                fname = f"raw_{idx:05d}.jpg"
                raw_img.save(samples_dir / "raw" / fname, "JPEG", quality=85)
                stage_samples["raw"].append({"url": f"samples/raw/{fname}", "label": "unlabeled", "id": idx})

            blur_scatter.append({
                "id": idx,
                "laplacian": round(blur_val, 1),
                "resolution": int((img_w * img_h) / 1000),
                "accepted": not is_blurry and not is_dup,
            })

            if is_blurry or is_dup:
                if is_blurry:
                    blur_filtered += 1
                    if not is_dup:
                        raw_img.save(blur_rejected_dir / f"blur_{idx:05d}.jpg", "JPEG", quality=92)
                if is_dup:
                    dup_filtered += 1
                if len(stage_samples["filtered"]) < 5:
                    reason = "blurry" if is_blurry else "duplicate"
                    fname  = f"filtered_{idx:05d}.jpg"
                    raw_img.save(samples_dir / "filtered" / fname, "JPEG", quality=85)
                    stage_samples["filtered"].append({"url": f"samples/filtered/{fname}", "label": "unlabeled", "reason": reason, "id": idx})
                continue

            seen_hashes.add(img_hash)

            # ── CLIP classification ───────────────────────────────────────────
            with torch.no_grad():
                inputs    = clip_processor(images=raw_img, return_tensors="pt").to(device)
                img_embed = clip_model.get_image_features(**inputs)
                img_embed = img_embed / img_embed.norm(p=2, dim=-1, keepdim=True)
                sims      = (img_embed @ class_tensors.T).squeeze(0)
                best_idx  = int(sims.argmax().item())
                best_score = float(sims[best_idx].item())
                assigned_class = classes[best_idx]
                class_id       = class_idx_map[assigned_class]

            # ── Confidence gate ───────────────────────────────────────────────
            if best_score < min_conf:
                low_conf_count += 1
                lc_dir  = low_conf_dir / assigned_class
                lc_dir.mkdir(parents=True, exist_ok=True)
                lc_name = f"lc_{idx:05d}.jpg"
                raw_img.save(lc_dir / lc_name, "JPEG", quality=90)
                if len(low_conf_samples) < 12:
                    low_conf_samples.append({
                        "url":        f"low_confidence/{assigned_class}/{lc_name}",
                        "label":      assigned_class,
                        "confidence": round(best_score, 3),
                    })
                if idx % 10 == 0 or idx == total_imgs - 1:
                    emit_event("progress", f"Processed {idx+1}/{total_imgs}", progress=int((idx+1)/total_imgs*100))
                continue

            # ── SAM segmentation → bounding box ──────────────────────────────
            input_points = [[[img_w // 2, img_h // 2]]]   # centre-point prompt
            sam_inputs   = sam_processor(raw_img, input_points=input_points, return_tensors="pt").to(device)
            with torch.no_grad():
                sam_out  = sam_model(**sam_inputs)

            masks = sam_processor.image_processor.post_process_masks(
                sam_out.pred_masks.cpu(),
                sam_inputs["original_sizes"].cpu(),
                sam_inputs["reshaped_input_sizes"].cpu(),
            )
            # masks[0] shape: (1, num_masks, H, W) — take best mask (index 0)
            best_mask = masks[0][0][0].numpy()   # (H, W) bool

            bbox_abs = mask_to_tight_bbox(best_mask)
            if bbox_abs is None:
                # Fallback: full-image bbox
                bbox_abs = (0, 0, img_w, img_h)

            x_abs, y_abs, w_abs, h_abs = bbox_abs
            cx, cy, nw, nh = bbox_to_yolo(x_abs, y_abs, w_abs, h_abs, img_w, img_h)

            # ── Determine train / val split (80 / 20 round-robin) ────────────
            split     = "train" if (idx % 5 != 4) else "val"
            img_fname = f"{assigned_class}_{idx:05d}.jpg"

            # Save image
            dest_img = (images_train_dir if split == "train" else images_val_dir) / img_fname
            raw_img.save(dest_img, "JPEG", quality=92)

            # Write YOLO label file
            dest_lbl  = (labels_train_dir if split == "train" else labels_val_dir) / f"{Path(img_fname).stem}.txt"
            dest_lbl.write_text(f"{class_id} {cx} {cy} {nw} {nh}\n", encoding="utf-8")

            # Accumulate COCO record
            coco_records.append({
                "file_name":  img_fname,
                "split":      split,
                "width":      img_w,
                "height":     img_h,
                "class_id":   class_id,
                "class_name": assigned_class,
                "bbox_abs":   [x_abs, y_abs, w_abs, h_abs],
                "score":      round(best_score, 4),
            })

            # Update counters
            class_counts[assigned_class] += 1
            if best_score > 0.8:    conf_dist["high"]   += 1
            elif best_score > 0.5:  conf_dist["medium"] += 1
            else:                   conf_dist["low"]     += 1

            # Preview sample for UI
            if len(stage_samples["processed"]) < 5:
                fname_p = f"proc_{idx:05d}.jpg"
                raw_img.save(samples_dir / "processed" / fname_p, "JPEG", quality=85)
                stage_samples["processed"].append({"url": f"samples/processed/{fname_p}", "label": assigned_class, "id": idx})

            if len(annotated_samples) < 24:
                annotated_samples.append({
                    "url":        f"dataset/images/{split}/{img_fname}",
                    "label":      assigned_class,
                    "confidence": round(best_score, 3),
                    "bbox":       [x_abs, y_abs, w_abs, h_abs],
                    "confidence_band": "high" if best_score > 0.8 else "medium" if best_score > 0.5 else "low",
                })

            if idx % 10 == 0 or idx == total_imgs - 1:
                emit_event("progress", f"Processed {idx+1}/{total_imgs}", progress=int((idx+1)/total_imgs*100))

        except Exception as exc:
            emit_event("log", f"Skipping {img_path.name}: {exc}")

    # ── Write data.yaml (YOLO) ────────────────────────────────────────────────
    yaml_path = out_dir / "dataset" / "data.yaml"
    yaml_path.write_text(
        f"path: ./dataset\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: {len(classes)}\n"
        f"names: {json.dumps(classes)}\n",
        encoding="utf-8",
    )

    # ── Write COCO JSON (train + val) ─────────────────────────────────────────
    coco_categories = [{"id": i, "name": c, "supercategory": "object"} for i, c in enumerate(classes)]

    for split_name in ("train", "val"):
        split_records = [r for r in coco_records if r["split"] == split_name]
        coco_images, coco_annos = [], []
        for img_id, rec in enumerate(split_records):
            x, y, w, h = rec["bbox_abs"]
            coco_images.append({
                "id":        img_id,
                "file_name": rec["file_name"],
                "width":     rec["width"],
                "height":    rec["height"],
            })
            coco_annos.append({
                "id":          img_id,
                "image_id":    img_id,
                "category_id": rec["class_id"],
                "bbox":        [x, y, w, h],   # COCO: [x, y, width, height]
                "area":        w * h,
                "iscrowd":     0,
            })

        coco_out = {
            "info":        {"description": "VisCurator auto-annotation", "version": "1.0"},
            "categories":  coco_categories,
            "images":      coco_images,
            "annotations": coco_annos,
        }
        (anno_dir / f"instances_{split_name}.json").write_text(
            json.dumps(coco_out, indent=2), encoding="utf-8"
        )

    total_annotated = sum(class_counts.values())

    # ── Legacy preprocessing_report.json (for pipeline UI compatibility) ──────
    preprocess_report = {
        "dataset_id":   args.job_id,
        "output_dir":   str(out_dir / "dataset"),
        "plan":         {"reasoning": "CLIP+SAM Auto-Annotation Pipeline"},
        "stage_samples": stage_samples,
        "before_stats": {"images": total_imgs, "class_distribution": {}},
        "after_stats":  {
            "images":             total_annotated,
            "blur_filtered":      blur_filtered,
            "duplicates_removed": dup_filtered,
            "class_distribution": class_counts,
        },
        "blur_scatter": blur_scatter[:200],
        "formats": {
            "yolo": "dataset/data.yaml + labels/train/*.txt + labels/val/*.txt",
            "coco": "dataset/annotations/instances_train.json + instances_val.json",
        },
    }
    (out_dir / "preprocessing_report.json").write_text(
        json.dumps(preprocess_report, indent=2), encoding="utf-8"
    )

    # ── annotation_report.json (for annotation panel UI) ─────────────────────
    anno_report = {
        "job_id":                  args.job_id,
        "classes":                 classes,
        "min_confidence":          min_conf,
        "total_processed":         total_imgs,
        "total_annotated":         total_annotated,
        "low_confidence_count":    low_conf_count,
        "blur_rejected":           blur_filtered,
        "duplicates_removed":      dup_filtered,
        "class_counts":            class_counts,
        "confidence_distribution": conf_dist,
        "annotated_samples":       annotated_samples,
        "low_confidence_samples":  low_conf_samples,
        "exports": {
            "yolo_data_yaml":          str(yaml_path.relative_to(out_dir)),
            "coco_train":              "dataset/annotations/instances_train.json",
            "coco_val":                "dataset/annotations/instances_val.json",
        },
    }
    (out_dir / "annotation_report.json").write_text(
        json.dumps(anno_report, indent=2), encoding="utf-8"
    )

    emit_event("report", "Pipeline finished", **preprocess_report)
    emit_event(
        "completed",
        f"Auto-annotation done. {total_annotated} annotated, "
        f"{low_conf_count} low-confidence, {blur_filtered} blur-rejected. "
        f"YOLO + COCO labels written.",
        report_path=str(out_dir / "annotation_report.json"),
        annotation_report=anno_report,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        emit_event("error", f"Fatal annotator error: {exc}")
        sys.exit(1)
