"""
VisCurator — Augmentation Agent
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Standalone agent that ingests any ImageFolder-style dataset and produces an
augmented superset, increasing dataset size by a configurable multiplier.

Features
--------
* Class-aware balancing  — minority classes get more augmentations so the
  output dataset is balanced (or uses the user's target multiplier for each).
* Rich augmentation strategies — 10+ spatial + photometric + noise transforms
  selectable by profile: 'light', 'medium', 'heavy', 'medical', 'adversarial'.
* Label-preserving  — augmentations never cross class boundaries; the original
  images are ALWAYS preserved in the output alongside variants.
* Duplicate-safe  — perceptual hash deduplication before saving.
* JSON progress events on stdout (prefix: AUGAGENT_EVENT:) so the FastAPI host
  can relay them to the UI WebSocket.
* Writes an augmentation_report.json summary at the end.

CLI usage (called by the FastAPI backend via subprocess):
    python augmentation_agent.py
        --job-id    <uuid>
        --input-dir <path/to/ImageFolder>
        --out-dir   <path/to/output>
        --strategy  medium          # light | medium | heavy | medical | adversarial
        --multiplier 3              # output = input × multiplier (default 3)
        --target-size 0             # overrides multiplier if > 0 (total images per class)
        --target-px  224            # resize all images to this square resolution
        --balance    true           # balance classes to the majority count (default true)
        --max-workers 4             # parallel worker threads
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

try:
    import albumentations as A
    ALBUMENTATIONS_OK = True
except ImportError:
    ALBUMENTATIONS_OK = False

try:
    import imagehash
    IMAGEHASH_OK = True
except ImportError:
    IMAGEHASH_OK = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("augmentation_agent")

VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


# ─── Event emitter ──────────────────────────────────────────────────────────

def emit(event: str, message: str, **data: Any) -> None:
    payload = {"event": event, "message": message, **data}
    print("AUGAGENT_EVENT:" + json.dumps(payload), flush=True)


# ─── Augmentation pipeline factory ──────────────────────────────────────────

def _build_pipeline(strategy: str, target_px: int) -> "A.Compose":
    """Return an albumentations Compose pipeline for the given strategy."""

    resize = A.Resize(target_px, target_px, interpolation=cv2.INTER_AREA)

    LIGHT = [
        resize,
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.03, scale_limit=0.08, rotate_limit=10,
                           border_mode=cv2.BORDER_REFLECT_101, p=0.5),
    ]

    MEDIUM = [
        resize,
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.15),
        A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.12, rotate_limit=20,
                           border_mode=cv2.BORDER_REFLECT_101, p=0.7),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=1.0),
            A.HueSaturationValue(hue_shift_limit=12, sat_shift_limit=25, val_shift_limit=20, p=1.0),
            A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=1.0),
        ], p=0.8),
        A.OneOf([
            A.GaussNoise(std_range=(0.01, 0.04), p=1.0),
            A.ISONoise(color_shift=(0.01, 0.04), intensity=(0.1, 0.25), p=1.0),
        ], p=0.35),
        A.CoarseDropout(num_holes_range=(1, 4),
                        hole_height_range=(8, 24), hole_width_range=(8, 24),
                        fill=0, p=0.3),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
    ]

    HEAVY = [
        resize,
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.25),
        A.Transpose(p=0.2),
        A.ShiftScaleRotate(shift_limit=0.10, scale_limit=0.20, rotate_limit=45,
                           border_mode=cv2.BORDER_REFLECT_101, p=0.8),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.35, p=1.0),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=40, val_shift_limit=30, p=1.0),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=1.0),
        ], p=0.9),
        A.OneOf([
            A.GaussNoise(std_range=(0.02, 0.07), p=1.0),
            A.ISONoise(color_shift=(0.02, 0.07), intensity=(0.2, 0.5), p=1.0),
            A.MultiplicativeNoise(multiplier=(0.85, 1.15), p=1.0),
        ], p=0.5),
        A.OneOf([
            A.MotionBlur(blur_limit=(3, 7), p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        ], p=0.25),
        A.CoarseDropout(num_holes_range=(2, 8),
                        hole_height_range=(8, 32), hole_width_range=(8, 32),
                        fill=0, p=0.4),
        A.RandomGamma(gamma_limit=(70, 130), p=0.4),
        A.ElasticTransform(alpha=30.0, sigma=5.0, p=0.2),
        A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.2),
        A.OpticalDistortion(distort_limit=0.15, p=0.15),
    ]

    # Medical imaging — avoid colour changes, keep structure
    MEDICAL = [
        resize,
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Rotate(limit=20, border_mode=cv2.BORDER_REFLECT_101, p=0.6),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.25, p=0.6),
        A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=0.5),
        A.GaussNoise(std_range=(0.01, 0.03), p=0.3),
        A.ElasticTransform(alpha=20.0, sigma=4.0, p=0.3),
        A.GridDistortion(num_steps=4, distort_limit=0.12, p=0.2),
    ]

    # Adversarial — pushes distribution very far to stress-test models
    ADVERSARIAL = [
        resize,
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.4),
        A.Transpose(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.30, rotate_limit=90,
                           border_mode=cv2.BORDER_REFLECT_101, p=0.9),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5, p=1.0),
            A.HueSaturationValue(hue_shift_limit=30, sat_shift_limit=60, val_shift_limit=40, p=1.0),
            A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.15, p=1.0),
            A.RandomToneCurve(scale=0.15, p=1.0),
        ], p=0.95),
        A.OneOf([
            A.GaussNoise(std_range=(0.03, 0.1), p=1.0),
            A.ISONoise(color_shift=(0.03, 0.1), intensity=(0.3, 0.7), p=1.0),
            A.MultiplicativeNoise(multiplier=(0.7, 1.3), p=1.0),
        ], p=0.6),
        A.OneOf([
            A.MotionBlur(blur_limit=(3, 11), p=1.0),
            A.Defocus(radius=(2, 6), p=1.0),
        ], p=0.35),
        A.CoarseDropout(num_holes_range=(4, 12),
                        hole_height_range=(8, 48), hole_width_range=(8, 48),
                        fill=0, p=0.5),
        A.ElasticTransform(alpha=50.0, sigma=8.0, p=0.3),
    ]

    profiles = {
        "light":       LIGHT,
        "medium":      MEDIUM,
        "heavy":       HEAVY,
        "medical":     MEDICAL,
        "adversarial": ADVERSARIAL,
    }
    chosen = profiles.get(strategy, MEDIUM)
    return A.Compose(chosen)


# ─── Perceptual-hash deduplication ──────────────────────────────────────────

class HashSet:
    """Thread-safe set of perceptual hashes to avoid near-duplicate outputs."""

    def __init__(self) -> None:
        self._hashes: set[str] = set()
        self._lock = threading.Lock()

    def add_and_check(self, img: Image.Image) -> bool:
        """Return True if the image is a near-duplicate (should be skipped)."""
        if not IMAGEHASH_OK:
            return False  # skip dedup if imagehash not installed
        h = str(imagehash.phash(img, hash_size=8))
        with self._lock:
            if h in self._hashes:
                return True
            self._hashes.add(h)
            return False


# ─── Per-image augmentation worker ──────────────────────────────────────────

def _augment_image(
    src_path: Path,
    dst_dir: Path,
    n_variants: int,
    pipeline: "A.Compose",
    hashset: HashSet,
    target_px: int,
) -> dict[str, int]:
    """
    Load src_path, apply pipeline n_variants times, save results to dst_dir.
    Returns {'saved': N, 'skipped_dup': M, 'failed': K}.
    """
    stats = {"saved": 0, "skipped_dup": 0, "failed": 0}
    try:
        pil_orig = Image.open(src_path).convert("RGB")
        arr_orig = np.array(pil_orig)
    except Exception as exc:
        logger.warning("Cannot open %s: %s", src_path.name, exc)
        stats["failed"] += 1
        return stats

    stem = src_path.stem

    for i in range(n_variants):
        try:
            result_arr = pipeline(image=arr_orig)["image"]
            pil_result = Image.fromarray(result_arr)

            if hashset.add_and_check(pil_result):
                stats["skipped_dup"] += 1
                continue

            # Build unique filename
            suffix = f"aug{i:03d}_{stem}"[:128]  # keep names manageable
            out_path = dst_dir / f"{suffix}.jpg"
            # Avoid clobbering on name collision
            counter = 0
            while out_path.exists():
                counter += 1
                out_path = dst_dir / f"{suffix}_{counter}.jpg"

            pil_result.save(out_path, "JPEG", quality=92, optimize=True)
            stats["saved"] += 1
        except Exception as exc:
            logger.warning("Augmentation variant %d for %s failed: %s", i, src_path.name, exc)
            stats["failed"] += 1

    return stats


# ─── Class-aware balancing ───────────────────────────────────────────────────

def compute_per_class_variants(
    class_counts: dict[str, int],
    multiplier: float,
    target_size: int,
    balance: bool,
) -> dict[str, int]:
    """
    Compute how many *augmented* variants to generate per image in each class.

    If balance=True: bring every class up to max(class_counts) × multiplier.
    If target_size > 0: bring every class up to target_size total images.
    Otherwise: apply a flat multiplier to every image.
    """
    if not class_counts:
        return {}

    per_image: dict[str, int] = {}
    max_count = max(class_counts.values())

    for cls, count in class_counts.items():
        if count == 0:
            per_image[cls] = 0
            continue

        if target_size > 0:
            # How many augmented images needed to reach target_size total?
            needed = max(0, target_size - count)
            per_image[cls] = math.ceil(needed / count)
        elif balance:
            # Bring minority classes up to majority × multiplier
            balanced_total = int(max_count * multiplier)
            needed = max(0, balanced_total - count)
            per_image[cls] = math.ceil(needed / count)
        else:
            # Flat multiplier — every image gets N variants
            per_image[cls] = max(1, int(multiplier) - 1)  # -1 because orig counts as 1

    return per_image


# ─── Main logic ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="VisCurator Augmentation Agent")
    parser.add_argument("--job-id",      required=True)
    parser.add_argument("--input-dir",   required=True,  help="ImageFolder root (class subdirs inside)")
    parser.add_argument("--out-dir",     required=True,  help="Output ImageFolder root")
    parser.add_argument("--strategy",    default="medium",
                        choices=["light", "medium", "heavy", "medical", "adversarial"])
    parser.add_argument("--multiplier",  type=float, default=3.0,
                        help="Target total images = original × multiplier (unless --target-size set)")
    parser.add_argument("--target-size", type=int, default=0,
                        help="Target total images PER CLASS; overrides --multiplier if > 0")
    parser.add_argument("--target-px",   type=int, default=224,
                        help="Resize all images to this square resolution")
    parser.add_argument("--balance",     type=str, default="true",
                        help="Balance minority classes to majority count (true/false)")
    parser.add_argument("--max-workers", type=int, default=4)
    args = parser.parse_args()

    balance = args.balance.lower() not in ("false", "0", "no")
    input_dir = Path(args.input_dir)
    out_dir   = Path(args.out_dir)

    if not input_dir.exists():
        emit("error", f"Input directory does not exist: {input_dir}")
        sys.exit(1)

    if not ALBUMENTATIONS_OK:
        emit("error", "albumentations is not installed. Run: pip install albumentations")
        sys.exit(1)

    # ── Discover classes ────────────────────────────────────────────────────
    class_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])
    if not class_dirs:
        # Flat dataset — no subdirectories; treat as single class
        class_dirs = [input_dir]

    class_counts: dict[str, int] = {}
    class_images: dict[str, list[Path]] = {}

    for cd in class_dirs:
        images = [f for f in cd.rglob("*")
                  if f.is_file() and f.suffix.lower() in VALID_EXTS]
        class_counts[cd.name] = len(images)
        class_images[cd.name] = images

    total_input = sum(class_counts.values())
    if total_input == 0:
        emit("error", f"No images found in {input_dir}. Supported: {', '.join(VALID_EXTS)}")
        sys.exit(1)

    emit("started",
         f"Augmentation Agent starting — {total_input} images across {len(class_counts)} class(es).",
         total_input=total_input,
         num_classes=len(class_counts),
         class_counts=class_counts,
         strategy=args.strategy,
         multiplier=args.multiplier,
         target_size=args.target_size,
         balance=balance,
         target_px=args.target_px)

    # ── Compute per-class variants ───────────────────────────────────────────
    per_image_variants = compute_per_class_variants(
        class_counts=class_counts,
        multiplier=args.multiplier,
        target_size=args.target_size,
        balance=balance,
    )

    # ── Build augmentation pipeline ─────────────────────────────────────────
    pipeline = _build_pipeline(args.strategy, args.target_px)
    hashset  = HashSet()

    # ── Create output class dirs & copy originals ────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    for cls_name in class_counts:
        (out_dir / cls_name).mkdir(parents=True, exist_ok=True)

    emit("log", "Copying original images to output directory…")
    copied_originals = 0
    for cls_name, images in class_images.items():
        dst_class_dir = out_dir / cls_name
        for img_path in images:
            try:
                pil = Image.open(img_path).convert("RGB")
                pil_resized = pil.resize((args.target_px, args.target_px), Image.LANCZOS)
                out_path = dst_class_dir / f"orig_{img_path.stem}.jpg"
                pil_resized.save(out_path, "JPEG", quality=95)
                hashset.add_and_check(pil_resized)  # register so dups aren't saved
                copied_originals += 1
            except Exception as exc:
                logger.warning("Failed to copy %s: %s", img_path.name, exc)

    emit("log", f"Copied {copied_originals} originals. Starting augmentation…",
         copied=copied_originals)

    # ── Parallel augmentation ────────────────────────────────────────────────
    global_stats = {
        "saved": 0, "skipped_dup": 0, "failed": 0,
        "classes": {}
    }
    processed_images = 0
    total_tasks = sum(
        len(imgs) for imgs in class_images.values()
        if per_image_variants.get(list(class_images.keys())[list(class_images.values()).index(imgs)], 0) > 0
    )
    # Recalculate total_tasks more accurately
    total_tasks = sum(
        len(class_images[cls]) for cls in class_counts if per_image_variants.get(cls, 0) > 0
    )

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {}
        for cls_name, images in class_images.items():
            n_variants = per_image_variants.get(cls_name, 0)
            if n_variants <= 0:
                emit("log", f"Class '{cls_name}': already at target, skipping augmentation.")
                global_stats["classes"][cls_name] = {
                    "original": len(images), "augmented": 0, "total": len(images)
                }
                continue

            dst_class_dir = out_dir / cls_name
            emit("log",
                 f"Class '{cls_name}': {len(images)} originals → {n_variants} variants each "
                 f"(target ~{len(images) * (1 + n_variants)} total)")
            global_stats["classes"][cls_name] = {
                "original": len(images), "augmented": 0, "total": len(images)
            }

            for img_path in images:
                future = executor.submit(
                    _augment_image,
                    img_path,
                    dst_class_dir,
                    n_variants,
                    pipeline,
                    hashset,
                    args.target_px,
                )
                futures[future] = (cls_name, img_path)

        for future in as_completed(futures):
            cls_name, img_path = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                logger.exception("Worker crashed for %s", img_path)
                result = {"saved": 0, "skipped_dup": 0, "failed": 1}

            global_stats["saved"]       += result["saved"]
            global_stats["skipped_dup"] += result["skipped_dup"]
            global_stats["failed"]      += result["failed"]
            global_stats["classes"][cls_name]["augmented"] += result["saved"]
            global_stats["classes"][cls_name]["total"]     += result["saved"]
            processed_images += 1

            # Emit progress every 10 images or when done
            if processed_images % 10 == 0 or processed_images == total_tasks:
                pct = int(processed_images / max(total_tasks, 1) * 100)
                emit("progress",
                     f"Augmented {processed_images}/{total_tasks} images "
                     f"({global_stats['saved']} variants saved, "
                     f"{global_stats['skipped_dup']} near-dups skipped)",
                     progress=pct,
                     processed=processed_images,
                     total=total_tasks,
                     saved=global_stats["saved"],
                     skipped_dup=global_stats["skipped_dup"],
                     failed=global_stats["failed"])

    # ── Compute final dataset stats ──────────────────────────────────────────
    total_output = sum(
        len([f for f in (out_dir / cls).rglob("*") if f.is_file() and f.suffix.lower() in VALID_EXTS])
        for cls in class_counts
    )

    expansion_ratio = round(total_output / max(total_input, 1), 2)

    # ── Write augmentation report ────────────────────────────────────────────
    report = {
        "job_id":          args.job_id,
        "strategy":        args.strategy,
        "multiplier":      args.multiplier,
        "target_size":     args.target_size,
        "target_px":       args.target_px,
        "balance":         balance,
        "input_dir":       str(input_dir),
        "output_dir":      str(out_dir),
        "total_input":     total_input,
        "total_output":    total_output,
        "expansion_ratio": expansion_ratio,
        "stats":           global_stats,
        "class_summary":   global_stats["classes"],
    }

    report_path = out_dir / "augmentation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    emit("completed",
         f"Done! {total_input} → {total_output} images ({expansion_ratio}× expansion). "
         f"Strategy: {args.strategy} | Balance: {balance}.",
         total_input=total_input,
         total_output=total_output,
         expansion_ratio=expansion_ratio,
         class_summary=global_stats["classes"],
         report_path=str(report_path),
         stats=global_stats)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        emit("error", f"Fatal augmentation agent error: {exc}")
        sys.exit(1)
