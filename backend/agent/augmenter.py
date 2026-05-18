"""
VisCurator / CVAgent — Anti-Blur & Augmentation Engine
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Runs as an isolated subprocess.
For every image rejected as "blurry" by the annotation pipeline:
  1. Applies Wiener / unsharp-mask deconvolution to recover sharpness
  2. Saves the recovered image
  3. Generates N augmented variants (flip, rotate, colour-jitter, cutout)
  4. Writes them into the curated dataset alongside the clean images
  5. Emits JSON progress events so the FastAPI host can relay them to the UI

Usage (called by main.py):
    python augmenter.py
        --job-id   <uuid>
        --base-dir <cvagent_output/<slug>>
        --out-dir  <cvagent_output/<slug>/augmented>
        --n-aug    4           # augmented variants per image (default 4)
        --blur-thresh 80       # Laplacian threshold used by annotator
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageFilter
import albumentations as A

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("augmenter")

# ── helpers ──────────────────────────────────────────────────────────────────

def emit(event: str, message: str, **data: Any) -> None:
    payload = {"event": event, "message": message, **data}
    print("AUG_EVENT:" + json.dumps(payload), flush=True)


def laplacian_var(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    return float(lap.var())


# ── Anti-blur (unsharp-mask + Wiener-style sharpening) ───────────────────────

def anti_blur(img_np: np.ndarray) -> np.ndarray:
    """
    Multi-pass sharpening pipeline:
      Pass 1 — Gaussian unsharp mask (fast, recovers mid-freq edges)
      Pass 2 — Iterative blind deconvolution via Richardson-Lucy (3 iters)
      Pass 3 — CLAHE on luminance channel to restore local contrast
    Returns uint8 RGB array.
    """
    # --- Pass 1: Unsharp mask ---
    blur_sigma = 1.0
    blurred = cv2.GaussianBlur(img_np, (0, 0), blur_sigma)
    sharp = cv2.addWeighted(img_np, 1.8, blurred, -0.8, 0)
    sharp = np.clip(sharp, 0, 255).astype(np.uint8)

    # --- Pass 2: Richardson-Lucy (3 iterations, Gaussian PSF) ---
    psf_size = 5
    psf = cv2.getGaussianKernel(psf_size, 1.0)
    psf = psf @ psf.T  # 2-D Gaussian PSF
    psf /= psf.sum()

    def rl_deconv(channel: np.ndarray, psf: np.ndarray, iters: int = 3) -> np.ndarray:
        u = channel.astype(np.float64) / 255.0
        u = np.clip(u, 1e-6, 1.0)
        for _ in range(iters):
            c = cv2.filter2D(u, -1, psf) + 1e-10
            ratio = (channel.astype(np.float64) / 255.0) / c
            u = u * cv2.filter2D(ratio, -1, np.flip(psf))
            u = np.clip(u, 1e-6, 1.0)
        return (u * 255).astype(np.uint8)

    channels = [rl_deconv(sharp[:, :, c], psf) for c in range(3)]
    deconvolved = np.stack(channels, axis=-1)

    # --- Pass 3: CLAHE on L-channel ---
    lab = cv2.cvtColor(deconvolved, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return result.astype(np.uint8)


# ── Augmentation pipeline ────────────────────────────────────────────────────

def build_aug_pipeline(target_size: int = 224) -> A.Compose:
    return A.Compose([
        A.Resize(target_size, target_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.ShiftScaleRotate(
            shift_limit=0.06, scale_limit=0.12, rotate_limit=15,
            border_mode=cv2.BORDER_REFLECT_101, p=0.7
        ),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=1.0),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=15, p=1.0),
            A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=1.0),
        ], p=0.8),
        A.OneOf([
            A.GaussNoise(std_range=(0.01, 0.05), p=1.0),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.3), p=1.0),
            A.MultiplicativeNoise(multiplier=(0.95, 1.05), p=1.0),
        ], p=0.4),
        A.CoarseDropout(
            num_holes_range=(1, 4), hole_height_range=(8, 24), hole_width_range=(8, 24),
            fill=0, p=0.3
        ),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
    ])


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id",      required=True)
    parser.add_argument("--base-dir",    required=True, help="cvagent_output/<slug>")
    parser.add_argument("--out-dir",     required=True, help="where to write augmented images")
    parser.add_argument("--n-aug",       type=int, default=4,  help="augmented variants per image")
    parser.add_argument("--blur-thresh", type=float, default=80.0)
    parser.add_argument("--target-size", type=int, default=224)
    args = parser.parse_args()

    base_dir  = Path(args.base_dir)
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Scan blur-rejected images from the annotator's filtered/ folder
    filtered_dir = base_dir / "samples" / "filtered"
    # Also scan from the HF-pipeline filtered samples if present
    hf_filtered  = base_dir / "processed" / ".."   # not used

    # Collect ALL images from the "filtered" sample folder (annotator keeps them)
    # AND from a dedicated "blur_rejected" dir if the augmenter was called directly
    blur_rejected_dir = base_dir / "blur_rejected"
    candidate_dirs = [filtered_dir, blur_rejected_dir]

    valid_exts = {".jpg", ".jpeg", ".png", ".webp"}

    blurry_images: list[Path] = []
    for cdir in candidate_dirs:
        if cdir.exists():
            for f in cdir.rglob("*"):
                if f.is_file() and f.suffix.lower() in valid_exts:
                    blurry_images.append(f)

    total = len(blurry_images)
    if total == 0:
        emit("warning", "No blur-rejected images found to augment. "
             "Run the annotator first or place images in blur_rejected/.")
        emit("completed", "Augmenter finished (0 images processed).",
             stats={"recovered": 0, "augmented": 0})
        return

    emit("started", f"Anti-Blur & Augmenter starting on {total} blurry image(s).",
         total=total, n_aug=args.n_aug, target_size=args.target_size)

    augmenter = build_aug_pipeline(args.target_size)

    stats = {"recovered": 0, "augmented": 0, "failed": 0,
             "avg_blur_before": 0.0, "avg_blur_after": 0.0}
    blur_before_list: list[float] = []
    blur_after_list:  list[float] = []

    recovered_dir = out_dir / "recovered"
    augmented_dir = out_dir / "augmented"
    recovered_dir.mkdir(parents=True, exist_ok=True)
    augmented_dir.mkdir(parents=True, exist_ok=True)

    for idx, img_path in enumerate(blurry_images):
        try:
            raw = np.array(Image.open(img_path).convert("RGB"))
            blur_before = laplacian_var(raw)
            blur_before_list.append(blur_before)

            # ── Anti-blur ──────────────────────────────────────────────
            recovered = anti_blur(raw)
            blur_after = laplacian_var(recovered)
            blur_after_list.append(blur_after)

            # Save recovered image
            rec_name = f"rec_{idx:05d}_{img_path.stem}.jpg"
            rec_path = recovered_dir / rec_name
            Image.fromarray(recovered).save(rec_path, "JPEG", quality=92)
            stats["recovered"] += 1

            # ── Augmentation ───────────────────────────────────────────
            for aug_idx in range(args.n_aug):
                aug_result = augmenter(image=recovered)["image"]
                aug_name   = f"aug_{idx:05d}_{aug_idx:02d}_{img_path.stem}.jpg"
                aug_path   = augmented_dir / aug_name
                Image.fromarray(aug_result).save(aug_path, "JPEG", quality=90)
                stats["augmented"] += 1

            # Progress every 5 images or at end
            if idx % 5 == 0 or idx == total - 1:
                progress = int((idx + 1) / total * 100)
                emit("progress",
                     f"Processed {idx + 1}/{total} images | "
                     f"Blur before={blur_before:.1f} → after={blur_after:.1f}",
                     progress=progress,
                     blur_before=round(blur_before, 1),
                     blur_after=round(blur_after, 1),
                     recovered=stats["recovered"],
                     augmented=stats["augmented"])

        except Exception as exc:
            logger.exception("Failed on %s", img_path.name)
            stats["failed"] += 1
            emit("log", f"Skipped {img_path.name}: {exc}")

    stats["avg_blur_before"] = round(float(np.mean(blur_before_list)), 1) if blur_before_list else 0.0
    stats["avg_blur_after"]  = round(float(np.mean(blur_after_list)),  1) if blur_after_list  else 0.0

    # Write summary JSON
    summary_path = out_dir / "augmentation_report.json"
    summary = {
        "job_id":      args.job_id,
        "total_input": total,
        "n_aug":       args.n_aug,
        "target_size": args.target_size,
        "blur_threshold": args.blur_thresh,
        "stats":       stats,
        "output_dirs": {
            "recovered": str(recovered_dir),
            "augmented": str(augmented_dir),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    emit("completed",
         f"Done! Recovered {stats['recovered']} images, "
         f"generated {stats['augmented']} augmented variants. "
         f"Avg Laplacian: {stats['avg_blur_before']} → {stats['avg_blur_after']}.",
         stats=stats,
         report_path=str(summary_path))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        emit("error", f"Fatal augmenter error: {exc}")
        sys.exit(1)
