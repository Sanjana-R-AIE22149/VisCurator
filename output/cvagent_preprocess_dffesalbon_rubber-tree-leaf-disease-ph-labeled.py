#!/usr/bin/env python3
# Fix HuggingFace datasets multiprocessing on Windows
import os, sys
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["DATASETS_VERBOSITY"] = "error"

import multiprocessing
try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass

import io
import json
import random
import threading
from collections import Counter
from pathlib import Path

import albumentations as A
import cv2
import imagehash
import numpy as np
from PIL import Image

DATASET_ID = 'dffesalbon/rubber-tree-leaf-disease-ph-labeled'
SOURCE = 'HuggingFace'
LABEL_COL = 'labels'
IMAGE_COL = 'image'
OUTPUT_DIR = Path('output')
TARGET_SIZE = 224
PLAN = {'blur_filtering': True, 'deduplication': False, 'augmentation': True, 'augmentations': ['resize_224', 'horizontal_flip', 'random_brightness_contrast', 'minority_class_oversampling'], 'needs_blur_filtering': True, 'needs_deduplication': False, 'needs_augmentation': True, 'recommended_augmentations': ['resize_224', 'horizontal_flip', 'random_brightness_contrast', 'minority_class_oversampling']}
RANDOM_SEED = 42
BLUR_THRESHOLD_LARGE = 80.0   # for images >= 128px
BLUR_THRESHOLD_SMALL = 8.0 

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def emit_event(kind, payload):
    print("JSON_EVENT:" + json.dumps({"kind": kind, **payload}), flush=True)

def laplacian_var(img):
    arr = np.array(img.convert("L"), dtype=np.float32)
    lap = cv2.Laplacian(arr, cv2.CV_32F)
    return float(lap.var())

def ensure_rgb(raw_img):
    if isinstance(raw_img, dict) and "bytes" in raw_img:
        return Image.open(io.BytesIO(raw_img["bytes"])).convert("RGB")
    if isinstance(raw_img, Image.Image):
        return raw_img.convert("RGB")
    return None

def build_augmenter():
    resize_side = 224 if "resize_224" in PLAN.get("recommended_augmentations", []) else 256
    ops = [A.Resize(resize_side, resize_side)]
    if "horizontal_flip" in PLAN.get("recommended_augmentations", []):
        ops.append(A.HorizontalFlip(p=1.0))
    if "random_brightness_contrast" in PLAN.get("recommended_augmentations", []):
        ops.append(A.RandomBrightnessContrast(brightness_limit=0.12, contrast_limit=0.12, p=1.0))
    if "shift_scale_rotate" in PLAN.get("recommended_augmentations", []):
        ops.append(A.ShiftScaleRotate(shift_limit=0.04, scale_limit=0.08, rotate_limit=8, border_mode=cv2.BORDER_REFLECT_101, p=1.0))
    return A.Compose(ops)

result_holder = {}
error_holder = {}

def _load():
    try:
        from datasets import load_dataset
        result_holder["ds"] = load_dataset(
            DATASET_ID, 
            num_proc=1,
            download_mode="reuse_cache_if_exists"
        )
    except Exception as e:
        msg = str(e)
        if "loading script" in msg or "trust_remote_code" in msg:
            msg += (
                "\nThis HuggingFace dataset appears to require a deprecated custom "
                "loading script. Choose a dataset published in standard imagefolder, "
                "Parquet, WebDataset, or COCO files."
            )
        error_holder["err"] = msg

print(f"> Loading {DATASET_ID} from {SOURCE} ...")
t = threading.Thread(target=_load)
t.start()
t.join(timeout=300)  # 5 minute timeout
if t.is_alive():
    print("ERR: Dataset download timed out after 5 minutes", file=sys.stderr)
    sys.exit(1)
if "err" in error_holder:
    print(f"ERR: {error_holder['err']}", file=sys.stderr)
    sys.exit(1)
ds = result_holder["ds"]

if not isinstance(ds, dict):
    ds = {"train": ds}

output_processed = OUTPUT_DIR / DATASET_ID.replace("/", "_") / "processed"
output_samples = OUTPUT_DIR / DATASET_ID.replace("/", "_") / "samples"

if output_processed.exists():
    import shutil
    shutil.rmtree(output_processed)
if output_samples.exists():
    import shutil
    shutil.rmtree(output_samples)

output_processed.mkdir(parents=True, exist_ok=True)
output_samples.mkdir(parents=True, exist_ok=True)
(output_samples / "raw").mkdir(exist_ok=True)
(output_samples / "filtered").mkdir(exist_ok=True)
(output_samples / "processed").mkdir(exist_ok=True)

before_counts = Counter()
after_counts = Counter()
seen_hashes = set()
blur_scatter = []
augmentation_summary = Counter()
accepted_records = []
rejected_blur = 0
rejected_dup = 0

sample_maps = {"raw": [], "filtered": [], "processed": []}

def row_generator():
    for split_name, split_ds in ds.items():
        print("> split {}: {} rows".format(split_name, len(split_ds)), flush=True)
        for r in split_ds:
            yield r

for idx, row in enumerate(row_generator()):
    if len(accepted_records) >= TARGET_SIZE:
        break
    img = ensure_rgb(row.get(IMAGE_COL))
    if img is None:
        continue

    label = str(row.get(LABEL_COL, "unknown"))
    before_counts[label] += 1
    blur_value = laplacian_var(img)
    img_hash = str(imagehash.dhash(img))
    _thresh = BLUR_THRESHOLD_SMALL if (img.width < 128 or img.height < 128) else BLUR_THRESHOLD_LARGE
    is_blurry = blur_value < _thresh
    is_dup = img_hash in seen_hashes

    # Save Raw Samples
    if len(sample_maps["raw"]) < 5:
        fname = f"raw_{idx:05d}.jpg"
        img.save(output_samples / "raw" / fname, "JPEG", quality=85)
        sample_maps["raw"].append({"url": f"samples/raw/{fname}", "label": label, "id": idx})

    blur_scatter.append({
        "id": idx,
        "laplacian": round(blur_value, 1),
        "resolution": int((img.width * img.height) / 1000),
        "accepted": not is_blurry and not is_dup,
    })

    if (PLAN.get("needs_blur_filtering") and is_blurry) or (PLAN.get("needs_deduplication") and is_dup):
        if PLAN.get("needs_blur_filtering") and is_blurry: rejected_blur += 1
        if PLAN.get("needs_deduplication") and is_dup: rejected_dup += 1

        # Save Filtered Samples
        if len(sample_maps["filtered"]) < 5:
            reason = "blurry" if is_blurry else "duplicate"
            fname = f"filtered_{idx:05d}.jpg"
            img.save(output_samples / "filtered" / fname, "JPEG", quality=85)
            sample_maps["filtered"].append({"url": f"samples/filtered/{fname}", "label": label, "reason": reason, "id": idx})
        continue

    seen_hashes.add(img_hash)
    accepted_records.append((img, label, idx))

augmenter = build_augmenter()
minority_target = max([count for count in before_counts.values()] or [0])
minority_labels = {label for label, count in before_counts.items() if count < minority_target}


export_total = 0
for sample_index, (img, label, original_idx) in enumerate(accepted_records):
    out_dir = output_processed / label
    out_dir.mkdir(parents=True, exist_ok=True)
    base_np = np.array(img)
    res = augmenter(image=base_np)
    resized = res["image"]

    save_name = f"sample_{sample_index:05d}.jpg"
    Image.fromarray(resized).save(out_dir / save_name, "JPEG", quality=92)

    # Save Processed Samples
    if len(sample_maps["processed"]) < 5:
        fname = f"proc_{sample_index:05d}.jpg"
        Image.fromarray(resized).save(output_samples / "processed" / fname, "JPEG", quality=85)
        sample_maps["processed"].append({"url": f"samples/processed/{fname}", "label": label, "id": original_idx})

    after_counts[label] += 1
    export_total += 1

    if PLAN.get("needs_augmentation") and label in minority_labels and after_counts[label] < min(minority_target, after_counts[label] + 2):
        augmented = augmenter(image=base_np)["image"]
        Image.fromarray(augmented).save(out_dir / f"sample_{sample_index:05d}_aug.jpg", "JPEG", quality=92)
        augmentation_summary["augmented_images"] += 1
        after_counts[label] += 1
        export_total += 1

preprocess_report = {
    "dataset_id": DATASET_ID,
    "output_dir": str(output_processed),
    "plan": PLAN,
    "stage_samples": sample_maps,
    "before_stats": {
        "images": int(sum(before_counts.values())),
        "class_distribution": dict(sorted(before_counts.items())),
    },
    "after_stats": {
        "images": int(export_total),
        "class_distribution": dict(sorted(after_counts.items())),
        "blur_filtered": int(rejected_blur),
        "duplicates_removed": int(rejected_dup),
    },
    "class_distribution": dict(sorted(after_counts.items())),
    "augmentation_summary": {
        "operations": PLAN.get("recommended_augmentations", []),
        "augmented_images": int(augmentation_summary.get("augmented_images", 0)),
        "synthetic_generation_recommended": bool(PLAN.get("needs_synthetic_generation")),
    },
    "blur_scatter": blur_scatter[:200],
}
report_path = output_processed / "preprocessing_report.json"
report_path.write_text(json.dumps(preprocess_report, indent=2), encoding="utf-8")
emit_event("report", preprocess_report)
print(f"> preprocessing report written to {report_path}", flush=True)
print(f"> exported {export_total} processed images", flush=True)
