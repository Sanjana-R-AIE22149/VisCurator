#!/usr/bin/env python3
"""CVAgent auto-generated download script — pantelism/cats-vs-dogs from huggingface"""
# Fix HuggingFace datasets multiprocessing on Windows
import os, sys
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["DATASETS_VERBOSITY"] = "error"

import multiprocessing
try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass

import io, math, shutil, hashlib, random, json, threading
from pathlib import Path
from PIL import Image
import numpy as np

DATASET_ID   = "pantelism/cats-vs-dogs"
SOURCE       = "huggingface"
LABEL_COL    = "labels"
IMAGE_COL    = "image"
OUTPUT_DIR   = Path("output")
TARGET_SIZE  = 224
SPLIT_RATIOS = (0.8, 0.1, 0.1)
BLUR_THRESH  = 80.0

def laplacian_var(img):
    arr = np.array(img.convert("L"), dtype=np.float32)
    lap = (arr[:-2,1:-1] + arr[2:,1:-1] + arr[1:-1,:-2] + arr[1:-1,2:] - 4*arr[1:-1,1:-1])
    return float(np.var(lap))

def dhash(img, size=8):
    resized = img.convert("L").resize((size+1, size), Image.LANCZOS)
    arr = np.array(resized)
    diff = arr[:, 1:] > arr[:, :-1]
    return hashlib.md5(diff.tobytes()).hexdigest()

def resolve_label(row, default_col, dataset_dict):
    col = default_col
    if col not in row:
        for fallback in ["label", "labels", "category", "class", "target", "fine_label"]:
            if fallback in row:
                col = fallback
                break
    val = row.get(col)
    if val is None:
        return "unknown"
    if isinstance(val, int):
        try:
            feats = next(iter(dataset_dict.values())).features
            if hasattr(feats[col], "int2str"):
                return feats[col].int2str(val)
            elif hasattr(feats[col], "names"):
                return feats[col].names[val]
        except Exception:
            pass
    return str(val)

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

print(f"> Splits: {list(ds.keys())}")

print("> Filtering (blur + dedup) ...")
accepted, rejected_blur, rejected_dup = [], 0, 0
seen_hashes = set()

def row_generator():
    for split_name, split_ds in ds.items():
        print(f">   {split_name}: {len(split_ds)} rows")
        for r in split_ds:
            yield r

for i, row in enumerate(row_generator()):
    if len(accepted) >= TARGET_SIZE:
        break
    raw_img = row.get(IMAGE_COL)
    if raw_img is None:
        continue
    try:
        if isinstance(raw_img, dict):
            img = Image.open(io.BytesIO(raw_img["bytes"])).convert("RGB")
        elif isinstance(raw_img, Image.Image):
            img = raw_img.convert("RGB")
        else:
            continue
    except Exception:
        continue

    lv = laplacian_var(img)
    if lv < BLUR_THRESH:
        rejected_blur += 1
        continue

    h = dhash(img)
    if h in seen_hashes:
        rejected_dup += 1
        continue
    seen_hashes.add(h)

    label = resolve_label(row, LABEL_COL, ds)
    accepted.append((img, label))

    if (i+1) % 500 == 0:
        print(f">   Scanned {i+1} | accepted {len(accepted)} | blur {rejected_blur} | dup {rejected_dup}")

print(f"> Filter done — accepted: {len(accepted)} | blur_rejected: {rejected_blur} | dup_rejected: {rejected_dup}")

random.seed(42)
random.shuffle(accepted)
n = len(accepted)
n_train = math.floor(n * SPLIT_RATIOS[0])
n_val   = math.floor(n * SPLIT_RATIOS[1])
splits  = {"train": accepted[:n_train], "val": accepted[n_train:n_train+n_val], "test": accepted[n_train+n_val:]}

print("> Writing splits ...")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
counts = {}
for split_name, rows in splits.items():
    print(f">   {split_name}: {len(rows)} images")
    for img, label in rows:
        dest = OUTPUT_DIR / split_name / label
        dest.mkdir(parents=True, exist_ok=True)
        idx = counts.get(f"{split_name}/{label}", 0)
        counts[f"{split_name}/{label}"] = idx + 1
        img.save(dest / f"{label}_{idx:05d}.jpg", "JPEG", quality=92)

total = sum(counts.values())
print(f"> Wrote {total} images to {OUTPUT_DIR.resolve()}")
print("> Class breakdown:")
for k, v in sorted(counts.items()):
    print(f">   {k}: {v}")
print("> Pipeline complete.")
