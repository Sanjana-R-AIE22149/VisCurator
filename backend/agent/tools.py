"""
VisCurator / CVAgent — Agent Tools
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Four production-grade async tool functions for the dataset curation
agent.  Each tool is fully self-contained with error handling, timeouts,
and structured return values.

Tools are registered in ``TOOL_REGISTRY`` so the agent loop can
discover, describe, and invoke them dynamically.
"""

from __future__ import annotations

import asyncio
import csv
import io
import logging
import os
import sys
import tempfile
import textwrap
from typing import Any, Callable, Coroutine

import aiohttp
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ── Types ────────────────────────────────────────────────────

ToolFn = Callable[..., Coroutine[Any, Any, dict[str, Any]]]

# ── Tool Registry ────────────────────────────────────────────

TOOL_REGISTRY: dict[str, dict[str, Any]] = {}

# Default timeout for all outbound HTTP requests (seconds).
_HTTP_TIMEOUT = aiohttp.ClientTimeout(total=45)


def _register(
    name: str,
    description: str,
    parameters: dict[str, Any],
    fn: ToolFn,
) -> None:
    """Insert a tool into the global registry."""
    TOOL_REGISTRY[name] = {
        "name": name,
        "description": description,
        "parameters": parameters,
        "function": fn,
    }


def get_tool_descriptions_for_prompt() -> str:
    """Render a human-readable tool catalogue for the LLM system prompt."""
    lines: list[str] = []
    for t in TOOL_REGISTRY.values():
        props = t["parameters"].get("properties", {})
        params = ", ".join(
            f"{k}: {v.get('type', 'any')}" for k, v in props.items()
        )
        lines.append(f"  - {t['name']}({params})")
        lines.append(f"    {t['description']}")
    return "\n".join(lines)


async def execute_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Look up a tool by *name* and execute it with *arguments*.

    Returns a dict that always contains ``"status"`` (``"success"`` or
    ``"error"``).  On failure, an ``"error"`` key carries the message.
    """
    entry = TOOL_REGISTRY.get(name)
    if entry is None:
        return {"status": "error", "error": f"Unknown tool '{name}'."}
    try:
        result = await asyncio.wait_for(
            entry["function"](**arguments),
            timeout=120,  # hard ceiling per tool invocation
        )
        return result
    except asyncio.TimeoutError:
        logger.error("Tool '%s' timed out after 120 s", name)
        return {"status": "error", "error": f"Tool '{name}' timed out."}
    except Exception as exc:
        logger.exception("Tool '%s' raised an exception", name)
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TOOL 1 — search_huggingface
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def search_huggingface(
    query: str,
    max_results: int = 5,
) -> dict[str, Any]:
    """Search the HuggingFace Hub for public datasets matching *query*.

    Uses the unauthenticated ``/api/datasets`` REST endpoint so no
    token is needed for public results.
    """
    url = "https://huggingface.co/api/datasets"
    params = {
        "search": query,
        "limit": min(max_results, 20),
        "sort": "likes",
        "direction": "-1",
        "full": "false",
    }
    logger.info("search_huggingface  query=%r  max_results=%d", query, max_results)

    async with aiohttp.ClientSession(timeout=_HTTP_TIMEOUT) as session:
        async with session.get(url, params=params) as resp:
            if resp.status != 200:
                body = await resp.text()
                return {
                    "status": "error",
                    "error": f"HuggingFace API returned HTTP {resp.status}: {body[:300]}",
                }
            data: list[dict[str, Any]] = await resp.json()

    results: list[dict[str, Any]] = []
    for ds in data:
        tags = ds.get("tags") or []
        card = ds.get("cardData") or {}

        # Estimate size from cardData or tags
        size_est = card.get("dataset_size") or card.get("size_categories")
        if isinstance(size_est, list):
            size_est = size_est[0] if size_est else "unknown"

        results.append({
            "dataset_id": ds.get("id", ""),
            "description": (ds.get("description") or "")[:250],
            "downloads": ds.get("downloads", 0),
            "likes": ds.get("likes", 0),
            "tags": tags[:15],
            "size_estimate": str(size_est) if size_est else "unknown",
            "last_modified": ds.get("lastModified", ""),
        })

    return {
        "status": "success",
        "count": len(results),
        "datasets": results,
    }


_register(
    name="search_huggingface",
    description=(
        "Search the HuggingFace Hub for public datasets matching a "
        "natural-language query.  Returns dataset IDs, download counts, "
        "tags, and size estimates.  No authentication required."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Free-text search query (e.g. 'apple leaf disease').",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return (1-20).",
                "default": 5,
            },
        },
        "required": ["query"],
    },
    fn=search_huggingface,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TOOL 2 — get_dataset_info
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def get_dataset_info(dataset_id: str) -> dict[str, Any]:
    """Retrieve detailed metadata for a specific HuggingFace dataset.

    Fetches split information, feature columns, size in bytes, and
    row counts per split via the Hub REST API.
    """
    url = f"https://huggingface.co/api/datasets/{dataset_id}"
    headers: dict[str, str] = {}
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"

    logger.info("get_dataset_info  dataset_id=%r", dataset_id)

    async with aiohttp.ClientSession(timeout=_HTTP_TIMEOUT) as session:
        # ── Main dataset metadata ──
        async with session.get(url, headers=headers) as resp:
            if resp.status == 404:
                return {"status": "error", "error": f"Dataset '{dataset_id}' not found."}
            if resp.status != 200:
                body = await resp.text()
                return {"status": "error", "error": f"HTTP {resp.status}: {body[:300]}"}
            meta: dict[str, Any] = await resp.json()

        # ── Parquet / split info (datasets-server v1) ──
        splits_info: dict[str, Any] = {}
        splits_url = f"https://datasets-server.huggingface.co/info?dataset={dataset_id}"
        try:
            async with session.get(splits_url, headers=headers, timeout=_HTTP_TIMEOUT) as resp2:
                if resp2.status == 200:
                    info_payload = await resp2.json()
                    # datasets-server returns { dataset_info: { <config>: { splits: ... } } }
                    ds_info = info_payload.get("dataset_info") or {}
                    for config_name, config_data in ds_info.items():
                        raw_splits = config_data.get("splits") or {}
                        for split_name, split_meta in raw_splits.items():
                            splits_info[split_name] = {
                                "num_rows": split_meta.get("num_examples", split_meta.get("num_bytes", 0)),
                                "num_bytes": split_meta.get("num_bytes", 0),
                            }
                        # Extract feature names from first config
                        if "features" in config_data:
                            features_raw = config_data["features"]
                            if isinstance(features_raw, dict):
                                splits_info["_features"] = list(features_raw.keys())
                        break  # only inspect the first (default) config
        except Exception as exc:
            logger.warning("Could not fetch splits info for '%s': %s", dataset_id, exc)

    features = splits_info.pop("_features", [])
    tags = meta.get("tags") or []
    card = meta.get("cardData") or {}

    total_bytes = sum(s.get("num_bytes", 0) for s in splits_info.values())
    total_rows = sum(s.get("num_rows", 0) for s in splits_info.values())

    return {
        "status": "success",
        "dataset_id": dataset_id,
        "description": (meta.get("description") or "")[:400],
        "tags": tags[:20],
        "license": card.get("license", "unknown"),
        "features": features,
        "splits": splits_info,
        "total_rows": total_rows,
        "total_size_bytes": total_bytes,
        "total_size_mb": round(total_bytes / (1024 * 1024), 2) if total_bytes else 0,
        "downloads": meta.get("downloads", 0),
        "likes": meta.get("likes", 0),
    }


_register(
    name="get_dataset_info",
    description=(
        "Get detailed metadata for a specific HuggingFace dataset: "
        "splits, features/columns, total size in bytes, row counts, "
        "license, and tags."
    ),
    parameters={
        "type": "object",
        "properties": {
            "dataset_id": {
                "type": "string",
                "description": "Full HuggingFace dataset identifier (e.g. 'keremberke/plant-disease-detection').",
            },
        },
        "required": ["dataset_id"],
    },
    fn=get_dataset_info,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TOOL 3 — search_open_images
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Cache the class-descriptions CSV in memory after the first fetch.
_OI_CLASS_CACHE: dict[str, str] | None = None


async def _load_open_images_classes() -> dict[str, str]:
    """Download & cache the Open Images v6 class-description CSV.

    Returns a dict mapping *lowercased display name* → *class MID*.
    """
    global _OI_CLASS_CACHE
    if _OI_CLASS_CACHE is not None:
        return _OI_CLASS_CACHE

    csv_url = (
        "https://storage.googleapis.com/openimages/v6/"
        "oidv6-class-descriptions.csv"
    )
    logger.info("Downloading Open Images class descriptions …")

    async with aiohttp.ClientSession(timeout=_HTTP_TIMEOUT) as session:
        async with session.get(csv_url) as resp:
            if resp.status != 200:
                raise RuntimeError(f"Failed to fetch OI classes: HTTP {resp.status}")
            text = await resp.text()

    mapping: dict[str, str] = {}
    reader = csv.reader(io.StringIO(text))
    for row in reader:
        if len(row) >= 2:
            mid, display_name = row[0].strip(), row[1].strip()
            mapping[display_name.lower()] = mid
    _OI_CLASS_CACHE = mapping
    logger.info("Loaded %d Open Images class descriptions.", len(mapping))
    return mapping


async def search_open_images(
    classes: list[str],
    max_images: int = 500,
) -> dict[str, Any]:
    """Look up class IDs in Open Images v7 and return a download plan.

    For each requested class name, fuzzy-matches against the OI class
    list and returns the class ID, estimated available count, and a
    handful of sample thumbnail URLs.
    """
    logger.info("search_open_images  classes=%r  max_images=%d", classes, max_images)

    try:
        class_map = await _load_open_images_classes()
    except Exception as exc:
        return {"status": "error", "error": f"Failed to load OI class list: {exc}"}

    results: list[dict[str, Any]] = []
    for cls_name in classes:
        key = cls_name.strip().lower()

        # Exact match first, then substring match
        mid: str | None = class_map.get(key)
        if mid is None:
            # Try substring / fuzzy match
            candidates = [
                (name, m) for name, m in class_map.items() if key in name
            ]
            if candidates:
                # Pick the shortest matching name (most specific)
                candidates.sort(key=lambda x: len(x[0]))
                mid = candidates[0][1]
                matched_name = candidates[0][0]
            else:
                results.append({
                    "class_name": cls_name,
                    "class_id": None,
                    "matched": False,
                    "available_count": 0,
                    "sample_urls": [],
                    "note": "No matching class found in Open Images v7.",
                })
                continue
        else:
            matched_name = key

        # Build sample thumbnail URLs via the OI visualiser
        sample_urls = [
            f"https://storage.googleapis.com/openimages/web/visualizer/image.html?set=train&id={mid}&idx={i}"
            for i in range(min(5, max_images))
        ]

        results.append({
            "class_name": cls_name,
            "matched_name": matched_name,
            "class_id": mid,
            "matched": True,
            "available_count": "varies (typically 500-50000 per class in OIv7)",
            "max_requested": max_images,
            "sample_urls": sample_urls,
            "download_command": (
                f"fiftyone zoo datasets load open-images-v7 "
                f"--split train --label-types detections "
                f"--classes {cls_name} --max-samples {max_images}"
            ),
        })

    return {
        "status": "success",
        "classes_requested": len(classes),
        "classes_found": sum(1 for r in results if r.get("matched")),
        "results": results,
    }


_register(
    name="search_open_images",
    description=(
        "Look up object classes in Google Open Images v7.  For each "
        "requested class, returns the MID class ID, whether a match was "
        "found, estimated availability, sample URLs, and a ready-to-run "
        "download command."
    ),
    parameters={
        "type": "object",
        "properties": {
            "classes": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of object class names to search for (e.g. ['Apple', 'Leaf']).",
            },
            "max_images": {
                "type": "integer",
                "description": "Max images to include in the download plan per class.",
                "default": 500,
            },
        },
        "required": ["classes"],
    },
    fn=search_open_images,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TOOL 4 — estimate_dataset_quality
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def estimate_dataset_quality(
    dataset_id: str,
    sample_size: int = 50,
) -> dict[str, Any]:
    """Download a small sample from a HuggingFace dataset and estimate
    quality metrics: resolution distribution, blur ratio, class balance,
    and an overall quality score (0-100).

    Uses the datasets-server ``/rows`` endpoint to fetch sample rows
    without downloading the entire dataset.
    """
    logger.info(
        "estimate_dataset_quality  dataset_id=%r  sample_size=%d",
        dataset_id, sample_size,
    )

    # ── 1. Fetch sample rows via datasets-server ──
    rows_url = (
        f"https://datasets-server.huggingface.co/rows"
        f"?dataset={dataset_id}&config=default&split=train"
        f"&offset=0&length={min(sample_size, 100)}"
    )
    headers: dict[str, str] = {}
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"

    async with aiohttp.ClientSession(timeout=_HTTP_TIMEOUT) as session:
        async with session.get(rows_url, headers=headers) as resp:
            if resp.status != 200:
                # Fallback: try first available config
                info_url = f"https://datasets-server.huggingface.co/info?dataset={dataset_id}"
                async with session.get(info_url, headers=headers) as info_resp:
                    if info_resp.status != 200:
                        return {
                            "status": "error",
                            "error": f"Cannot access dataset '{dataset_id}' via datasets-server (HTTP {resp.status}).",
                        }
                    info_data = await info_resp.json()
                    ds_info = info_data.get("dataset_info") or {}
                    if not ds_info:
                        return {"status": "error", "error": "No configs found."}
                    first_config = next(iter(ds_info))
                    splits = ds_info[first_config].get("splits") or {}
                    first_split = next(iter(splits), "train")

                rows_url2 = (
                    f"https://datasets-server.huggingface.co/rows"
                    f"?dataset={dataset_id}&config={first_config}&split={first_split}"
                    f"&offset=0&length={min(sample_size, 100)}"
                )
                async with session.get(rows_url2, headers=headers) as resp2:
                    if resp2.status != 200:
                        return {
                            "status": "error",
                            "error": f"Rows endpoint failed: HTTP {resp2.status}.",
                        }
                    payload = await resp2.json()
            else:
                payload = await resp.json()

    rows = payload.get("rows") or []
    if not rows:
        return {
            "status": "error",
            "error": "No rows returned from datasets-server.",
        }

    # ── 2. Analyse images in the sample ──
    widths: list[int] = []
    heights: list[int] = []
    blur_scores: list[float] = []
    labels: list[str] = []

    image_urls: list[str] = []
    label_keys = ("label", "labels", "class", "category", "target")

    for row_wrapper in rows:
        row = row_wrapper.get("row") or row_wrapper
        # Find image URL
        for key in ("image", "img", "image_url", "file"):
            val = row.get(key)
            if isinstance(val, dict) and "src" in val:
                image_urls.append(val["src"])
                break
            elif isinstance(val, str) and val.startswith("http"):
                image_urls.append(val)
                break
        # Find labels
        for lk in label_keys:
            lv = row.get(lk)
            if lv is not None:
                labels.append(str(lv))
                break

    # Download and measure a subset of images
    sample_limit = min(len(image_urls), sample_size, 30)  # cap downloads
    sampled_urls = image_urls[:sample_limit]

    async def _measure(session: aiohttp.ClientSession, url: str) -> dict[str, Any] | None:
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as r:
                if r.status != 200:
                    return None
                data = await r.read()
            img = Image.open(io.BytesIO(data)).convert("RGB")
            arr = np.array(img)
            gray = np.mean(arr, axis=2).astype(np.float64)
            # Laplacian variance (approx blur detection without cv2 import overhead)
            lap = (
                gray[:-2, 1:-1] + gray[2:, 1:-1] +
                gray[1:-1, :-2] + gray[1:-1, 2:] -
                4 * gray[1:-1, 1:-1]
            )
            blur = float(np.var(lap))
            return {"w": img.width, "h": img.height, "blur": blur}
        except Exception:
            return None

    async with aiohttp.ClientSession() as session:
        tasks = [_measure(session, u) for u in sampled_urls]
        measurements = await asyncio.gather(*tasks, return_exceptions=False)

    for m in measurements:
        if m is not None:
            widths.append(m["w"])
            heights.append(m["h"])
            blur_scores.append(m["blur"])

    # ── 3. Compute aggregate metrics ──
    n_measured = len(widths)
    if n_measured == 0:
        return {
            "status": "partial",
            "dataset_id": dataset_id,
            "images_sampled": 0,
            "note": "No images could be downloaded for analysis.",
            "label_distribution": _class_dist(labels),
            "quality_score": 0,
        }

    w_arr = np.array(widths)
    h_arr = np.array(heights)
    b_arr = np.array(blur_scores)

    # Blur threshold: images with Laplacian var < 100 are likely blurry
    blur_threshold = 100.0
    blurry_count = int(np.sum(b_arr < blur_threshold))
    blur_ratio = round(blurry_count / n_measured, 3)

    # Quality scoring (heuristic, 0-100)
    resolution_score = min(float(np.median(w_arr * h_arr)) / (640 * 480), 1.0) * 30
    sharpness_score = (1 - blur_ratio) * 40
    class_dist = _class_dist(labels)
    balance_score = _balance_score(class_dist) * 30
    quality_score = round(resolution_score + sharpness_score + balance_score, 1)
    quality_score = max(0, min(100, quality_score))

    return {
        "status": "success",
        "dataset_id": dataset_id,
        "images_sampled": n_measured,
        "avg_resolution": {
            "width": int(np.mean(w_arr)),
            "height": int(np.mean(h_arr)),
        },
        "resolution_range": {
            "min_w": int(w_arr.min()), "max_w": int(w_arr.max()),
            "min_h": int(h_arr.min()), "max_h": int(h_arr.max()),
        },
        "blur_ratio": blur_ratio,
        "blurry_images": blurry_count,
        "avg_sharpness": round(float(b_arr.mean()), 1),
        "class_distribution": class_dist,
        "quality_score": quality_score,
        "quality_grade": (
            "A" if quality_score >= 80 else
            "B" if quality_score >= 60 else
            "C" if quality_score >= 40 else
            "D"
        ),
    }


def _class_dist(labels: list[str]) -> dict[str, int]:
    """Count label occurrences."""
    dist: dict[str, int] = {}
    for lbl in labels:
        dist[lbl] = dist.get(lbl, 0) + 1
    return dict(sorted(dist.items(), key=lambda x: -x[1]))


def _balance_score(dist: dict[str, int]) -> float:
    """Return 0-1 score for class balance (1 = perfectly balanced)."""
    if not dist:
        return 0.5  # unknown → neutral
    counts = list(dist.values())
    if len(counts) == 1:
        return 1.0
    arr = np.array(counts, dtype=float)
    cv = float(arr.std() / arr.mean()) if arr.mean() > 0 else 1.0
    return max(0.0, 1.0 - cv)


_register(
    name="estimate_dataset_quality",
    description=(
        "Download a small sample from a HuggingFace dataset and estimate "
        "quality: average resolution, blur ratio (Laplacian variance), "
        "class balance, and an overall quality score from 0-100."
    ),
    parameters={
        "type": "object",
        "properties": {
            "dataset_id": {
                "type": "string",
                "description": "HuggingFace dataset ID to evaluate.",
            },
            "sample_size": {
                "type": "integer",
                "description": "Number of sample images to analyse (max 100).",
                "default": 50,
            },
        },
        "required": ["dataset_id"],
    },
    fn=estimate_dataset_quality,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TOOL 5 — generate_and_run_script
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Module-level emit callback — injected by DatasetAgent before calling this tool.
# Signature: async (line: str, stream: str) -> None
_script_emit_callback: Any = None


def set_script_emit_callback(cb: Any) -> None:
    """Register the async callback used to stream script output lines."""
    global _script_emit_callback
    _script_emit_callback = cb


def _build_download_script(
    dataset_id: str,
    label_column: str,
    image_column: str,
    output_dir: str,
    target_size: int,
    split_ratios: tuple[float, float, float],
) -> str:
    """Return a self-contained Python script that downloads a HuggingFace
    dataset, applies basic quality filtering, and writes train/val/test
    splits to *output_dir* as JPEG images organised by class label.
    """
    train_r, val_r, test_r = split_ratios
    return textwrap.dedent(f"""\
        #!/usr/bin/env python3
        \"\"\"
        CVAgent auto-generated download script
        Dataset : {dataset_id}
        Generated by VisCurator DatasetAgent
        \"\"\"
        import os, sys, io, math, shutil, hashlib
        from pathlib import Path
        from PIL import Image
        import numpy as np

        DATASET_ID   = "{dataset_id}"
        LABEL_COL    = "{label_column}"
        IMAGE_COL    = "{image_column}"
        OUTPUT_DIR   = Path("{output_dir}")
        TARGET_SIZE  = {target_size}
        SPLIT_RATIOS = ({train_r}, {val_r}, {test_r})
        BLUR_THRESH  = 80.0   # Laplacian variance — below this = blurry

        # ── helpers ──────────────────────────────────────────
        def laplacian_var(img: Image.Image) -> float:
            arr = np.array(img.convert("L"), dtype=np.float32)
            lap = (
                arr[:-2, 1:-1] + arr[2:, 1:-1] +
                arr[1:-1, :-2] + arr[1:-1, 2:] -
                4 * arr[1:-1, 1:-1]
            )
            return float(np.var(lap))

        def dhash(img: Image.Image, size: int = 8) -> str:
            resized = img.convert("L").resize((size + 1, size), Image.LANCZOS)
            arr = np.array(resized)
            diff = arr[:, 1:] > arr[:, :-1]
            return hashlib.md5(diff.tobytes()).hexdigest()

        # ── load dataset ─────────────────────────────────────
        print(f"> Loading dataset {{DATASET_ID}} from HuggingFace Hub …")
        try:
            from datasets import load_dataset
        except ImportError:
            print("✖ 'datasets' package not installed. Run: pip install datasets", file=sys.stderr)
            sys.exit(1)

        ds = load_dataset(DATASET_ID, trust_remote_code=True)
        print(f"> Available splits: {{list(ds.keys())}}")

        # Flatten all splits into one pool, then re-split ourselves
        all_rows = []
        for split_name, split_ds in ds.items():
            for row in split_ds:
                all_rows.append(row)
            print(f">   {{split_name}}: {{len(split_ds)}} rows")

        print(f"> Total rows in pool: {{len(all_rows)}}")

        # ── filter & deduplicate ──────────────────────────────
        print("> Filtering: blur detection + deduplication …")
        accepted, rejected_blur, rejected_dup = [], 0, 0
        seen_hashes: set[str] = set()

        for i, row in enumerate(all_rows):
            if len(accepted) >= TARGET_SIZE:
                break
            raw_img = row.get(IMAGE_COL)
            if raw_img is None:
                continue
            # HuggingFace image column can be a PIL Image or a dict with 'bytes'
            if isinstance(raw_img, dict):
                try:
                    img = Image.open(io.BytesIO(raw_img["bytes"])).convert("RGB")
                except Exception:
                    continue
            elif isinstance(raw_img, Image.Image):
                img = raw_img.convert("RGB")
            else:
                continue

            # Blur check
            lv = laplacian_var(img)
            if lv < BLUR_THRESH:
                rejected_blur += 1
                continue

            # Duplicate check
            h = dhash(img)
            if h in seen_hashes:
                rejected_dup += 1
                continue
            seen_hashes.add(h)

            label = str(row.get(LABEL_COL, "unknown"))
            accepted.append((img, label))

            if (i + 1) % 200 == 0:
                print(f">   Scanned {{i+1}}/{{len(all_rows)}} | accepted {{len(accepted)}} | blur {{rejected_blur}} | dup {{rejected_dup}}")

        print(f"> Filtering complete — accepted: {{len(accepted)}} | rejected blur: {{rejected_blur}} | rejected dup: {{rejected_dup}}")

        # ── split ─────────────────────────────────────────────
        import random
        random.seed(42)
        random.shuffle(accepted)
        n = len(accepted)
        n_train = math.floor(n * SPLIT_RATIOS[0])
        n_val   = math.floor(n * SPLIT_RATIOS[1])
        splits  = {{
            "train": accepted[:n_train],
            "val":   accepted[n_train:n_train + n_val],
            "test":  accepted[n_train + n_val:],
        }}
        for sname, rows in splits.items():
            print(f">   {{sname}}: {{len(rows)}} images")

        # ── write to disk ─────────────────────────────────────
        print(f"> Writing images to {{OUTPUT_DIR}} …")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        counts: dict[str, int] = {{}}
        for split_name, rows in splits.items():
            for img, label in rows:
                dest = OUTPUT_DIR / split_name / label
                dest.mkdir(parents=True, exist_ok=True)
                idx = counts.get(f"{{split_name}}/{{label}}", 0)
                counts[f"{{split_name}}/{{label}}"] = idx + 1
                img.save(dest / f"{{label}}_{{idx:05d}}.jpg", "JPEG", quality=92)

        total_written = sum(counts.values())
        print(f"> ✓ Wrote {{total_written}} images across {{len(splits)}} splits")
        print(f"> Output directory: {{OUTPUT_DIR.resolve()}}")
        print("> Class distribution:")
        for k, v in sorted(counts.items()):
            print(f">   {{k}}: {{v}}")
        print("> Pipeline complete.")
    """)


async def generate_and_run_script(
    dataset_id: str,
    label_column: str = "label",
    image_column: str = "image",
    output_dir: str = "./cvagent_output",
    target_size: int = 1000,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
) -> dict[str, Any]:
    """Generate a Python download/split script for a HuggingFace dataset,
    write it to a temp file, execute it as a subprocess, and stream every
    stdout/stderr line back to the frontend terminal in real time.

    Returns a summary dict with the script path, exit code, and last 40
    lines of output.
    """
    logger.info(
        "generate_and_run_script  dataset_id=%r  output_dir=%r  target=%d",
        dataset_id, output_dir, target_size,
    )

    # Normalise ratios
    total = train_ratio + val_ratio + test_ratio
    if total <= 0:
        train_ratio, val_ratio, test_ratio = 0.7, 0.2, 0.1
        total = 1.0
    split_ratios = (train_ratio / total, val_ratio / total, test_ratio / total)

    script_code = _build_download_script(
        dataset_id=dataset_id,
        label_column=label_column,
        image_column=image_column,
        output_dir=output_dir,
        target_size=target_size,
        split_ratios=split_ratios,
    )

    # Write script to a named temp file so the user can inspect it
    script_dir = Path(output_dir).parent
    script_dir.mkdir(parents=True, exist_ok=True)
    script_path = script_dir / f"cvagent_download_{dataset_id.replace('/', '_')}.py"
    script_path.write_text(script_code, encoding="utf-8")
    logger.info("Script written to %s", script_path)

    emit = _script_emit_callback  # may be None if called outside agent context

    async def _stream_line(line: str, stream: str = "stdout") -> None:
        if emit is not None:
            try:
                await emit(line.rstrip(), stream)
            except Exception:
                pass

    await _stream_line(f"> Script written to: {script_path}")
    await _stream_line(f"> Executing: {sys.executable} {script_path}")
    await _stream_line("> " + "─" * 55)

    # ── Execute the script as a subprocess ──
    output_lines: list[str] = []
    exit_code = -1

    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, str(script_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,  # merge stderr into stdout
            env={**os.environ},
        )

        assert proc.stdout is not None
        async for raw_line in proc.stdout:
            line = raw_line.decode("utf-8", errors="replace").rstrip()
            output_lines.append(line)
            await _stream_line(line)

        await proc.wait()
        exit_code = proc.returncode or 0

    except Exception as exc:
        err_line = f"✖ Subprocess error: {exc}"
        output_lines.append(err_line)
        await _stream_line(err_line, "stderr")
        return {
            "status": "error",
            "error": err_line,
            "script_path": str(script_path),
            "exit_code": -1,
            "output_tail": output_lines[-40:],
        }

    await _stream_line("> " + "─" * 55)
    status_line = f"> Script exited with code {exit_code}"
    await _stream_line(status_line)
    output_lines.append(status_line)

    return {
        "status": "success" if exit_code == 0 else "error",
        "script_path": str(script_path),
        "exit_code": exit_code,
        "output_dir": output_dir,
        "output_tail": output_lines[-40:],
        "error": None if exit_code == 0 else f"Script exited with code {exit_code}",
    }


_register(
    name="generate_and_run_script",
    description=(
        "Generate a Python script that downloads a specific HuggingFace dataset, "
        "applies blur filtering and deduplication, then splits it into train/val/test "
        "directories. The script is written to disk and executed immediately; every "
        "stdout/stderr line is streamed to the terminal in real time. "
        "Call this AFTER you have identified the best dataset_id via search_huggingface "
        "and get_dataset_info."
    ),
    parameters={
        "type": "object",
        "properties": {
            "dataset_id": {
                "type": "string",
                "description": "HuggingFace dataset ID to download (e.g. 'sasha/dog-food').",
            },
            "label_column": {
                "type": "string",
                "description": "Name of the label/class column in the dataset.",
                "default": "label",
            },
            "image_column": {
                "type": "string",
                "description": "Name of the image column in the dataset.",
                "default": "image",
            },
            "output_dir": {
                "type": "string",
                "description": "Local directory to write the split dataset into.",
                "default": "./cvagent_output",
            },
            "target_size": {
                "type": "integer",
                "description": "Maximum total images to include after filtering.",
                "default": 1000,
            },
            "train_ratio": {"type": "number", "default": 0.7},
            "val_ratio":   {"type": "number", "default": 0.2},
            "test_ratio":  {"type": "number", "default": 0.1},
        },
        "required": ["dataset_id"],
    },
    fn=generate_and_run_script,
)
