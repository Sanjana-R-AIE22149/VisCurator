"""
VisCurator / CVAgent — Agent Tools
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Async tool functions for the dataset curation agent.
Supports HuggingFace, Kaggle, Roboflow, OpenImages, and Papers With Code.
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
import logging
import os
import random
import shutil
import sys
import tempfile
import textwrap
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Coroutine

import aiohttp
import albumentations as A
import cv2
import imagehash
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

ToolFn = Callable[..., Coroutine[Any, Any, dict[str, Any]]]
TOOL_REGISTRY: dict[str, dict[str, Any]] = {}
_HTTP_TIMEOUT = aiohttp.ClientTimeout(total=45)


def _register(name: str, description: str, parameters: dict[str, Any], fn: ToolFn) -> None:
    TOOL_REGISTRY[name] = {"name": name, "description": description, "parameters": parameters, "function": fn}


async def execute_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    entry = TOOL_REGISTRY.get(name)
    if entry is None:
        return {"status": "error", "error": f"Unknown tool '{name}'."}
    try:
        result = await asyncio.wait_for(entry["function"](**arguments), timeout=120)
        return result
    except asyncio.TimeoutError:
        return {"status": "error", "error": f"Tool '{name}' timed out."}
    except Exception as exc:
        logger.exception("Tool '%s' raised an exception", name)
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TOOL 1 — search_datasets  (multi-source)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def _search_huggingface(query: str, max_results: int) -> list[dict[str, Any]]:
    url = "https://huggingface.co/api/datasets"
    params = {"search": query, "limit": min(max_results, 20), "sort": "likes", "direction": "-1", "full": "false"}
    async with aiohttp.ClientSession(timeout=_HTTP_TIMEOUT) as session:
        async with session.get(url, params=params) as resp:
            if resp.status != 200:
                return []
            data = await resp.json()
    results = []
    for ds in data:
        tags = ds.get("tags") or []
        results.append({
            "source": "HuggingFace",
            "dataset_id": ds.get("id", ""),
            "name": ds.get("id", "").split("/")[-1],
            "description": (ds.get("description") or "")[:300],
            "downloads": ds.get("downloads", 0),
            "likes": ds.get("likes", 0),
            "tags": tags[:10],
            "url": f"https://huggingface.co/datasets/{ds.get('id', '')}",
            "size_estimate": "unknown",
        })
    return results


_KAGGLE_UNAVAILABLE = [{
    "source": "Kaggle",
    "dataset_id": "unavailable",
    "name": "Kaggle search unavailable",
    "description": "Set KAGGLE_USERNAME and KAGGLE_KEY in .env to enable Kaggle search. Get them at https://www.kaggle.com/account.",
    "downloads": 0,
    "likes": 0,
    "tags": [],
    "url": "https://www.kaggle.com/account",
    "size_estimate": "unknown",
    "unavailable": True,
}]


async def _search_kaggle(query: str, max_results: int) -> list[dict[str, Any]]:
    """Search Kaggle datasets via the v1 API using HTTP Basic Auth."""
    username = os.getenv("KAGGLE_USERNAME", "").strip()
    key = os.getenv("KAGGLE_KEY", "").strip()
    if not username or not key:
        return _KAGGLE_UNAVAILABLE

    url = "https://www.kaggle.com/api/v1/datasets/list"
    params = {"search": query, "sortBy": "votes", "pageSize": min(max_results, 20)}
    auth = aiohttp.BasicAuth(username, key)
    try:
        async with aiohttp.ClientSession(timeout=_HTTP_TIMEOUT) as session:
            async with session.get(url, params=params, auth=auth) as resp:
                if resp.status == 401:
                    logger.warning("Kaggle auth failed — check KAGGLE_USERNAME / KAGGLE_KEY")
                    return _KAGGLE_UNAVAILABLE
                if resp.status != 200:
                    return []
                data = await resp.json()
    except Exception as exc:
        logger.warning("Kaggle search error: %s", exc)
        return []
    results = []
    for ds in data:
        owner = ds.get("ownerUser", ds.get("creatorName", ""))
        slug = ds.get("datasetSlug", ds.get("slug", ""))
        results.append({
            "source": "Kaggle",
            "dataset_id": f"{owner}/{slug}",
            "name": ds.get("title", slug),
            "description": (ds.get("description") or ds.get("subtitle", ""))[:300],
            "downloads": ds.get("downloadCount", 0),
            "likes": ds.get("voteCount", 0),
            "tags": [t.get("name", "") for t in (ds.get("tags") or [])[:8]],
            "url": f"https://www.kaggle.com/datasets/{owner}/{slug}",
            "size_estimate": f"{round(ds.get('totalBytes', 0) / 1e6, 1)} MB" if ds.get("totalBytes") else "unknown",
        })
    return results


_ROBOFLOW_UNAVAILABLE = [{
    "source": "Roboflow",
    "dataset_id": "unavailable",
    "name": "Roboflow search unavailable",
    "description": "Set ROBOFLOW_API_KEY in .env to enable Roboflow Universe search. Get a free key at https://roboflow.com.",
    "downloads": 0,
    "likes": 0,
    "tags": [],
    "url": "https://roboflow.com",
    "size_estimate": "unknown",
    "unavailable": True,
}]


async def _search_roboflow(query: str, max_results: int) -> list[dict[str, Any]]:
    """Search Roboflow Universe via the universeSearch API endpoint."""
    rf_key = os.getenv("ROBOFLOW_API_KEY", "").strip()
    if not rf_key:
        return _ROBOFLOW_UNAVAILABLE

    url = "https://api.roboflow.com/universeSearch"
    params = {"q": query, "limit": min(max_results, 20), "api_key": rf_key}
    try:
        async with aiohttp.ClientSession(timeout=_HTTP_TIMEOUT) as session:
            async with session.get(url, params=params) as resp:
                if resp.status == 401:
                    logger.warning("Roboflow auth failed — check ROBOFLOW_API_KEY")
                    return _ROBOFLOW_UNAVAILABLE
                if resp.status != 200:
                    return []
                data = await resp.json()
    except Exception as exc:
        logger.warning("Roboflow search error: %s", exc)
        return []
    results = []
    for ds in (data.get("results") or data.get("datasets") or []):
        workspace = ds.get("workspace", ds.get("workspaceName", ""))
        project = ds.get("project", ds.get("projectName", ds.get("slug", "")))
        results.append({
            "source": "Roboflow",
            "dataset_id": f"{workspace}/{project}",
            "name": ds.get("name", project),
            "description": (ds.get("description") or "")[:300],
            "downloads": ds.get("images", 0),
            "likes": ds.get("stars", 0),
            "tags": ds.get("classes", [])[:10],
            "url": f"https://universe.roboflow.com/{workspace}/{project}",
            "size_estimate": f"{ds.get('images', 0):,} images",
        })
    return results


async def _search_paperswithcode(query: str, max_results: int) -> list[dict[str, Any]]:
    """Search Papers With Code datasets."""
    url = "https://paperswithcode.com/api/v1/datasets/"
    params = {"q": query, "page_size": min(max_results, 20)}
    try:
        async with aiohttp.ClientSession(timeout=_HTTP_TIMEOUT) as session:
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()
    except Exception:
        return []
    results = []
    for ds in (data.get("results") or []):
        results.append({
            "source": "PapersWithCode",
            "dataset_id": ds.get("name", ""),
            "name": ds.get("name", ""),
            "description": (ds.get("description") or "")[:300],
            "downloads": ds.get("paper_count", 0),
            "likes": ds.get("paper_count", 0),
            "tags": [],
            "url": ds.get("url", f"https://paperswithcode.com/dataset/{ds.get('name', '').lower().replace(' ', '-')}"),
            "size_estimate": "unknown",
        })
    return results


async def search_datasets(
    query: str,
    sources: list[str] | None = None,
    max_results_per_source: int = 5,
) -> dict[str, Any]:
    """Search for datasets across multiple sources simultaneously."""
    if sources is None:
        sources = ["huggingface", "kaggle", "roboflow", "paperswithcode"]

    logger.info("search_datasets  query=%r  sources=%r", query, sources)

    tasks = {}
    if "huggingface" in sources:
        tasks["huggingface"] = _search_huggingface(query, max_results_per_source)
    if "kaggle" in sources:
        tasks["kaggle"] = _search_kaggle(query, max_results_per_source)
    if "roboflow" in sources:
        tasks["roboflow"] = _search_roboflow(query, max_results_per_source)
    if "paperswithcode" in sources:
        tasks["paperswithcode"] = _search_paperswithcode(query, max_results_per_source)

    results_by_source = await asyncio.gather(*tasks.values(), return_exceptions=True)
    all_results = []
    source_status: dict[str, str] = {}
    for src, res in zip(tasks.keys(), results_by_source):
        if isinstance(res, Exception):
            source_status[src] = "error"
        elif isinstance(res, list):
            unavailable = res and res[0].get("unavailable")
            if unavailable:
                source_status[src] = "unavailable"
            else:
                source_status[src] = f"{len(res)} results"
            # Include unavailable sentinel in results so agent can report it,
            # but exclude from ranking
            real = [r for r in res if not r.get("unavailable")]
            all_results.extend(real)
            if unavailable:
                all_results.extend(res)  # append the single sentinel at the end
        else:
            source_status[src] = "error"

    # Rank real results by downloads+likes; sentinels stay at the end
    real_results = [r for r in all_results if not r.get("unavailable")]
    sentinel_results = [r for r in all_results if r.get("unavailable")]
    real_results.sort(key=lambda x: x.get("downloads", 0) + x.get("likes", 0) * 5, reverse=True)
    all_results = real_results + sentinel_results

    return {
        "status": "success",
        "query": query,
        "total_found": len(real_results),
        "sources_searched": list(tasks.keys()),
        "source_status": source_status,
        "datasets": all_results,
    }


_register(
    name="search_datasets",
    description=(
        "Search for computer vision datasets across multiple sources: HuggingFace, Kaggle, Roboflow, and Papers With Code. "
        "Returns ranked results with source, size, description, and URL. "
        "Use this FIRST to find candidate datasets matching the user's task."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Natural language search query e.g. 'apple leaf disease detection'"},
            "sources": {
                "type": "array",
                "items": {"type": "string", "enum": ["huggingface", "kaggle", "roboflow", "paperswithcode"]},
                "description": "Which sources to search. Defaults to all four.",
            },
            "max_results_per_source": {"type": "integer", "default": 5, "description": "Max results per source (1-20)."},
        },
        "required": ["query"],
    },
    fn=search_datasets,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TOOL 2 — get_dataset_info
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def get_dataset_info(dataset_id: str, source: str = "huggingface") -> dict[str, Any]:
    """Retrieve detailed metadata for a specific dataset."""
    logger.info("get_dataset_info  dataset_id=%r  source=%r", dataset_id, source)

    if source == "huggingface":
        url = f"https://huggingface.co/api/datasets/{dataset_id}"
        headers: dict[str, str] = {}
        hf_token = os.getenv("HF_TOKEN")
        if hf_token and hf_token != "optional_huggingface_token":
            headers["Authorization"] = f"Bearer {hf_token}"

        async with aiohttp.ClientSession(timeout=_HTTP_TIMEOUT) as session:
            async with session.get(url, headers=headers) as resp:
                if resp.status == 404:
                    return {"status": "error", "error": f"Dataset '{dataset_id}' not found."}
                if resp.status != 200:
                    return {"status": "error", "error": f"HTTP {resp.status}"}
                meta = await resp.json()

            splits_info: dict[str, Any] = {}
            splits_url = f"https://datasets-server.huggingface.co/info?dataset={dataset_id}"
            try:
                async with session.get(splits_url, headers=headers) as resp2:
                    if resp2.status == 200:
                        info_payload = await resp2.json()
                        ds_info = info_payload.get("dataset_info") or {}
                        for config_name, config_data in ds_info.items():
                            raw_splits = config_data.get("splits") or {}
                            for split_name, split_meta in raw_splits.items():
                                splits_info[split_name] = {
                                    "num_rows": split_meta.get("num_examples", 0),
                                    "num_bytes": split_meta.get("num_bytes", 0),
                                }
                            if "features" in config_data:
                                features_raw = config_data["features"]
                                if isinstance(features_raw, dict):
                                    splits_info["_features"] = list(features_raw.keys())
                            break
            except Exception as exc:
                logger.warning("Could not fetch splits for '%s': %s", dataset_id, exc)

        features = splits_info.pop("_features", [])
        tags = meta.get("tags") or []
        card = meta.get("cardData") or {}
        total_bytes = sum(s.get("num_bytes", 0) for s in splits_info.values())
        total_rows = sum(s.get("num_rows", 0) for s in splits_info.values())

        return {
            "status": "success",
            "source": "huggingface",
            "dataset_id": dataset_id,
            "description": (meta.get("description") or "")[:500],
            "tags": tags[:20],
            "license": card.get("license", "unknown"),
            "features": features,
            "splits": splits_info,
            "total_rows": total_rows,
            "total_size_mb": round(total_bytes / (1024 * 1024), 2) if total_bytes else 0,
            "downloads": meta.get("downloads", 0),
            "likes": meta.get("likes", 0),
            "url": f"https://huggingface.co/datasets/{dataset_id}",
        }

    elif source == "kaggle":
        return {
            "status": "success",
            "source": "kaggle",
            "dataset_id": dataset_id,
            "description": f"Kaggle dataset: {dataset_id}",
            "url": f"https://www.kaggle.com/datasets/{dataset_id}",
            "note": "Use the Kaggle CLI to download: kaggle datasets download -d " + dataset_id,
        }

    elif source == "roboflow":
        parts = dataset_id.split("/")
        if len(parts) >= 2:
            workspace, project = parts[0], parts[1]
            url = f"https://universe.roboflow.com/{workspace}/{project}"
        else:
            url = f"https://universe.roboflow.com/search?q={dataset_id}"
        return {
            "status": "success",
            "source": "roboflow",
            "dataset_id": dataset_id,
            "url": url,
            "note": f"Visit {url} to download or use the Roboflow Python SDK.",
        }

    return {"status": "error", "error": f"Unknown source: {source}"}


_register(
    name="get_dataset_info",
    description="Get detailed metadata for a specific dataset: splits, features, size, license. Supports huggingface, kaggle, roboflow sources.",
    parameters={
        "type": "object",
        "properties": {
            "dataset_id": {"type": "string", "description": "Dataset ID e.g. 'keremberke/plant-disease-detection'"},
            "source": {"type": "string", "enum": ["huggingface", "kaggle", "roboflow"], "default": "huggingface"},
        },
        "required": ["dataset_id"],
    },
    fn=get_dataset_info,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TOOL 3 — estimate_dataset_quality
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def estimate_dataset_quality(dataset_id: str, sample_size: int = 30) -> dict[str, Any]:
    """Download a small sample from a HuggingFace dataset and estimate image quality."""
    logger.info("estimate_dataset_quality  dataset_id=%r  sample_size=%d", dataset_id, sample_size)

    rows_url = (
        f"https://datasets-server.huggingface.co/rows"
        f"?dataset={dataset_id}&config=default&split=train"
        f"&offset=0&length={min(sample_size, 100)}"
    )
    headers: dict[str, str] = {}
    hf_token = os.getenv("HF_TOKEN")
    if hf_token and hf_token != "optional_huggingface_token":
        headers["Authorization"] = f"Bearer {hf_token}"

    payload = None
    async with aiohttp.ClientSession(timeout=_HTTP_TIMEOUT) as session:
        async with session.get(rows_url, headers=headers) as resp:
            if resp.status == 200:
                payload = await resp.json()
            else:
                # Try to find the right config
                info_url = f"https://datasets-server.huggingface.co/info?dataset={dataset_id}"
                async with session.get(info_url, headers=headers) as info_resp:
                    if info_resp.status != 200:
                        return {"status": "error", "error": f"Cannot access dataset '{dataset_id}' via datasets-server."}
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
                            return {"status": "error", "error": f"Rows endpoint failed: HTTP {resp2.status}."}
                        payload = await resp2.json()

    rows = (payload or {}).get("rows") or []
    if not rows:
        return {"status": "error", "error": "No rows returned."}

    image_urls: list[str] = []
    labels: list[str] = []
    label_keys = ("label", "labels", "class", "category", "target")

    for row_wrapper in rows:
        row = row_wrapper.get("row") or row_wrapper
        for key in ("image", "img", "image_url", "file"):
            val = row.get(key)
            if isinstance(val, dict) and "src" in val:
                image_urls.append(val["src"])
                break
            elif isinstance(val, str) and val.startswith("http"):
                image_urls.append(val)
                break
        for lk in label_keys:
            lv = row.get(lk)
            if lv is not None:
                labels.append(str(lv))
                break

    sample_limit = min(len(image_urls), 20)
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
            lap = (gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:] - 4 * gray[1:-1, 1:-1])
            blur = float(np.var(lap))
            return {"w": img.width, "h": img.height, "blur": blur}
        except Exception:
            return None

    async with aiohttp.ClientSession() as session:
        measurements = await asyncio.gather(*[_measure(session, u) for u in sampled_urls])

    widths, heights, blur_scores = [], [], []
    hashes: list[str] = []
    for m in measurements:
        if m:
            widths.append(m["w"])
            heights.append(m["h"])
            blur_scores.append(m["blur"])

    n = len(widths)
    if n == 0:
        return {"status": "partial", "dataset_id": dataset_id, "images_sampled": 0, "quality_score": 0}

    w_arr, h_arr, b_arr = np.array(widths), np.array(heights), np.array(blur_scores)
    blurry_count = int(np.sum(b_arr < 80.0))
    blur_ratio = round(blurry_count / n, 3)
    for url in sampled_urls:
        try:
            async with aiohttp.ClientSession(timeout=_HTTP_TIMEOUT) as session:
                async with session.get(url) as resp:
                    if resp.status != 200:
                        continue
                    payload_bytes = await resp.read()
            hashes.append(str(imagehash.dhash(Image.open(io.BytesIO(payload_bytes)).convert("RGB"))))
        except Exception:
            continue
    unique_hashes = len(set(hashes))
    duplicate_percentage = round(((len(hashes) - unique_hashes) / len(hashes)) * 100, 1) if hashes else 0.0

    class_dist = _class_dist(labels)
    class_balance = _describe_class_balance(class_dist)
    resolution_score = min(float(np.median(w_arr * h_arr)) / (640 * 480), 1.0) * 30
    sharpness_score = (1 - blur_ratio) * 40
    balance_score = _balance_score(class_dist) * 30
    quality_score = round(max(0, min(100, resolution_score + sharpness_score + balance_score)), 1)

    # Build blur scatter data for the frontend chart
    blur_scatter = [
        {
            "id": i,
            "laplacian": round(b_arr[i], 1),
            "resolution": widths[i] * heights[i] // 1000,
            "accepted": bool(b_arr[i] >= 80.0),
        }
        for i in range(n)
    ]

    return {
        "status": "success",
        "dataset_id": dataset_id,
        "images_sampled": n,
        "avg_resolution": {"width": int(np.mean(w_arr)), "height": int(np.mean(h_arr))},
        "blur_ratio": blur_ratio,
        "blurry_images": blurry_count,
        "avg_sharpness": round(float(b_arr.mean()), 1),
        "class_distribution": class_dist,
        "class_balance": class_balance,
        "duplicate_percentage": duplicate_percentage,
        "quality_score": quality_score,
        "quality_grade": "A" if quality_score >= 80 else "B" if quality_score >= 60 else "C" if quality_score >= 40 else "D",
        "blur_scatter": blur_scatter,
        "resolution_stats": {
            "avg_width": int(np.mean(w_arr)),
            "avg_height": int(np.mean(h_arr)),
            "median_kpx": int(np.median(w_arr * h_arr) // 1000),
        },
    }


def _class_dist(labels: list[str]) -> dict[str, int]:
    dist: dict[str, int] = {}
    for lbl in labels:
        dist[lbl] = dist.get(lbl, 0) + 1
    return dict(sorted(dist.items(), key=lambda x: -x[1]))


def _balance_score(dist: dict[str, int]) -> float:
    if not dist:
        return 0.5
    counts = list(dist.values())
    if len(counts) == 1:
        return 1.0
    arr = np.array(counts, dtype=float)
    cv = float(arr.std() / arr.mean()) if arr.mean() > 0 else 1.0
    return max(0.0, 1.0 - cv)


def _describe_class_balance(dist: dict[str, int]) -> str:
    if not dist:
        return "Unknown"
    counts = np.array(list(dist.values()), dtype=float)
    if len(counts) == 1:
        return "Single-class"
    ratio = counts.max() / max(1.0, counts.min())
    if ratio <= 1.35:
        return "Balanced"
    if ratio <= 2.25:
        return "Moderately Imbalanced"
    return "Highly Imbalanced"


_register(
    name="estimate_dataset_quality",
    description="Download a sample from a HuggingFace dataset and measure image quality: resolution, blur (Laplacian variance), class balance, and overall quality score 0-100.",
    parameters={
        "type": "object",
        "properties": {
            "dataset_id": {"type": "string"},
            "sample_size": {"type": "integer", "default": 30},
        },
        "required": ["dataset_id"],
    },
    fn=estimate_dataset_quality,
)


async def analyze_dataset_and_plan_processing(
    blur_score: float,
    duplicate_percentage: float,
    class_balance: str,
    dataset_size: int,
    resolution_stats: dict[str, Any],
) -> dict[str, Any]:
    """Deterministically convert quality stats into a preprocessing plan."""
    avg_width = int(resolution_stats.get("avg_width", resolution_stats.get("width", 0)) or 0)
    avg_height = int(resolution_stats.get("avg_height", resolution_stats.get("height", 0)) or 0)
    median_kpx = int(resolution_stats.get("median_kpx", max(1, (avg_width * avg_height) // 1000)) or 1)

    needs_blur_filtering = blur_score < 65
    needs_deduplication = duplicate_percentage >= 8
    needs_augmentation = class_balance in {"Moderately Imbalanced", "Highly Imbalanced"} or dataset_size < 1200
    needs_synthetic_generation = dataset_size < 300 or class_balance == "Highly Imbalanced"

    recommended_augmentations: list[str] = []
    if median_kpx < 180:
        recommended_augmentations.append("resize_224")
    else:
        recommended_augmentations.append("resize_256")
    if needs_augmentation:
        recommended_augmentations.extend(["horizontal_flip", "random_brightness_contrast"])
    if class_balance in {"Moderately Imbalanced", "Highly Imbalanced"}:
        recommended_augmentations.append("minority_class_oversampling")
    if blur_score >= 70 and median_kpx >= 160:
        recommended_augmentations.append("shift_scale_rotate")
    recommended_model = "YOLOv8n" if dataset_size >= 1200 else "ResNet18"

    reasons = []
    reasons.append(f"Blur score {blur_score:.1f} {'requires' if needs_blur_filtering else 'does not require'} blur filtering.")
    reasons.append(f"Estimated duplicates at {duplicate_percentage:.1f}% {'trigger' if needs_deduplication else 'do not trigger'} deduplication.")
    reasons.append(f"Class balance is {class_balance.lower()}, so augmentation is {'recommended' if needs_augmentation else 'optional'}.")
    reasons.append(f"Dataset size of {dataset_size:,} samples {'benefits from' if needs_synthetic_generation else 'does not require'} synthetic support.")
    reasons.append(f"Average resolution is {avg_width}x{avg_height}, so standardized resizing is included.")

    return {
        "status": "success",
        "needs_blur_filtering": needs_blur_filtering,
        "needs_deduplication": needs_deduplication,
        "needs_augmentation": needs_augmentation,
        "needs_synthetic_generation": needs_synthetic_generation,
        "recommended_augmentations": recommended_augmentations,
        "recommended_model": recommended_model,
        "reasoning": " ".join(reasons),
        "input_summary": {
            "blur_score": blur_score,
            "duplicate_percentage": duplicate_percentage,
            "class_balance": class_balance,
            "dataset_size": dataset_size,
            "resolution_stats": {
                "avg_width": avg_width,
                "avg_height": avg_height,
                "median_kpx": median_kpx,
            },
        },
    }


_register(
    name="analyze_dataset_and_plan_processing",
    description=(
        "Use deterministic rule-based logic to decide whether a dataset needs blur filtering, deduplication, augmentation, "
        "synthetic generation support, and which augmentations/model are recommended."
    ),
    parameters={
        "type": "object",
        "properties": {
            "blur_score": {"type": "number"},
            "duplicate_percentage": {"type": "number"},
            "class_balance": {"type": "string"},
            "dataset_size": {"type": "integer"},
            "resolution_stats": {"type": "object"},
        },
        "required": ["blur_score", "duplicate_percentage", "class_balance", "dataset_size", "resolution_stats"],
    },
    fn=analyze_dataset_and_plan_processing,
)


def _build_preprocess_script(
    dataset_id: str,
    source: str,
    label_column: str,
    image_column: str,
    output_dir: str,
    target_size: int,
    plan: dict[str, Any],
) -> str:
    return textwrap.dedent(
        f"""\
        #!/usr/bin/env python3
        import io
        import json
        import random
        from collections import Counter
        from pathlib import Path

        import albumentations as A
        import cv2
        import imagehash
        import numpy as np
        from PIL import Image

        DATASET_ID = {dataset_id!r}
        SOURCE = {source!r}
        LABEL_COL = {label_column!r}
        IMAGE_COL = {image_column!r}
        OUTPUT_DIR = Path({output_dir!r})
        TARGET_SIZE = {target_size}
        PLAN = {json.dumps(plan)}
        RANDOM_SEED = 42
        BLUR_THRESHOLD = 80.0

        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

        def emit_event(kind, payload):
            print("JSON_EVENT:" + json.dumps({{"kind": kind, **payload}}), flush=True)

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

        from datasets import load_dataset
        ds = load_dataset(DATASET_ID, trust_remote_code=True)
        all_rows = []
        for split_name, split_ds in ds.items():
            print(f"> split {{split_name}}: {{len(split_ds)}} rows", flush=True)
            all_rows.extend(list(split_ds))

        output_processed = OUTPUT_DIR / DATASET_ID.replace("/", "_") / "processed"
        if output_processed.exists():
            import shutil
            shutil.rmtree(output_processed)
        output_processed.mkdir(parents=True, exist_ok=True)

        before_counts = Counter()
        after_counts = Counter()
        seen_hashes = set()
        blur_scatter = []
        augmentation_summary = Counter()
        accepted_records = []
        rejected_blur = 0
        rejected_dup = 0

        for idx, row in enumerate(all_rows):
            if len(accepted_records) >= TARGET_SIZE:
                break
            img = ensure_rgb(row.get(IMAGE_COL))
            if img is None:
                continue
            label = str(row.get(LABEL_COL, "unknown"))
            before_counts[label] += 1
            blur_value = laplacian_var(img)
            img_hash = str(imagehash.dhash(img))
            is_blurry = blur_value < BLUR_THRESHOLD
            is_dup = img_hash in seen_hashes
            blur_scatter.append({{
                "id": idx,
                "laplacian": round(blur_value, 1),
                "resolution": int((img.width * img.height) / 1000),
                "accepted": not is_blurry and not is_dup,
            }})
            if PLAN.get("needs_blur_filtering") and is_blurry:
                rejected_blur += 1
                continue
            if PLAN.get("needs_deduplication") and is_dup:
                rejected_dup += 1
                continue
            seen_hashes.add(img_hash)
            accepted_records.append((img, label))

        augmenter = build_augmenter()
        minority_target = max([count for count in before_counts.values()] or [0])
        minority_labels = {{label for label, count in before_counts.items() if count < minority_target}}

        export_total = 0
        for sample_index, (img, label) in enumerate(accepted_records):
            out_dir = output_processed / label
            out_dir.mkdir(parents=True, exist_ok=True)
            base_np = np.array(img)
            resized = augmenter(image=base_np)["image"]
            Image.fromarray(resized).save(out_dir / f"sample_{{sample_index:05d}}.jpg", "JPEG", quality=92)
            after_counts[label] += 1
            export_total += 1

            if PLAN.get("needs_augmentation") and label in minority_labels and after_counts[label] < min(minority_target, after_counts[label] + 2):
                augmented = augmenter(image=base_np)["image"]
                Image.fromarray(augmented).save(out_dir / f"sample_{{sample_index:05d}}_aug.jpg", "JPEG", quality=92)
                augmentation_summary["augmented_images"] += 1
                after_counts[label] += 1
                export_total += 1

        preprocess_report = {{
            "dataset_id": DATASET_ID,
            "output_dir": str(output_processed),
            "plan": PLAN,
            "before_stats": {{
                "images": int(sum(before_counts.values())),
                "class_distribution": dict(sorted(before_counts.items())),
            }},
            "after_stats": {{
                "images": int(export_total),
                "class_distribution": dict(sorted(after_counts.items())),
                "blur_filtered": int(rejected_blur),
                "duplicates_removed": int(rejected_dup),
            }},
            "class_distribution": dict(sorted(after_counts.items())),
            "augmentation_summary": {{
                "operations": PLAN.get("recommended_augmentations", []),
                "augmented_images": int(augmentation_summary.get("augmented_images", 0)),
                "synthetic_generation_recommended": bool(PLAN.get("needs_synthetic_generation")),
            }},
            "blur_scatter": blur_scatter[:200],
        }}
        report_path = output_processed / "preprocessing_report.json"
        report_path.write_text(json.dumps(preprocess_report, indent=2), encoding="utf-8")
        emit_event("report", preprocess_report)
        print(f"> preprocessing report written to {{report_path}}", flush=True)
        print(f"> exported {{export_total}} processed images", flush=True)
        """
    )


async def clean_and_augment_dataset(
    dataset_id: str,
    source: str = "huggingface",
    label_column: str = "label",
    image_column: str = "image",
    output_dir: str = "./cvagent_output",
    target_size: int = 1000,
    processing_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run deterministic filtering, deduplication, normalization, and augmentation."""
    plan = processing_plan or {
        "needs_blur_filtering": True,
        "needs_deduplication": True,
        "needs_augmentation": False,
        "needs_synthetic_generation": False,
        "recommended_augmentations": ["resize_224"],
        "recommended_model": "ResNet18",
        "reasoning": "Default preprocessing plan.",
    }
    logger.info("clean_and_augment_dataset dataset_id=%r source=%r target=%d", dataset_id, source, target_size)

    script_code = _build_preprocess_script(
        dataset_id=dataset_id,
        source=source,
        label_column=label_column,
        image_column=image_column,
        output_dir=output_dir,
        target_size=target_size,
        plan=plan,
    )
    script_dir = Path(output_dir).parent
    script_dir.mkdir(parents=True, exist_ok=True)
    script_path = script_dir / f"cvagent_preprocess_{dataset_id.replace('/', '_')}.py"
    script_path.write_text(script_code, encoding="utf-8")

    emit = _script_emit_callback
    report_payload: dict[str, Any] | None = None

    async def _stream(line: str) -> None:
        if emit:
            try:
                await emit(line.rstrip(), "stdout")
            except Exception:
                pass

    await _stream(f"> Preprocessing script written: {script_path}")
    await _stream(f"> Running preprocessing for {dataset_id}")

    output_lines: list[str] = []
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            str(script_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env={**os.environ},
        )
        assert proc.stdout is not None
        async for raw_line in proc.stdout:
            line = raw_line.decode("utf-8", errors="replace").rstrip()
            output_lines.append(line)
            if line.startswith("JSON_EVENT:"):
                payload = json.loads(line[len("JSON_EVENT:"):])
                if payload.get("kind") == "report":
                    report_payload = payload
                    continue
            await _stream(line)
        await proc.wait()
        exit_code = proc.returncode or 0
    except Exception as exc:
        err = f"Preprocessing subprocess failed: {exc}"
        await _stream(f"> {err}")
        return {"status": "error", "error": err, "output_tail": output_lines[-40:]}

    result = {
        "status": "success" if exit_code == 0 else "error",
        "script_path": str(script_path),
        "exit_code": exit_code,
        "output_tail": output_lines[-40:],
        "preprocessing_report": report_payload or {},
    }
    if report_payload:
        result.update(report_payload)
    return result


_register(
    name="clean_and_augment_dataset",
    description=(
        "Process a selected dataset with deterministic blur filtering, dHash deduplication, resize normalization, "
        "and Albumentations-based augmentation, then export a preprocessing report and processed dataset."
    ),
    parameters={
        "type": "object",
        "properties": {
            "dataset_id": {"type": "string"},
            "source": {"type": "string", "default": "huggingface"},
            "label_column": {"type": "string", "default": "label"},
            "image_column": {"type": "string", "default": "image"},
            "output_dir": {"type": "string", "default": "./cvagent_output"},
            "target_size": {"type": "integer", "default": 1000},
            "processing_plan": {"type": "object"},
        },
        "required": ["dataset_id"],
    },
    fn=clean_and_augment_dataset,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TOOL 4 — present_dataset_options
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def present_dataset_options(
    options: list[dict[str, Any]],
    recommendation: str,
    question: str,
) -> dict[str, Any]:
    """Present curated dataset options to the user and ask which one to download.
    
    This tool PAUSES the pipeline and waits for user input.
    The frontend will render the options as clickable cards.
    """
    logger.info("present_dataset_options  count=%d", len(options))
    return {
        "status": "waiting_for_user",
        "type": "dataset_selection",
        "question": question,
        "recommendation": recommendation,
        "options": options,
    }


_register(
    name="present_dataset_options",
    description=(
        "Show the user a list of found datasets and ASK them which one to download. "
        "ALWAYS call this after searching — never silently pick a dataset yourself. "
        "The pipeline pauses here until the user selects an option. "
        "Include source, name, size, quality info in each option."
    ),
    parameters={
        "type": "object",
        "properties": {
            "options": {
                "type": "array",
                "description": "List of dataset options to present. Each must have: source, dataset_id, name, description, size_estimate, url, pros, cons.",
                "items": {
                    "type": "object",
                    "properties": {
                        "source": {"type": "string"},
                        "dataset_id": {"type": "string"},
                        "name": {"type": "string"},
                        "description": {"type": "string"},
                        "size_estimate": {"type": "string"},
                        "url": {"type": "string"},
                        "pros": {"type": "array", "items": {"type": "string"}},
                        "cons": {"type": "array", "items": {"type": "string"}},
                    },
                },
            },
            "recommendation": {"type": "string", "description": "Which option you recommend and why (1-2 sentences)."},
            "question": {"type": "string", "description": "The question to ask the user, e.g. 'Which dataset would you like me to download and prepare?'"},
        },
        "required": ["options", "recommendation", "question"],
    },
    fn=present_dataset_options,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TOOL 5 — ask_clarification
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

async def ask_clarification(
    question: str,
    options: list[str] | None = None,
    context: str = "",
) -> dict[str, Any]:
    """Ask the user a clarifying question before proceeding.
    
    Use this when the user's request is ambiguous — e.g. they said 'disease detection'
    but you need to know if they mean plant, human, or animal disease.
    """
    logger.info("ask_clarification  question=%r", question)
    return {
        "status": "waiting_for_user",
        "type": "clarification",
        "question": question,
        "options": options or [],
        "context": context,
    }


_register(
    name="ask_clarification",
    description=(
        "Ask the user a clarifying question when their request is ambiguous. "
        "Use this BEFORE searching if the query is vague (e.g. 'plant disease' — which crop? which disease?). "
        "Provide quick-reply options when possible."
    ),
    parameters={
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "The clarifying question to ask."},
            "options": {"type": "array", "items": {"type": "string"}, "description": "Quick-reply choices (optional but recommended)."},
            "context": {"type": "string", "description": "Why you need this info (shown to user)."},
        },
        "required": ["question"],
    },
    fn=ask_clarification,
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TOOL 6 — generate_and_run_script
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_script_emit_callback: Any = None


def set_script_emit_callback(cb: Any) -> None:
    global _script_emit_callback
    _script_emit_callback = cb


def _build_download_script(
    dataset_id: str,
    source: str,
    label_column: str,
    image_column: str,
    output_dir: str,
    target_size: int,
    split_ratios: tuple[float, float, float],
) -> str:
    train_r, val_r, test_r = split_ratios
    return textwrap.dedent(f"""\
        #!/usr/bin/env python3
        \"\"\"CVAgent auto-generated download script — {dataset_id} from {source}\"\"\"
        import os, sys, io, math, shutil, hashlib, random
        from pathlib import Path
        from PIL import Image
        import numpy as np

        DATASET_ID   = "{dataset_id}"
        SOURCE       = "{source}"
        LABEL_COL    = "{label_column}"
        IMAGE_COL    = "{image_column}"
        OUTPUT_DIR   = Path("{output_dir}")
        TARGET_SIZE  = {target_size}
        SPLIT_RATIOS = ({train_r}, {val_r}, {test_r})
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

        print(f"> Loading {{DATASET_ID}} from {{SOURCE}} ...")
        try:
            from datasets import load_dataset
        except ImportError:
            print("ERR: pip install datasets", file=sys.stderr)
            sys.exit(1)

        try:
            ds = load_dataset(DATASET_ID, trust_remote_code=True)
        except Exception as e:
            print(f"ERR: Could not load dataset: {{e}}", file=sys.stderr)
            sys.exit(1)

        print(f"> Splits: {{list(ds.keys())}}")
        all_rows = []
        for split_name, split_ds in ds.items():
            all_rows.extend(list(split_ds))
            print(f">   {{split_name}}: {{len(split_ds)}} rows")
        print(f"> Pool size: {{len(all_rows)}} rows")

        print("> Filtering (blur + dedup) ...")
        accepted, rejected_blur, rejected_dup = [], 0, 0
        seen_hashes = set()

        for i, row in enumerate(all_rows):
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

            label = str(row.get(LABEL_COL, "unknown"))
            accepted.append((img, label))

            if (i+1) % 500 == 0:
                print(f">   Scanned {{i+1}}/{{len(all_rows)}} | accepted {{len(accepted)}} | blur {{rejected_blur}} | dup {{rejected_dup}}")

        print(f"> Filter done — accepted: {{len(accepted)}} | blur_rejected: {{rejected_blur}} | dup_rejected: {{rejected_dup}}")

        random.seed(42)
        random.shuffle(accepted)
        n = len(accepted)
        n_train = math.floor(n * SPLIT_RATIOS[0])
        n_val   = math.floor(n * SPLIT_RATIOS[1])
        splits  = {{"train": accepted[:n_train], "val": accepted[n_train:n_train+n_val], "test": accepted[n_train+n_val:]}}

        print("> Writing splits ...")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        counts = {{}}
        for split_name, rows in splits.items():
            print(f">   {{split_name}}: {{len(rows)}} images")
            for img, label in rows:
                dest = OUTPUT_DIR / split_name / label
                dest.mkdir(parents=True, exist_ok=True)
                idx = counts.get(f"{{split_name}}/{{label}}", 0)
                counts[f"{{split_name}}/{{label}}"] = idx + 1
                img.save(dest / f"{{label}}_{{idx:05d}}.jpg", "JPEG", quality=92)

        total = sum(counts.values())
        print(f"> Wrote {{total}} images to {{OUTPUT_DIR.resolve()}}")
        print("> Class breakdown:")
        for k, v in sorted(counts.items()):
            print(f">   {{k}}: {{v}}")
        print("> Pipeline complete.")
    """)


async def generate_and_run_script(
    dataset_id: str,
    source: str = "huggingface",
    label_column: str = "label",
    image_column: str = "image",
    output_dir: str = "./cvagent_output",
    target_size: int = 1000,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
) -> dict[str, Any]:
    """Generate and execute a download+filter+split script for a dataset."""
    logger.info("generate_and_run_script  dataset_id=%r  source=%r  target=%d", dataset_id, source, target_size)

    total = train_ratio + val_ratio + test_ratio
    if total <= 0:
        train_ratio, val_ratio, test_ratio = 0.7, 0.2, 0.1
        total = 1.0
    split_ratios = (train_ratio / total, val_ratio / total, test_ratio / total)

    script_code = _build_download_script(
        dataset_id=dataset_id, source=source,
        label_column=label_column, image_column=image_column,
        output_dir=output_dir, target_size=target_size, split_ratios=split_ratios,
    )

    script_dir = Path(output_dir).parent
    script_dir.mkdir(parents=True, exist_ok=True)
    script_path = script_dir / f"cvagent_download_{dataset_id.replace('/', '_')}.py"
    script_path.write_text(script_code, encoding="utf-8")

    emit = _script_emit_callback

    async def _stream(line: str) -> None:
        if emit:
            try:
                await emit(line.rstrip(), "stdout")
            except Exception:
                pass

    await _stream(f"> Script written: {script_path}")
    await _stream(f"> Running: {sys.executable} {script_path}")
    await _stream("> " + "─" * 50)

    output_lines: list[str] = []
    exit_code = -1

    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, str(script_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env={**os.environ},
        )
        assert proc.stdout is not None
        async for raw_line in proc.stdout:
            line = raw_line.decode("utf-8", errors="replace").rstrip()
            output_lines.append(line)
            await _stream(line)
        await proc.wait()
        exit_code = proc.returncode or 0
    except Exception as exc:
        err = f"✖ Subprocess error: {exc}"
        await _stream(err)
        return {"status": "error", "error": err, "exit_code": -1, "output_tail": output_lines[-40:]}

    await _stream(f"> Exited with code {exit_code}")

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
        "Generate a Python script that downloads the chosen dataset from HuggingFace, "
        "applies blur filtering + perceptual deduplication, then splits into train/val/test. "
        "Call this ONLY after the user has confirmed which dataset to download via present_dataset_options."
    ),
    parameters={
        "type": "object",
        "properties": {
            "dataset_id": {"type": "string"},
            "source": {"type": "string", "default": "huggingface"},
            "label_column": {"type": "string", "default": "label"},
            "image_column": {"type": "string", "default": "image"},
            "output_dir": {"type": "string", "default": "./cvagent_output"},
            "target_size": {"type": "integer", "default": 1000},
            "train_ratio": {"type": "number", "default": 0.7},
            "val_ratio": {"type": "number", "default": 0.2},
            "test_ratio": {"type": "number", "default": 0.1},
        },
        "required": ["dataset_id"],
    },
    fn=generate_and_run_script,
)
