"""
VisCurator / CVAgent — Agent Tools
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Async tool functions for the dataset curation agent.
Supports HuggingFace, Kaggle, Roboflow, OpenImages, and Papers With Code.
"""

from __future__ import annotations

import asyncio
import ast
import csv
import io
import json
import logging
import os
import random
import shutil
import subprocess
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
_SEARCH_TIMEOUT = aiohttp.ClientTimeout(total=15.0)  # HF API needs time
_HEADERS = {"User-Agent": "VisCurator/1.0"}


def _register(name: str, description: str, parameters: dict[str, Any], fn: ToolFn) -> None:
    TOOL_REGISTRY[name] = {"name": name, "description": description, "parameters": parameters, "function": fn}


async def execute_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    entry = TOOL_REGISTRY.get(name)
    if entry is None:
        return {"status": "error", "error": f"Unknown tool '{name}'."}

    parameters = entry.get("parameters", {}).get("properties", {})
    normalized_args = dict(arguments)
    for arg_name, arg_value in list(normalized_args.items()):
        expected_type = parameters.get(arg_name, {}).get("type")
        if isinstance(arg_value, str):
            if expected_type in ("array", "object"):
                try:
                    parsed = json.loads(arg_value)
                    if expected_type == "array" and isinstance(parsed, list):
                        normalized_args[arg_name] = parsed
                    elif expected_type == "object" and isinstance(parsed, dict):
                        normalized_args[arg_name] = parsed
                except json.JSONDecodeError:
                    pass
            elif expected_type == "integer":
                try:
                    normalized_args[arg_name] = int(arg_value)
                except ValueError:
                    pass
            elif expected_type == "number":
                try:
                    normalized_args[arg_name] = float(arg_value)
                except ValueError:
                    pass
    
    # Use longer timeout for heavy operations (clean/augment), 
    # but tools.py functions should handle their own internal timeouts where specific.
    tout = 300 if name == "clean_and_augment_dataset" else 120
    try:
        result = await asyncio.wait_for(entry["function"](**normalized_args), timeout=tout)
        return result
    except asyncio.TimeoutError:
        return {"status": "error", "error": f"Tool '{name}' timed out after {tout}s."}
    except Exception as exc:
        logger.exception("Tool '%s' failed", name)
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


async def _run_python_script_streaming(
    script_path: Path,
    output_lines: list[str],
    on_line: Callable[[str], Coroutine[Any, Any, None]],
) -> int:
    """Run a Python script with streaming output using threads for Windows safety."""
    queue: asyncio.Queue[str | tuple[str, int | None]] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def _worker() -> None:
        exit_code = -1
        try:
            proc = subprocess.Popen(
                [sys.executable, str(script_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env={**os.environ},
            )
            if proc.stdout is not None:
                for raw_line in proc.stdout:
                    line = raw_line.decode("utf-8", errors="replace").rstrip()
                    loop.call_soon_threadsafe(queue.put_nowait, line)
            exit_code = proc.wait()
        except Exception as exc:
            loop.call_soon_threadsafe(queue.put_nowait, f"ERR: Subprocess failed: {exc}")
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, ("DONE", exit_code))

    import threading
    threading.Thread(target=_worker, daemon=True).start()

    while True:
        item = await queue.get()
        if isinstance(item, tuple) and item[0] == "DONE":
            return int(item[1] or 0)
        line = str(item)
        output_lines.append(line)
        await on_line(line)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TOOL 1 — search_datasets  (multi-source)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _expand_query(query: str) -> list[str]:
    """
    Convert a user's natural language query into multiple targeted search 
    terms that HuggingFace's API will actually return results for.
    
    HuggingFace search is keyword-based and case-insensitive, but:
    - It matches against dataset ID, tags, and description
    - Spaces work better than hyphens for multi-word searches  
    - Short (1-2 word) queries return more results than long phrases
    - Domain-specific synonyms matter a lot
    
    Returns a deduplicated list of queries, most specific first.
    """
    q = query.lower().strip()
    words = q.split()
    expansions: list[str] = []

    # ── Domain synonym table ──────────────────────────────────
    # Maps common user phrases → HuggingFace-friendly search terms
    DOMAIN_MAP: dict[str, list[str]] = {
        # Plant / agriculture
        "apple leaf":         ["plant-disease", "apple disease", "leaf disease"],
        "leaf disease":       ["plant-disease", "plant disease", "leaf"],
        "plant disease":      ["plant-disease", "plant disease", "agriculture"],
        "crop disease":       ["plant-disease", "agriculture", "crop"],
        "apple scab":         ["plant-disease", "apple", "scab disease"],
        "powdery mildew":     ["plant-disease", "mildew", "plant"],
        "tomato disease":     ["plant-disease", "tomato", "tomato leaf"],
        "rice disease":       ["plant-disease", "rice", "paddy"],
        "wheat disease":      ["plant-disease", "wheat", "cereal"],
        "corn disease":       ["plant-disease", "corn", "maize"],
        "cassava":            ["plant-disease", "cassava", "agriculture"],

        # Medical imaging
        "chest xray":         ["chest-xray", "chest xray", "pneumonia", "medical"],
        "xray":               ["xray", "chest-xray", "medical imaging"],
        "mri":                ["mri", "brain mri", "medical imaging"],
        "skin lesion":        ["skin-lesion", "skin disease", "dermatology", "isic"],
        "skin cancer":        ["skin-lesion", "melanoma", "dermoscopy"],
        "eye disease":        ["retinal", "fundus", "diabetic retinopathy", "eye"],
        "retinal":            ["retinal", "fundus", "diabetic retinopathy"],
        "cancer":             ["cancer", "tumor", "pathology", "histology"],
        "brain tumor":        ["brain tumor", "brain mri", "mri segmentation"],
        "pneumonia":          ["chest-xray", "pneumonia", "chest xray"],
        "covid":              ["covid", "chest-xray", "covid-19"],

        # Object detection / general CV
        "object detection":   ["detection", "coco", "object-detection"],
        "face detection":     ["face", "face detection", "facial"],
        "face recognition":   ["face", "facial recognition", "lfw"],
        "pedestrian":         ["pedestrian", "person detection", "crowd"],
        "vehicle":            ["vehicle", "car detection", "traffic"],
        "license plate":      ["license plate", "ocr", "vehicle"],
        "traffic sign":       ["traffic", "sign detection", "road"],
        "drone":              ["aerial", "drone", "uav", "satellite"],
        "satellite":          ["satellite", "aerial", "remote sensing"],
        "underwater":         ["underwater", "marine", "aquatic"],
        "wildfire":           ["fire", "wildfire", "smoke detection"],
        "garbage":            ["waste", "garbage", "trash", "recycling"],
        "crack":              ["crack detection", "defect", "surface inspection"],
        "weld":               ["welding", "defect", "industrial"],
        "pcb":                ["pcb", "circuit board", "defect detection"],

        # Animals
        "cat dog":            ["cats-vs-dogs", "pet", "animal"],
        "cats dogs":          ["cats-vs-dogs", "cat dog", "pet"],
        "bird":               ["bird", "cub-200", "ornithology"],
        "flower":             ["flower", "flowers102", "plant"],
        "fruit":              ["fruit", "food", "grocery"],
        "food":               ["food", "food101", "recipe"],
        "dog breed":          ["stanford-dogs", "dog breed", "dog"],
        "fish":               ["fish", "aquatic", "marine"],

        # Text / document
        "handwriting":        ["handwriting", "mnist", "handwritten"],
        "digit":              ["mnist", "digit recognition", "handwritten"],
        "mnist":              ["mnist", "digit", "handwritten"],
        "document":           ["document", "ocr", "text detection"],
        "invoice":            ["invoice", "document", "ocr"],

        # Scene / classification
        "scene":              ["scene", "places", "landscape"],
        "indoor":             ["indoor", "scene", "room"],
        "outdoor":            ["outdoor", "scene", "landscape"],
        "weather":            ["weather", "fog", "rain", "snow"],
        "nighttime":          ["nighttime", "night", "low light"],

        # Specific well-known datasets (user might describe without knowing the name)
        "imagenet":           ["imagenet", "image classification"],
        "cifar":              ["cifar", "image classification"],
        "fashion":            ["fashion-mnist", "clothing", "fashion"],
        "emotion":            ["emotion", "facial expression", "fer"],
        "gesture":            ["gesture", "hand gesture", "sign language"],
        "pose":               ["pose", "human pose", "skeleton"],
        "depth":              ["depth estimation", "depth", "stereo"],
        "segmentation":       ["segmentation", "semantic segmentation", "mask"],
    }

    # 1. Check domain map for any matching phrase
    for phrase, synonyms in DOMAIN_MAP.items():
        if phrase in q:
            expansions.extend(synonyms)
            break  # one match is enough for domain routing

    # 2. Always add the original query (lowercased, cleaned)
    expansions.append(q)

    # 3. Add hyphenated version (HF IDs often use hyphens)
    hyphenated = "-".join(words)
    if hyphenated != q:
        expansions.append(hyphenated)

    # 4. Add first two words (most specific without being too long)
    if len(words) >= 2:
        expansions.append(" ".join(words[:2]))

    # 5. Add each individual content word (skip stopwords)
    STOPWORDS = {"for", "in", "the", "a", "an", "of", "with", "and", "or",
                 "to", "on", "at", "from", "by", "is", "are", "image",
                 "images", "dataset", "data", "classification", "detection",
                 "recognition", "model", "training", "deep", "learning"}
    content_words = [w for w in words if w not in STOPWORDS and len(w) > 2]
    expansions.extend(content_words)

    # 6. Deduplicate preserving order
    seen: set[str] = set()
    result: list[str] = []
    for e in expansions:
        e = e.strip()
        if e and e not in seen:
            seen.add(e)
            result.append(e)

    logger.info("Query expansion: %r → %r", query, result)
    return result


async def _search_huggingface(query: str, max_results: int) -> list[dict[str, Any]]:
    """
    Search HuggingFace with intelligent query expansion.
    
    Runs up to 4 expanded queries in parallel (not sequentially),
    deduplicates by dataset_id, and scores results by relevance
    to the original query before returning.
    """
    headers = {**_HEADERS}
    hf_token = os.getenv("HF_TOKEN", "").strip()
    if hf_token and hf_token not in ("", "optional_huggingface_token"):
        headers["Authorization"] = f"Bearer {hf_token}"

    expanded = _expand_query(query)
    # Take at most 4 queries to keep latency under 15 seconds
    queries_to_run = expanded[:4]
    logger.info("HF parallel search: %r", queries_to_run)

    # Tags that indicate an image dataset
    IMAGE_TAGS = {
        "image-classification", "object-detection", "image-segmentation",
        "image-to-image", "visual-question-answering", "image-feature-extraction",
        "image", "images", "computer-vision", "cv", "task_categories:image-classification",
        "task_categories:object-detection", "task_categories:image-segmentation",
        "modality:image", "modalities:image",
    }
    # Tags that definitively mark a NON-image dataset
    NON_IMAGE_TAGS = {
        "text-classification", "text-generation", "question-answering",
        "summarization", "translation", "audio", "speech", "audio-classification",
        "automatic-speech-recognition", "text", "tabular", "token-classification",
        "modality:text", "modality:audio", "modality:tabular",
    }

    async def _single_search(session: aiohttp.ClientSession, q: str) -> list[dict]:
        results = []
        for sort in ("downloads", "likes"):
            params = {"search": q, "limit": max_results * 3, "sort": sort, "direction": "-1"}
            try:
                async with session.get(
                    "https://huggingface.co/api/datasets",
                    params=params
                ) as resp:
                    if resp.status != 200:
                        continue
                    data = await resp.json()
                for ds in data:
                    ds_id = ds.get("id", "")
                    if not ds_id:
                        continue
                    tags_raw: list[str] = ds.get("tags", []) or []
                    tags_lower = {t.lower() for t in tags_raw}

                    # Drop datasets that are explicitly non-image
                    if tags_lower & NON_IMAGE_TAGS:
                        continue

                    # Accept if any image tag present, OR if no modality tags at all
                    # (many small CV datasets have no tags — don't exclude them)
                    has_image_tag = bool(tags_lower & IMAGE_TAGS)
                    has_any_modality = bool(tags_lower & (IMAGE_TAGS | NON_IMAGE_TAGS))
                    if has_any_modality and not has_image_tag:
                        continue  # has modality tags but none are image → skip

                    results.append({
                        "source": "HuggingFace",
                        "dataset_id": ds_id,
                        "name": ds_id.split("/")[-1],
                        "description": (ds.get("description") or "")[:200],
                        "downloads": ds.get("downloads", 0),
                        "likes": ds.get("likes", 0),
                        "tags": tags_raw,
                        "has_image_tag": has_image_tag,
                        "url": f"https://huggingface.co/datasets/{ds_id}",
                        "size_estimate": "metadata-only",
                        "_search_query": q,
                    })
                    if len(results) >= max_results:
                        break
                if results:
                    break
            except asyncio.TimeoutError:
                logger.warning("HF timeout for q=%r sort=%r", q, sort)
            except Exception as e:
                logger.warning("HF error for q=%r: %s", q, e)
        return results

    async with aiohttp.ClientSession(timeout=_SEARCH_TIMEOUT, headers=headers) as session:
        all_batches = await asyncio.gather(
            *[_single_search(session, q) for q in queries_to_run],
            return_exceptions=True,
        )

    # Merge and deduplicate
    seen_ids: set[str] = set()
    merged: list[dict] = []
    for batch in all_batches:
        if isinstance(batch, Exception):
            continue
        for ds in batch:
            if ds["dataset_id"] not in seen_ids:
                seen_ids.add(ds["dataset_id"])
                merged.append(ds)

    # Score by relevance to ORIGINAL query. For crop-specific searches
    # ("apple leaf disease"), keep popularity as a small tiebreaker so
    # unrelated but popular datasets (cassava, tomato, etc.) do not win.
    q_lower = query.lower()
    q_words = {w for w in q_lower.split() if len(w) > 2}
    crop_terms = {
        "apple", "cassava", "mango", "grape", "tomato", "rice", "wheat",
        "corn", "maize", "potato", "pepper", "banana", "citrus", "soybean",
    }
    requested_crops = crop_terms & q_words
    conflicting_crops = crop_terms - requested_crops

    def _relevance(ds: dict) -> float:
        score = 0.0
        ds_id_lower = ds["dataset_id"].lower()
        desc_lower = (ds.get("description") or "").lower()
        tags_lower = " ".join(ds.get("tags", [])).lower()
        combined = ds_id_lower + " " + desc_lower + " " + tags_lower

        # Bonus for explicitly image-tagged datasets
        if ds.get("has_image_tag"):
            score += 4.0

        # Word overlap with original query
        for w in q_words:
            if w in combined:
                score += 3.0

        for crop in requested_crops:
            if crop in combined:
                score += 12.0

        if requested_crops and not any(crop in combined for crop in requested_crops):
            score -= 12.0

        for crop in conflicting_crops:
            if crop in combined:
                score -= 8.0

        if "plantvillage" in combined:
            score += 5.0

        # Exact phrase match in ID is gold
        if q_lower.replace(" ", "-") in ds_id_lower:
            score += 10.0
        if q_lower.replace(" ", "") in ds_id_lower.replace("-", "").replace("_", ""):
            score += 8.0
        # Popularity tiebreaker (small weight)
        score += min(ds.get("downloads", 0) / 2_000_000, 1.0)
        score += min(ds.get("likes", 0) / 2_000, 0.5)
        ds["_relevance_score"] = round(score, 3)
        return score

    merged.sort(key=_relevance, reverse=True)
    
    # Remove internal tracking fields before returning
    for ds in merged:
        ds.pop("_search_query", None)
        ds.pop("tags", None)  # tags can be large, not needed downstream
        ds.pop("has_image_tag", None)

    logger.info("HF adaptive search for %r: %d unique results", query, len(merged))
    return merged[:max_results]


async def _search_kaggle(query: str, max_results: int) -> list[dict[str, Any]]:
    username = os.getenv("KAGGLE_USERNAME", "").strip()
    key = os.getenv("KAGGLE_KEY", "").strip()
    if not username or not key:
        return []

    url = "https://www.kaggle.com/api/v1/datasets/list"
    params = {"search": query, "sortBy": "votes", "pageSize": min(max_results, 20)}
    try:
        async with aiohttp.ClientSession(timeout=_SEARCH_TIMEOUT, auth=aiohttp.BasicAuth(username, key)) as session:
            async with session.get(url, params=params) as resp:
                if resp.status != 200: return []
                data = await resp.json()
        results = []
        for ds in data:
            owner = ds.get("ownerUser", "")
            slug = ds.get("datasetSlug", "")
            results.append({
                "source": "Kaggle",
                "dataset_id": f"{owner}/{slug}",
                "name": ds.get("title", slug),
                "description": (ds.get("description") or "")[:200],
                "downloads": ds.get("downloadCount", 0),
                "likes": ds.get("voteCount", 0),
                "url": f"https://www.kaggle.com/datasets/{owner}/{slug}",
                "size_estimate": "metadata-only",
            })
        return results
    except Exception:
        return []


async def search_datasets(
    query: str,
    sources: Any = None,
    max_results_per_source: Any = 5,
) -> dict[str, Any]:
    """Search for datasets across multiple sources simultaneously with strict timeouts."""
    start_time = asyncio.get_event_loop().time()
    
    # Robust type casting for LLM-provided arguments
    try:
        max_results = int(max_results_per_source)
    except (ValueError, TypeError):
        max_results = 8
    max_results = max(max_results, 8)  # always fetch at least 8

    if isinstance(sources, str):
        try:
            # Handle if LLM sends "['hf', 'kaggle']" or "huggingface"
            import ast
            parsed = ast.literal_eval(sources)
            sources = parsed if isinstance(parsed, list) else [sources]
        except Exception:
            sources = [sources]
    
    if not isinstance(sources, list) or not sources:
        sources = ["huggingface"]
    
    # Always include huggingface — it has the most datasets
    sources_lower = [s.lower() for s in sources]
    if "huggingface" not in sources_lower:
        sources_lower = ["huggingface"] + sources_lower
    sources = sources_lower

    logger.info("Search started: query=%r sources=%r max_results=%d", query, sources, max_results)

    tasks = []
    source_names = []
    if "huggingface" in sources:
        tasks.append(_search_huggingface(query, max_results))
        source_names.append("huggingface")
    if "kaggle" in sources:
        tasks.append(_search_kaggle(query, max_results))
        source_names.append("kaggle")

    results_by_source = await asyncio.gather(*tasks, return_exceptions=True)
    
    all_results = []
    source_status = {}
    
    for src, res in zip(source_names, results_by_source):
        if isinstance(res, Exception):
            logger.error("%s search failed: %s", src, res)
            source_status[src] = "failed"
        elif isinstance(res, list):
            all_results.extend(res)
            source_status[src] = f"ok ({len(res)})"
        else:
            source_status[src] = "empty"

    all_results.sort(
        key=lambda x: (
            x.get("_relevance_score", 0),
            x.get("downloads", 0) + x.get("likes", 0) * 5,
        ),
        reverse=True,
    )
    elapsed = asyncio.get_event_loop().time() - start_time
    
    logger.info("Search finished in %.2fs: %d results found. Status: %s", 
                elapsed, len(all_results), source_status)

    return {
        "status": "success",
        "query": query,
        "total_found": len(all_results),
        "elapsed_time": round(elapsed, 3),
        "source_status": source_status,
        "datasets": all_results[:15],
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
    logger.info("[STEP] get_dataset_info  dataset_id=%r  source=%r", dataset_id, source)

    if source == "huggingface":
        url = f"https://huggingface.co/api/datasets/{dataset_id}"
        headers: dict[str, str] = {}
        hf_token = os.getenv("HF_TOKEN")
        if hf_token and hf_token != "optional_huggingface_token":
            headers["Authorization"] = f"Bearer {hf_token}"
        headers.update(_HEADERS)

        logger.info("[INFO] Fetching metadata from HuggingFace API...")
        async with aiohttp.ClientSession(timeout=_HTTP_TIMEOUT) as session:
            async with session.get(url, headers=headers) as resp:
                if resp.status == 404:
                    logger.warning("[FAIL] Dataset '%s' not found.", dataset_id)
                    return {"status": "error", "error": f"Dataset '{dataset_id}' not found."}
                if resp.status != 200:
                    logger.warning("[FAIL] HF metadata API returned HTTP %d", resp.status)
                    return {"status": "error", "error": f"HTTP {resp.status}"}
                meta = await resp.json()

            logger.info("[INFO] Fetching split/row information...")
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
                    else:
                        logger.warning("[INFO] datasets-server returned %d, proceeding with partial meta.", resp2.status)
            except Exception as exc:
                logger.warning("[INFO] datasets-server info unavailable for '%s': %s", dataset_id, exc)

        features = splits_info.pop("_features", [])
        tags = meta.get("tags") or []
        card = meta.get("cardData") or {}
        total_bytes = sum(s.get("num_bytes", 0) for s in splits_info.values())
        total_rows = sum(s.get("num_rows", 0) for s in splits_info.values())
        logger.info("[OK] Dataset info retrieved. Found %d features and %d total rows.", len(features), total_rows)

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

async def estimate_dataset_quality(dataset_id: str, sample_size: Any = 30) -> dict[str, Any]:
    """Download a small sample from a HuggingFace dataset and estimate image quality."""
    try:
        sample_size = int(sample_size)
    except (ValueError, TypeError):
        sample_size = 30

    logger.info("[STEP] Starting quality estimation for %r (sample_size=%d)", dataset_id, sample_size)

    rows_url = (
        f"https://datasets-server.huggingface.co/rows"
        f"?dataset={dataset_id}&config=default&split=train"
        f"&offset=0&length={min(sample_size, 100)}"
    )
    headers: dict[str, str] = {}
    hf_token = os.getenv("HF_TOKEN")
    if hf_token and hf_token != "optional_huggingface_token":
        headers["Authorization"] = f"Bearer {hf_token}"
    headers.update(_HEADERS)

    payload = None
    logger.info("[INFO] Fetching rows from datasets-server...")
    async with aiohttp.ClientSession(timeout=_HTTP_TIMEOUT) as session:
        async with session.get(rows_url, headers=headers) as resp:
            if resp.status == 200:
                payload = await resp.json()
                logger.info("[INFO] Succeeded with 'default' config.")
            else:
                logger.info("[INFO] 'default' config failed (HTTP %d), searching for other configs...", resp.status)
                # Try to find the right config
                info_url = f"https://datasets-server.huggingface.co/info?dataset={dataset_id}"
                async with session.get(info_url, headers=headers) as info_resp:
                    if info_resp.status != 200:
                        logger.warning("[FAIL] Cannot access dataset '%s' via datasets-server.", dataset_id)
                        return {"status": "error", "error": f"Cannot access dataset '{dataset_id}' via datasets-server."}
                    info_data = await info_resp.json()
                    ds_info = info_data.get("dataset_info") or {}
                    if not ds_info:
                        logger.warning("[FAIL] No configs found for '%s'.", dataset_id)
                        return {"status": "error", "error": "No configs found."}
                    first_config = next(iter(ds_info))
                    splits = ds_info[first_config].get("splits") or {}
                    first_split = next(iter(splits), "train")
                    rows_url2 = (
                        f"https://datasets-server.huggingface.co/rows"
                        f"?dataset={dataset_id}&config={first_config}&split={first_split}"
                        f"&offset=0&length={min(sample_size, 100)}"
                    )
                    logger.info("[INFO] Retrying with config=%r split=%r", first_config, first_split)
                    async with session.get(rows_url2, headers=headers) as resp2:
                        if resp2.status != 200:
                            logger.warning("[FAIL] Rows endpoint failed with config=%r", first_config)
                            return {"status": "error", "error": f"Rows endpoint failed: HTTP {resp2.status}."}
                        payload = await resp2.json()

    rows = (payload or {}).get("rows") or []
    if not rows:
        logger.warning("[FAIL] No rows returned for '%s'.", dataset_id)
        return {"status": "error", "error": "No rows returned."}

    logger.info("[INFO] Successfully retrieved %d rows. Analyzing image quality...", len(rows))

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
    blur_score: float | str,
    duplicate_percentage: float | str,
    class_balance: str,
    dataset_size: int | str,
    resolution_stats: dict[str, Any] | str,
) -> dict[str, Any]:
    """Deterministically convert quality stats into a preprocessing plan."""
    # Defensive type coercion for LLM-generated args that may be strings
    try:
        blur_score = float(blur_score)
    except (ValueError, TypeError):
        blur_score = 50.0
    
    try:
        duplicate_percentage = float(duplicate_percentage)
    except (ValueError, TypeError):
        duplicate_percentage = 5.0
    
    try:
        dataset_size = int(dataset_size)
    except (ValueError, TypeError):
        dataset_size = 100
    
    # Parse resolution_stats if it's a string (from LLM JSON)
    if isinstance(resolution_stats, str):
        try:
            resolution_stats = json.loads(resolution_stats)
        except (json.JSONDecodeError, ValueError):
            resolution_stats = {}
    
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

        DATASET_ID = {dataset_id!r}
        SOURCE = {source!r}
        LABEL_COL = {label_column!r}
        IMAGE_COL = {image_column!r}
        OUTPUT_DIR = Path({output_dir!r})
        TARGET_SIZE = {target_size}
        PLAN = {plan!r}
        RANDOM_SEED = 42
        BLUR_THRESHOLD_LARGE = 80.0   # for images >= 128px
        BLUR_THRESHOLD_SMALL = 8.0 

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

        def resolve_label(row, default_col, dataset_dict):
            col = default_col
            if col not in row:
                for fallback in ["label", "labels", "category", "class", "target"]:
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

        result_holder = {{}}
        error_holder = {{}}

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
                        "\\nThis HuggingFace dataset appears to require a deprecated custom "
                        "loading script. Choose a dataset published in standard imagefolder, "
                        "Parquet, WebDataset, or COCO files."
                    )
                error_holder["err"] = msg

        print(f"> Loading {{DATASET_ID}} from {{SOURCE}} ...")
        t = threading.Thread(target=_load)
        t.start()
        t.join(timeout=300)  # 5 minute timeout
        if t.is_alive():
            print("ERR: Dataset download timed out after 5 minutes", file=sys.stderr)
            sys.exit(1)
        if "err" in error_holder:
            print(f"ERR: {{error_holder['err']}}", file=sys.stderr)
            sys.exit(1)
        ds = result_holder["ds"]

        if not isinstance(ds, dict):
            ds = {{"train": ds}}

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
        
        sample_maps = {{"raw": [], "filtered": [], "processed": []}}

        def row_generator():
            for split_name, split_ds in ds.items():
                print("> split {{}}: {{}} rows".format(split_name, len(split_ds)), flush=True)
                for r in split_ds:
                    yield r

        for idx, row in enumerate(row_generator()):
            if len(accepted_records) >= TARGET_SIZE:
                break
            img = ensure_rgb(row.get(IMAGE_COL))
            if img is None:
                continue
            
            label = resolve_label(row, LABEL_COL, ds)
            before_counts[label] += 1
            blur_value = laplacian_var(img)
            img_hash = str(imagehash.dhash(img))
            _thresh = BLUR_THRESHOLD_SMALL if (img.width < 128 or img.height < 128) else BLUR_THRESHOLD_LARGE
            is_blurry = blur_value < _thresh
            is_dup = img_hash in seen_hashes
            
            # Save Raw Samples
            if len(sample_maps["raw"]) < 8:
                fname = f"raw_{{idx:05d}}.jpg"
                img.save(output_samples / "raw" / fname, "JPEG", quality=85)
                sinfo = {{"url": f"samples/raw/{{fname}}", "label": label, "id": idx}}
                sample_maps["raw"].append(sinfo)
                emit_event("sample", {{"stage": "raw", "sample": sinfo}})

            blur_scatter.append({{
                "id": idx,
                "laplacian": round(blur_value, 1),
                "resolution": int((img.width * img.height) / 1000),
                "accepted": not is_blurry and not is_dup,
            }})
            
            if (PLAN.get("needs_blur_filtering") and is_blurry) or (PLAN.get("needs_deduplication") and is_dup):
                if PLAN.get("needs_blur_filtering") and is_blurry: rejected_blur += 1
                if PLAN.get("needs_deduplication") and is_dup: rejected_dup += 1
                
                # Save Filtered Samples
                if len(sample_maps["filtered"]) < 8:
                    reason = "blurry" if is_blurry else "duplicate"
                    fname = f"filtered_{{idx:05d}}.jpg"
                    img.save(output_samples / "filtered" / fname, "JPEG", quality=85)
                    sinfo = {{"url": f"samples/filtered/{{fname}}", "label": label, "reason": reason, "id": idx}}
                    sample_maps["filtered"].append(sinfo)
                    emit_event("sample", {{"stage": "filtered", "sample": sinfo}})
                continue
                
            seen_hashes.add(img_hash)
            accepted_records.append((img, label, idx))

        augmenter = build_augmenter()
        minority_target = max([count for count in before_counts.values()] or [0])
        minority_labels = {{label for label, count in before_counts.items() if count < minority_target}}


        export_total = 0
        for sample_index, (img, label, original_idx) in enumerate(accepted_records):
            out_dir = output_processed / label
            out_dir.mkdir(parents=True, exist_ok=True)
            base_np = np.array(img)
            res = augmenter(image=base_np)
            resized = res["image"]
            
            save_name = f"sample_{{sample_index:05d}}.jpg"
            Image.fromarray(resized).save(out_dir / save_name, "JPEG", quality=92)
            
            # Save Processed Samples
            if len(sample_maps["processed"]) < 8:
                fname = f"proc_{{sample_index:05d}}.jpg"
                Image.fromarray(resized).save(output_samples / "processed" / fname, "JPEG", quality=85)
                sinfo = {{"url": f"samples/processed/{{fname}}", "label": label, "id": original_idx}}
                sample_maps["processed"].append(sinfo)
                emit_event("sample", {{"stage": "processed", "sample": sinfo}})
            
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
            "stage_samples": sample_maps,
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
    target_size: int | str = 1000,
    processing_plan: dict[str, Any] | str | None = None,
) -> dict[str, Any]:
    """Run deterministic filtering, deduplication, normalization, and augmentation."""
    # ALWAYS force output to the canonical location regardless of what the LLM passed.
    # This prevents files being scattered across curated_dataset/, output/, etc.
    output_dir = "./cvagent_output"

    # Defensive type coercion for LLM-generated args that may be strings
    try:
        target_size = int(target_size)
    except (ValueError, TypeError):
        target_size = 1000

    
    # Parse processing_plan if it's a string (from LLM JSON)
    if isinstance(processing_plan, str):
        try:
            processing_plan = json.loads(processing_plan)
        except (json.JSONDecodeError, ValueError):
            try:
                parsed_plan = ast.literal_eval(processing_plan)
                processing_plan = parsed_plan if isinstance(parsed_plan, dict) else None
            except (ValueError, SyntaxError):
                processing_plan = None

    if isinstance(processing_plan, dict):
        if "needs_blur_filtering" not in processing_plan and "blur_filtering" in processing_plan:
            processing_plan["needs_blur_filtering"] = bool(processing_plan.get("blur_filtering"))
        if "needs_deduplication" not in processing_plan and "deduplication" in processing_plan:
            processing_plan["needs_deduplication"] = bool(processing_plan.get("deduplication"))
        if "needs_augmentation" not in processing_plan and "augmentation" in processing_plan:
            processing_plan["needs_augmentation"] = bool(processing_plan.get("augmentation"))
        if "recommended_augmentations" not in processing_plan and "augmentations" in processing_plan:
            processing_plan["recommended_augmentations"] = processing_plan.get("augmentations") or []
    
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
    script_dir = Path(output_dir)
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
        async def _handle_line(line: str) -> None:
            if line.startswith("JSON_EVENT:"):
                payload = json.loads(line[len("JSON_EVENT:"):])
                if payload.get("kind") == "report":
                    nonlocal report_payload
                    report_payload = payload
                    return
            await _stream(line)

        exit_code = await _run_python_script_streaming(script_path, output_lines, _handle_line)
    except Exception as exc:
        import traceback
        err = f"Preprocessing subprocess failed: {exc}\n{traceback.format_exc()}"
        logger.error(err)
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

        result_holder = {{}}
        error_holder = {{}}

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
                        "\\nThis HuggingFace dataset appears to require a deprecated custom "
                        "loading script. Choose a dataset published in standard imagefolder, "
                        "Parquet, WebDataset, or COCO files."
                    )
                error_holder["err"] = msg

        print(f"> Loading {{DATASET_ID}} from {{SOURCE}} ...")
        t = threading.Thread(target=_load)
        t.start()
        t.join(timeout=300)  # 5 minute timeout
        if t.is_alive():
            print("ERR: Dataset download timed out after 5 minutes", file=sys.stderr)
            sys.exit(1)
        if "err" in error_holder:
            print(f"ERR: {{error_holder['err']}}", file=sys.stderr)
            sys.exit(1)
        ds = result_holder["ds"]

        if not isinstance(ds, dict):
            ds = {{"train": ds}}

        print(f"> Splits: {{list(ds.keys())}}")

        print("> Filtering (blur + dedup) ...")
        accepted, rejected_blur, rejected_dup = [], 0, 0
        seen_hashes = set()

        def row_generator():
            for split_name, split_ds in ds.items():
                print(f">   {{split_name}}: {{len(split_ds)}} rows")
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
                print(f">   Scanned {{i+1}} | accepted {{len(accepted)}} | blur {{rejected_blur}} | dup {{rejected_dup}}")

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
    target_size: int | str = 1000,
    train_ratio: float | str = 0.7,
    val_ratio: float | str = 0.2,
    test_ratio: float | str = 0.1,
) -> dict[str, Any]:
    """Generate and execute a download+filter+split script for a dataset."""
    # Defensive type coercion for LLM-generated args that may be strings
    try:
        target_size = int(target_size)
    except (ValueError, TypeError):
        target_size = 1000
    
    try:
        train_ratio = float(train_ratio)
    except (ValueError, TypeError):
        train_ratio = 0.7
    
    try:
        val_ratio = float(val_ratio)
    except (ValueError, TypeError):
        val_ratio = 0.2
    
    try:
        test_ratio = float(test_ratio)
    except (ValueError, TypeError):
        test_ratio = 0.1
    
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

    script_dir = Path(output_dir)
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
        async def _handle_line(line: str) -> None:
            await _stream(line)

        exit_code = await _run_python_script_streaming(script_path, output_lines, _handle_line)
    except Exception as exc:
        import traceback
        err = f"✖ Subprocess error: {exc}\n{traceback.format_exc()}"
        logger.error(err)
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
