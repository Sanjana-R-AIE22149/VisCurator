from __future__ import annotations

import logging
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from backend.agent.tools import annotate_with_clip, download_and_clean, inspect_dataset, search_huggingface

logger = logging.getLogger(__name__)


class DatasetAgent:
    def __init__(self, websocket, job_id, nim_client):
        self.ws = websocket
        self.job_id = str(job_id)
        self.nim = nim_client
        self.output_base = Path("./cvagent_datasets")

    async def emit(self, type: str, message: str, data: dict = {}):
        try:
            payload = {
                "id": uuid4().hex,
                "type": type,
                "message": message,
                "data": data,
                "timestamp": datetime.utcnow().isoformat(),
            }
            await self.ws.send_json(payload)
        except Exception:
            pass

    async def run(self, query: str, source: str, target_size: int):
        output_dir = self.output_base / self.job_id
        output_dir.mkdir(parents=True, exist_ok=True)
        anno_result: dict[str, Any] = {}

        try:
            await self.emit("thought", f"Searching HuggingFace for: {query}")
            search_result = await search_huggingface(query, max_results=8)
            if search_result.get("status") == "error" or not search_result.get("datasets"):
                await self.emit("error", f"No datasets found for '{query}'. Try a different query.")
                return

            datasets = search_result["datasets"]
            await self.emit("tool_result", f"Found {len(datasets)} datasets", {"datasets": datasets})

            await self.emit("thought", "Analyzing results to find the best match...")
            best = await self._pick_best_dataset(query, datasets)
            await self.emit("tool_result", f"Selected dataset: {best['dataset_id']}", {"selected": best})

            await self.emit("thought", f"Inspecting {best['dataset_id']} to understand its structure...")
            info = await inspect_dataset(best["dataset_id"])
            if info.get("status") == "error":
                await self.emit("error", f"Could not inspect dataset: {info.get('error')}")
                return

            image_col = info.get("image_column", "image")
            label_col = info.get("label_column", "label")
            class_names = info.get("class_names") or []

            await self.emit(
                "log",
                f"Dataset structure: image_column='{image_col}', label_column='{label_col}', classes={len(class_names)}",
            )

            await self.emit("thought", f"Downloading up to {target_size} images...")

            async def stream_download(msg: str):
                await self.emit("log", msg)

            clean_result = await download_and_clean(
                dataset_id=best["dataset_id"],
                image_column=image_col,
                label_column=label_col,
                output_dir=str(output_dir / "images"),
                max_images=target_size,
                emit_callback=stream_download,
            )

            if clean_result.get("status") == "error":
                await self.emit("error", f"Download failed: {clean_result.get('error')}")
                return

            await self.emit("tool_result", f"Downloaded {clean_result['total_downloaded']} images", clean_result)

            if not class_names:
                class_names = list(clean_result.get("class_distribution", {}).keys())

            if class_names:
                await self.emit("thought", f"Running CLIP annotation on {clean_result['total_downloaded']} images...")

                async def stream_annotation(msg: str):
                    await self.emit("log", msg)

                anno_result = await annotate_with_clip(
                    image_dir=str(output_dir / "images"),
                    class_names=class_names,
                    output_dir=str(output_dir),
                    emit_callback=stream_annotation,
                )

                if anno_result.get("status") == "error":
                    await self.emit("error", f"Annotation failed: {anno_result.get('error')}")
                    return

                await self.emit(
                    "tool_result",
                    f"Annotation complete. Verification rate: {anno_result.get('verification_rate', 0):.1%}",
                    anno_result,
                )
            else:
                await self.emit("log", "No class names available — skipping CLIP annotation")

            await self.emit("thought", "Packaging dataset...")
            zip_path = await self._create_zip(output_dir, best["dataset_id"])

            await self.emit(
                "done",
                f"Dataset ready: {clean_result['total_downloaded']} images, annotated and packaged.",
                {
                    "download_url": f"/api/dataset/download/{self.job_id}",
                    "dataset_id": best["dataset_id"],
                    "total_images": clean_result["total_downloaded"],
                    "class_distribution": clean_result.get("class_distribution", {}),
                    "annotation_summary": anno_result.get("summary") if class_names else None,
                    "zip_path": str(zip_path),
                    "preprocessing_report": {
                        "dataset_id": best["dataset_id"],
                        "job_id": self.job_id,
                        "hf_pipeline": True,
                        "before_stats": {
                            "images": clean_result.get("total_downloaded", 0)
                                      + clean_result.get("rejected_blur", 0)
                                      + clean_result.get("rejected_dup", 0)
                                      + clean_result.get("rejected_small", 0),
                        },
                        "after_stats": {
                            "images": clean_result["total_downloaded"],
                            "blur_filtered": clean_result.get("rejected_blur", 0),
                            "duplicates_removed": clean_result.get("rejected_dup", 0),
                            "class_distribution": clean_result.get("class_distribution", {}),
                        },
                        "class_distribution": clean_result.get("class_distribution", {}),
                        "stage_samples": {
                            "raw": [],
                            "filtered": [
                                {"url": p.replace("\\", "/"), "label": "blurry", "reason": "blurry", "id": i}
                                for i, p in enumerate(clean_result.get("blurry_preview_paths", []))
                            ] + [
                                {"url": p.replace("\\", "/"), "label": "duplicate", "reason": "duplicate", "id": i + 100}
                                for i, p in enumerate(clean_result.get("dup_preview_paths", []))
                            ],
                            "processed": [
                                {"url": p.replace("\\\\", "/"), "label": "accepted", "id": i}
                                for i, p in enumerate(clean_result.get("processed_preview_paths", []))
                            ],
                        },
                        "deblur_preview": anno_result.get("deblur_preview", []),
                        "deblur_summary": {
                            "recovered_for_export": len(anno_result.get("deblur_preview", [])),
                            "avg_before": round(
                                sum(d["before_blur"] for d in anno_result.get("deblur_preview", [])) /
                                max(len(anno_result.get("deblur_preview", [])), 1), 1
                            ),
                            "avg_after": round(
                                sum(d["after_blur"] for d in anno_result.get("deblur_preview", [])) /
                                max(len(anno_result.get("deblur_preview", [])), 1), 1
                            ),
                        } if anno_result.get("deblur_preview") else None,
                        "annotation_summary": {
                            "available": bool(class_names and anno_result.get("status") == "success"),
                            "count": anno_result.get("verified_count", 0),
                            "generator": "CLIP + GrabCut BBox",
                        },
                        "formats": {
                            "yolo": "data.yaml + labels/train/*.txt + labels/val/*.txt",
                            "coco": "annotations/instances_train.json + instances_val.json",
                        },
                    },
                },
            )
        except Exception as e:
            import traceback

            logger.exception("Dataset pipeline failed")
            await self.emit("error", f"Pipeline failed: {str(e)}\n{traceback.format_exc()[:500]}")

    async def _pick_best_dataset(self, query: str, datasets: list[dict]) -> dict:
        try:
            if not self.nim:
                return datasets[0]
            dataset_list = "\n".join(
                [
                    f"{i + 1}. {d['dataset_id']} — {d.get('description', '')[:100]} (downloads: {d.get('downloads', 0)})"
                    for i, d in enumerate(datasets[:6])
                ]
            )
            response = await self.nim.chat(
                [
                    {
                        "role": "system",
                        "content": "You are a dataset selection expert. Reply with ONLY the dataset_id of the best match. Nothing else.",
                    },
                    {
                        "role": "user",
                        "content": f"Query: '{query}'\n\nDatasets:\n{dataset_list}\n\nBest dataset_id:",
                    },
                ]
            )
            chosen_id = response.choices[0].message.content.strip().strip('"').strip("'")
            match = next((d for d in datasets if d["dataset_id"] == chosen_id), None)
            return match or datasets[0]
        except Exception:
            return datasets[0]

    async def _create_zip(self, output_dir: Path, dataset_id: str) -> Path:
        try:
            safe_name = dataset_id.replace("/", "_")
            zip_path = output_dir.parent / f"{safe_name}_{self.job_id[:8]}.zip"
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for file in output_dir.rglob("*"):
                    if file.is_file():
                        zf.write(file, file.relative_to(output_dir))
            return zip_path
        except Exception as e:
            logger.exception("Zip creation failed")
            raise RuntimeError(f"Failed to create zip: {e}") from e
