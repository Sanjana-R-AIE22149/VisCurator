import sys
import re

with open('backend/agent/tools.py', 'r', encoding='utf-8') as f:
    content = f.read()

match = re.search(r'async def annotate_with_clip\(.*', content, re.DOTALL)
if not match:
    print("Could not find annotate_with_clip")
    sys.exit(1)

prefix = content[:match.start()]

new_func = '''async def annotate_with_clip(
    image_dir: str,
    class_names: list[str],
    output_dir: str,
    emit_callback: Any = None,
) -> dict[str, Any]:
    try:
        import cv2

        def _as_feature_tensor(output: Any) -> torch.Tensor:
            if isinstance(output, torch.Tensor):
                return output
            if hasattr(output, "text_embeds") and output.text_embeds is not None:
                return output.text_embeds
            if hasattr(output, "image_embeds") and output.image_embeds is not None:
                return output.image_embeds
            if hasattr(output, "pooler_output") and output.pooler_output is not None:
                return output.pooler_output
            if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
                hidden = output.last_hidden_state
                return hidden[:, 0, :] if hidden.ndim == 3 else hidden
            raise TypeError(f"Unsupported CLIP feature output type: {type(output).__name__}")

        def fallback_bbox(img: Image.Image) -> tuple[int, int, int, int]:
            arr = np.array(img.convert("RGB"))
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            rect = (max(1, img.width // 20), max(1, img.height // 20), max(2, img.width - img.width // 10), max(2, img.height - img.height // 10))
            try:
                mask = np.zeros(gray.shape[:2], np.uint8)
                bgd = np.zeros((1, 65), np.float64)
                fgd = np.zeros((1, 65), np.float64)
                cv2.grabCut(arr, mask, rect, bgd, fgd, 3, cv2.GC_INIT_WITH_RECT)
                fg = np.where((mask == 1) | (mask == 3), 255, 0).astype("uint8")
            except Exception:
                _, fg = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return (0, 0, img.width, img.height)
            largest = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest)
            if w <= 1 or h <= 1:
                return (0, 0, img.width, img.height)
            return int(x), int(y), int(w), int(h)

        def bbox_to_yolo(x: int, y: int, w: int, h: int, img_w: int, img_h: int) -> tuple[float, float, float, float]:
            cx = (x + w / 2) / img_w
            cy = (y + h / 2) / img_h
            nw = w / img_w
            nh = h / img_h
            return round(cx, 6), round(cy, 6), round(nw, 6), round(nh, 6)

        image_root = Path(image_dir)
        out_root = Path(output_dir)
        out_root.mkdir(parents=True, exist_ok=True)
        labels_root = out_root / "labels"
        labels_root.mkdir(parents=True, exist_ok=True)
        anno_dir = out_root / "annotations"
        anno_dir.mkdir(parents=True, exist_ok=True)

        image_paths = sorted([p for p in image_root.rglob("*") if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
        if not image_paths:
            return {"status": "error", "error": "No images found for annotation."}

        await _safe_emit(emit_callback, "> Downloading CLIP model (first run only, ~350MB)...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        model.eval()

        prompts = [f"a photo of a {name}" for name in class_names]
        with torch.no_grad():
            text_inputs = processor(text=prompts, return_tensors="pt", padding=True)
            text_inputs = {key: value.to(device) for key, value in text_inputs.items()}
            text_features = _as_feature_tensor(model.get_text_features(**text_inputs))
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        annotations: list[dict[str, Any]] = []
        coco_records: list[dict[str, Any]] = []
        class_confidence: dict[str, list[float]] = defaultdict(list)
        verified_count = 0
        low_confidence_count = 0

        for batch_start in range(0, len(image_paths), 16):
            batch_paths = image_paths[batch_start:batch_start + 16]
            images = [Image.open(path).convert("RGB") for path in batch_paths]
            with torch.no_grad():
                image_inputs = processor(images=images, return_tensors="pt")
                image_inputs = {key: value.to(device) for key, value in image_inputs.items()}
                image_features = _as_feature_tensor(model.get_image_features(**image_inputs))
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                logits = image_features @ text_features.T
                probs = logits.softmax(dim=-1).cpu().numpy()

            for local_index, image_path in enumerate(batch_paths):
                rel_path = image_path.relative_to(image_root)
                split = rel_path.parts[0] if len(rel_path.parts) > 1 else "train"
                original_label = image_path.parent.name
                prob_vector = probs[local_index]
                predicted_index = int(prob_vector.argmax())
                predicted_class = class_names[predicted_index]
                confidence = float(prob_vector[predicted_index])
                verified = original_label == predicted_class
                if verified:
                    verified_count += 1
                if confidence < 0.6:
                    low_confidence_count += 1
                class_confidence[predicted_class].append(confidence)

                # Bounding box
                img_obj = images[local_index]
                img_w, img_h = img_obj.size
                x_abs, y_abs, w_abs, h_abs = fallback_bbox(img_obj)
                cx, cy, nw, nh = bbox_to_yolo(x_abs, y_abs, w_abs, h_abs, img_w, img_h)

                labels_dir = labels_root / split
                labels_dir.mkdir(parents=True, exist_ok=True)
                (labels_dir / f"{image_path.stem}.txt").write_text(f"{predicted_index} {cx} {cy} {nw} {nh}\\n", encoding="utf-8")

                coco_records.append({
                    "file_name": f"images/{rel_path}".replace("\\\\", "/"),
                    "split": split,
                    "width": img_w,
                    "height": img_h,
                    "class_id": predicted_index,
                    "bbox_abs": [x_abs, y_abs, w_abs, h_abs]
                })

                annotations.append(
                    {
                        "image_path": str(rel_path).replace("\\\\", "/"),
                        "original_label": original_label,
                        "clip_predicted_class": predicted_class,
                        "clip_confidence": confidence,
                        "verified": verified,
                    }
                )

            processed = batch_start + len(batch_paths)
            if processed % 50 == 0 or processed == len(image_paths):
                await _safe_emit(emit_callback, f"> Annotated {processed}/{len(image_paths)} images with CLIP + Bboxes...")

        # Write YOLO data.yaml
        yaml_path = out_root / "data.yaml"
        yaml_path.write_text(
            f"path: .\\n"
            f"train: images/train\\n"
            f"val: images/val\\n"
            f"test: images/test\\n"
            f"nc: {len(class_names)}\\n"
            f"names: {json.dumps(class_names)}\\n",
            encoding="utf-8",
        )

        # Write COCO JSONs
        coco_categories = [{"id": i, "name": c, "supercategory": "object"} for i, c in enumerate(class_names)]
        for split_name in ("train", "val", "test"):
            split_records = [r for r in coco_records if r["split"] == split_name]
            if not split_records:
                continue
            coco_images, coco_annos = [], []
            for img_id, rec in enumerate(split_records):
                x, y, w, h = rec["bbox_abs"]
                coco_images.append({
                    "id": img_id,
                    "file_name": rec["file_name"],
                    "width": rec["width"],
                    "height": rec["height"],
                })
                coco_annos.append({
                    "id": img_id,
                    "image_id": img_id,
                    "category_id": rec["class_id"],
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                })
            
            coco_out = {
                "info": {"description": "VisCurator CLIP Dataset", "version": "1.0"},
                "categories": coco_categories,
                "images": coco_images,
                "annotations": coco_annos,
            }
            (anno_dir / f"instances_{split_name}.json").write_text(
                json.dumps(coco_out, indent=2), encoding="utf-8"
            )

        class_breakdown = {
            class_name: {
                "count": len(scores),
                "avg_confidence": (sum(scores) / len(scores)) if scores else 0.0,
            }
            for class_name, scores in class_confidence.items()
        }
        summary = {
            "verified_count": verified_count,
            "verification_rate": verified_count / len(annotations),
            "low_confidence_count": low_confidence_count,
            "class_breakdown": class_breakdown,
        }
        payload = {
            "metadata": {
                "tool": "CLIP zero-shot + OpenCV GrabCut",
                "model": "openai/clip-vit-base-patch32",
                "class_names": class_names,
                "total_images": len(annotations),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            "annotations": annotations,
            "summary": summary,
        }
        (anno_dir / "annotations.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return {"status": "success", **summary, "summary": summary, "output_dir": str(out_root)}
    except Exception as e:
        logger.exception("annotate_with_clip failed")
        return {"status": "error", "error": str(e)}
'''

with open('backend/agent/tools.py', 'w', encoding='utf-8') as f:
    f.write(prefix + new_func)

print("Updated tools.py")
