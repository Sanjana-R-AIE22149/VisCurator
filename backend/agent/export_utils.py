"""
VisCurator — Dataset Export Utilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Converts the processed ImageFolder structure into COCO JSON,
YOLO classification, and YOLO detection (dummy bbox) formats.
"""
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path


def _iter_images(processed_dir: Path):
    """Yield (class_name, img_path) pairs from an ImageFolder directory."""
    for class_dir in sorted(processed_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        for img_path in sorted(class_dir.glob("*.jpg")):
            yield class_dir.name, img_path
        for img_path in sorted(class_dir.glob("*.png")):
            yield class_dir.name, img_path


def export_to_yolo_classification(processed_dir: Path, output_zip: Path):
    """
    YOLO Classification format:
      dataset/
        train/
          class_a/
            img1.jpg
          class_b/
            img2.jpg
        data.yaml

    Splits 80% train / 20% val automatically.
    """
    classes = sorted([d.name for d in processed_dir.iterdir() if d.is_dir()])

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # data.yaml
        yaml_lines = [
            f"path: ./dataset",
            f"train: train",
            f"val: val",
            f"nc: {len(classes)}",
            f"names: {classes}",
        ]
        (tmp_path / "data.yaml").write_text("\n".join(yaml_lines))

        all_items: list[tuple[str, Path]] = list(_iter_images(processed_dir))
        from collections import defaultdict
        by_class: dict[str, list[Path]] = defaultdict(list)
        for cls, p in all_items:
            by_class[cls].append(p)

        for cls, paths in by_class.items():
            split = max(1, int(len(paths) * 0.8))
            train_imgs, val_imgs = paths[:split], paths[split:]
            for split_name, imgs in (("train", train_imgs), ("val", val_imgs)):
                dest_dir = tmp_path / "dataset" / split_name / cls
                dest_dir.mkdir(parents=True, exist_ok=True)
                for img in imgs:
                    shutil.copy(img, dest_dir / img.name)

        shutil.make_archive(str(output_zip).replace(".zip", ""), "zip", tmp)


def export_to_coco_classification(processed_dir: Path, output_zip: Path):
    """
    COCO classification format:
      annotations.json   — COCO-style JSON with categories, images, annotations
      images/
        class_a/img1.jpg
        class_b/img2.jpg
    """
    try:
        from PIL import Image as PilImage
    except ImportError:
        PilImage = None  # type: ignore

    categories = []
    class_map: dict[str, int] = {}
    for i, cls in enumerate(sorted([d.name for d in processed_dir.iterdir() if d.is_dir()])):
        categories.append({"id": i, "name": cls, "supercategory": "none"})
        class_map[cls] = i

    images_meta = []
    annotations_meta = []
    img_id = 0

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        images_dir = tmp_path / "images"
        images_dir.mkdir()

        for cls, img_path in _iter_images(processed_dir):
            w, h = 224, 224
            if PilImage:
                try:
                    with PilImage.open(img_path) as im:
                        w, h = im.size
                except Exception:
                    pass

            rel_name = f"{cls}/{img_path.name}"
            dest = images_dir / cls
            dest.mkdir(exist_ok=True)
            shutil.copy(img_path, dest / img_path.name)

            images_meta.append({
                "id": img_id,
                "file_name": rel_name,
                "width": w,
                "height": h,
            })
            annotations_meta.append({
                "id": img_id,
                "image_id": img_id,
                "category_id": class_map[cls],
                "area": w * h,
                "bbox": [0, 0, w, h],
                "iscrowd": 0,
            })
            img_id += 1

        coco = {
            "info": {"description": "VisCurator curated dataset", "version": "1.0"},
            "categories": categories,
            "images": images_meta,
            "annotations": annotations_meta,
        }
        (tmp_path / "annotations.json").write_text(
            json.dumps(coco, indent=2), encoding="utf-8"
        )

        shutil.make_archive(str(output_zip).replace(".zip", ""), "zip", tmp)


def export_to_yolo_detection(processed_dir: Path, output_zip: Path):
    """
    YOLO detection format (classification → full-image bboxes).
    Kept for backwards compatibility. For classification use
    export_to_yolo_classification instead.

      dataset/
        images/
          class_a_img1.jpg
        labels/
          class_a_img1.txt   ← '<class_idx> 0.5 0.5 1.0 1.0'
        data.yaml
    """
    classes = sorted([d.name for d in processed_dir.iterdir() if d.is_dir()])

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "images").mkdir()
        (tmp_path / "labels").mkdir()

        with open(tmp_path / "data.yaml", "w") as f:
            f.write(f"names: {classes}\n")
            f.write(f"nc: {len(classes)}\n")
            f.write("train: ./images\nval: ./images\n")

        for class_idx, class_name in enumerate(classes):
            for img_path in (processed_dir / class_name).glob("*.jpg"):
                new_name = f"{class_name}_{img_path.name}"
                shutil.copy(img_path, tmp_path / "images" / new_name)
                label_file = tmp_path / "labels" / f"{Path(new_name).stem}.txt"
                label_file.write_text(f"{class_idx} 0.5 0.5 1.0 1.0\n")

        shutil.make_archive(str(output_zip).replace(".zip", ""), "zip", tmp)
