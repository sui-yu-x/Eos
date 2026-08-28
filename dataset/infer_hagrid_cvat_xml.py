#!/usr/bin/env python3
"""Run HaGRIDv2 YOLOv10n on one 8_1 dataset and emit CVAT 1.1 bbox XML."""

from __future__ import annotations

import argparse
import math
import os
import re
import shutil
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


import cv2
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm


DATASET_CHOICES = ("dragon", "peak", "soar")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_ROOT = PROJECT_ROOT / "external"
INPUT_BASE = EXTERNAL_ROOT / "data/images"
OUTPUT_BASE = EXTERNAL_ROOT / "data/annotations"
DEFAULT_WEIGHTS = EXTERNAL_ROOT / "models/hagrid/YOLOv10n_hands.pt"
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
PORTRAIT_WIDTH = 720
PORTRAIT_HEIGHT = 1280
MAX_DETECTIONS = 2
NEW_XML_NAME = "annotations_new.xml"


@dataclass(frozen=True)
class Detection:
    box: tuple[float, float, float, float]


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    name: str
    width: int
    height: int
    detections: tuple[Detection, ...]


@dataclass(frozen=True)
class Scene:
    name: str
    path: Path
    images: tuple[Path, ...]


def natural_key(value: str) -> tuple:
    return tuple(
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", value)
    )


def is_hidden_relative(path: Path, root: Path) -> bool:
    return any(part.startswith(".") for part in path.relative_to(root).parts)


def discover_scenes(input_root: Path) -> list[Scene]:
    scene_dirs = sorted(
        (
            path
            for path in input_root.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        ),
        key=lambda path: natural_key(path.name),
    )
    if not scene_dirs:
        raise ValueError(f"No non-hidden scene directories found in: {input_root}")

    scenes: list[Scene] = []
    empty_scenes: list[str] = []
    for scene_dir in scene_dirs:
        images = sorted(
            (
                path
                for path in scene_dir.rglob("*")
                if path.is_file()
                and path.suffix.lower() in VALID_EXTS
                and not is_hidden_relative(path, scene_dir)
            ),
            key=lambda path: natural_key(path.relative_to(scene_dir).as_posix()),
        )
        if not images:
            empty_scenes.append(scene_dir.name)
            continue
        scenes.append(Scene(scene_dir.name, scene_dir, tuple(images)))

    if empty_scenes:
        preview = ", ".join(empty_scenes[:10])
        suffix = "" if len(empty_scenes) <= 10 else f" ... (+{len(empty_scenes) - 10})"
        raise ValueError(f"Scene directories contain no supported images: {preview}{suffix}")
    if not scenes:
        raise ValueError(f"No supported images found in: {input_root}")
    return scenes


def ensure_unique_basenames(scenes: Sequence[Scene]) -> None:
    exact = Counter(path.name for scene in scenes for path in scene.images)
    duplicates = sorted(name for name, count in exact.items() if count > 1)
    folded = Counter(path.name.casefold() for scene in scenes for path in scene.images)
    duplicate_folded = {name.casefold() for name in duplicates}
    case_collisions = sorted(
        name
        for name, count in folded.items()
        if count > 1 and name not in duplicate_folded
    )
    if duplicates or case_collisions:
        details: list[str] = []
        if duplicates:
            details.append("duplicate names: " + ", ".join(duplicates[:10]))
        if case_collisions:
            details.append(
                "case-insensitive collisions: " + ", ".join(case_collisions[:10])
            )
        raise ValueError(
            "Image basenames must be unique for basename-only CVAT XML; "
            + "; ".join(details)
        )


def to_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    if np.issubdtype(image.dtype, np.integer):
        maximum = int(image.max()) if image.size else 0
        if maximum <= 255:
            return image.astype(np.uint8)
        scale = 255.0 / float(np.iinfo(image.dtype).max)
        return np.clip(image.astype(np.float32) * scale, 0, 255).astype(np.uint8)
    array = np.nan_to_num(image.astype(np.float32), nan=0.0, posinf=255.0, neginf=0.0)
    maximum = float(array.max()) if array.size else 0.0
    if maximum <= 1.0:
        array *= 255.0
    elif maximum > 255.0:
        array *= 255.0 / maximum
    return np.clip(array, 0, 255).astype(np.uint8)


def load_image(path: Path) -> np.ndarray:
    try:
        with Image.open(path) as pil_image:
            pil_image = ImageOps.exif_transpose(pil_image)
            image = np.array(pil_image)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Failed to read image: {path}") from exc

    if image.ndim < 2:
        raise RuntimeError(f"Unsupported image shape for {path}: {image.shape}")
    height, width = image.shape[:2]
    if width == PORTRAIT_WIDTH and height == PORTRAIT_HEIGHT:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    image = to_uint8(image)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 1:
        return cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    raise RuntimeError(f"Unsupported image shape for {path}: {image.shape}")


def infer_scene(
    model: YOLO,
    scene: Scene,
    confidence: float,
    batch_size: int,
) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    progress = tqdm(total=len(scene.images), desc=scene.name, ncols=110)
    try:
        for start in range(0, len(scene.images), batch_size):
            batch_paths = scene.images[start : start + batch_size]
            batch_images = [load_image(path) for path in batch_paths]
            try:
                results = model.predict(
                    source=batch_images,
                    conf=confidence,
                    max_det=MAX_DETECTIONS,
                    agnostic_nms=True,
                    verbose=False,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Inference failed in scene {scene.name}, batch starting at {start}: {exc}"
                ) from exc
            if len(results) != len(batch_paths):
                raise RuntimeError(
                    f"Inference result count mismatch in {scene.name}: "
                    f"{len(results)} != {len(batch_paths)}"
                )

            for path, image, result in zip(batch_paths, batch_images, results):
                height, width = image.shape[:2]
                boxes = result.boxes.xyxy.detach().cpu().numpy()
                scores = result.boxes.conf.detach().cpu().numpy()
                order = np.argsort(scores)[::-1][:MAX_DETECTIONS]
                detections: list[Detection] = []
                for index in order:
                    values = tuple(float(value) for value in boxes[index])
                    if len(values) != 4 or not all(math.isfinite(value) for value in values):
                        raise RuntimeError(f"Invalid bbox returned for {path}: {values}")
                    x1, y1, x2, y2 = values
                    if x2 < x1 or y2 < y1:
                        raise RuntimeError(f"Inverted bbox returned for {path}: {values}")
                    detections.append(Detection(values))
                records.append(
                    ImageRecord(
                        path=path,
                        name=path.name,
                        width=int(width),
                        height=int(height),
                        detections=tuple(detections),
                    )
                )
            progress.update(len(batch_paths))
    finally:
        progress.close()
    return records


def add_text(parent: ET.Element, tag: str, text: object) -> ET.Element:
    element = ET.SubElement(parent, tag)
    element.text = str(text)
    return element


def build_labels(parent: ET.Element) -> None:
    labels = ET.SubElement(parent, "labels")
    bbox = ET.SubElement(labels, "label")
    add_text(bbox, "name", "bbox")
    add_text(bbox, "color", "#66ff66")
    add_text(bbox, "type", "rectangle")
    ET.SubElement(bbox, "attributes")


def fmt_coord(value: float) -> str:
    return f"{value:.2f}"


def clip_coord(value: float, limit: int) -> float:
    return min(max(value, 0.0), float(limit))


def build_cvat_tree(task_name: str, records: Sequence[ImageRecord]) -> ET.ElementTree:
    now = datetime.now(timezone.utc).isoformat()
    root = ET.Element("annotations")
    add_text(root, "version", "1.1")
    meta = ET.SubElement(root, "meta")
    task = ET.SubElement(meta, "task")
    add_text(task, "id", 0)
    add_text(task, "name", task_name)
    add_text(task, "size", len(records))
    add_text(task, "mode", "annotation")
    add_text(task, "overlap", 0)
    ET.SubElement(task, "bugtracker")
    add_text(task, "created", now)
    add_text(task, "updated", now)
    add_text(task, "subset", "default")
    add_text(task, "start_frame", 0)
    add_text(task, "stop_frame", len(records) - 1)
    ET.SubElement(task, "frame_filter")
    segments = ET.SubElement(task, "segments")
    segment = ET.SubElement(segments, "segment")
    add_text(segment, "id", 0)
    add_text(segment, "start", 0)
    add_text(segment, "stop", len(records) - 1)
    ET.SubElement(segment, "url")
    ET.SubElement(task, "owner")
    ET.SubElement(task, "assignee")
    build_labels(task)

    for image_id, record in enumerate(records):
        image = ET.SubElement(
            root,
            "image",
            {
                "id": str(image_id),
                "name": record.name,
                "width": str(record.width),
                "height": str(record.height),
            },
        )
        for detection in record.detections:
            x1, y1, x2, y2 = detection.box
            ET.SubElement(
                image,
                "box",
                {
                    "label": "bbox",
                    "source": "auto",
                    "occluded": "0",
                    "xtl": fmt_coord(clip_coord(x1, record.width)),
                    "ytl": fmt_coord(clip_coord(y1, record.height)),
                    "xbr": fmt_coord(clip_coord(x2, record.width)),
                    "ybr": fmt_coord(clip_coord(y2, record.height)),
                    "z_order": "0",
                },
            )
    return ET.ElementTree(root)


def indent_xml(element: ET.Element, level: int = 0) -> None:
    indentation = "\n" + level * "  "
    if len(element):
        if not element.text or not element.text.strip():
            element.text = indentation + "  "
        for child in element:
            indent_xml(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = indentation
    if level and (not element.tail or not element.tail.strip()):
        element.tail = indentation


def write_xml(tree: ET.ElementTree, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    indent_xml(tree.getroot())
    tree.write(path, encoding="utf-8", xml_declaration=True)


def validate_xml(path: Path, expected: Sequence[ImageRecord]) -> None:
    root = ET.parse(path).getroot()
    if root.tag != "annotations" or root.findtext("version") != "1.1":
        raise ValueError(f"Invalid CVAT version/root in {path}")
    labels = [element.text for element in root.findall("meta/task/labels/label/name")]
    if labels != ["bbox"]:
        raise ValueError(f"Unexpected label schema in {path}: {labels}")
    if root.findtext("meta/task/size") != str(len(expected)):
        raise ValueError(f"Incorrect task size in {path}")
    if root.findall(".//skeleton") or root.findall(".//points"):
        raise ValueError(f"Non-bbox annotations found in {path}")

    images = root.findall("image")
    if len(images) != len(expected):
        raise ValueError(f"Incorrect image count in {path}: {len(images)} != {len(expected)}")
    for index, (image, record) in enumerate(zip(images, expected)):
        if image.get("id") != str(index) or image.get("name") != record.name:
            raise ValueError(f"Image identity mismatch at index {index} in {path}")
        if image.get("width") != str(record.width) or image.get("height") != str(record.height):
            raise ValueError(f"Image size mismatch for {record.name} in {path}")
        boxes = image.findall("box")
        if len(boxes) != len(record.detections) or len(boxes) > MAX_DETECTIONS:
            raise ValueError(f"Detection count mismatch for {record.name} in {path}")
        for box in boxes:
            if box.get("label") != "bbox":
                raise ValueError(f"Unexpected box label in {record.name}")
            coords = [float(box.get(name, "nan")) for name in ("xtl", "ytl", "xbr", "ybr")]
            if not all(math.isfinite(value) for value in coords):
                raise ValueError(f"Non-finite bbox in {record.name}")
            if not (
                0 <= coords[0] <= coords[2] <= record.width
                and 0 <= coords[1] <= coords[3] <= record.height
            ):
                raise ValueError(f"Out-of-range bbox in {record.name}: {coords}")


def load_existing_records(path: Path) -> dict[str, ImageRecord]:
    if not path.is_file():
        return {}

    root = ET.parse(path).getroot()
    if root.tag != "annotations" or root.findtext("version") != "1.1":
        raise ValueError(f"Existing output is not CVAT 1.1 XML: {path}")
    labels = [element.text for element in root.findall("meta/task/labels/label/name")]
    if labels != ["bbox"]:
        raise ValueError(f"Unexpected label schema in existing output {path}: {labels}")

    records: dict[str, ImageRecord] = {}
    for image in root.findall("image"):
        name = image.get("name", "")
        if not name:
            raise ValueError(f"Existing output contains an image without a name: {path}")
        if name in records:
            raise ValueError(f"Existing output contains duplicate image name {name}: {path}")
        width = int(image.get("width", "0"))
        height = int(image.get("height", "0"))
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid existing image dimensions for {name}: {width}x{height}")

        detections: list[Detection] = []
        boxes = image.findall("box")
        if len(boxes) > MAX_DETECTIONS:
            raise ValueError(f"Too many existing detections for {name}: {len(boxes)}")
        for box in boxes:
            if box.get("label") != "bbox":
                raise ValueError(f"Unexpected existing box label for {name}: {box.get('label')}")
            values = tuple(
                float(box.get(attribute, "nan"))
                for attribute in ("xtl", "ytl", "xbr", "ybr")
            )
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"Non-finite existing bbox for {name}: {values}")
            x1, y1, x2, y2 = values
            if not (0.0 <= x1 <= x2 <= width and 0.0 <= y1 <= y2 <= height):
                raise ValueError(f"Out-of-range existing bbox for {name}: {values}")
            detections.append(Detection(values))
        records[name] = ImageRecord(
            path=Path(name),
            name=name,
            width=width,
            height=height,
            detections=tuple(detections),
        )
    if root.findtext("meta/task/size") != str(len(records)):
        raise ValueError(f"Existing CVAT task size mismatch in {path}")
    return records


def stage_outputs(
    output_root: Path,
    dataset_name: str,
    scene_records: Sequence[tuple[Scene, list[ImageRecord], list[ImageRecord]]],
) -> Path:
    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.staging-", dir=output_root.parent)
    )
    try:
        all_records: list[ImageRecord] = []
        all_new_records: list[ImageRecord] = []
        for scene, records, new_records in scene_records:
            path = staging / scene.name / "annotations.xml"
            write_xml(build_cvat_tree(scene.name, records), path)
            validate_xml(path, records)
            all_records.extend(records)

            new_path = staging / scene.name / NEW_XML_NAME
            write_xml(build_cvat_tree(f"{scene.name}-new", new_records), new_path)
            validate_xml(new_path, new_records)
            all_new_records.extend(new_records)

        total_path = staging / "annotations.xml"
        write_xml(build_cvat_tree(dataset_name, all_records), total_path)
        validate_xml(total_path, all_records)

        new_total_path = staging / NEW_XML_NAME
        write_xml(build_cvat_tree(f"{dataset_name}-new", all_new_records), new_total_path)
        validate_xml(new_total_path, all_new_records)

        names = [record.name for record in all_records]
        if len(names) != len(set(names)):
            raise ValueError("Combined XML contains duplicate image names")
        return staging
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def publish_staging(staging: Path, output_root: Path) -> None:
    if output_root.is_symlink():
        raise ValueError(f"Refusing to write through symlink output root: {output_root}")
    if output_root.exists():
        if not output_root.is_dir():
            raise ValueError(f"Output root exists and is not a directory: {output_root}")
    else:
        output_root.mkdir(parents=True)

    for source in sorted(path for path in staging.rglob("*") if path.is_file()):
        relative = source.relative_to(staging)
        destination = output_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(source, destination)
    shutil.rmtree(staging)


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return number


def probability(value: str) -> float:
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return number


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run HaGRIDv2 YOLOv10n on dragon, peak, or soar and create "
            "per-scene plus combined CVAT 1.1 bbox XML."
        )
    )
    parser.add_argument("--dataset", required=True, choices=DATASET_CHOICES)
    parser.add_argument(
        "--input-root",
        type=Path,
        help="Dataset image directory; defaults to external/data/images/<dataset>",
    )
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--confidence", type=probability, default=0.30)
    parser.add_argument("--batch-size", type=positive_int, default=16)
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Output directory; defaults to external/data/annotations/<dataset>",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_root = (
        args.input_root.expanduser().resolve()
        if args.input_root is not None
        else (INPUT_BASE / args.dataset).resolve()
    )
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else (OUTPUT_BASE / args.dataset).resolve()
    )
    weights = args.weights.expanduser().resolve()

    if not input_root.is_dir():
        raise FileNotFoundError(f"Input root not found: {input_root}")
    if not weights.is_file():
        raise FileNotFoundError(f"Weights not found: {weights}")
    if output_root == input_root or input_root in output_root.parents:
        raise ValueError(f"Output root must not be the input root or inside it: {output_root}")
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as error:
        raise RuntimeError("Ultralytics is not installed in the current Python environment") from error

    scenes = discover_scenes(input_root)
    ensure_unique_basenames(scenes)
    image_count = sum(len(scene.images) for scene in scenes)
    print(f"Dataset: {args.dataset}")
    print(f"Input: {input_root}")
    print(f"Output: {output_root}")
    print(f"Weights: {weights}")
    print(f"Scenes: {len(scenes)}, images: {image_count}")
    for scene in scenes:
        print(f"  {scene.name}: {len(scene.images)} images")

    existing_records = load_existing_records(output_root / "annotations.xml")
    current_paths = {
        path.name: path
        for scene in scenes
        for path in scene.images
    }
    removed_names = sorted(set(existing_records) - set(current_paths), key=natural_key)
    if removed_names:
        preview = ", ".join(removed_names[:10])
        suffix = "" if len(removed_names) <= 10 else f" ... (+{len(removed_names) - 10})"
        raise ValueError(
            "Existing annotations reference images no longer present on disk; "
            f"refusing to silently drop them: {preview}{suffix}"
        )

    new_names = set(current_paths) - set(existing_records)
    print(f"Existing annotated images: {len(existing_records)}")
    print(f"New images to infer: {len(new_names)}")

    model = YOLO(str(weights)) if new_names else None
    scene_records: list[tuple[Scene, list[ImageRecord], list[ImageRecord]]] = []
    total_detections = 0
    images_with_detections = 0
    for scene in scenes:
        pending_scene = Scene(
            scene.name,
            scene.path,
            tuple(path for path in scene.images if path.name in new_names),
        )
        new_records = (
            infer_scene(model, pending_scene, args.confidence, args.batch_size)
            if pending_scene.images
            else []
        )
        new_by_name = {record.name: record for record in new_records}
        records: list[ImageRecord] = []
        for path in scene.images:
            if path.name in new_by_name:
                records.append(new_by_name[path.name])
            else:
                existing = existing_records[path.name]
                records.append(
                    ImageRecord(
                        path=path,
                        name=existing.name,
                        width=existing.width,
                        height=existing.height,
                        detections=existing.detections,
                    )
                )
        scene_records.append((scene, records, new_records))
        scene_detections = sum(len(record.detections) for record in new_records)
        scene_detected_images = sum(bool(record.detections) for record in new_records)
        total_detections += scene_detections
        images_with_detections += scene_detected_images
        print(
            f"Scene done: {scene.name}, total_images={len(records)}, "
            f"new_images={len(new_records)}, new_images_with_detections={scene_detected_images}, "
            f"new_detections={scene_detections}"
        )

    staging = stage_outputs(output_root, args.dataset, scene_records)
    try:
        publish_staging(staging, output_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    print(
        f"Done. scenes={len(scenes)}, total_images={image_count}, "
        f"new_images={len(new_names)}, new_images_with_detections={images_with_detections}, "
        f"new_detections={total_detections}"
    )
    print(f"Combined XML: {output_root / 'annotations.xml'}")
    print(f"New-only XML: {output_root / NEW_XML_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
