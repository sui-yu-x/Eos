#!/usr/bin/env python3
"""Run RTMPose Hand5 on checked CVAT bboxes using an expanded square ROI."""

from __future__ import annotations

import argparse
import copy
import math
import os
import re
import tempfile
import time
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from PIL import Image, ImageOps


DATASET_CHOICES = ("dragon", "peak", "soar")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_ROOT = PROJECT_ROOT / "external"
IMAGE_BASE = EXTERNAL_ROOT / "data/images"
ANNOTATION_BASE = EXTERNAL_ROOT / "data/annotations"
CHECKED_XML_NAME = "annotations校验后.xml"
OUTPUT_XML_NAME = "annotations_square_roi_rtmpose.xml"
DEFAULT_CONFIG = EXTERNAL_ROOT / (
    "frameworks/mmpose/configs/hand_2d_keypoint/rtmpose/hand5/"
    "rtmpose-m_8xb256-210e_hand5-256x256.py"
)
DEFAULT_CHECKPOINT = EXTERNAL_ROOT / "models/rtmpose/rtmpose-m_simcc-hand5.pth"
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
OUTPUT_LABELS = ("bbox", "keypoints", "1", "2")
KEYPOINT_SVG = """<line x1="30.232179641723633" y1="30.68145179748535" x2="48.93446731567383" y2="44.42190933227539" data-type="edge" data-node-from="1" data-node-to="2"></line>
<circle r="0.75" cx="30.232179641723633" cy="30.68145179748535" data-type="element node" data-element-id="1" data-node-id="1" data-label-name="1"></circle>
<circle r="0.75" cx="48.93446731567383" cy="44.42190933227539" data-type="element node" data-element-id="2" data-node-id="2" data-label-name="2"></circle>"""
PORTRAIT_SIZE = (720, 1280)
LANDSCAPE_SIZE = (1280, 720)
INPUT_SIZE = 256
ROI_EXPANSION = 1.2


@dataclass(frozen=True)
class ExpectedImage:
    image_id: str
    name: str
    width: int
    height: int
    boxes: tuple[tuple[float, float, float, float], ...]


@dataclass
class SourceData:
    tree: ET.ElementTree
    images: list[ET.Element]
    expected: list[ExpectedImage]
    image_paths: dict[str, Path]
    source_non_bbox_annotations_removed: int


def natural_key(value: str) -> tuple:
    return tuple(
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", value)
    )


def is_hidden_relative(path: Path, root: Path) -> bool:
    return any(part.startswith(".") for part in path.relative_to(root).parts)


def discover_images(image_root: Path) -> dict[str, Path]:
    paths = sorted(
        (
            path
            for path in image_root.rglob("*")
            if path.is_file()
            and path.suffix.lower() in VALID_EXTENSIONS
            and not is_hidden_relative(path, image_root)
        ),
        key=lambda path: natural_key(path.relative_to(image_root).as_posix()),
    )
    if not paths:
        raise ValueError(f"No supported images found in: {image_root}")

    counts = Counter(path.name for path in paths)
    duplicates = sorted(name for name, count in counts.items() if count > 1)
    folded = Counter(path.name.casefold() for path in paths)
    case_collisions = sorted(name for name, count in folded.items() if count > 1)
    if duplicates or case_collisions:
        raise ValueError(
            "Image basenames must be unique for CVAT basename mapping; "
            f"duplicates={duplicates[:10]}, case_collisions={case_collisions[:10]}"
        )
    return {path.name: path for path in paths}


def validate_input_label_schema(root: ET.Element) -> None:
    label_nodes = root.findall("./meta/task/labels/label")
    bbox_nodes = [node for node in label_nodes if node.findtext("name") == "bbox"]
    if len(bbox_nodes) != 1:
        names = tuple(node.findtext("name") for node in label_nodes)
        raise ValueError(f"Expected exactly one CVAT bbox label, got {names}")
    if bbox_nodes[0].findtext("type") != "rectangle":
        raise ValueError(
            f"Expected bbox label type rectangle, got {bbox_nodes[0].findtext('type')}"
        )


def normalize_output_label_schema(root: ET.Element) -> None:
    labels = root.find("./meta/task/labels")
    if labels is None:
        raise ValueError("Missing CVAT task labels metadata")
    bbox = next(
        node for node in labels.findall("label") if node.findtext("name") == "bbox"
    )
    bbox = copy.deepcopy(bbox)
    for node in list(labels):
        labels.remove(node)
    labels.append(bbox)

    skeleton = ET.SubElement(labels, "label")
    ET.SubElement(skeleton, "name").text = "keypoints"
    ET.SubElement(skeleton, "color").text = "#fa3253"
    ET.SubElement(skeleton, "type").text = "skeleton"
    ET.SubElement(skeleton, "attributes")
    ET.SubElement(skeleton, "svg").text = KEYPOINT_SVG
    for name in ("1", "2"):
        point = ET.SubElement(labels, "label")
        ET.SubElement(point, "name").text = name
        ET.SubElement(point, "color").text = "#fa3253"
        ET.SubElement(point, "type").text = "points"
        ET.SubElement(point, "attributes")
        ET.SubElement(point, "parent").text = "keypoints"


def validate_output_label_schema(root: ET.Element) -> None:
    label_nodes = root.findall("./meta/task/labels/label")
    names = tuple(node.findtext("name") for node in label_nodes)
    if names != OUTPUT_LABELS:
        raise ValueError(f"Expected output CVAT labels {OUTPUT_LABELS}, got {names}")
    types = tuple(node.findtext("type") for node in label_nodes)
    if types != ("rectangle", "skeleton", "points", "points"):
        raise ValueError(f"Unexpected CVAT label types: {types}")
    parents = tuple(node.findtext("parent") for node in label_nodes[2:])
    if parents != ("keypoints", "keypoints"):
        raise ValueError(f"Point labels must be children of keypoints, got {parents}")


def parse_box(box: ET.Element, image_name: str, width: int, height: int) -> tuple[float, float, float, float]:
    if box.get("label") != "bbox":
        raise ValueError(f"Unexpected box label in {image_name}: {box.get('label')}")
    values = tuple(float(box.get(key, "nan")) for key in ("xtl", "ytl", "xbr", "ybr"))
    x1, y1, x2, y2 = values
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"Non-finite bbox in {image_name}: {values}")
    if not (0.0 <= x1 < x2 <= width and 0.0 <= y1 < y2 <= height):
        raise ValueError(f"Invalid or out-of-range bbox in {image_name}: {values}")
    return values


def load_source(source_xml: Path, image_root: Path) -> SourceData:
    if not source_xml.is_file():
        raise FileNotFoundError(f"Checked annotation XML not found: {source_xml}")
    if not image_root.is_dir():
        raise FileNotFoundError(f"Image dataset directory not found: {image_root}")

    image_paths = discover_images(image_root)
    tree = ET.parse(source_xml)
    root = tree.getroot()
    if root.tag != "annotations" or root.findtext("version") != "1.1":
        raise ValueError(f"Input is not CVAT 1.1 XML: {source_xml}")
    validate_input_label_schema(root)
    normalize_output_label_schema(root)

    images = root.findall("image")
    if not images:
        raise ValueError(f"No image nodes in: {source_xml}")
    task_size = root.findtext("./meta/task/size")
    if task_size != str(len(images)):
        raise ValueError(f"CVAT task size mismatch: meta={task_size}, images={len(images)}")

    xml_names = [image.get("name", "") for image in images]
    if any(not name for name in xml_names):
        raise ValueError("Every CVAT image must have a non-empty name")
    if len(xml_names) != len(set(xml_names)):
        raise ValueError("CVAT XML contains duplicate image names")
    disk_names = set(image_paths)
    xml_name_set = set(xml_names)
    missing = sorted(xml_name_set - disk_names)
    extra = sorted(disk_names - xml_name_set)
    if missing:
        raise ValueError(
            "XML/disk image-set mismatch: "
            f"missing_on_disk={missing[:10]} ({len(missing)}), "
            f"extra_on_disk={extra[:10]} ({len(extra)})"
        )
    if extra:
        print(
            f"Ignoring {len(extra)} disk images not referenced by the input XML; "
            "inference follows the XML image set only."
        )

    expected: list[ExpectedImage] = []
    non_bbox_annotations_removed = 0
    total_boxes = 0
    for image in images:
        name = image.attrib["name"]
        width = int(image.attrib["width"])
        height = int(image.attrib["height"])
        if (width, height) != LANDSCAPE_SIZE:
            raise ValueError(f"Expected XML dimensions 1280x720 for {name}, got {width}x{height}")

        ignored = [
            child
            for child in list(image)
            if not (child.tag == "box" and child.get("label") == "bbox")
        ]
        for child in ignored:
            image.remove(child)
        non_bbox_annotations_removed += len(ignored)

        boxes = image.findall("box")
        coordinates = tuple(parse_box(box, name, width, height) for box in boxes)
        total_boxes += len(boxes)
        for group_id, box in enumerate(boxes, start=1):
            box.set("group_id", str(group_id))
        expected.append(
            ExpectedImage(
                image_id=image.attrib["id"],
                name=name,
                width=width,
                height=height,
                boxes=coordinates,
            )
        )
    if total_boxes == 0:
        raise ValueError(f"No bbox annotations found in: {source_xml}")
    return SourceData(tree, images, expected, image_paths, non_bbox_annotations_removed)


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
            image = np.asarray(pil_image)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Failed to read image: {path}") from exc

    if image.ndim < 2:
        raise RuntimeError(f"Unsupported image shape for {path}: {image.shape}")
    height, width = image.shape[:2]
    if (width, height) == PORTRAIT_SIZE:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif (width, height) != LANDSCAPE_SIZE:
        raise RuntimeError(f"Unexpected image dimensions for {path}: {width}x{height}")

    image = to_uint8(image)
    if image.ndim == 2:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 1:
        bgr = cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 3:
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        bgr = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    else:
        raise RuntimeError(f"Unsupported image shape for {path}: {image.shape}")
    if (bgr.shape[1], bgr.shape[0]) != LANDSCAPE_SIZE:
        raise RuntimeError(f"Orientation failed for {path}: {bgr.shape}")
    return bgr


def square_roi(box: tuple[float, float, float, float]) -> tuple[float, float, float]:
    x1, y1, x2, y2 = box
    center_x = (x1 + x2) * 0.5
    center_y = (y1 + y2) * 0.5
    side = max(x2 - x1, y2 - y1) * ROI_EXPANSION
    return center_x - side * 0.5, center_y - side * 0.5, side


def make_square_crop(image: np.ndarray, left: float, top: float, side: float) -> np.ndarray:
    if not math.isfinite(side) or side <= 0:
        raise ValueError(f"Invalid square ROI side: {side}")
    scale = INPUT_SIZE / side
    matrix = np.array(
        [[scale, 0.0, -left * scale], [0.0, scale, -top * scale]],
        dtype=np.float32,
    )
    return cv2.warpAffine(
        image,
        matrix,
        (INPUT_SIZE, INPUT_SIZE),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def configure_model(config: Path, checkpoint: Path, device: str):
    register_all_modules()
    model = init_model(str(config), str(checkpoint), device=device)
    model.eval()
    pipeline_config = copy.deepcopy(model.cfg.test_dataloader.dataset.pipeline)
    padding_nodes = 0
    for transform in pipeline_config:
        if transform.get("type") == "GetBBoxCenterScale":
            transform["padding"] = 1.0
            padding_nodes += 1
    if padding_nodes != 1:
        raise RuntimeError(f"Expected one GetBBoxCenterScale transform, found {padding_nodes}")
    return model, Compose(pipeline_config), pipeline_config


def pipeline_item(crop: np.ndarray, dataset_meta: dict) -> dict:
    item = {
        "img": crop,
        "bbox": np.array([[0.0, 0.0, INPUT_SIZE, INPUT_SIZE]], dtype=np.float32),
        "bbox_score": np.ones(1, dtype=np.float32),
    }
    item.update(dataset_meta)
    return item


def clip_point(value: float, limit: int) -> float:
    return min(max(value, 0.0), float(limit))


def format_coordinate(value: float) -> str:
    return f"{value:.2f}"


def add_prediction(
    image: ET.Element,
    group_id: int,
    points: np.ndarray,
    left: float,
    top: float,
    side: float,
    width: int,
    height: int,
) -> None:
    if points.shape != (21, 2) or not np.isfinite(points).all():
        raise RuntimeError(f"Unexpected RTMPose keypoints: shape={points.shape}")
    skeleton = ET.SubElement(
        image,
        "skeleton",
        {
            "label": "keypoints",
            "source": "auto",
            "outside": "0",
            "occluded": "0",
            "z_order": "0",
            "group_id": str(group_id),
        },
    )
    for output_label, model_index in (("1", 0), ("2", 9)):
        x = left + float(points[model_index, 0]) * side / INPUT_SIZE
        y = top + float(points[model_index, 1]) * side / INPUT_SIZE
        x = clip_point(x, width)
        y = clip_point(y, height)
        ET.SubElement(
            skeleton,
            "points",
            {
                "label": output_label,
                "source": "auto",
                "outside": "0",
                "occluded": "0",
                "points": f"{format_coordinate(x)},{format_coordinate(y)}",
                "z_order": "0",
            },
        )


def run_inference(
    source: SourceData,
    model,
    pipeline: Compose,
    batch_size: int,
) -> tuple[int, int]:
    pending_inputs: list[dict] = []
    pending_meta: list[tuple[ET.Element, int, float, float, float, int, int]] = []
    inferred_boxes = 0
    loaded_images = 0
    started = time.time()

    def flush_batch() -> None:
        nonlocal inferred_boxes
        if not pending_inputs:
            return
        with torch.no_grad():
            results = model.test_step(pseudo_collate(pending_inputs))
        if len(results) != len(pending_meta):
            raise RuntimeError(
                f"RTMPose result count mismatch: {len(results)} != {len(pending_meta)}"
            )
        for result, meta in zip(results, pending_meta):
            image, group_id, left, top, side, width, height = meta
            keypoints = np.asarray(result.pred_instances.keypoints)[0]
            scores = np.asarray(result.pred_instances.keypoint_scores)[0]
            if keypoints.shape != (21, 2) or scores.shape != (21,):
                raise RuntimeError(
                    f"Unexpected RTMPose output shapes: {keypoints.shape}, {scores.shape}"
                )
            if not np.isfinite(scores).all():
                raise RuntimeError("Non-finite RTMPose keypoint scores")
            add_prediction(image, group_id, keypoints, left, top, side, width, height)
            inferred_boxes += 1
        pending_inputs.clear()
        pending_meta.clear()

    for image_index, (image, expected) in enumerate(zip(source.images, source.expected), start=1):
        boxes = image.findall("box")
        if boxes:
            pixels = load_image(source.image_paths[expected.name])
            loaded_images += 1
            for group_id, (box_element, coordinates) in enumerate(
                zip(boxes, expected.boxes), start=1
            ):
                if box_element.get("group_id") != str(group_id):
                    raise RuntimeError(f"Internal group_id mismatch for {expected.name}")
                left, top, side = square_roi(coordinates)
                crop = make_square_crop(pixels, left, top, side)
                pending_inputs.append(pipeline(pipeline_item(crop, model.dataset_meta)))
                pending_meta.append(
                    (image, group_id, left, top, side, expected.width, expected.height)
                )
                if len(pending_inputs) >= batch_size:
                    flush_batch()
        if image_index % 500 == 0 or image_index == len(source.images):
            print(
                f"Progress: images={image_index}/{len(source.images)}, "
                f"inferred_bboxes={inferred_boxes}, elapsed={time.time() - started:.1f}s",
                flush=True,
            )
    flush_batch()
    return loaded_images, inferred_boxes


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


def validate_output(path: Path, expected: Sequence[ExpectedImage]) -> None:
    root = ET.parse(path).getroot()
    if root.tag != "annotations" or root.findtext("version") != "1.1":
        raise ValueError(f"Output is not CVAT 1.1 XML: {path}")
    validate_output_label_schema(root)
    images = root.findall("image")
    if len(images) != len(expected):
        raise ValueError(f"Output image count mismatch: {len(images)} != {len(expected)}")

    total_boxes = 0
    total_skeletons = 0
    total_points = 0
    for image, reference in zip(images, expected):
        if image.get("id") != reference.image_id or image.get("name") != reference.name:
            raise ValueError(f"Output image identity mismatch for {reference.name}")
        if image.get("width") != str(reference.width) or image.get("height") != str(reference.height):
            raise ValueError(f"Output dimensions changed for {reference.name}")
        boxes = image.findall("box")
        skeletons = image.findall("skeleton")
        if len(boxes) != len(reference.boxes) or len(skeletons) != len(reference.boxes):
            raise ValueError(
                f"bbox/skeleton count mismatch for {reference.name}: "
                f"{len(boxes)}/{len(skeletons)}/{len(reference.boxes)}"
            )
        box_groups = []
        skeleton_groups = []
        for index, (box, original_coordinates) in enumerate(zip(boxes, reference.boxes), start=1):
            coordinates = parse_box(box, reference.name, reference.width, reference.height)
            if coordinates != original_coordinates:
                raise ValueError(f"Source bbox coordinates changed for {reference.name}")
            if box.get("group_id") != str(index):
                raise ValueError(f"Incorrect bbox group_id for {reference.name}: {box.get('group_id')}")
            box_groups.append(box.get("group_id"))
        for index, skeleton in enumerate(skeletons, start=1):
            if skeleton.get("label") != "keypoints":
                raise ValueError(f"Incorrect skeleton label for {reference.name}")
            if skeleton.get("group_id") != str(index):
                raise ValueError(f"Incorrect skeleton group_id for {reference.name}")
            if skeleton.get("outside") != "0" or skeleton.get("occluded") != "0":
                raise ValueError(f"Missing skeleton visibility attributes for {reference.name}")
            skeleton_groups.append(skeleton.get("group_id"))
            points = skeleton.findall("points")
            if [point.get("label") for point in points] != ["1", "2"]:
                raise ValueError(f"Incorrect keypoint labels for {reference.name}")
            for point in points:
                if point.get("outside") != "0" or point.get("occluded") != "0":
                    raise ValueError(f"Missing point visibility attributes for {reference.name}")
                values = [float(value) for value in point.get("points", "").split(",")]
                if len(values) != 2 or not all(math.isfinite(value) for value in values):
                    raise ValueError(f"Invalid point coordinates for {reference.name}")
                if not (0.0 <= values[0] <= reference.width and 0.0 <= values[1] <= reference.height):
                    raise ValueError(f"Out-of-range point for {reference.name}: {values}")
            total_points += len(points)
        if box_groups != skeleton_groups:
            raise ValueError(f"bbox/skeleton group pairing mismatch for {reference.name}")
        total_boxes += len(boxes)
        total_skeletons += len(skeletons)
    if total_skeletons != total_boxes or total_points != total_boxes * 2:
        raise ValueError(
            f"Global output count mismatch: boxes={total_boxes}, "
            f"skeletons={total_skeletons}, points={total_points}"
        )


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return number


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use checked bbox annotations and expanded square ROIs to run RTMPose Hand5, "
            "then write CVAT 1.1 bbox plus two-keypoint XML."
        )
    )
    parser.add_argument("--dataset", required=True, choices=DATASET_CHOICES)
    parser.add_argument(
        "--image-root",
        type=Path,
        help="Dataset image directory; defaults to external/data/images/<dataset>",
    )
    parser.add_argument(
        "--annotation-root",
        type=Path,
        help="Dataset annotation directory; defaults to external/data/annotations/<dataset>",
    )
    parser.add_argument("--batch-size", type=positive_int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--output",
        type=Path,
        help="Output XML path; defaults to external/data/annotations/<dataset>/annotations_square_roi_rtmpose.xml",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow atomic replacement of an existing output XML.",
    )
    return parser.parse_args()


def main() -> int:
    global Compose, init_model, pseudo_collate, register_all_modules, torch
    args = parse_args()
    image_root = (
        args.image_root.expanduser().resolve()
        if args.image_root is not None
        else (IMAGE_BASE / args.dataset).resolve()
    )
    annotation_root = (
        args.annotation_root.expanduser().resolve()
        if args.annotation_root is not None
        else (ANNOTATION_BASE / args.dataset).resolve()
    )
    source_xml = (annotation_root / CHECKED_XML_NAME).resolve()
    output_xml = (
        args.output.expanduser().resolve()
        if args.output is not None
        else (annotation_root / OUTPUT_XML_NAME).resolve()
    )
    config = args.config.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()

    if not image_root.is_dir():
        raise FileNotFoundError(f"Image root not found: {image_root}")
    if not annotation_root.is_dir():
        raise FileNotFoundError(f"Annotation root not found: {annotation_root}")
    if output_xml == source_xml:
        raise ValueError("Output XML must not overwrite annotations校验后.xml")
    if output_xml.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {output_xml}; pass --overwrite to replace it"
        )
    if not config.is_file():
        raise FileNotFoundError(f"RTMPose config not found: {config}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"RTMPose checkpoint not found: {checkpoint}")
    try:
        import torch
        from mmcv.transforms import Compose
        from mmengine.dataset import pseudo_collate
        from mmpose.apis import init_model
        from mmpose.utils import register_all_modules
    except ModuleNotFoundError as error:
        raise RuntimeError("MMPose, MMCV, MMEngine and PyTorch must be installed in the current Python environment") from error
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {args.device}")

    print(f"Dataset: {args.dataset}")
    print(f"Images: {image_root}")
    print(f"Input XML: {source_xml}")
    print(f"Output XML: {output_xml}")
    print(f"Config: {config}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Device: {args.device}, batch_size={args.batch_size}")
    print(f"ROI: centered square, side=max(width,height)*{ROI_EXPANSION}")

    source = load_source(source_xml, image_root)
    total_boxes = sum(len(item.boxes) for item in source.expected)
    print(
        f"Validated source: images={len(source.expected)}, bboxes={total_boxes}, "
        "ignored_non_bbox_annotations_in_memory="
        f"{source.source_non_bbox_annotations_removed}"
    )
    model, pipeline, pipeline_config = configure_model(config, checkpoint, args.device)
    print(f"RTMPose test pipeline: {pipeline_config}")
    if not any(
        item.get("type") == "GetBBoxCenterScale" and item.get("padding") == 1.0
        for item in pipeline_config
    ):
        raise RuntimeError("RTMPose pipeline padding is not 1.0")

    started = time.time()
    loaded_images, inferred_boxes = run_inference(source, model, pipeline, args.batch_size)
    if inferred_boxes != total_boxes:
        raise RuntimeError(f"Inference count mismatch: {inferred_boxes} != {total_boxes}")

    output_xml.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_xml.name}.staging-",
        suffix=".xml",
        dir=output_xml.parent,
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        indent_xml(source.tree.getroot())
        source.tree.write(temporary_path, encoding="utf-8", xml_declaration=True)
        validate_output(temporary_path, source.expected)
        if output_xml.exists() and not args.overwrite:
            raise FileExistsError(f"Output appeared during inference: {output_xml}")
        os.replace(temporary_path, output_xml)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    print(
        f"Done: images={len(source.expected)}, images_with_bboxes={loaded_images}, "
        f"bboxes={total_boxes}, skeletons={total_boxes}, points={total_boxes * 2}, "
        f"elapsed={time.time() - started:.1f}s"
    )
    print(f"CVAT 1.1 XML: {output_xml}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
