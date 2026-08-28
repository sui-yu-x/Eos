#!/usr/bin/env python3
"""Resumable seventh Palm training with ATSS and fixed balanced hard sampling."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import itertools
import json
import math
import os
import random
import tempfile
import time
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tqdm import tqdm

from anchor_utils import FEATURE_NAMES, NUM_POINTS, decode_outputs, encode_instances, pairwise_iou
from losses import loss_components
from model import palm_detection_model
from seventh_config import BASE, DATASETS, IMAGE_ROOT, SEED, SIXTH_WEIGHTS, SPLIT_ROOT


SPLIT_PATH = SPLIT_ROOT / "fixed_split_manifest.json"
GRAD_CLIPNORM = 1.0
SELECTION_SCORE_THRESHOLD = 0.25
SELECTION_NMS_THRESHOLD = 0.10
MATCH_IOU = 0.50
NO_MATCH_PIXEL_ERROR = float(math.hypot(1280, 720))


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=".staging-", suffix=".json", dir=str(path.parent))
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gpu_snapshot() -> dict:
    try:
        output = subprocess.check_output([
            "nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ], text=True).strip().splitlines()[0]
        used, total, utilization = [int(value.strip()) for value in output.split(",")]
        return {"memory_used_mib": used, "memory_total_mib": total, "utilization_percent": utilization}
    except Exception as error:
        return {"error": str(error)}


def normalized_image_path(path: str) -> str:
    normalized = path.replace("\\", "/")
    if normalized.startswith("/") or (len(normalized) >= 2 and normalized[1] == ":"):
        raise ValueError(f"Image path must be relative to {IMAGE_ROOT}: {path}")
    parts = PurePosixPath(normalized).parts
    if len(parts) < 3 or parts[0] not in DATASETS or any(part in ("", ".", "..") for part in parts):
        raise ValueError(f"Expected dataset/source-folder/image relative path: {path}")
    return PurePosixPath(*parts).as_posix()


def resolve_image_path(path: str) -> Path:
    return IMAGE_ROOT.joinpath(*PurePosixPath(normalized_image_path(path)).parts)


def domain_of(path: str) -> str:
    return PurePosixPath(normalized_image_path(path)).parts[0]


def folder_of(path: str) -> str:
    return PurePosixPath(normalized_image_path(path)).parts[1]


def parse_line(line: str):
    tokens = line.split()
    if not tokens:
        raise ValueError("Empty split line")
    path = normalized_image_path(tokens[0])
    values = np.asarray([float(value) for value in tokens[1:]], dtype=np.float32)
    if values.size % 8 or not np.isfinite(values).all():
        raise ValueError(f"Invalid annotation row width/value: {path}")
    instances = []
    for row in values.reshape(-1, 8):
        x1, y1, x2, y2 = row[:4]
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1 and np.all((row[4:] >= 0) & (row[4:] <= 1))):
            raise ValueError(f"Coordinate violation: {path}")
        instances.append(([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], row[4:].tolist()))
    return path, instances


def normalized_split_line(line: str) -> str:
    tokens = line.split()
    if not tokens:
        raise ValueError("Empty split line")
    tokens[0] = normalized_image_path(tokens[0])
    return " ".join(tokens)


def load_image(path: str):
    relative_path = normalized_image_path(path)
    source_path = resolve_image_path(relative_path)
    encoded = np.fromfile(source_path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found or unreadable: {relative_path} ({source_path})")
    height, width = image.shape[:2]
    if (width, height) == (720, 1280):
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif (width, height) != (1280, 720):
        raise ValueError(f"Unsupported image size {width}x{height}: {relative_path}")
    corrected_height, corrected_width = image.shape[:2]
    image = cv2.resize(image, (384, 224), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    return image[None, :, :], (float(corrected_width), float(corrected_height))


def safe_random_crop(image, instances, rng, minimum_scale=0.85):
    """Crop synchronously; never retain a keypoint outside the kept region."""
    height, width = image.shape[1:]
    scale = float(rng.uniform(minimum_scale, 1.0))
    crop_height = max(1, int(round(height * scale)))
    crop_width = max(1, int(round(width * scale)))
    y0 = int(rng.integers(0, height - crop_height + 1))
    x0 = int(rng.integers(0, width - crop_width + 1))
    x0_norm, y0_norm = x0 / width, y0 / height
    width_norm, height_norm = crop_width / width, crop_height / height
    retained = []
    for box, keypoints in instances:
        cx, cy, box_width, box_height = box
        x1 = (cx - box_width / 2 - x0_norm) / width_norm
        y1 = (cy - box_height / 2 - y0_norm) / height_norm
        x2 = (cx + box_width / 2 - x0_norm) / width_norm
        y2 = (cy + box_height / 2 - y0_norm) / height_norm
        x1, y1, x2, y2 = np.clip([x1, y1, x2, y2], 0.0, 1.0)
        points = np.asarray(keypoints, dtype=np.float32).reshape(NUM_POINTS, 2)
        points[:, 0] = (points[:, 0] - x0_norm) / width_norm
        points[:, 1] = (points[:, 1] - y0_norm) / height_norm
        points_inside = bool(np.all((points >= 0.0) & (points <= 1.0)))
        if points_inside and x2 - x1 > 1e-4 and y2 - y1 > 1e-4:
            retained.append(([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], points.reshape(-1).tolist()))
    if not retained:
        return image, instances, False
    cropped = image[0, y0 : y0 + crop_height, x0 : x0 + crop_width]
    resized = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)[None, :, :]
    return resized.astype(np.float32), retained, True


def augment(image, instances, rng):
    if rng.random() < 0.5:
        image = image[:, :, ::-1].copy()
        flipped = []
        for box, keypoints in instances:
            points = np.asarray(keypoints, np.float32).reshape(NUM_POINTS, 2)
            points[:, 0] = 1.0 - points[:, 0]
            flipped.append(([1.0 - box[0], box[1], box[2], box[3]], points.reshape(-1).tolist()))
        instances = flipped
    crop_applied = False
    if rng.random() < 0.35:
        image, instances, crop_applied = safe_random_crop(image, instances, rng)
    image = np.clip(image * rng.uniform(0.88, 1.12) + rng.uniform(-0.06, 0.06), 0.0, 1.0)
    if rng.random() < 0.15:
        image[0] = cv2.GaussianBlur(image[0], (3, 3), 0)
    if rng.random() < 0.20:
        image = np.clip(image + rng.normal(0.0, 0.01, size=image.shape), 0.0, 1.0)
    return image.astype(np.float32), instances, crop_applied


def process_line(task):
    line, augment_data, sample_seed = task
    path, instances = parse_line(line)
    image, image_size = load_image(path)
    crop_applied = False
    if augment_data:
        image, instances, crop_applied = augment(image, instances, np.random.default_rng(sample_seed))
    targets, diagnostics = encode_instances(instances, return_diagnostics=True)
    if not all(np.isfinite(value).all() for value in (image,) + targets):
        raise ValueError(f"Non-finite batch data: {path}")
    return image, targets, image_size, path, instances, diagnostics, crop_applied


def load_split():
    payload = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload.get("records"), list):
        raise ValueError("Split manifest must contain a records list")
    for record in payload["records"]:
        record["path"] = normalized_image_path(record["path"])
    train = [normalized_split_line(line) for line in (SPLIT_ROOT / "train.txt").read_text(encoding="utf-8").splitlines()]
    legacy_val = [normalized_split_line(line) for line in (SPLIT_ROOT / "legacy_val.txt").read_text(encoding="utf-8").splitlines()]
    new_val = [normalized_split_line(line) for line in (SPLIT_ROOT / "new_val.txt").read_text(encoding="utf-8").splitlines()]
    combined_val = [normalized_split_line(line) for line in (SPLIT_ROOT / "combined_val.txt").read_text(encoding="utf-8").splitlines()]
    if (len(train) != payload["train_images"] or len(legacy_val) != payload["legacy_val_images"]
            or len(new_val) != payload["new_val_images"] or len(combined_val) != payload["val_images"]):
        raise ValueError("Split text/manifest count mismatch")
    train_paths = {line.split()[0] for line in train}
    legacy_paths = {line.split()[0] for line in legacy_val}
    new_paths = {line.split()[0] for line in new_val}
    val_paths = {line.split()[0] for line in combined_val}
    if train_paths & val_paths or legacy_paths & new_paths or val_paths != legacy_paths | new_paths:
        raise ValueError("Train/val path leakage")
    expected = set(DATASETS)
    if {domain_of(line.split()[0]) for line in train} != expected:
        raise ValueError("Every dataset must occur in train")
    if {domain_of(line.split()[0]) for line in legacy_val} != expected:
        raise ValueError("Every dataset must occur in legacy validation")
    if {domain_of(line.split()[0]) for line in new_val} != {"dragon"}:
        raise ValueError("New validation must contain only Dragon")
    return payload, train, legacy_val, new_val, combined_val


class BalancedFolderStream:
    def __init__(self, lines, dataset, rng):
        self.rng = rng
        self.by_folder = defaultdict(list)
        for line in lines:
            path = line.split()[0]
            if domain_of(path) == dataset:
                self.by_folder[folder_of(path)].append(line)
        if not self.by_folder:
            raise ValueError(f"Empty balanced stream for {dataset}")
        self.folders = sorted(self.by_folder)
        self.rng.shuffle(self.folders)
        self.folder_index = 0
        self.indices = {}
        for folder, values in self.by_folder.items():
            self.rng.shuffle(values)
            self.indices[folder] = 0

    def draw(self, count):
        output = []
        for _ in range(count):
            folder = self.folders[self.folder_index % len(self.folders)]
            self.folder_index += 1
            values = self.by_folder[folder]
            index = self.indices[folder]
            if index and index % len(values) == 0:
                self.rng.shuffle(values)
            output.append(values[index % len(values)])
            self.indices[folder] = index + 1
        return output


def balanced_train_entries(lines, hard_paths, epoch, batch_size, limit=0):
    if batch_size < 96 or batch_size % 96:
        raise ValueError("Training batch size must be a positive multiple of the 96-sample quota block")
    hard_paths = set(hard_paths)
    regular_lines = [line for line in lines if line.split()[0] not in hard_paths]
    hard_lines = [line for line in lines if line.split()[0] in hard_paths]
    rng = random.Random(SEED * 1_000_003 + epoch)
    regular = {dataset: BalancedFolderStream(regular_lines, dataset, rng) for dataset in DATASETS}
    hard = {dataset: BalancedFolderStream(hard_lines, dataset, rng) for dataset in DATASETS}
    regular_counts = {
        dataset: sum(domain_of(line.split()[0]) == dataset for line in regular_lines) for dataset in DATASETS
    }
    quota_units = math.ceil(max(regular_counts.values()) / 24)
    if limit:
        quota_units = max(1, math.ceil(limit / 96))
    units_per_batch = batch_size // 96
    quota_units = math.ceil(quota_units / units_per_batch) * units_per_batch
    entries = []
    for batch_start in range(0, quota_units, units_per_batch):
        selected = []
        for _ in range(units_per_batch):
            for dataset in DATASETS:
                selected.extend((line, "regular") for line in regular[dataset].draw(24))
            for dataset in DATASETS:
                selected.extend((line, "hard") for line in hard[dataset].draw(8))
        rng.shuffle(selected)
        if len(selected) != batch_size:
            raise RuntimeError(f"Balanced batch construction differs: {len(selected)} != {batch_size}")
        entries.extend(selected)
    paths = [line.split()[0] for line, _ in entries]
    stream_counts = Counter(stream for _, stream in entries)
    dataset_counts = Counter(domain_of(path) for path in paths)
    folder_counts = Counter(f"{domain_of(path)}/{folder_of(path)}" for path in paths)
    unique = len(set(paths))
    report = {
        "epoch": epoch + 1, "effective_batch_size": batch_size,
        "quota_block_size": 96, "quota_blocks": quota_units,
        "samples": len(entries), "unique_images": unique,
        "duplicate_samples": len(entries) - unique,
        "repeat_rate": (len(entries) - unique) / len(entries),
        "stream_counts": dict(sorted(stream_counts.items())),
        "hard_ratio": stream_counts["hard"] / len(entries),
        "dataset_counts": dict(sorted(dataset_counts.items())),
        "folder_counts": dict(sorted(folder_counts.items())),
    }
    expected_per_dataset = len(entries) // 3
    if stream_counts != {"regular": len(entries) * 3 // 4, "hard": len(entries) // 4}:
        raise RuntimeError(f"72+24 stream quota differs: {stream_counts}")
    if any(dataset_counts[dataset] != expected_per_dataset for dataset in DATASETS):
        raise RuntimeError(f"Dataset quota differs: {dataset_counts}")
    return entries, report


def stratified_limit(lines, limit):
    if not limit or len(lines) <= limit:
        return list(lines)
    buckets = {dataset: [] for dataset in DATASETS}
    for line in lines:
        buckets[domain_of(line.split()[0])].append(line)
    selected = []
    while len(selected) < limit and any(buckets.values()):
        for dataset in DATASETS:
            if buckets[dataset] and len(selected) < limit:
                selected.append(buckets[dataset].pop(0))
    return selected


def iter_batches(lines, batch_size, workers, augment_data, epoch, require_full=False):
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for start in range(0, len(lines), batch_size):
            selected = lines[start : start + batch_size]
            if require_full and len(selected) != batch_size:
                raise RuntimeError(f"Partial training batch is forbidden: {len(selected)} != {batch_size}")
            normalized = [item if isinstance(item, tuple) else (item, "validation") for item in selected]
            tasks = [(line, augment_data, SEED * 1_000_003 + epoch * 100_003 + start + index) for index, (line, _) in enumerate(normalized)]
            rows = list(executor.map(process_line, tasks))
            images = np.stack([row[0] for row in rows]).astype(np.float32)
            targets = tuple(np.stack([row[1][index] for row in rows]).astype(np.float32) for index in range(4))
            image_sizes = np.asarray([row[2] for row in rows], dtype=np.float32)
            metadata = [(row[3], row[4], row[5], row[6], row[2], normalized[index][1]) for index, row in enumerate(rows)]
            yield images, targets, image_sizes, metadata


@tf.function(reduce_retracing=True)
def train_step(model, optimizer, images, targets, image_sizes):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        components = loss_components(targets, predictions, image_sizes)
        scaled_loss = optimizer.get_scaled_loss(components["total_loss"])
    scaled_gradients = tape.gradient(scaled_loss, model.trainable_variables)
    gradients = optimizer.get_unscaled_gradients(scaled_gradients)
    finite_checks = [tf.reduce_all(tf.math.is_finite(value)) for value in gradients if value is not None]
    all_finite = tf.reduce_all(tf.stack(finite_checks))

    def apply_finite_gradients():
        clipped, gradient_norm = tf.clip_by_global_norm(gradients, GRAD_CLIPNORM)
        optimizer.apply_gradients(zip(clipped, model.trainable_variables))
        return gradient_norm, tf.constant(False)

    def back_off_dynamic_loss_scale():
        # LossScaleOptimizer detects the overflow, skips the weight update, and
        # lowers its dynamic scale. The batch is excluded from epoch averages.
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return tf.constant(0.0, tf.float32), tf.constant(True)

    gradient_norm, overflow_skipped = tf.cond(all_finite, apply_finite_gradients, back_off_dynamic_loss_scale)
    tf.debugging.assert_all_finite(components["total_loss"], "Non-finite training loss")
    return components, gradient_norm, overflow_skipped


@tf.function(reduce_retracing=True)
def validation_step(model, images, targets, image_sizes):
    predictions = model(images, training=False)
    components = loss_components(targets, predictions, image_sizes)
    tf.debugging.assert_all_finite(components["total_loss"], "Non-finite validation loss")
    return components, predictions


def new_metrics():
    return {"tp": 0, "fp": 0, "fn": 0, "bbox": [], "bbox_tl": [], "bbox_br": [], "keypoint": [], "iou": []}


def optimal_matches(gt_boxes, pred_boxes):
    if not len(gt_boxes) or not len(pred_boxes):
        return []
    ious = pairwise_iou(gt_boxes, pred_boxes)
    best_key, best = (-1, -1.0), []
    for size in range(min(len(gt_boxes), len(pred_boxes)) + 1):
        for gt_indices in itertools.combinations(range(len(gt_boxes)), size):
            for pred_indices in itertools.permutations(range(len(pred_boxes)), size):
                matches = [(gi, pi, float(ious[gi, pi])) for gi, pi in zip(gt_indices, pred_indices) if ious[gi, pi] >= MATCH_IOU]
                key = (len(matches), sum(item[2] for item in matches))
                if key > best_key:
                    best_key, best = key, matches
    return best


def update_metrics(instances, decoded, image_size, accumulator):
    gt_boxes = np.asarray([item[0] for item in instances], np.float32).reshape(-1, 4)
    gt_keypoints = np.asarray([item[1] for item in instances], np.float32).reshape(-1, NUM_POINTS, 2)
    pred_boxes, pred_keypoints = decoded["boxes"], decoded["keypoints"]
    matches = optimal_matches(gt_boxes, pred_boxes)
    accumulator["tp"] += len(matches)
    accumulator["fp"] += len(pred_boxes) - len(matches)
    accumulator["fn"] += len(gt_boxes) - len(matches)
    width, height = image_size
    corner_scale = np.asarray([width, height, width, height], np.float32)
    point_scale = np.asarray([width, height], np.float32)
    for gt_index, pred_index, iou in matches:
        gt = gt_boxes[gt_index]
        pred = pred_boxes[pred_index]
        gt_corners = np.asarray([gt[0] - gt[2] / 2, gt[1] - gt[3] / 2, gt[0] + gt[2] / 2, gt[1] + gt[3] / 2])
        pred_corners = np.asarray([pred[0] - pred[2] / 2, pred[1] - pred[3] / 2, pred[0] + pred[2] / 2, pred[1] + pred[3] / 2])
        difference = (pred_corners - gt_corners) * corner_scale
        tl_error = float(np.linalg.norm(difference[:2]))
        br_error = float(np.linalg.norm(difference[2:]))
        accumulator["bbox_tl"].append(tl_error)
        accumulator["bbox_br"].append(br_error)
        accumulator["bbox"].append((tl_error + br_error) / 2)
        point_error = np.linalg.norm((pred_keypoints[pred_index] - gt_keypoints[gt_index]) * point_scale, axis=1)
        accumulator["keypoint"].extend(point_error.astype(float).tolist())
        accumulator["iou"].append(iou)


def summarize_metrics(values):
    precision = values["tp"] / max(values["tp"] + values["fp"], 1)
    recall = values["tp"] / max(values["tp"] + values["fn"], 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "precision": precision, "recall": recall, "f1": f1, "tp": values["tp"], "fp": values["fp"], "fn": values["fn"],
        "bbox_corner_mean_px": float(np.mean(values["bbox"])) if values["bbox"] else NO_MATCH_PIXEL_ERROR,
        "bbox_tl_mean_px": float(np.mean(values["bbox_tl"])) if values["bbox_tl"] else NO_MATCH_PIXEL_ERROR,
        "bbox_br_mean_px": float(np.mean(values["bbox_br"])) if values["bbox_br"] else NO_MATCH_PIXEL_ERROR,
        "keypoint_mean_px": float(np.mean(values["keypoint"])) if values["keypoint"] else NO_MATCH_PIXEL_ERROR,
        "bbox_iou": float(np.mean(values["iou"])) if values["iou"] else 0.0,
        "matched_instances_for_error": len(values["bbox"]),
        "no_match_error_policy": "1280x720 diagonal penalty" if not values["bbox"] else None,
    }


def mean_components(rows):
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def display_loss_components(values):
    box = 1.5 * (0.5 * values["box_pixel_huber_loss"] + 0.5 * values["box_giou_loss"])
    keypoint = 0.25 * values["keypoint_pixel_huber_loss"]
    regression = box + keypoint
    classification = 0.7 * values["focal_loss"]
    return regression, classification, box, keypoint


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 4))
    parser.add_argument("--initial-lr", type=float, default=1e-4)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-augment", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.experiment.replace("_", "").replace("-", "").isalnum():
        raise ValueError("Experiment name must be alphanumeric with _ or -")
    cv2.setNumThreads(1)
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    mixed_precision.set_global_policy("mixed_float16")
    random.seed(SEED)
    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)
    split_payload, raw_train, raw_legacy_val, raw_new_val, raw_val = load_split()
    raw_val = stratified_limit(raw_val, args.max_val_samples)
    legacy_val_paths = {line.split()[0] for line in raw_legacy_val}
    new_val_paths = {line.split()[0] for line in raw_new_val}
    hard_pool_path = BASE / "hard_samples/hard_pool.json"
    hard_payload = json.loads(hard_pool_path.read_text(encoding="utf-8"))
    if hard_payload.get("source_split") != "seventh_train_train_only":
        raise ValueError("Hard pool is not signed as seventh_train train-only")
    hard_paths = [row["path"] for row in hard_payload["selected"]]
    raw_train_paths = {line.split()[0] for line in raw_train}
    if not set(hard_paths) <= raw_train_paths:
        raise ValueError("Hard pool contains a path outside seventh_train train split")
    checkpoint_root = BASE / "checkpoints" / args.experiment
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    tf_checkpoint_root = checkpoint_root / "training_state"
    history_path = BASE / "logs" / f"{args.experiment}_history.csv"
    state_path = BASE / "status" / f"{args.experiment}.json"
    model = palm_detection_model()
    if not args.resume:
        model.load_weights(str(SIXTH_WEIGHTS))
    optimizer = mixed_precision.LossScaleOptimizer(
        tf.keras.optimizers.Adam(args.initial_lr),
        dynamic=True,
        initial_scale=128.0,
        dynamic_growth_steps=2000,
    )
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    manager = tf.train.CheckpointManager(checkpoint, str(tf_checkpoint_root), max_to_keep=None)
    start_epoch, best_f1, no_improve, lr_wait = 0, -1.0, 0, 0
    if args.resume:
        if not manager.latest_checkpoint:
            raise FileNotFoundError(f"No seventh_train checkpoint to resume: {tf_checkpoint_root}")
        checkpoint.restore(manager.latest_checkpoint).assert_existing_objects_matched()
        state = json.loads(state_path.read_text(encoding="utf-8"))
        start_epoch, best_f1 = int(state["last_epoch"]), float(state["best_macro_f1"])
        no_improve, lr_wait = int(state["early_stop_wait"]), int(state["lr_wait"])
    else:
        if manager.latest_checkpoint or list(checkpoint_root.glob("epoch_*.weights.h5")):
            raise RuntimeError("Experiment already contains checkpoints; use --resume or a new name")
        # Sixth weights initialize model parameters only. The optimizer above
        # is new and epoch/learning-rate/loss-scale/callback state start fresh.

    counts = {dataset: sum(domain_of(line.split()[0]) == dataset for line in raw_train) for dataset in DATASETS}
    config = {
        "experiment": args.experiment,
        "datasets": list(DATASETS),
        "architecture": "two-head non-FPN 14x24 + 7x12",
        "input": [None, 1, 224, 384], "parameters": model.count_params(), "anchors": 840,
        "initialization": {
            "mode": "sixth_best_weights", "weights": str(SIXTH_WEIGHTS),
            "hash_check_required": False, "resume_model_weights": False,
            "restore_sixth_optimizer": False, "restore_sixth_epoch": False,
        },
        "split": {"path": str(SPLIT_PATH), "seed": SEED, "train_images": len(raw_train),
                  "legacy_val_images": len(raw_legacy_val), "new_val_images": len(raw_new_val),
                  "combined_val_images": len(raw_val), "primary_selection": "new_val_dragon_f1"},
        "natural_training_counts": counts, "forced_dataset_balancing": True,
        "balanced_sampling": {
            "quota_block": 96, "regular_per_block": {dataset: 24 for dataset in DATASETS},
            "hard_per_block": {dataset: 8 for dataset in DATASETS},
            "folder_balanced_round_robin": True, "validation_resampled": False,
            "hard_pool": str(hard_pool_path), "hard_pool_sha256": sha256_file(hard_pool_path),
        },
        "training": {"batch_size": args.batch_size, "max_epochs": args.max_epochs, "initial_lr": args.initial_lr, "optimizer": "Adam", "mixed_precision": "mixed_float16", "dynamic_loss_scale_overflow": "skip_batch_and_back_off", "gradient_clipnorm": GRAD_CLIPNORM, "reduce_lr_patience": 4, "early_stop_patience": 12},
        "loss": {"box_pixel_huber_delta": 16, "box_pixel_huber_divisor": 16, "giou": True, "keypoint_pixel_huber_delta": 16, "keypoint_pixel_huber_divisor": 16, "regression": "1.5*(0.5*box_pixel_huber+0.5*box_giou)+0.25*keypoint_pixel_huber", "total": "regression+0.7*focal"},
        "augmentation": {"new_types_added": False, "horizontal_flip": 0.5, "safe_random_crop": 0.35, "crop_min_scale": 0.85, "brightness_contrast": True, "blur": 0.15, "noise": 0.20},
        "checkpoint_every_epoch": True, "resume_restricted_to": str(tf_checkpoint_root),
        "max_train_samples": args.max_train_samples, "max_val_samples": args.max_val_samples,
    }
    atomic_json(BASE / "logs" / f"{args.experiment}_config.json", config)
    columns = ["epoch", "train_total_loss", "train_box_pixel_huber_loss", "train_box_giou_loss", "train_keypoint_pixel_huber_loss", "train_focal_loss", "train_gradient_norm", "val_total_loss", "val_box_pixel_huber_loss", "val_box_giou_loss", "val_keypoint_pixel_huber_loss", "val_focal_loss", "macro_f1", "new_dragon_f1", "legacy_macro_f1", "combined_macro_f1", "dragon_f1", "peak_f1", "soar_f1", "macro_bbox_px", "macro_keypoint_px", "learning_rate", "atss_positive_14x24", "atss_positive_7x12", "hard_ratio", "unique_images", "repeat_rate", "seconds"]
    history_exists = history_path.exists()
    history_stream = history_path.open("a", encoding="utf-8", newline="")
    history_writer = csv.DictWriter(history_stream, fieldnames=columns)
    if not history_exists:
        history_writer.writeheader()
        history_stream.flush()
    try:
        for epoch in range(start_epoch, args.max_epochs):
            started = time.time()
            train_rows, gradient_norms = [], []
            crop_attempted, crop_applied, overflow_batches = 0, 0, 0
            epoch_train_lines, sampling_report = balanced_train_entries(
                raw_train, hard_paths, epoch, args.batch_size, args.max_train_samples
            )
            atomic_json(BASE / "logs" / f"sampling_epoch_{epoch + 1:03d}.json", sampling_report)
            train_batches = len(epoch_train_lines) // args.batch_size
            train_samples_seen = 0
            train_iterator = iter_batches(epoch_train_lines, args.batch_size, args.workers, not args.no_augment, epoch, require_full=True)
            atss_positive_by_level = [0, 0]
            atss_positive_per_gt = []
            epoch_pbar = tqdm(
                total=train_batches,
                desc=f"Epoch {epoch + 1}/{args.max_epochs}",
                leave=False,
                dynamic_ncols=True,
                ascii=True,
            )
            for batch_index, (images, targets, image_sizes, metadata) in enumerate(train_iterator, 1):
                step_started = time.time()
                components, gradient_norm, overflow_skipped = train_step(model, optimizer, images, targets, image_sizes)
                if epoch == start_epoch and batch_index == 1:
                    print(f"[Warmup] first train_step took {time.time() - step_started:.2f}s", flush=True)
                epoch_pbar.update(1)
                if bool(overflow_skipped.numpy()):
                    overflow_batches += 1
                    print(f"Dynamic loss scale overflow at epoch {epoch + 1}; skipped batch {overflow_batches}", flush=True)
                    continue
                train_rows.append({key: float(value.numpy()) for key, value in components.items() if key not in {"positive_anchors", "regression_loss"}})
                gradient_norms.append(float(gradient_norm.numpy()))
                train_samples_seen += len(metadata)
                for item in metadata:
                    diagnostic = item[2]
                    atss_positive_by_level[0] += diagnostic["positive_by_level"][0]
                    atss_positive_by_level[1] += diagnostic["positive_by_level"][1]
                    atss_positive_per_gt.extend(diagnostic["positive_per_gt"])
                if not args.no_augment:
                    crop_attempted += len(metadata)
                    crop_applied += sum(item[3] for item in metadata)
                if batch_index % 20 == 0 or batch_index == train_batches:
                    running = mean_components(train_rows)
                    running_reg, running_cls, _, _ = display_loss_components(running)
                    elapsed = time.time() - started
                    speed = train_samples_seen / max(elapsed, 1e-6)
                    epoch_pbar.set_postfix(
                        loss=f"{running['total_loss']:.4f}",
                        reg=f"{running_reg:.4f}",
                        cls=f"{running_cls:.4f}",
                        spd=f"{speed:.1f}/s",
                    )
            epoch_pbar.close()
            val_rows = []
            combined_accumulators = {dataset: new_metrics() for dataset in DATASETS}
            legacy_accumulators = {dataset: new_metrics() for dataset in DATASETS}
            new_accumulator = new_metrics()
            for images, targets, image_sizes, metadata in iter_batches(raw_val, args.batch_size, args.workers, False, epoch):
                components, predictions = validation_step(model, images, targets, image_sizes)
                val_rows.append({key: float(value.numpy()) for key, value in components.items() if key not in {"positive_anchors", "regression_loss"}})
                arrays = [value.numpy() for value in predictions]
                for index, (path, instances, _, _, image_size, _) in enumerate(metadata):
                    decoded = decode_outputs([value[index] for value in arrays], SELECTION_SCORE_THRESHOLD, SELECTION_NMS_THRESHOLD, 2)
                    dataset = domain_of(path)
                    update_metrics(instances, decoded, image_size, combined_accumulators[dataset])
                    if path in legacy_val_paths:
                        update_metrics(instances, decoded, image_size, legacy_accumulators[dataset])
                    elif path in new_val_paths:
                        update_metrics(instances, decoded, image_size, new_accumulator)
                    else:
                        raise RuntimeError(f"Combined validation contains an unclassified path: {path}")
            train_mean, val_mean = mean_components(train_rows), mean_components(val_rows)
            metrics = {dataset: summarize_metrics(values) for dataset, values in combined_accumulators.items()}
            legacy_metrics = {dataset: summarize_metrics(values) for dataset, values in legacy_accumulators.items()}
            new_dragon_metrics = summarize_metrics(new_accumulator)
            legacy_macro_f1 = float(np.mean([legacy_metrics[dataset]["f1"] for dataset in DATASETS]))
            combined_macro_f1 = float(np.mean([metrics[dataset]["f1"] for dataset in DATASETS]))
            macro_f1 = new_dragon_metrics["f1"]
            macro_bbox = float(np.mean([metrics[dataset]["bbox_corner_mean_px"] for dataset in metrics if metrics[dataset]["bbox_corner_mean_px"] is not None]))
            macro_bbox_tl = float(np.mean([metrics[dataset]["bbox_tl_mean_px"] for dataset in metrics if metrics[dataset]["bbox_tl_mean_px"] is not None]))
            macro_bbox_br = float(np.mean([metrics[dataset]["bbox_br_mean_px"] for dataset in metrics if metrics[dataset]["bbox_br_mean_px"] is not None]))
            macro_keypoint = float(np.mean([metrics[dataset]["keypoint_mean_px"] for dataset in metrics if metrics[dataset]["keypoint_mean_px"] is not None]))
            if not np.isfinite([*train_mean.values(), *val_mean.values(), *gradient_norms, macro_f1, macro_bbox, macro_bbox_tl, macro_bbox_br, macro_keypoint]).all():
                raise FloatingPointError("Non-finite epoch result")
            train_reg, train_cls, _, _ = display_loss_components(train_mean)
            val_reg, val_cls, val_reg_box, val_reg_kp = display_loss_components(val_mean)
            epoch_weights = checkpoint_root / f"epoch_{epoch + 1:03d}.weights.h5"
            model.save_weights(epoch_weights)
            manager.save(checkpoint_number=epoch + 1)
            improved = macro_f1 > best_f1 + 1e-8
            if improved:
                best_f1, no_improve, lr_wait = macro_f1, 0, 0
                model.save_weights(checkpoint_root / "best_running_new_val_f1.weights.h5")
            else:
                no_improve += 1
                lr_wait += 1
            if lr_wait >= 4:
                optimizer.inner_optimizer.learning_rate.assign(float(optimizer.inner_optimizer.learning_rate.numpy()) * 0.5)
                lr_wait = 0
            seconds = time.time() - started
            row = {
                "epoch": epoch + 1,
                **{f"train_{key}": train_mean[key] for key in ("total_loss", "box_pixel_huber_loss", "box_giou_loss", "keypoint_pixel_huber_loss", "focal_loss")},
                "train_gradient_norm": float(np.mean(gradient_norms)),
                **{f"val_{key}": val_mean[key] for key in ("total_loss", "box_pixel_huber_loss", "box_giou_loss", "keypoint_pixel_huber_loss", "focal_loss")},
                "macro_f1": macro_f1, "new_dragon_f1": macro_f1,
                "legacy_macro_f1": legacy_macro_f1, "combined_macro_f1": combined_macro_f1,
                "dragon_f1": metrics["dragon"]["f1"], "peak_f1": metrics["peak"]["f1"], "soar_f1": metrics["soar"]["f1"],
                "macro_bbox_px": macro_bbox, "macro_keypoint_px": macro_keypoint,
                "learning_rate": float(optimizer.inner_optimizer.learning_rate.numpy()),
                "atss_positive_14x24": atss_positive_by_level[0], "atss_positive_7x12": atss_positive_by_level[1],
                "hard_ratio": sampling_report["hard_ratio"], "unique_images": sampling_report["unique_images"],
                "repeat_rate": sampling_report["repeat_rate"], "seconds": seconds,
            }
            history_writer.writerow(row)
            history_stream.flush()
            state = {
                "status": "training", "experiment": args.experiment, "last_epoch": epoch + 1, "best_macro_f1": best_f1,
                "early_stop_wait": no_improve, "lr_wait": lr_wait, "learning_rate": row["learning_rate"], "last_epoch_metrics": row,
                "dataset_metrics": {"legacy_val": legacy_metrics, "new_val": {"dragon": new_dragon_metrics},
                                    "combined_val": metrics},
                "selection_metric": "new_val.dragon.f1", "crop_applied": crop_applied,
                "samples_seen_for_crop": crop_attempted,
                "atss": {
                    "positive_by_level": atss_positive_by_level,
                    "positive_per_gt_count": len(atss_positive_per_gt),
                    "positive_per_gt_min": min(atss_positive_per_gt) if atss_positive_per_gt else None,
                    "positive_per_gt_mean": float(np.mean(atss_positive_per_gt)) if atss_positive_per_gt else None,
                    "positive_per_gt_max": max(atss_positive_per_gt) if atss_positive_per_gt else None,
                },
                "sampling": sampling_report,
                "dynamic_loss_scale_overflow_batches": overflow_batches,
                "last_checkpoint": str(epoch_weights), "tf_checkpoint": manager.latest_checkpoint,
                "gpu": gpu_snapshot(), "estimated_remaining_seconds": seconds * max(args.max_epochs - (epoch + 1), 0),
                "updated_at_unix": time.time(),
            }
            atomic_json(state_path, state)
            atomic_json(BASE / "status/current_training.json", state)
            print(
                f"Epoch {epoch + 1}/{args.max_epochs} - "
                f"train_loss: {train_mean['total_loss']:.4f} - "
                f"train_reg: {train_reg:.4f} - "
                f"train_cls: {train_cls:.4f} - "
                f"val_loss: {val_mean['total_loss']:.4f} - "
                f"val_reg: {val_reg:.4f} - "
                f"val_reg_box: {val_reg_box:.4f} - "
                f"val_reg_kp: {val_reg_kp:.4f} - "
                f"val_cls: {val_cls:.4f} - "
                f"lr: {row['learning_rate']:.6f} - "
                f"time: {seconds:.2f}s",
                flush=True,
            )
            print(
                f"  -> pixel_error: val_kp_avg={macro_keypoint:.2f}px, "
                f"val_bbox_avg={macro_bbox:.2f}px "
                f"(tl={macro_bbox_tl:.2f}px, br={macro_bbox_br:.2f}px) "
                f"[1280x720]",
                flush=True,
            )
            print(
                f"  -> f1: new_dragon={macro_f1:.6f}, legacy_macro={legacy_macro_f1:.6f}, "
                f"combined_macro={combined_macro_f1:.6f}, combined_dragon={metrics['dragon']['f1']:.6f}, "
                f"combined_peak={metrics['peak']['f1']:.6f}, combined_soar={metrics['soar']['f1']:.6f}, "
                f"overflow={overflow_batches}",
                flush=True,
            )
            if no_improve >= 12:
                print(f"Early stopping at epoch {epoch + 1}", flush=True)
                break
        final_state = json.loads(state_path.read_text(encoding="utf-8"))
        final_state["status"] = "training_complete"
        atomic_json(state_path, final_state)
        atomic_json(BASE / "status/current_training.json", final_state)
    finally:
        history_stream.close()


if __name__ == "__main__":
    main()
