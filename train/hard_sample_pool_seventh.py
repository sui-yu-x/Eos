from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import tensorflow as tf

from anchor_utils import center_to_corners, decode_outputs, pairwise_iou
from model import palm_detection_model
from seventh_config import (
    BASE, DATASETS, SIXTH_WEIGHTS, SPLIT_ROOT, HARD_CONFIDENCE,
    HARD_NMS_IOU, MATCH_IOU, MAX_DETECTIONS,
)
from train_seventh import load_image, normalized_image_path, optimal_matches


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record_instances(record):
    output = []
    for item in record["instances"]:
        x1, y1, x2, y2 = item["bbox_norm_xyxy"]
        output.append(([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], np.asarray(item["keypoints_norm_xy"]).reshape(-1).tolist()))
    return output


def score_record(record, decoded):
    instances = record_instances(record)
    gt_boxes = np.asarray([item[0] for item in instances], np.float32).reshape(-1, 4)
    gt_keypoints = np.asarray([item[1] for item in instances], np.float32).reshape(-1, 2, 2)
    pred_boxes = decoded["boxes"]
    pred_keypoints = decoded["keypoints"]
    scores = decoded["scores"]
    matches = optimal_matches(gt_boxes, pred_boxes)
    matched_gt = {item[0] for item in matches}
    matched_pred = {item[1] for item in matches}
    tp, fp, fn = len(matches), len(pred_boxes) - len(matches), len(gt_boxes) - len(matches)
    ious, bbox_errors, kp1_errors, kp2_errors, match_rows = [], [], [], [], []
    for gt_index, pred_index, iou in matches:
        gt_xyxy = center_to_corners(gt_boxes[[gt_index]])[0]
        pred_xyxy = center_to_corners(pred_boxes[[pred_index]])[0]
        difference = (pred_xyxy - gt_xyxy) * np.asarray([1280, 720, 1280, 720], np.float32)
        tl = float(np.linalg.norm(difference[:2]))
        br = float(np.linalg.norm(difference[2:]))
        bbox_error = (tl + br) / 2.0
        point_error = np.linalg.norm(
            (pred_keypoints[pred_index] - gt_keypoints[gt_index]) * np.asarray([1280, 720], np.float32), axis=1
        )
        ious.append(float(iou)); bbox_errors.append(bbox_error)
        kp1_errors.append(float(point_error[0])); kp2_errors.append(float(point_error[1]))
        match_rows.append({
            "gt_index": gt_index, "pred_index": pred_index, "iou": float(iou),
            "bbox_tl_error_px": tl, "bbox_br_error_px": br, "bbox_mean_error_px": bbox_error,
            "keypoint_1_error_px": float(point_error[0]), "keypoint_2_error_px": float(point_error[1]),
            "confidence": float(scores[pred_index]),
        })
    unmatched_scores = [float(scores[index]) for index in range(len(scores)) if index not in matched_pred]
    matched_scores = [float(scores[index]) for index in matched_pred]
    confidence_abnormality = max(
        max(unmatched_scores, default=0.0),
        1.0 - min(matched_scores, default=1.0),
    )
    no_match_penalty = math.hypot(1280, 720)
    minimum_iou = min(ious) if ious else (0.0 if len(gt_boxes) else 1.0)
    bbox_error = float(np.mean(bbox_errors)) if bbox_errors else (no_match_penalty if fn else 0.0)
    kp1_error = float(np.mean(kp1_errors)) if kp1_errors else (no_match_penalty if fn else 0.0)
    kp2_error = float(np.mean(kp2_errors)) if kp2_errors else (no_match_penalty if fn else 0.0)
    kp_error = (kp1_error + kp2_error) / 2.0
    ranking_key = [-fn, -fp, minimum_iou, -bbox_error, -kp_error, -confidence_abnormality, record["path"]]
    return {
        "dataset": record["dataset"], "folder": record["folder"], "sequence": record["sequence"],
        "image_id": record["image_id"], "name": record["name"], "path": record["path"],
        "gt": len(gt_boxes), "predictions": len(pred_boxes), "tp": tp, "fp": fp, "fn": fn,
        "minimum_matched_iou": minimum_iou, "mean_matched_iou": float(np.mean(ious)) if ious else None,
        "bbox_mean_pixel_error": bbox_error, "keypoint_1_pixel_error": kp1_error,
        "keypoint_2_pixel_error": kp2_error, "keypoint_mean_pixel_error": kp_error,
        "prediction_confidences": [float(value) for value in scores],
        "confidence_abnormality": confidence_abnormality,
        "ranking_key": ranking_key, "matches": match_rows,
        "pred_boxes_center": pred_boxes.astype(float).tolist(),
        "pred_keypoints": pred_keypoints.astype(float).tolist(),
        "annotation_contract_anomalies": [],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    manifest_path = SPLIT_ROOT / "fixed_split_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = [record for record in manifest["records"] if record["split"] == "train"]
    if any(record["split"] != "train" for record in records):
        raise RuntimeError("Validation record entered hard scoring")
    for record in records:
        record["path"] = normalized_image_path(record["path"])
    model = palm_detection_model()
    model.load_weights(str(SIXTH_WEIGHTS))
    scored = []
    for start in range(0, len(records), args.batch_size):
        selected = records[start:start + args.batch_size]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            loaded = list(pool.map(load_image, [record["path"] for record in selected]))
        images = np.stack([item[0] for item in loaded]).astype(np.float32)
        outputs = [value.numpy() for value in model(images, training=False)]
        for index, record in enumerate(selected):
            decoded = decode_outputs(
                [value[index] for value in outputs], HARD_CONFIDENCE, HARD_NMS_IOU, MAX_DETECTIONS
            )
            scored.append(score_record(record, decoded))
        batch_index = start // args.batch_size + 1
        total_batches = math.ceil(len(records) / args.batch_size)
        if batch_index == 1 or batch_index % 20 == 0 or batch_index == total_batches:
            print(f"HARD_SCORE {batch_index}/{total_batches}", flush=True)

    grouped = defaultdict(list)
    for row in scored:
        grouped[(row["dataset"], row["folder"])].append(row)
    selected_rows, folder_summary = [], []
    for (dataset, folder), rows in sorted(grouped.items()):
        rows.sort(key=lambda row: tuple(row["ranking_key"]))
        count = max(1, int(round(len(rows) * 0.20)))
        for rank, row in enumerate(rows, 1):
            row["folder_rank"] = rank
            row["selected"] = rank <= count and not row["annotation_contract_anomalies"]
            row["selection_reason"] = (
                "top approximately 20% by lexicographic FN, FP, minimum IoU, bbox error, keypoint error, confidence abnormality"
                if row["selected"] else ("annotation anomaly" if row["annotation_contract_anomalies"] else "below folder cutoff")
            )
            if row["selected"]:
                selected_rows.append(row)
        folder_summary.append({
            "dataset": dataset, "folder": folder, "train_images": len(rows),
            "target_fraction": 0.20, "selected_images": sum(row["selected"] for row in rows),
            "selected_fraction": sum(row["selected"] for row in rows) / len(rows),
        })
    output_dir = BASE / "hard_samples"
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_split": "seventh_train_train_only", "source_manifest": str(manifest_path),
        "source_manifest_sha256": sha256_file(manifest_path),
        "scoring_weights": str(SIXTH_WEIGHTS), "hash_check_required": False,
        "inference": {"confidence_threshold": HARD_CONFIDENCE, "nms_iou_threshold": HARD_NMS_IOU,
                      "max_detections": MAX_DETECTIONS, "gt_matching_iou": MATCH_IOU},
        "ranking_priority": ["FN descending", "FP descending", "minimum matched IoU ascending",
                             "bbox pixel error descending", "keypoint pixel error descending", "confidence abnormality descending"],
        "selection": "approximately 20% independently within dataset x folder",
        "train_images_scored": len(scored), "selected_images": len(selected_rows),
        "folders": folder_summary,
        "selected": [{key: row[key] for key in (
            "dataset", "folder", "sequence", "image_id", "name", "path", "tp", "fp", "fn",
            "minimum_matched_iou", "bbox_mean_pixel_error", "keypoint_1_pixel_error",
            "keypoint_2_pixel_error", "keypoint_mean_pixel_error", "prediction_confidences",
            "confidence_abnormality", "ranking_key", "folder_rank", "selection_reason",
        )} for row in selected_rows],
    }
    pool_path = output_dir / "hard_pool.json"
    pool_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output_dir / "hard_pool.sha256").write_text(f"{sha256_file(pool_path)}  hard_pool.json\n", encoding="utf-8")
    print(json.dumps({
        "passed_scoring": True, "train_images_scored": len(scored), "selected_images": len(selected_rows),
        "folders": len(folder_summary), "hard_pool_sha256": sha256_file(pool_path),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
