"""Two-level rectangular anchors with sixth_train ATSS target assignment."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


INPUT_HEIGHT = 224
INPUT_WIDTH = 384
NUM_POINTS = 2
VALUES_PER_ANCHOR = 4 + NUM_POINTS * 2
FEATURE_SHAPES = ((14, 24), (7, 12))
FEATURE_NAMES = ("14x24", "7x12")
NUM_ANCHORS_PER_LEVEL = 2
ATSS_TOP_K = 9
REG_OFFSET_CLIP = 8.0
_LOCAL_CONFIG = Path(__file__).resolve().parent / "anchor_config.json"
_CONFIG_PATH = _LOCAL_CONFIG if _LOCAL_CONFIG.is_file() else Path(__file__).resolve().parents[1] / "data_audit" / "anchor_candidates.json"


def _load_shapes():
    payload = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
    return tuple(np.asarray(payload["levels"][name], dtype=np.float32) for name in FEATURE_NAMES)


ANCHOR_SHAPES = _load_shapes()


def generate_anchors(feature_shape, shapes):
    height, width = feature_shape
    values = []
    for y in range(height):
        for x in range(width):
            for anchor_width, anchor_height in shapes:
                values.append(((x + 0.5) / width, (y + 0.5) / height, anchor_width, anchor_height))
    return np.asarray(values, dtype=np.float32)


ANCHORS_BY_LEVEL = tuple(generate_anchors(shape, anchors) for shape, anchors in zip(FEATURE_SHAPES, ANCHOR_SHAPES))
ANCHORS_14, ANCHORS_7 = ANCHORS_BY_LEVEL
ANCHORS_MULTI = np.concatenate(ANCHORS_BY_LEVEL, axis=0)
assert len(ANCHORS_MULTI) == 840


def center_to_corners(boxes):
    boxes = np.asarray(boxes, dtype=np.float32)
    output = np.empty_like(boxes)
    output[..., 0] = boxes[..., 0] - boxes[..., 2] / 2.0
    output[..., 1] = boxes[..., 1] - boxes[..., 3] / 2.0
    output[..., 2] = boxes[..., 0] + boxes[..., 2] / 2.0
    output[..., 3] = boxes[..., 1] + boxes[..., 3] / 2.0
    return output


def pairwise_iou(boxes_a, boxes_b):
    a = center_to_corners(boxes_a)
    b = center_to_corners(boxes_b)
    intersection_width = np.maximum(0.0, np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]))
    intersection_height = np.maximum(0.0, np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]))
    intersection = intersection_width * intersection_height
    area_a = np.maximum(0.0, a[:, 2] - a[:, 0]) * np.maximum(0.0, a[:, 3] - a[:, 1])
    area_b = np.maximum(0.0, b[:, 2] - b[:, 0]) * np.maximum(0.0, b[:, 3] - b[:, 1])
    return intersection / np.maximum(area_a[:, None] + area_b[None, :] - intersection, 1e-9)


def _empty_targets():
    outputs = []
    for height, width in FEATURE_SHAPES:
        outputs.extend((
            np.zeros((NUM_ANCHORS_PER_LEVEL * VALUES_PER_ANCHOR, height, width), np.float32),
            np.zeros((NUM_ANCHORS_PER_LEVEL, height, width), np.float32),
        ))
    return tuple(outputs)


def encode_instances(instances, return_diagnostics=False, top_k=ATSS_TOP_K):
    """Assign positives with ATSS while preserving the four-output contract.

    Positive anchors have classification target 1 and regression targets.
    Every otherwise valid dense anchor is a negative with target 0.  The
    dataset contract contains no crowd/invalid regions, so the ignore set is
    explicitly empty.  Regression masks are therefore exactly cls_target > .5.
    """
    if not instances:
        targets = _empty_targets()
        diagnostics = {
            "assignment": "ATSS", "top_k": int(top_k), "gt": 0,
            "positive_by_level": [0, 0],
            "negative_by_level": [len(level) for level in ANCHORS_BY_LEVEL],
            "ignore_by_level": [0, 0], "positive_per_gt": [],
            "definitions": {
                "positive": "ATSS-selected anchor owned by one GT",
                "negative": "valid anchor not assigned to any GT",
                "ignore": "none: audited data has no crowd or invalid regions",
            },
        }
        return (targets, diagnostics) if return_diagnostics else targets
    if int(top_k) <= 0:
        raise ValueError("ATSS top_k must be positive")
    boxes = np.asarray([item[0] for item in instances], dtype=np.float32)
    keypoints = np.asarray([item[1] for item in instances], dtype=np.float32).reshape(-1, NUM_POINTS, 2)
    if not np.isfinite(boxes).all() or not np.isfinite(keypoints).all():
        raise ValueError("Non-finite ground truth")
    if np.any(boxes[:, 2:] <= 0.0):
        raise ValueError("Ground-truth boxes must have positive width and height")

    all_anchors = np.concatenate(ANCHORS_BY_LEVEL, axis=0)
    all_ious = pairwise_iou(boxes, all_anchors)
    level_offsets = np.cumsum([0] + [len(values) for values in ANCHORS_BY_LEVEL])
    candidate_mask = np.zeros_like(all_ious, dtype=bool)
    thresholds = []
    candidate_counts = []
    for gt_index, gt in enumerate(boxes):
        candidates = []
        for level, anchors in enumerate(ANCHORS_BY_LEVEL):
            distances = np.square(anchors[:, 0] - gt[0]) + np.square(anchors[:, 1] - gt[1])
            count = min(int(top_k), len(anchors))
            local = np.argsort(distances, kind="stable")[:count]
            global_indices = local + level_offsets[level]
            candidate_mask[gt_index, global_indices] = True
            candidates.extend(global_indices.tolist())
        candidate_ious = all_ious[gt_index, candidates]
        thresholds.append(float(np.mean(candidate_ious) + np.std(candidate_ious)))
        candidate_counts.append(len(candidates))

    corners = center_to_corners(boxes)
    centers_inside = (
        (all_anchors[None, :, 0] > corners[:, None, 0])
        & (all_anchors[None, :, 0] < corners[:, None, 2])
        & (all_anchors[None, :, 1] > corners[:, None, 1])
        & (all_anchors[None, :, 1] < corners[:, None, 3])
    )
    positive_matrix = candidate_mask & centers_inside & (all_ious >= np.asarray(thresholds)[:, None])
    fallback_used = [False] * len(boxes)
    for gt_index in range(len(boxes)):
        if not np.any(positive_matrix[gt_index]):
            candidates = np.where(candidate_mask[gt_index])[0]
            chosen = int(candidates[np.argmax(all_ious[gt_index, candidates])])
            positive_matrix[gt_index, chosen] = True
            fallback_used[gt_index] = True

    owner_all = np.full(len(all_anchors), -1, dtype=np.int32)
    for anchor_index in np.where(np.any(positive_matrix, axis=0))[0]:
        eligible = np.where(positive_matrix[:, anchor_index])[0]
        owner_all[anchor_index] = int(eligible[np.argmax(all_ious[eligible, anchor_index])])

    # A collision can remove a GT's only provisional positive. Reassign the
    # best candidate that is free or owned by a GT with another positive.
    for _ in range(len(boxes) + 1):
        counts = np.bincount(owner_all[owner_all >= 0], minlength=len(boxes))
        missing = np.where(counts == 0)[0]
        if not len(missing):
            break
        changed = False
        for gt_index in missing:
            candidates = np.where(candidate_mask[gt_index])[0]
            ranked = candidates[np.argsort(-all_ious[gt_index, candidates], kind="stable")]
            for anchor_index in ranked:
                previous = owner_all[anchor_index]
                if previous < 0 or counts[previous] > 1:
                    if previous >= 0:
                        counts[previous] -= 1
                    owner_all[anchor_index] = gt_index
                    counts[gt_index] += 1
                    fallback_used[gt_index] = True
                    changed = True
                    break
        if not changed:
            break
    final_counts = np.bincount(owner_all[owner_all >= 0], minlength=len(boxes))
    if np.any(final_counts == 0):
        raise RuntimeError(f"ATSS failed to give every GT a positive anchor: {final_counts.tolist()}")
    owners = [owner_all[level_offsets[level] : level_offsets[level + 1]] for level in range(2)]

    outputs = []
    positive_by_level = []
    for level, ((height, width), anchors) in enumerate(zip(FEATURE_SHAPES, ANCHORS_BY_LEVEL)):
        regression = np.zeros((len(anchors), VALUES_PER_ANCHOR), dtype=np.float32)
        classification = (owners[level] >= 0).astype(np.float32)
        positive = np.where(owners[level] >= 0)[0]
        positive_by_level.append(int(len(positive)))
        if positive.size:
            gt_indices = owners[level][positive]
            selected_boxes = boxes[gt_indices]
            selected_keypoints = keypoints[gt_indices]
            selected_anchors = anchors[positive]
            encoded = np.empty((len(positive), VALUES_PER_ANCHOR), dtype=np.float32)
            encoded[:, 0] = (selected_boxes[:, 0] - selected_anchors[:, 0]) / selected_anchors[:, 2]
            encoded[:, 1] = (selected_boxes[:, 1] - selected_anchors[:, 1]) / selected_anchors[:, 3]
            encoded[:, 2] = np.log(np.maximum(selected_boxes[:, 2], 1e-8) / selected_anchors[:, 2])
            encoded[:, 3] = np.log(np.maximum(selected_boxes[:, 3], 1e-8) / selected_anchors[:, 3])
            encoded[:, 4::2] = (selected_keypoints[:, :, 0] - selected_anchors[:, None, 0]) / selected_anchors[:, None, 2]
            encoded[:, 5::2] = (selected_keypoints[:, :, 1] - selected_anchors[:, None, 1]) / selected_anchors[:, None, 3]
            regression[positive] = np.clip(encoded, -REG_OFFSET_CLIP, REG_OFFSET_CLIP)
        outputs.extend((
            regression.reshape(height, width, NUM_ANCHORS_PER_LEVEL * VALUES_PER_ANCHOR).transpose(2, 0, 1),
            classification.reshape(height, width, NUM_ANCHORS_PER_LEVEL).transpose(2, 0, 1),
        ))
    diagnostics = {
        "assignment": "ATSS", "top_k": int(top_k), "gt": int(len(boxes)),
        "candidate_count_per_gt": candidate_counts,
        "dynamic_iou_threshold_per_gt": thresholds,
        "positive_by_level": positive_by_level,
        "negative_by_level": [len(owners[level]) - positive_by_level[level] for level in range(2)],
        "ignore_by_level": [0, 0],
        "positive_per_gt": final_counts.astype(int).tolist(),
        "fallback_used_per_gt": fallback_used,
        "assigned_gt_by_level": [int(len(set(owners[level][owners[level] >= 0].tolist()))) for level in range(2)],
        "definitions": {
            "positive": "ATSS-selected anchor owned by highest-IoU GT after collision resolution",
            "negative": "valid anchor not assigned to any GT",
            "ignore": "none: audited data has no crowd or invalid regions",
        },
    }
    targets = tuple(outputs)
    return (targets, diagnostics) if return_diagnostics else targets


def decode_level(regression, classification, level_index):
    height, width = FEATURE_SHAPES[level_index]
    anchors = ANCHORS_BY_LEVEL[level_index]
    reg = np.asarray(regression, dtype=np.float32).transpose(1, 2, 0).reshape(-1, VALUES_PER_ANCHOR)
    scores = np.asarray(classification, dtype=np.float32).transpose(1, 2, 0).reshape(-1)
    boxes = np.empty((len(anchors), 4), dtype=np.float32)
    boxes[:, 0] = reg[:, 0] * anchors[:, 2] + anchors[:, 0]
    boxes[:, 1] = reg[:, 1] * anchors[:, 3] + anchors[:, 1]
    boxes[:, 2] = np.exp(np.clip(reg[:, 2], -REG_OFFSET_CLIP, REG_OFFSET_CLIP)) * anchors[:, 2]
    boxes[:, 3] = np.exp(np.clip(reg[:, 3], -REG_OFFSET_CLIP, REG_OFFSET_CLIP)) * anchors[:, 3]
    keypoints = np.empty((len(anchors), NUM_POINTS, 2), dtype=np.float32)
    keypoints[:, :, 0] = reg[:, 4::2] * anchors[:, None, 2] + anchors[:, None, 0]
    keypoints[:, :, 1] = reg[:, 5::2] * anchors[:, None, 3] + anchors[:, None, 1]
    return np.clip(boxes, 0.0, 1.0), np.clip(keypoints, 0.0, 1.0), scores


def nms_center_boxes(boxes, scores, threshold=0.3, max_detections=2):
    order = np.argsort(-scores)
    keep = []
    corners = center_to_corners(boxes)
    while order.size and len(keep) < max_detections:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        rest = order[1:]
        a, b = corners[current], corners[rest]
        intersection = np.maximum(0.0, np.minimum(a[2], b[:, 2]) - np.maximum(a[0], b[:, 0])) * np.maximum(
            0.0, np.minimum(a[3], b[:, 3]) - np.maximum(a[1], b[:, 1])
        )
        area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
        area_b = np.maximum(0.0, b[:, 2] - b[:, 0]) * np.maximum(0.0, b[:, 3] - b[:, 1])
        iou = intersection / np.maximum(area_a + area_b - intersection, 1e-9)
        order = rest[iou < threshold]
    return np.asarray(keep, dtype=np.int64)


def decode_outputs(outputs, score_threshold=0.3, nms_threshold=0.3, max_detections=2):
    all_boxes, all_keypoints, all_scores, all_levels = [], [], [], []
    for level in range(2):
        boxes, keypoints, scores = decode_level(outputs[level * 2], outputs[level * 2 + 1], level)
        mask = scores >= score_threshold
        all_boxes.append(boxes[mask])
        all_keypoints.append(keypoints[mask])
        all_scores.append(scores[mask])
        all_levels.append(np.full(np.sum(mask), level, dtype=np.int32))
    if not any(len(values) for values in all_scores):
        return {"boxes": np.zeros((0, 4), np.float32), "keypoints": np.zeros((0, NUM_POINTS, 2), np.float32), "scores": np.zeros(0, np.float32), "levels": np.zeros(0, np.int32)}
    boxes = np.concatenate(all_boxes)
    keypoints = np.concatenate(all_keypoints)
    scores = np.concatenate(all_scores)
    levels = np.concatenate(all_levels)
    keep = nms_center_boxes(boxes, scores, nms_threshold, max_detections)
    return {"boxes": boxes[keep], "keypoints": keypoints[keep], "scores": scores[keep], "levels": levels[keep]}
