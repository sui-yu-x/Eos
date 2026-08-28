"""Numerically safe bbox-directed fourth-training losses."""
from __future__ import annotations

import tensorflow as tf

from anchor_utils import ANCHORS_BY_LEVEL, NUM_POINTS, VALUES_PER_ANCHOR


HUBER_DELTA_PX = 16.0
DW_DH_CLIP = 8.0
EPSILON = 1e-7


def _flatten(regression, classification):
    regression = tf.cast(regression, tf.float32)
    classification = tf.cast(classification, tf.float32)
    reg_flat = tf.reshape(tf.transpose(regression, [0, 2, 3, 1]), [tf.shape(regression)[0], -1, VALUES_PER_ANCHOR])
    cls_flat = tf.reshape(tf.transpose(classification, [0, 2, 3, 1]), [tf.shape(classification)[0], -1])
    return reg_flat, cls_flat


def decode_regression(regression, anchors):
    """Decode encoded regression into normalized xyxy boxes and xy keypoints."""
    anchors = tf.cast(anchors, tf.float32)[None, :, :]
    regression = tf.cast(regression, tf.float32)
    center_x = anchors[:, :, 0] + regression[:, :, 0] * anchors[:, :, 2]
    center_y = anchors[:, :, 1] + regression[:, :, 1] * anchors[:, :, 3]
    width = anchors[:, :, 2] * tf.exp(tf.clip_by_value(regression[:, :, 2], -DW_DH_CLIP, DW_DH_CLIP))
    height = anchors[:, :, 3] * tf.exp(tf.clip_by_value(regression[:, :, 3], -DW_DH_CLIP, DW_DH_CLIP))
    boxes = tf.stack((center_x - width * 0.5, center_y - height * 0.5, center_x + width * 0.5, center_y + height * 0.5), axis=-1)
    keypoint_x = anchors[:, :, 0, None] + regression[:, :, 4::2] * anchors[:, :, 2, None]
    keypoint_y = anchors[:, :, 1, None] + regression[:, :, 5::2] * anchors[:, :, 3, None]
    keypoints = tf.stack((keypoint_x, keypoint_y), axis=-1)
    return boxes, keypoints


def huber_delta16(values):
    absolute = tf.abs(tf.cast(values, tf.float32))
    delta = tf.cast(HUBER_DELTA_PX, tf.float32)
    return tf.where(absolute <= delta, 0.5 * tf.square(absolute), delta * (absolute - 0.5 * delta))


def giou(boxes_a, boxes_b):
    boxes_a = tf.cast(boxes_a, tf.float32)
    boxes_b = tf.cast(boxes_b, tf.float32)
    intersection_left = tf.maximum(boxes_a[..., :2], boxes_b[..., :2])
    intersection_right = tf.minimum(boxes_a[..., 2:], boxes_b[..., 2:])
    intersection_size = tf.maximum(intersection_right - intersection_left, 0.0)
    intersection = intersection_size[..., 0] * intersection_size[..., 1]
    size_a = tf.maximum(boxes_a[..., 2:] - boxes_a[..., :2], 0.0)
    size_b = tf.maximum(boxes_b[..., 2:] - boxes_b[..., :2], 0.0)
    area_a = size_a[..., 0] * size_a[..., 1]
    area_b = size_b[..., 0] * size_b[..., 1]
    union = area_a + area_b - intersection
    iou = intersection / (union + EPSILON)
    enclosure_left = tf.minimum(boxes_a[..., :2], boxes_b[..., :2])
    enclosure_right = tf.maximum(boxes_a[..., 2:], boxes_b[..., 2:])
    enclosure_size = tf.maximum(enclosure_right - enclosure_left, 0.0)
    enclosure = enclosure_size[..., 0] * enclosure_size[..., 1]
    return iou - (enclosure - union) / (enclosure + EPSILON)


def focal_sums(classification_true, classification_pred, alpha=0.25, gamma=2.0):
    true = tf.cast(classification_true, tf.float32)
    pred = tf.clip_by_value(tf.cast(classification_pred, tf.float32), 1e-6, 1.0 - 1e-6)
    probability = tf.where(true > 0.5, pred, 1.0 - pred)
    alpha_factor = tf.where(true > 0.5, alpha, 1.0 - alpha)
    losses = -alpha_factor * tf.pow(1.0 - probability, gamma) * tf.math.log(probability)
    return tf.reduce_sum(losses), tf.cast(tf.size(losses), tf.float32)


def loss_components(targets, predictions, image_sizes):
    """Return the four requested components, combined regression, and total.

    image_sizes is [batch, 2] in corrected-original (width, height) pixels.
    Regression terms are summed across heads and independently normalized by
    positive-anchor coordinate counts, so head density cannot change scale.
    """
    image_sizes = tf.cast(image_sizes, tf.float32)
    pixel_scale_box = tf.concat((image_sizes, image_sizes), axis=1)[:, None, :]
    pixel_scale_kp = image_sizes[:, None, None, :]
    box_huber_sum = tf.constant(0.0, tf.float32)
    box_coordinate_count = tf.constant(0.0, tf.float32)
    giou_sum = tf.constant(0.0, tf.float32)
    positive_count = tf.constant(0.0, tf.float32)
    keypoint_huber_sum = tf.constant(0.0, tf.float32)
    keypoint_coordinate_count = tf.constant(0.0, tf.float32)
    focal_sum = tf.constant(0.0, tf.float32)
    focal_count = tf.constant(0.0, tf.float32)

    for level, anchors_np in enumerate(ANCHORS_BY_LEVEL):
        reg_true, cls_true = _flatten(targets[level * 2], targets[level * 2 + 1])
        reg_pred, cls_pred = _flatten(predictions[level * 2], predictions[level * 2 + 1])
        anchors = tf.convert_to_tensor(anchors_np, dtype=tf.float32)
        true_boxes, true_keypoints = decode_regression(reg_true, anchors)
        pred_boxes, pred_keypoints = decode_regression(reg_pred, anchors)
        mask = tf.cast(cls_true > 0.5, tf.float32)
        mask_column = mask[..., None]
        box_difference_px = (pred_boxes - true_boxes) * pixel_scale_box
        box_huber_sum += tf.reduce_sum(huber_delta16(box_difference_px) * mask_column)
        positives = tf.reduce_sum(mask)
        box_coordinate_count += positives * 4.0
        giou_sum += tf.reduce_sum((1.0 - giou(pred_boxes, true_boxes)) * mask)
        positive_count += positives
        keypoint_difference_px = (pred_keypoints - true_keypoints) * pixel_scale_kp
        keypoint_huber_sum += tf.reduce_sum(huber_delta16(keypoint_difference_px) * mask_column[..., None])
        keypoint_coordinate_count += positives * float(NUM_POINTS * 2)
        level_focal_sum, level_focal_count = focal_sums(cls_true, cls_pred)
        focal_sum += level_focal_sum
        focal_count += level_focal_count

    box_pixel_huber_loss = tf.math.divide_no_nan(box_huber_sum, box_coordinate_count) / HUBER_DELTA_PX
    box_giou_loss = tf.math.divide_no_nan(giou_sum, positive_count)
    keypoint_pixel_huber_loss = tf.math.divide_no_nan(keypoint_huber_sum, keypoint_coordinate_count) / HUBER_DELTA_PX
    focal_loss = tf.math.divide_no_nan(focal_sum, focal_count)
    regression_loss = 1.5 * (0.5 * box_pixel_huber_loss + 0.5 * box_giou_loss) + 0.25 * keypoint_pixel_huber_loss
    total_loss = regression_loss + 0.7 * focal_loss
    return {
        "box_pixel_huber_loss": box_pixel_huber_loss,
        "box_giou_loss": box_giou_loss,
        "keypoint_pixel_huber_loss": keypoint_pixel_huber_loss,
        "focal_loss": focal_loss,
        "regression_loss": regression_loss,
        "total_loss": total_loss,
        "positive_anchors": positive_count,
    }
