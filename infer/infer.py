from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

try:
    import cv2
except ImportError as exc:
    raise SystemExit(
        "缺少 OpenCV。请先运行：python -m pip install opencv-python"
    ) from exc

# Ignore non-fatal TIFF metadata warnings while keeping OpenCV errors visible.
cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)

import numpy as np

from anchor_utils import center_to_corners, decode_outputs
from model import palm_detection_model


BASE = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = BASE / "images_original"
DEFAULT_OUTPUT_DIR = BASE / "images_palm"
DEFAULT_WEIGHTS = BASE / "best_model.weights.h5"
DEFAULT_CONFIG = BASE / "recommended_inference.json"

SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def verify_runtime_files(weights: Path) -> None:
    required = {
        "best_model.weights.h5": weights,
        "anchor_config.json": BASE / "anchor_config.json",
        "model.py": BASE / "model.py",
        "anchor_utils.py": BASE / "anchor_utils.py",
    }
    for path in required.values():
        if not path.is_file():
            raise FileNotFoundError(f"缺少推理文件：{path}")


def load_inference_config(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"缺少推理配置：{path}")
    config = json.loads(path.read_text(encoding="utf-8-sig"))
    expected = {
        "input_nchw": [1, 1, 224, 384],
        "grayscale": True,
        "normalization": "[0,1]",
        "anchors": 840,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(
                f"推理配置不符合第六次训练约定：{key}={config.get(key)!r}，期望 {value!r}"
            )
    return config


def decode_image(path: Path) -> tuple[np.ndarray, np.ndarray, str]:
    encoded = np.fromfile(path, dtype=np.uint8)
    gray = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    color = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if gray is None or color is None:
        raise ValueError(f"无法读取图片：{path}")

    height, width = gray.shape[:2]
    if (width, height) == (1280, 720):
        orientation = "1280x720_no_rotation"
    elif (width, height) == (720, 1280):
        gray = cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)
        color = cv2.rotate(color, cv2.ROTATE_90_CLOCKWISE)
        orientation = "720x1280_rotate_clockwise_90"
    else:
        raise ValueError(
            f"不支持的图片尺寸 {width}x{height}：{path}。"
            "只接受 1280x720 或 720x1280。"
        )

    # 与服务器 train_sixth.load_image 完全一致：灰度、INTER_AREA、[0,1]、NCHW。
    resized = cv2.resize(gray, (384, 224), interpolation=cv2.INTER_AREA)
    model_input = resized.astype(np.float32) / 255.0
    return model_input[None, :, :], color, orientation


def draw_predictions(image: np.ndarray, decoded: dict) -> np.ndarray:
    canvas = image.copy()
    height, width = canvas.shape[:2]
    scale = np.asarray([width, height, width, height], dtype=np.float32)
    corners = center_to_corners(decoded["boxes"])
    bbox_color = (0, 255, 0)
    keypoint_color = (0, 0, 255)

    for box, keypoints, score in zip(
        corners, decoded["keypoints"], decoded["scores"]
    ):
        x1, y1, x2, y2 = np.rint(box * scale).astype(int)
        x1, x2 = np.clip([x1, x2], 0, width - 1)
        y1, y2 = np.clip([y1, y2], 0, height - 1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), bbox_color, 3)
        cv2.putText(
            canvas,
            f"{float(score):.3f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            bbox_color,
            2,
        )
        for x, y in keypoints:
            px = int(np.clip(round(float(x) * width), 0, width - 1))
            py = int(np.clip(round(float(y) * height), 0, height - 1))
            cv2.circle(canvas, (px, py), 5, keypoint_color, -1)
    return canvas


def write_jpeg(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not success:
        raise IOError(f"图片编码失败：{path}")
    encoded.tofile(path)


def collect_images(input_dir: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in input_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
        ),
        key=lambda path: str(path.relative_to(input_dir)).casefold(),
    )


def load_weights_compatibly(model, weights: Path) -> str:
    """Load TF 2.10 legacy HDF5 weights under both Keras 2 and Keras 3."""
    try:
        model.load_weights(weights)
        return "native_keras_load_weights"
    except ValueError as native_error:
        try:
            import h5py
            from keras.src.legacy.saving import legacy_h5_format

            with h5py.File(weights, "r") as stream:
                legacy_h5_format.load_weights_from_hdf5_group(stream, model)
            return "keras3_legacy_hdf5_topology_loader"
        except Exception as legacy_error:
            raise RuntimeError(
                "权重加载失败。该权重由服务器 TensorFlow 2.10 保存；"
                "本地 Keras 原生加载和旧版 HDF5 兼容加载均失败。\n"
                f"原生加载错误：{native_error}\n"
                f"兼容加载错误：{legacy_error}"
            ) from legacy_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="第六次 Palm 模型的本地批量推理与可视化脚本"
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--confidence", type=float, default=None)
    parser.add_argument("--nms", type=float, default=None)
    parser.add_argument("--max-detections", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    weights = args.weights.resolve()
    config_path = args.config.resolve()

    if not input_dir.is_dir():
        raise FileNotFoundError(f"输入文件夹不存在：{input_dir}")
    if input_dir == output_dir:
        raise ValueError("输入和输出文件夹不能相同，以免覆盖原始图片。")

    verify_runtime_files(weights)
    config = load_inference_config(config_path)
    confidence = float(
        config["confidence_threshold"] if args.confidence is None else args.confidence
    )
    nms = float(config["nms_iou_threshold"] if args.nms is None else args.nms)
    max_detections = int(
        config["max_detections"]
        if args.max_detections is None
        else args.max_detections
    )
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence 必须在 [0,1] 内。")
    if not 0.0 <= nms <= 1.0:
        raise ValueError("nms 必须在 [0,1] 内。")
    if max_detections <= 0:
        raise ValueError("max-detections 必须为正整数。")

    images = collect_images(input_dir)
    if not images:
        print(f"输入文件夹中没有支持的图片：{input_dir}")
        return 0

    model = palm_detection_model()
    weight_loader = load_weights_compatibly(model, weights)
    if model.count_params() != 1_367_620:
        raise RuntimeError(
            f"模型参数量异常：{model.count_params()}，期望 1367620。"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    total_detections = 0
    started = time.perf_counter()

    for index, source in enumerate(images, start=1):
        model_input, oriented_color, orientation = decode_image(source)
        outputs = [
            tensor.numpy()[0]
            for tensor in model(model_input[None, :, :, :], training=False)
        ]
        decoded = decode_outputs(outputs, confidence, nms, max_detections)
        rendered = draw_predictions(oriented_color, decoded)

        relative = source.relative_to(input_dir)
        destination = (output_dir / relative).with_suffix(".jpg")
        write_jpeg(destination, rendered)

        detections = []
        for box, keypoints, score, level in zip(
            decoded["boxes"],
            decoded["keypoints"],
            decoded["scores"],
            decoded["levels"],
        ):
            detections.append(
                {
                    "score": float(score),
                    "feature_level": "14x24" if int(level) == 0 else "7x12",
                    "bbox_center_normalized": [float(value) for value in box],
                    "keypoints_normalized": [
                        [float(x), float(y)] for x, y in keypoints
                    ],
                }
            )
        total_detections += len(detections)
        manifest.append(
            {
                "input": str(source),
                "output": str(destination),
                "orientation": orientation,
                "detections": detections,
            }
        )
        print(
            f"[{index}/{len(images)}] {relative} -> "
            f"{len(detections)} detection(s)"
        )

    elapsed = time.perf_counter() - started
    result = {
        "model": str(weights),
        "weight_loader": weight_loader,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "confidence_threshold": confidence,
        "nms_iou_threshold": nms,
        "max_detections": max_detections,
        "images": len(images),
        "detections": total_detections,
        "elapsed_seconds": elapsed,
        "items": manifest,
    }
    manifest_path = output_dir / "inference_results.json"
    manifest_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(
        f"完成：{len(images)} 张图片，{total_detections} 个检测，"
        f"耗时 {elapsed:.2f} 秒。"
    )
    print(f"输出目录：{output_dir}")
    print(f"结果清单：{manifest_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"错误：{exc}", file=sys.stderr)
        raise SystemExit(1) from exc
