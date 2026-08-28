from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
EXTERNAL_ROOT = PROJECT_ROOT / "external"
IMAGE_ROOT = EXTERNAL_ROOT / "data/images"
ANNOTATION_ROOT = EXTERNAL_ROOT / "data/annotations"
SPLIT_ROOT = EXTERNAL_ROOT / "data/splits/seventh_train"
RUN_ROOT = PROJECT_ROOT / "train/runs/seventh_train"
# BASE is retained as the runtime-output root for the existing training code.
BASE = RUN_ROOT
DATASETS = ("dragon", "peak", "soar")
SEED = 17

XML_SHA256 = {
    "dragon": "cb03f78f71f81acdf4399be4e4694f2309e99cd789f63fecb6d22fdb39e30865",
    "peak": "b610297f39996110d032a25a74fceb60d26abe8ddbb7ef0c9306599e1996f77b",
    "soar": "d8b901371dbd6a44c6c0eadda718bbfe04b43b1604d27e60d53af3c3dcd07f3d",
}
EXPECTED_RETAINED = {
    "dragon": {"images": 17_409, "bboxes": 34_109},
    "peak": {"images": 16_567, "bboxes": 32_746},
    "soar": {"images": 14_400, "bboxes": 28_515},
}

SIXTH_WEIGHTS = SCRIPT_DIR / "best_model_six_train.weights.h5"
SIXTH_WEIGHTS_SHA256 = "cdfa2a48d683378497a92aaf7cf1b6b459a61d0070f83bbdc7faf0fd07e7e6c8"

INPUT_WIDTH = 384
INPUT_HEIGHT = 224
CORRECTED_WIDTH = 1280
CORRECTED_HEIGHT = 720
ATSS_TOP_K = 9

HARD_CONFIDENCE = 0.25
HARD_NMS_IOU = 0.10
MAX_DETECTIONS = 2
MATCH_IOU = 0.50
