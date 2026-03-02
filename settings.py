# By Abdullah As-Sadeed

import sys
from pathlib import Path

file_path = Path(__file__).resolve()

root_path = file_path.parent

if root_path not in sys.path:
    sys.path.append(str(root_path))

ROOT = root_path.relative_to(Path.cwd())

MODEL_DIRECTORY = ROOT / "Ultralytics_YOLO_Weights"

IMAGES_DIRECTORY = ROOT / "demo_files/images"
DEFAULT_IMAGE = IMAGES_DIRECTORY / "default.jpg"
DEFAULT_RESULT_IMAGE = IMAGES_DIRECTORY / "default_result.jpg"
