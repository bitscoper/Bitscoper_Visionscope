#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# By Abdullah As-Sadeed

import sys
from pathlib import Path

file_path = Path(__file__).resolve()
root_path = file_path.parent

if root_path not in sys.path:
    sys.path.append(str(root_path))

ROOT = root_path.relative_to(Path.cwd())

MODEL_DIRECTORY = ROOT / "Ultralytics_YOLO_Weights"

DEFAULT_WEBCAM_NUMBER = 0
DEFAULT_VIDEO_WIDTH = 640
DEFAULT_VIDEO_HEIGHT = 480

DEFAULT_CONFIDENCE = 0.40
