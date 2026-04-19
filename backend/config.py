import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

POSE_CONFIDENCE_THRESHOLD = 0.5
POSE_TRACKING_CONFIDENCE = 0.5

MODEL_PATH = BASE_DIR / "models" / "checkpoints"
UPLOAD_FOLDER = BASE_DIR / "uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)

SHOT_CLASSES = {
    0: "drive",
    1: "pull",
    2: "cut",
    3: "defense",
    4: "backfoot_play",
    5: "cover_drive",
    6: "off_drive",
    7: "on_drive",
    8: "lofted_shot"
}

POSE_LANDMARKS_COUNT = 33
FEATURES_SIZE = 33 * 3

MAX_UPLOAD_SIZE = 50 * 1024 * 1024
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2

FRAME_SKIP = 2
FPS_TARGET = 30

DEVICE = "cuda" if os.environ.get("DEVICE", "cuda") else "cpu"
