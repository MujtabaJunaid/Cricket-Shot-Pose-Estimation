import os
from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parent.parent

POSE_CONFIDENCE_THRESHOLD = 0.5
POSE_TRACKING_CONFIDENCE = 0.5

MODEL_PATH = BASE_DIR / "models" / "checkpoints"
UPLOAD_FOLDER = BASE_DIR / "uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)

SHOT_CLASSES = {
    'cover': 0, 'defense': 1, 'flick': 2, 'hook': 3, 
    'late_cut': 4, 'lofted': 5, 'pull': 6, 
    'square_cut': 7, 'straight': 8, 'sweep': 9
}

REVERSE_SHOT_CLASSES = {v: k for k, v in SHOT_CLASSES.items()}

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

DEVICE = "cuda" if torch.cuda.is_available() and os.environ.get("DEVICE", "cpu").lower() != "cpu" else "cpu"

HUGGINGFACE_DATASET = "rokmr/cricket-shot"
USE_HUGGINGFACE = os.environ.get("USE_HUGGINGFACE", "false").lower() == "true"
HUGGINGFACE_CACHE_DIR = BASE_DIR / "hf_cache"

