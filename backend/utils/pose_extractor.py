import cv2
import numpy as np
from typing import Tuple, List, Optional, Dict
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import MediaPipe
MEDIAPIPE_AVAILABLE = False
POSE_MODEL_PATH = Path(__file__).parent.parent / "models" / "pose_landmarker.tflite"

try:
    # MediaPipe 0.10+ uses new tasks API
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions
    from mediapipe import Image, ImageFormat
    MEDIAPIPE_AVAILABLE = True
    logger.info("✓ MediaPipe API available")
except ImportError as e:
    logger.warning(f"MediaPipe not available: {e}")
    MEDIAPIPE_AVAILABLE = False

class PoseExtractor:
    def __init__(self, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        self.pose_detector = None
        self.using_fallback = False
        
        if not MEDIAPIPE_AVAILABLE:
            logger.warning("⚠ MediaPipe not installed - using random landmarks")
            self.using_fallback = True
            return
        
        # Check if model exists
        if not POSE_MODEL_PATH.exists():
            logger.warning(f"⚠ Pose model not found at {POSE_MODEL_PATH}")
            logger.info("  To enable real pose detection, download from:")
            logger.info("  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite.tflite")
            logger.info("  And save to: backend/models/pose_landmarker.tflite")
            logger.info("  Using random landmarks for now...")
            self.using_fallback = True
            return
        
        try:
            # Initialize with the downloaded model
            base_options = BaseOptions(model_asset_path=str(POSE_MODEL_PATH))
            options = PoseLandmarkerOptions(
                base_options=base_options,
                output_segmentation_masks=False
            )
            self.pose_detector = PoseLandmarker.create_from_options(options)
            logger.info("✓ MediaPipe PoseLandmarker initialized successfully")
        except Exception as e:
            logger.warning(f"Could not initialize MediaPipe: {e}")
            logger.info("Using random landmarks...")
            self.using_fallback = True
            self.pose_detector = None

    def extract_landmarks(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
        """Extract pose landmarks from a frame.
        
        Returns:
            (landmarks_array, landmark_info) - landmarks_array is shape (99,) or None
        """
        if self.using_fallback or self.pose_detector is None:
            # Fallback: return random landmarks (simulates pose detection)
            return np.random.randn(99).astype(np.float32), None
        
        try:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Create MediaPipe Image
            mp_image = Image(image_format=ImageFormat.SRGB, data=frame_rgb)
            
            # Run detection
            detection_result = self.pose_detector.detect(mp_image)
            
            # Check if pose was detected
            if not detection_result.pose_landmarks or len(detection_result.pose_landmarks) == 0:
                return None, None
            
            # Extract landmarks (33 keypoints × 3 coordinates = 99 values)
            landmarks_list = []
            for landmark in detection_result.pose_landmarks[0]:
                landmarks_list.append(landmark.x)
                landmarks_list.append(landmark.y)
                landmarks_list.append(landmark.z)
            
            landmarks_array = np.array(landmarks_list, dtype=np.float32)
            
            return landmarks_array, {
                'pose_landmarks': detection_result.pose_landmarks[0],
                'landmarks': landmarks_array
            }
            
        except Exception as e:
            logger.error(f"Pose extraction error: {e}")
            return None, None

    def extract_from_video(self, video_path: str, frame_skip: int = 2) -> List[np.ndarray]:
        cap = cv2.VideoCapture(video_path)
        landmarks_sequence = []
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_skip == 0:
                landmarks, _ = self.extract_landmarks(frame)
                if landmarks is not None:
                    landmarks_sequence.append(landmarks)
            
            frame_count += 1
        
        cap.release()
        return landmarks_sequence

    def extract_from_image(self, image_path: str) -> Optional[np.ndarray]:
        frame = cv2.imread(image_path)
        if frame is None:
            logger.error(f"Cannot read image: {image_path}")
            return None
        
        landmarks, _ = self.extract_landmarks(frame)
        return landmarks

    def draw_landmarks(self, frame: np.ndarray, landmark_info: Dict) -> np.ndarray:
        if landmark_info is None:
            return frame
        
        try:
            from mediapipe.tasks.python.vision import pose_landmarker
            from mediapipe.framework.formats import landmark_pb2
            
            # Draw circles at each landmark
            landmarks = landmark_info['pose_landmarks']
            h, w, _ = frame.shape
            
            for lm in landmarks:
                x = int(lm.x * w)
                y = int(lm.y * h)
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
            
            # Draw skeleton connections
            POSE_CONNECTIONS = [
                (0, 1), (1, 2), (2, 3), (3, 7),
                (0, 4), (4, 5), (5, 6), (6, 8),
                (9, 10), (11, 12), (11, 13), (13, 15),
                (12, 14), (14, 16), (11, 23), (12, 24),
                (23, 24), (23, 25), (24, 26), (25, 27),
                (26, 28), (27, 29), (28, 30), (29, 31),
                (30, 32), (27, 31), (28, 32)
            ]
            
            for connection in POSE_CONNECTIONS:
                start_lm = landmarks[connection[0]]
                end_lm = landmarks[connection[1]]
                start_pos = (int(start_lm.x * w), int(start_lm.y * h))
                end_pos = (int(end_lm.x * w), int(end_lm.y * h))
                cv2.line(frame, start_pos, end_pos, (0, 255, 0), 2)
        except Exception as e:
            logger.warning(f"Could not draw landmarks: {e}")
        
        return frame

    def get_landmark_names(self) -> List[str]:
        # MediaPipe landmark names
        return [
            "NOSE", "LEFT_EYE_INNER", "LEFT_EYE", "LEFT_EYE_OUTER",
            "RIGHT_EYE_INNER", "RIGHT_EYE", "RIGHT_EYE_OUTER", "LEFT_EAR",
            "RIGHT_EAR", "MOUTH_LEFT", "MOUTH_RIGHT", "LEFT_SHOULDER",
            "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW", "LEFT_WRIST",
            "RIGHT_WRIST", "LEFT_PINKY", "RIGHT_PINKY", "LEFT_INDEX",
            "RIGHT_INDEX", "LEFT_THUMB", "RIGHT_THUMB", "LEFT_HIP",
            "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE",
            "RIGHT_ANKLE", "LEFT_HEEL", "RIGHT_HEEL", "LEFT_FOOT_INDEX",
            "RIGHT_FOOT_INDEX"
        ]

    def close(self):
        if self.pose_detector is not None:
            try:
                # New API doesn't require explicit close, but being safe
                pass
            except:
                pass
