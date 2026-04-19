import cv2
import mediapipe as mp
import numpy as np
from typing import Tuple, List, Optional, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PoseExtractor:
    def __init__(self, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils

    def extract_landmarks(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        
        if not results.pose_landmarks:
            return None, None
        
        landmarks = []
        for lm in results.pose_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
        
        landmark_info = {
            'landmarks': np.array(landmarks),
            'pose_landmarks': results.pose_landmarks,
            'world_landmarks': results.world_landmarks
        }
        
        return np.array(landmarks), landmark_info

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
        
        self.mp_draw.draw_landmarks(
            frame,
            landmark_info['pose_landmarks'],
            self.mp_pose.POSE_CONNECTIONS
        )
        return frame

    def get_landmark_names(self) -> List[str]:
        return [lm.name for lm in self.mp_pose.PoseLandmark]

    def close(self):
        self.pose.close()
