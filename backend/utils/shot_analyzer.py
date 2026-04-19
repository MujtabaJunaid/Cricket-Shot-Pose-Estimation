import numpy as np
from typing import Dict, Tuple
import math

class ShotAnalyzer:
    
    @staticmethod
    def calculate_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        a = np.array(p1)
        b = np.array(p2)
        c = np.array(p3)
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle)

    @staticmethod
    def extract_angle_features(landmarks: np.ndarray) -> np.ndarray:
        landmarks = landmarks.reshape(33, 3)
        
        angles = []
        
        left_shoulder = landmarks[11][:2]
        right_shoulder = landmarks[12][:2]
        left_elbow = landmarks[13][:2]
        right_elbow = landmarks[14][:2]
        left_wrist = landmarks[15][:2]
        right_wrist = landmarks[16][:2]
        left_hip = landmarks[23][:2]
        right_hip = landmarks[24][:2]
        left_knee = landmarks[25][:2]
        right_knee = landmarks[26][:2]
        left_ankle = landmarks[27][:2]
        right_ankle = landmarks[28][:2]
        
        left_arm_angle = ShotAnalyzer.calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_arm_angle = ShotAnalyzer.calculate_angle(right_shoulder, right_elbow, right_wrist)
        
        left_leg_angle = ShotAnalyzer.calculate_angle(left_hip, left_knee, left_ankle)
        right_leg_angle = ShotAnalyzer.calculate_angle(right_hip, right_knee, right_ankle)
        
        shoulder_angle = np.degrees(np.arctan2(
            right_shoulder[1] - left_shoulder[1],
            right_shoulder[0] - left_shoulder[0]
        ))
        
        hip_angle = np.degrees(np.arctan2(
            right_hip[1] - left_hip[1],
            right_hip[0] - left_hip[0]
        ))
        
        angles = np.array([
            left_arm_angle, right_arm_angle,
            left_leg_angle, right_leg_angle,
            shoulder_angle, hip_angle
        ])
        
        return angles

    @staticmethod
    def calculate_velocity(landmarks_seq: list) -> np.ndarray:
        if len(landmarks_seq) < 2:
            return np.zeros(6)
        
        velocities = []
        for i in range(1, len(landmarks_seq)):
            diff = landmarks_seq[i] - landmarks_seq[i-1]
            velocity = np.linalg.norm(diff)
            velocities.append(velocity)
        
        return np.array([np.mean(velocities), np.max(velocities), np.std(velocities), 0, 0, 0])

    @staticmethod
    def extract_temporal_features(landmarks_sequence: list) -> Dict[str, np.ndarray]:
        features = {
            'static': [],
            'dynamic': []
        }
        
        for landmarks in landmarks_sequence:
            angles = ShotAnalyzer.extract_angle_features(landmarks)
            features['static'].append(angles)
        
        if len(landmarks_sequence) > 1:
            velocity = ShotAnalyzer.calculate_velocity(landmarks_sequence)
            features['dynamic'] = velocity
        else:
            features['dynamic'] = np.zeros(6)
        
        features['static'] = np.array(features['static'])
        
        return features

    @staticmethod
    def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
        landmarks = landmarks.reshape(33, 3)
        
        nose = landmarks[0]
        shoulder_center = (landmarks[11] + landmarks[12]) / 2
        
        landmarks = landmarks - shoulder_center
        
        scale = np.linalg.norm(landmarks[11] - landmarks[12]) + 1e-6
        landmarks = landmarks / scale
        
        return landmarks.reshape(-1)
