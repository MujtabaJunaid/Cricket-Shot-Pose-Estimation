#!/usr/bin/env python3

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import logging
import torch
import numpy as np
from backend.models.classifier import EnsembleClassifier
from backend.utils.pose_extractor import PoseExtractor
from backend.utils.shot_analyzer import ShotAnalyzer
from backend.config import SHOT_CLASSES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def predict_image(image_path: str, model_path: str, device: str = 'cuda'):
    logger.info(f"Predicting shot for image: {image_path}")
    
    pose_extractor = PoseExtractor()
    classifier = EnsembleClassifier(device=device)
    classifier.load_checkpoint(model_path)
    
    landmarks, _ = pose_extractor.extract_landmarks(image_path)
    
    if landmarks is None:
        logger.warning("No pose detected in image")
        return None
    
    normalized = ShotAnalyzer.normalize_landmarks(landmarks)
    prediction, confidence, probabilities = classifier.forward(normalized)
    
    shot_class = SHOT_CLASSES.get(int(prediction), "unknown")
    
    logger.info(f"Predicted shot: {shot_class} (confidence: {confidence:.4f})")
    
    pose_extractor.close()
    
    return {
        'shot': shot_class,
        'confidence': confidence,
        'probabilities': probabilities
    }

def predict_video(video_path: str, model_path: str, device: str = 'cuda'):
    logger.info(f"Predicting shots for video: {video_path}")
    
    pose_extractor = PoseExtractor()
    classifier = EnsembleClassifier(device=device)
    classifier.load_checkpoint(model_path)
    
    landmarks_sequence = pose_extractor.extract_from_video(video_path)
    
    if not landmarks_sequence:
        logger.warning("No poses detected in video")
        return None
    
    predictions = []
    for landmarks in landmarks_sequence:
        normalized = ShotAnalyzer.normalize_landmarks(landmarks)
        prediction, confidence, probabilities = classifier.forward(normalized)
        shot_class = SHOT_CLASSES.get(int(prediction), "unknown")
        predictions.append({
            'shot': shot_class,
            'confidence': float(confidence)
        })
    
    most_common = max(
        set(p['shot'] for p in predictions),
        key=lambda x: sum(1 for p in predictions if p['shot'] == x)
    )
    avg_confidence = np.mean([p['confidence'] for p in predictions])
    
    logger.info(f"Dominant shot: {most_common} (avg confidence: {avg_confidence:.4f})")
    logger.info(f"Processed {len(predictions)} frames")
    
    pose_extractor.close()
    
    return {
        'dominant_shot': most_common,
        'average_confidence': avg_confidence,
        'frames': len(predictions),
        'predictions': predictions
    }

def main():
    parser = argparse.ArgumentParser(description='Pose Prediction CLI')
    parser.add_argument('--image', type=str, help='Path to image file')
    parser.add_argument('--video', type=str, help='Path to video file')
    parser.add_argument('--model', type=str, default='models/checkpoints/best_model.pt', help='Model checkpoint path')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    
    args = parser.parse_args()
    
    if args.image:
        result = predict_image(args.image, args.model, args.device)
        if result:
            print(f"Shot: {result['shot']}, Confidence: {result['confidence']:.4f}")
    elif args.video:
        result = predict_video(args.video, args.model, args.device)
        if result:
            print(f"Dominant Shot: {result['dominant_shot']}")
            print(f"Average Confidence: {result['average_confidence']:.4f}")
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
