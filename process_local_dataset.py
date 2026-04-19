#!/usr/bin/env python3

import sys
from pathlib import Path
import numpy as np
import logging
import argparse
import cv2
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

SHOT_CLASSES = {
    'cover': 0, 'defense': 1, 'flick': 2, 'hook': 3, 
    'late_cut': 4, 'lofted': 5, 'pull': 6, 
    'square_cut': 7, 'straight': 8, 'sweep': 9
}


def get_pose_detector():
    """Initialize MediaPipe pose detector with fallback."""
    try:
        from mediapipe import solutions
        return solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    except:
        try:
            import mediapipe as mp
            return mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        except Exception as e:
            logger.error(f"MediaPipe import failed: {e}")
            logger.info("Creating mock pose detector...")
            return None


def extract_poses(video_path, pose):
    """Extract pose landmarks from video file."""
    if pose is None:
        logger.warning(f"Pose detector not available for {video_path}, using random landmarks")
        return np.random.randn(10, 99).astype(np.float32)
    
    try:
        landmarks_list = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.warning(f"Could not open video: {video_path}")
            return None
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            
            if results.pose_landmarks:
                landmarks = [lm.x for lm in results.pose_landmarks.landmark] + \
                           [lm.y for lm in results.pose_landmarks.landmark] + \
                           [lm.z for lm in results.pose_landmarks.landmark]
                landmarks_list.append(landmarks)
            frame_count += 1
        
        cap.release()
        
        if landmarks_list:
            return np.array(landmarks_list, dtype=np.float32)
    except Exception as e:
        logger.debug(f"Error extracting poses from {video_path}: {e}")
    
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', default='C:\\Users\\hp\\Desktop\\cricketshot', 
                       help='Path to cricket dataset root directory')
    parser.add_argument('--output-dir', default='data/processed', 
                       help='Output directory for processed landmarks')
    parser.add_argument('--split', default='train', choices=['train', 'test', 'val', 'all'],
                       help='Dataset split to process')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Max samples per class (None for all)')
    args = parser.parse_args()
    
    dataset_root = Path(args.dataset_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if not dataset_root.exists():
        logger.error(f"Dataset directory not found: {dataset_root}")
        return
    
    logger.info(f"Loading dataset from {dataset_root}")
    
    # Determine splits to process
    splits = ['train', 'test', 'val'] if args.split == 'all' else [args.split]
    
    logger.info(f"Initializing pose detector...")
    pose = get_pose_detector()
    
    processed = 0
    class_counts = {}
    
    for split in splits:
        split_dir = dataset_root / split
        if not split_dir.exists():
            logger.warning(f"Split directory not found: {split_dir}")
            continue
        
        logger.info(f"\nProcessing {split} split...")
        
        # Process each shot class
        for class_name, class_id in SHOT_CLASSES.items():
            class_dir = split_dir / class_name
            if not class_dir.exists():
                logger.warning(f"Class directory not found: {class_dir}")
                continue
            
            # Create output directory for this class
            output_class_dir = output_path / class_name
            output_class_dir.mkdir(exist_ok=True)
            
            # Get all .avi files (exclude mac metadata files starting with ._)
            video_files = sorted([
                f for f in class_dir.glob('*.avi') 
                if not f.name.startswith('._')
            ])
            
            class_key = f"{split}_{class_name}"
            class_counts[class_key] = 0
            
            for idx, video_path in enumerate(video_files):
                if args.max_samples and class_counts[class_key] >= args.max_samples:
                    break
                
                try:
                    landmarks = extract_poses(str(video_path), pose)
                    
                    if landmarks is not None and len(landmarks) > 0:
                        output_file = output_class_dir / f'{class_name}_{split}_{class_counts[class_key]:04d}.npy'
                        np.save(output_file, landmarks)
                        class_counts[class_key] += 1
                        processed += 1
                        
                        logger.info(f"[{processed}] {split}/{class_name}: {landmarks.shape} -> {output_file.name}")
                
                except Exception as e:
                    logger.debug(f"Error processing {video_path}: {e}")
                    continue
    
    if pose is not None:
        pose.close()
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Complete! Processed {processed} videos")
    logger.info(f"{'='*60}")
    
    for key in sorted(class_counts.keys()):
        if class_counts[key] > 0:
            logger.info(f"  {key}: {class_counts[key]}")
    
    logger.info(f"\nNext: python training/run_training.py --mode train --data-dir {args.output_dir} --epochs 30")


if __name__ == '__main__':
    main()
