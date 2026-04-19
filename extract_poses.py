#!/usr/bin/env python3

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import logging
import cv2
import numpy as np
from backend.utils.pose_extractor import PoseExtractor
from backend.utils.shot_analyzer import ShotAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_poses(video_path: str, output_dir: str, shot_class: str):
    logger.info(f"Extracting poses from: {video_path}")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    pose_extractor = PoseExtractor()
    landmarks_sequence = pose_extractor.extract_from_video(video_path)
    
    if not landmarks_sequence:
        logger.warning("No poses detected in video")
        return 0
    
    landmarks_array = np.array(landmarks_sequence)
    output_path = Path(output_dir) / shot_class / f"{Path(video_path).stem}.npy"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    np.save(output_path, landmarks_array)
    logger.info(f"Saved {len(landmarks_sequence)} frames to {output_path}")
    
    pose_extractor.close()
    return len(landmarks_sequence)

def batch_extract(data_dir: str, output_dir: str):
    logger.info(f"Batch extracting poses from: {data_dir}")
    
    data_path = Path(data_dir)
    total_frames = 0
    
    for shot_dir in data_path.iterdir():
        if not shot_dir.is_dir():
            continue
        
        shot_class = shot_dir.name
        logger.info(f"Processing {shot_class}...")
        
        for video_file in shot_dir.glob("*.mp4"):
            frames = extract_poses(str(video_file), output_dir, shot_class)
            total_frames += frames
    
    logger.info(f"Total frames extracted: {total_frames}")

def main():
    parser = argparse.ArgumentParser(description='Pose Extraction Utility')
    parser.add_argument('--video', type=str, help='Path to video file')
    parser.add_argument('--output-dir', type=str, default='data/processed', help='Output directory')
    parser.add_argument('--shot-class', type=str, help='Shot class name')
    parser.add_argument('--batch', type=str, help='Batch process directory')
    
    args = parser.parse_args()
    
    if args.batch:
        batch_extract(args.batch, args.output_dir)
    elif args.video and args.shot_class:
        extract_poses(args.video, args.output_dir, args.shot_class)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
