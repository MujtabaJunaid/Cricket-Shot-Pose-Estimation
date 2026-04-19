#!/usr/bin/env python3

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import cv2
import logging
from backend.utils.pose_extractor import PoseExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def annotate_video(video_path: str, output_path: str):
    logger.info(f"Annotating video: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    pose_extractor = PoseExtractor()
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        landmarks, landmark_info = pose_extractor.extract_landmarks(frame)
        
        if landmark_info:
            frame = pose_extractor.draw_landmarks(frame, landmark_info)
        
        out.write(frame)
        frame_count += 1
        
        if frame_count % 30 == 0:
            logger.info(f"Processed {frame_count} frames")
    
    cap.release()
    out.release()
    pose_extractor.close()
    
    logger.info(f"Annotation completed. Output: {output_path}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Video Annotation Tool')
    parser.add_argument('--input', type=str, required=True, help='Input video path')
    parser.add_argument('--output', type=str, required=True, help='Output video path')
    
    args = parser.parse_args()
    annotate_video(args.input, args.output)

if __name__ == '__main__':
    main()
