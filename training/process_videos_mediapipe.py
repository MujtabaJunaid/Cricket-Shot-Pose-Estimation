"""
Extract MediaPipe poses from cricket shot videos and save as numpy arrays for training.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# MediaPipe setup
POSE_MODEL_PATH = Path(__file__).parent.parent / "backend" / "models" / "pose_landmarker_lite.task"

SHOT_CLASSES = {
    'cover': 0, 'defense': 1, 'flick': 2, 'hook': 3, 
    'late_cut': 4, 'lofted': 5, 'pull': 6, 
    'square_cut': 7, 'straight': 8, 'sweep': 9
}

class VideoProcessor:
    def __init__(self, model_path: str):
        logger.info("Initializing MediaPipe PoseLandmarker...")
        
        base_options = python.BaseOptions(model_asset_path=str(model_path))
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=False
        )
        self.pose_detector = vision.PoseLandmarker.create_from_options(options)
        logger.info("✓ PoseLandmarker initialized")

    def extract_poses_from_video(self, video_path: str, max_frames: int = 30) -> np.ndarray:
        """Extract pose landmarks from video frames."""
        cap = cv2.VideoCapture(video_path)
        poses = []
        frame_count = 0

        while frame_count < max_frames and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Create MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            
            # Detect poses
            detection_result = self.pose_detector.detect(mp_image)
            
            # Extract landmarks
            if detection_result.pose_landmarks:
                landmarks = detection_result.pose_landmarks[0]
                pose_array = []
                
                for landmark in landmarks:
                    pose_array.extend([landmark.x, landmark.y, landmark.z])
                
                poses.append(np.array(pose_array))
            else:
                # Skip frames without detection
                logger.debug(f"No pose detected in frame {frame_count}")
                continue

            frame_count += 1

        cap.release()
        
        if poses:
            return np.array(poses)
        else:
            return np.array([])

    def process_dataset(self, input_dir: str, output_dir: str):
        """Process all videos in dataset."""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for shot_class, class_id in SHOT_CLASSES.items():
            class_dir = input_path / shot_class
            output_class_dir = output_path / shot_class
            
            if not class_dir.exists():
                logger.warning(f"Class directory not found: {class_dir}")
                continue

            output_class_dir.mkdir(exist_ok=True)
            
            # Get all AVI files
            video_files = sorted([f for f in class_dir.glob("*.avi") if not f.name.startswith("._")])
            logger.info(f"\nProcessing {shot_class}: {len(video_files)} videos")

            for idx, video_file in enumerate(tqdm(video_files, desc=f"Processing {shot_class}")):
                try:
                    poses = self.extract_poses_from_video(str(video_file))
                    
                    if poses.size > 0:
                        # Save as numpy array
                        output_file = output_class_dir / f"{shot_class}_{idx:04d}.npy"
                        np.save(output_file, poses)
                    else:
                        logger.warning(f"No poses extracted from {video_file.name}")
                
                except Exception as e:
                    logger.error(f"Error processing {video_file.name}: {e}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Process cricket videos with MediaPipe poses')
    parser.add_argument('--input-dir', default=r'C:\Users\hp\Desktop\cricketshot\train', 
                       help='Input directory with video files')
    parser.add_argument('--output-dir', default='data/processed', 
                       help='Output directory for processed data')
    parser.add_argument('--model-path', default=str(POSE_MODEL_PATH),
                       help='Path to MediaPipe pose model')
    
    args = parser.parse_args()
    
    if not Path(args.model_path).exists():
        logger.error(f"Model file not found: {args.model_path}")
        return
    
    processor = VideoProcessor(args.model_path)
    processor.process_dataset(args.input_dir, args.output_dir)
    
    logger.info("\n✓ Dataset processing complete!")
    logger.info(f"Processed data saved to: {args.output_dir}")

if __name__ == '__main__':
    main()
