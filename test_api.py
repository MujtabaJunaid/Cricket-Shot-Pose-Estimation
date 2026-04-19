#!/usr/bin/env python3

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import requests
import json
import logging
import argparse
from io import BytesIO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class APITester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def test_health(self):
        logger.info("Testing /health endpoint...")
        try:
            response = self.session.get(f"{self.base_url}/health")
            if response.status_code == 200:
                data = response.json()
                logger.info(f"Health status: {data['status']}")
                logger.info(f"Device: {data['device']}")
                logger.info(f"Model loaded: {data['model_loaded']}")
                return True
            else:
                logger.error(f"Health check failed: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Connection error: {e}")
            return False
    
    def test_classes(self):
        logger.info("Testing /classes endpoint...")
        try:
            response = self.session.get(f"{self.base_url}/classes")
            if response.status_code == 200:
                data = response.json()
                logger.info(f"Available classes: {len(data['shot_classes'])}")
                for class_id, class_name in data['shot_classes'].items():
                    logger.info(f"  {class_id}: {class_name}")
                return True
            else:
                logger.error(f"Failed to get classes: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Error: {e}")
            return False
    
    def test_model_info(self):
        logger.info("Testing /model/info endpoint...")
        try:
            response = self.session.get(f"{self.base_url}/model/info")
            if response.status_code == 200:
                data = response.json()
                logger.info(f"Device: {data['device']}")
                logger.info(f"Confidence threshold: {data['confidence_threshold']}")
                logger.info(f"Max upload size: {data['max_upload_size']} bytes")
                logger.info(f"Video formats: {', '.join(data['supported_video_formats'])}")
                return True
            else:
                logger.error(f"Failed to get model info: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Error: {e}")
            return False
    
    def test_image_prediction(self, image_path: str):
        logger.info(f"Testing /predict/image endpoint with {image_path}...")
        try:
            if not Path(image_path).exists():
                logger.error(f"Image file not found: {image_path}")
                return False
            
            with open(image_path, 'rb') as f:
                files = {'file': f}
                response = self.session.post(f"{self.base_url}/predict/image", files=files)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    logger.info(f"Prediction: {data['predicted_shot']}")
                    logger.info(f"Confidence: {data['confidence']:.4f}")
                    logger.info(f"Pose found: {data['pose_found']}")
                    return True
                else:
                    logger.warning(f"No pose detected: {data.get('message')}")
                    return True
            else:
                logger.error(f"Prediction failed: {response.status_code}")
                logger.error(response.text)
                return False
        except Exception as e:
            logger.error(f"Error: {e}")
            return False
    
    def test_video_prediction(self, video_path: str):
        logger.info(f"Testing /predict/video endpoint with {video_path}...")
        try:
            if not Path(video_path).exists():
                logger.error(f"Video file not found: {video_path}")
                return False
            
            with open(video_path, 'rb') as f:
                files = {'file': f}
                response = self.session.post(f"{self.base_url}/predict/video", files=files)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    logger.info(f"Prediction: {data['predicted_shot']}")
                    logger.info(f"Confidence: {data.get('confidence', data.get('average_confidence')):.4f}")
                    logger.info(f"Frames processed: {data['frames_processed']}")
                    return True
                else:
                    logger.warning(f"Analysis failed: {data.get('message')}")
                    return True
            else:
                logger.error(f"Prediction failed: {response.status_code}")
                logger.error(response.text)
                return False
        except Exception as e:
            logger.error(f"Error: {e}")
            return False
    
    def run_full_test(self, image_path: str = None, video_path: str = None):
        logger.info("\n" + "="*60)
        logger.info("RUNNING FULL API TEST SUITE")
        logger.info("="*60 + "\n")
        
        results = {
            'health': self.test_health(),
            'classes': self.test_classes(),
            'model_info': self.test_model_info(),
            'image_prediction': self.test_image_prediction(image_path) if image_path else None,
            'video_prediction': self.test_video_prediction(video_path) if video_path else None
        }
        
        logger.info("\n" + "="*60)
        logger.info("TEST RESULTS SUMMARY")
        logger.info("="*60)
        for test_name, result in results.items():
            if result is not None:
                status = "PASSED" if result else "FAILED"
                logger.info(f"{test_name}: {status}")
        logger.info("="*60 + "\n")
        
        return all(v for v in results.values() if v is not None)

def main():
    parser = argparse.ArgumentParser(description='API Testing Tool')
    parser.add_argument('--url', type=str, default='http://localhost:8000', help='API base URL')
    parser.add_argument('--image', type=str, help='Path to test image')
    parser.add_argument('--video', type=str, help='Path to test video')
    parser.add_argument('--test', choices=['health', 'classes', 'info', 'image', 'video', 'all'], 
                       default='all', help='Specific test to run')
    
    args = parser.parse_args()
    
    tester = APITester(args.url)
    
    if args.test == 'all':
        success = tester.run_full_test(args.image, args.video)
    elif args.test == 'health':
        success = tester.test_health()
    elif args.test == 'classes':
        success = tester.test_classes()
    elif args.test == 'info':
        success = tester.test_model_info()
    elif args.test == 'image':
        success = tester.test_image_prediction(args.image) if args.image else False
    elif args.test == 'video':
        success = tester.test_video_prediction(args.video) if args.video else False
    
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
