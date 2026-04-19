from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
import torch
from pathlib import Path
import logging
import io
import uuid
from typing import List, Dict

from backend.config import (
    UPLOAD_FOLDER, MODEL_PATH, SHOT_CLASSES, ALLOWED_VIDEO_EXTENSIONS,
    ALLOWED_IMAGE_EXTENSIONS, MAX_UPLOAD_SIZE, DEVICE, POSE_CONFIDENCE_THRESHOLD,
    POSE_TRACKING_CONFIDENCE
)
from backend.utils.pose_extractor import PoseExtractor
from backend.utils.shot_analyzer import ShotAnalyzer
from backend.models.classifier import EnsembleClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Cricket Pose Classification API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pose_extractor = None
classifier = None
inference_cache = {}

@app.on_event("startup")
async def startup():
    global pose_extractor, classifier
    
    pose_extractor = PoseExtractor(
        min_detection_confidence=POSE_CONFIDENCE_THRESHOLD,
        min_tracking_confidence=POSE_TRACKING_CONFIDENCE
    )
    
    classifier = EnsembleClassifier(num_classes=len(SHOT_CLASSES), device=DEVICE)
    
    checkpoint = MODEL_PATH / "best_model.pt"
    if checkpoint.exists():
        classifier.load_checkpoint(str(checkpoint))
        logger.info("Model loaded successfully")
    else:
        logger.warning("No trained model found. Model inference will use untrained weights.")

@app.on_event("shutdown")
async def shutdown():
    global pose_extractor
    if pose_extractor:
        pose_extractor.close()

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "device": DEVICE,
        "model_loaded": classifier is not None,
        "shot_classes": len(SHOT_CLASSES)
    }

@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Invalid image format")
    
    content = await file.read()
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail="File too large")
    
    try:
        nparr = np.frombuffer(content, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image")
        
        landmarks, landmark_info = pose_extractor.extract_landmarks(frame)
        
        if landmarks is None:
            return {
                "success": False,
                "message": "No pose detected in image",
                "pose_found": False
            }
        
        normalized_landmarks = ShotAnalyzer.normalize_landmarks(landmarks)
        
        with torch.no_grad():
            prediction, confidence, probabilities = classifier.forward(normalized_landmarks)
        
        shot_class = SHOT_CLASSES.get(int(prediction), "unknown")
        
        angle_features = ShotAnalyzer.extract_angle_features(landmarks)
        
        return {
            "success": True,
            "pose_found": True,
            "predicted_shot": shot_class,
            "confidence": float(confidence),
            "all_predictions": {
                SHOT_CLASSES[i]: float(probabilities[i])
                for i in range(len(SHOT_CLASSES))
            },
            "angle_features": {
                "left_arm_angle": float(angle_features[0]),
                "right_arm_angle": float(angle_features[1]),
                "left_leg_angle": float(angle_features[2]),
                "right_leg_angle": float(angle_features[3]),
                "shoulder_angle": float(angle_features[4]),
                "hip_angle": float(angle_features[5])
            }
        }
    
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/predict/video")
async def predict_video(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Invalid video format")
    
    content = await file.read()
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail="File too large")
    
    try:
        video_path = UPLOAD_FOLDER / f"{uuid.uuid4()}{file_ext}"
        video_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(video_path, 'wb') as f:
            f.write(content)
        
        landmarks_sequence = pose_extractor.extract_from_video(str(video_path))
        
        if not landmarks_sequence:
            return {
                "success": False,
                "message": "No pose detected in video",
                "frames_processed": 0
            }
        
        landmarks_array = np.array(landmarks_sequence)
        
        temporal_features = ShotAnalyzer.extract_temporal_features(landmarks_sequence)
        
        if landmarks_array.shape[0] >= 10:
            normalized_landmarks = ShotAnalyzer.normalize_landmarks(landmarks_array[-1])
            landmarks_tensor = torch.FloatTensor(normalized_landmarks).unsqueeze(0)
        else:
            landmarks_tensor = torch.FloatTensor(landmarks_array[0]).unsqueeze(0)
        
        with torch.no_grad():
            prediction, confidence, probabilities = classifier.forward(landmarks_tensor.numpy())
        
        shot_class = SHOT_CLASSES.get(int(prediction), "unknown")
        
        video_path.unlink()
        
        return {
            "success": True,
            "frames_processed": len(landmarks_sequence),
            "predicted_shot": shot_class,
            "confidence": float(confidence),
            "all_predictions": {
                SHOT_CLASSES[i]: float(probabilities[i])
                for i in range(len(SHOT_CLASSES))
            },
            "temporal_features": {
                "sequence_length": len(landmarks_sequence)
            }
        }
    
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")

@app.post("/predict/stream")
async def predict_stream(frames: List[dict]):
    if not frames:
        raise HTTPException(status_code=400, detail="No frames provided")
    
    try:
        all_predictions = []
        
        for frame_data in frames:
            frame_bytes = np.frombuffer(bytes(frame_data['data']), np.uint8)
            frame = cv2.imdecode(frame_bytes, cv2.IMREAD_COLOR)
            
            if frame is None:
                continue
            
            landmarks, _ = pose_extractor.extract_landmarks(frame)
            
            if landmarks is not None:
                normalized = ShotAnalyzer.normalize_landmarks(landmarks)
                with torch.no_grad():
                    pred, conf, probs = classifier.forward(normalized)
                
                all_predictions.append({
                    "shot": SHOT_CLASSES.get(int(pred), "unknown"),
                    "confidence": float(conf)
                })
        
        if not all_predictions:
            return {"success": False, "message": "No poses detected"}
        
        most_common_shot = max(
            set(p['shot'] for p in all_predictions),
            key=lambda x: sum(1 for p in all_predictions if p['shot'] == x)
        )
        avg_confidence = sum(p['confidence'] for p in all_predictions) / len(all_predictions)
        
        return {
            "success": True,
            "dominant_shot": most_common_shot,
            "average_confidence": avg_confidence,
            "frames_analyzed": len(all_predictions),
            "predictions_detail": all_predictions
        }
    
    except Exception as e:
        logger.error(f"Error in stream processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/classes")
async def get_classes():
    return {
        "shot_classes": SHOT_CLASSES,
        "total_classes": len(SHOT_CLASSES)
    }

@app.get("/model/info")
async def get_model_info():
    return {
        "device": DEVICE,
        "confidence_threshold": POSE_CONFIDENCE_THRESHOLD,
        "tracking_confidence": POSE_TRACKING_CONFIDENCE,
        "model_path": str(MODEL_PATH),
        "max_upload_size": MAX_UPLOAD_SIZE,
        "supported_video_formats": list(ALLOWED_VIDEO_EXTENSIONS),
        "supported_image_formats": list(ALLOWED_IMAGE_EXTENSIONS)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
