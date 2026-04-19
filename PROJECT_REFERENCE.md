CRICKET POSE CLASSIFICATION - COMPLETE PROJECT REFERENCE

PROJECT OVERVIEW
===============
A full-stack AI system for real-time cricket shot pose classification using computer vision,
deep learning, and REST APIs. Built for AMD Hackathon with $100 cloud credits.

TECH STACK
==========
Backend: FastAPI, Python, PyTorch, MediaPipe
Frontend: React, JavaScript, CSS
Models: LSTM + Attention, Fully Connected Networks, Ensemble
Database: File-based (NumPy, JSON)
Deployment: Docker, Docker Compose
ML Pipeline: TensorFlow/PyTorch, ONNX export

CORE FEATURES
=============
1. Real-time Pose Detection (33 keypoints)
2. Shot Classification (9 cricket shots)
3. Angle Analysis (arm, leg, shoulder, hip)
4. Video & Image Analysis
5. REST API Endpoints
6. Web-based UI
7. Model Training Pipeline
8. Model Export (ONNX, TorchScript, Quantized)
9. Performance Monitoring
10. Data Validation

DIRECTORY STRUCTURE
===================
amd-hackathon/
├── backend/                     # FastAPI server
│   ├── __init__.py
│   ├── app.py                   # Main API application
│   ├── config.py                # Configuration
│   ├── models/
│   │   ├── __init__.py
│   │   └── classifier.py        # Neural network models
│   ├── routes/                  # API routes (modular)
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── pose_extractor.py    # MediaPipe integration
│   │   └── shot_analyzer.py     # Pose analysis utilities
│
├── frontend/                    # React application
│   ├── public/
│   │   └── index.html
│   ├── src/
│   │   ├── App.js               # Main component
│   │   ├── App.css
│   │   ├── index.js
│   │   ├── index.css
│   │   ├── components/
│   │   │   ├── ImagePredictor.js    # Image upload & analysis
│   │   │   ├── VideoPredictor.js    # Video upload & analysis
│   │   │   ├── LiveAnalytics.js     # Analytics dashboard
│   │   │   └── ModelInfo.js         # Model information
│   │   └── styles/
│   │       ├── predictor.css
│   │       ├── analytics.css
│   │       └── model-info.css
│   ├── package.json
│   ├── Dockerfile
│   └── .gitignore
│
├── training/                    # Model training
│   ├── __init__.py
│   ├── dataset.py               # Data loading & preprocessing
│   ├── train.py                 # Training loop & validation
│   ├── export.py                # Model export utilities
│   └── run_training.py          # Training entry point
│
├── data/                        # Data storage
│   ├── raw/
│   │   ├── drive/
│   │   ├── pull/
│   │   ├── cut/
│   │   ├── defense/
│   │   ├── backfoot_play/
│   │   ├── cover_drive/
│   │   ├── off_drive/
│   │   ├── on_drive/
│   │   └── lofted_shot/
│   └── processed/               # Processed pose data
│
├── models/                      # Model storage
│   └── checkpoints/             # Trained weights
│
├── logs/                        # Application logs
├── uploads/                     # Temporary uploads
│
├── setup.py                     # Project initialization
├── start_dev.py                 # Development server
├── extract_poses.py             # Pose extraction utility
├── predict.py                   # Prediction CLI
├── annotate_video.py            # Video annotation
├── validate_data.py             # Data validation
├── performance_monitor.py       # Performance tracking
├── requirements.txt             # Python dependencies
├── docker-compose.yml           # Docker orchestration
├── Dockerfile.backend           # Backend container
├── .env.example                 # Environment template
├── .gitignore                   # Git ignore
└── SETUP_GUIDE.txt              # This guide

QUICK START GUIDE
=================

1. INITIAL SETUP
   python setup.py init
   pip install -r requirements.txt
   cd frontend && npm install && cd ..

2. PREPARE DATA
   - Place cricket videos in data/raw/{shot_class}/
   - Run: python extract_poses.py --batch data/raw --output-dir data/processed

3. VALIDATE DATA
   python validate_data.py

4. TRAIN MODEL
   python training/run_training.py --mode train --epochs 50 --batch-size 32

5. START DEVELOPMENT SERVER
   python start_dev.py

6. ACCESS APPLICATION
   Frontend: http://localhost:3000
   Backend API: http://localhost:8000
   API Documentation: http://localhost:8000/docs

REST API ENDPOINTS
==================

Health & Info:
  GET /health                     - API health status
  GET /classes                    - Available shot classes
  GET /model/info                 - Model configuration

Predictions:
  POST /predict/image             - Classify static image
  POST /predict/video             - Classify video file
  POST /predict/stream            - Real-time frame analysis

Request Format:
  Content-Type: multipart/form-data
  File parameter: file

Response Format (Success):
  {
    "success": true,
    "pose_found": true,
    "predicted_shot": "drive",
    "confidence": 0.95,
    "all_predictions": {
      "drive": 0.95,
      "pull": 0.03,
      ...
    },
    "angle_features": {
      "left_arm_angle": 45.2,
      "right_arm_angle": 38.1,
      ...
    }
  }

SHOT CLASSES (9 total)
======================
1. drive          - Forward defensive shot
2. pull           - Back of the bat shot
3. cut            - Horizontal bat shot
4. defense        - Defensive stance
5. backfoot_play  - Backfoot positioning
6. cover_drive    - Cover region shot
7. off_drive      - Off side shot
8. on_drive       - On side shot
9. lofted_shot    - Elevated shot

COMMAND LINE TOOLS
==================

Project Setup:
  python setup.py init            - Initialize directories
  python setup.py setup-backend   - Setup backend env
  python setup.py setup-frontend  - Setup frontend env
  python setup.py setup           - Full setup

Data Processing:
  python extract_poses.py --batch data/raw --output-dir data/processed
  python extract_poses.py --video video.mp4 --shot-class drive
  python validate_data.py
  python annotate_video.py --input video.mp4 --output annotated.mp4

Model Operations:
  python training/run_training.py --mode train --epochs 50
  python training/run_training.py --mode export --export-format ensemble
  python predict.py --image image.jpg
  python predict.py --video video.mp4

Monitoring:
  python performance_monitor.py --report

MODEL ARCHITECTURE
==================

Temporal Model (Sequence Analysis):
  - Input: Sequence of poses (batch_size, seq_len, 99)
  - LSTM Layer 1: 256 units, dropout 0.3
  - Layer Norm
  - LSTM Layer 2: 128 units, dropout 0.3
  - MultiheadAttention: 8 heads
  - FC: 128 -> 64 -> 32 -> num_classes
  - Output: Class probabilities

Static Model (Single Frame):
  - Input: Single pose (batch_size, 99)
  - FC: 99 -> 256 -> 128 -> 64 -> num_classes
  - BatchNorm after each FC
  - Dropout: 0.4
  - Output: Class probabilities

Ensemble:
  - Combines both models (average)
  - Input processing & normalization
  - Hybrid predictions for better accuracy

TRAINING CONFIGURATION
======================
Batch Size: 32
Epochs: 50 (with early stopping)
Learning Rate: 0.001 (AdamW)
Weight Decay: 1e-5
Scheduler: ReduceLROnPlateau
Validation Split: 20%
Loss Function: CrossEntropyLoss
Device: CUDA (GPU)

Pose Features:
- 33 keypoints × 3 coordinates (x, y, z) = 99 features
- Normalized by shoulder distance
- Centered at shoulder midpoint

Angle Features (6 total):
- Left arm angle
- Right arm angle
- Left leg angle
- Right leg angle
- Shoulder angle
- Hip angle

DEPLOYMENT OPTIONS
==================

Local Development:
  python start_dev.py

Docker Compose:
  docker-compose up

Manual Docker:
  docker build -f Dockerfile.backend -t cricket-pose-backend .
  docker build -f frontend/Dockerfile -t cricket-pose-frontend ./frontend
  docker run -p 8000:8000 cricket-pose-backend
  docker run -p 3000:3000 cricket-pose-frontend

Cloud Deployment (AMD Developer Cloud):
- Build Docker images
- Push to container registry
- Deploy using AMD cloud platform
- Configure GPU resources

PERFORMANCE METRICS
===================
Inference Time: ~50-100ms per frame
FPS: 10-20 fps real-time
Confidence Range: 0.0-1.0
Model Size: ~15-20MB
Memory Usage: 1-2GB (with GPU)

TROUBLESHOOTING
===============

No Pose Detected:
  - Ensure good lighting
  - Full body visible in frame
  - Adjust pose confidence threshold

API Connection Failed:
  - Check backend is running on port 8000
  - Verify network configuration
  - Check firewall settings

Training Issues:
  - Validate data with: python validate_data.py
  - Check data shape: (batch, seq_len, 99)
  - Ensure GPU has enough VRAM

Dependencies:
  - CUDA 11.8+ for GPU support
  - Python 3.9+
  - Node.js 18+

OPTIMIZATION TIPS
=================
1. Use GPU for training and inference
2. Preprocess and cache pose data
3. Use model quantization for deployment
4. Implement batch inference
5. Add input validation and error handling
6. Monitor performance metrics
7. Tune hyperparameters based on validation results
8. Use data augmentation for better generalization

NEXT STEPS
==========
1. Collect cricket shot videos
2. Annotate and organize data
3. Extract poses and validate
4. Train model on your data
5. Export and deploy model
6. Integrate with live camera feed
7. Build mobile app if needed
8. Deploy to AMD cloud with GPU

RESOURCES
=========
- MediaPipe Pose: https://mediapipe.dev
- PyTorch: https://pytorch.org
- FastAPI: https://fastapi.tiangolo.com
- React: https://react.dev
- Docker: https://docker.com

SUPPORT & CONTACT
=================
For issues, refer to:
- API Documentation: http://localhost:8000/docs
- SETUP_GUIDE.txt: Quick reference
- Source code comments
- Training logs in logs/ directory
