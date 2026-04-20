# Cricket Shot Classification - Local Setup

**Status:** ✅ Production Ready (Local Development)

## Quick Start

### 1. Backend (FastAPI)
```bash
cd c:\Users\hp\amd-hackathon
python -m uvicorn backend.app:app --host 0.0.0.0 --port 5000
```
- Runs on: `http://localhost:5000`
- API endpoints: `/health`, `/classes`, `/predict/image`, `/predict/video`

### 2. Frontend (React)
```bash
cd c:\Users\hp\amd-hackathon\frontend
npm start
```
- Runs on: `http://localhost:3000`
- Auto-connects to backend at `http://localhost:5000`

## Project Structure

```
c:\Users\hp\amd-hackathon/
├── backend/                    # FastAPI server
│   ├── app.py                 # Main app with all endpoints
│   ├── config.py              # Configuration
│   ├── models/                # Pre-trained PyTorch models + MediaPipe task file
│   └── utils/
│       ├── pose_extractor.py  # MediaPipe pose detection
│       ├── shot_analyzer.py   # Analysis logic
│       └── classifier.py      # LSTM + FC models
├── frontend/                  # React UI
│   ├── src/
│   │   ├── config.js          # API endpoint config
│   │   ├── App.js             # Main component
│   │   └── components/        # UI components
│   └── public/
├── training/                  # Model training scripts
│   ├── run_training.py        # Main training script
│   ├── train.py               # Trainer class
│   ├── dataset.py             # Dataset loader
│   └── process_videos_mediapipe.py  # Extract poses from videos
├── data/
│   └── processed/             # Processed pose data (organized by shot class)
├── models/
│   ├── checkpoints/           # Saved model checkpoints
│   └── pose_landmarker_lite.task  # MediaPipe pose model
└── uploads/                   # Temp storage for uploaded images
```

## Key Features

✅ **Real-time Pose Detection** - MediaPipe 0.10.33 with .task model
✅ **Cricket Shot Classification** - 10 shot types (cover, defense, flick, hook, late_cut, lofted, pull, square_cut, straight, sweep)
✅ **Temporal & Static Models** - LSTM for sequences + FC for single frames
✅ **REST API** - FastAPI with CORS enabled for local development
✅ **Web UI** - React frontend for image/video uploads
✅ **Live Analytics** - Model information and prediction results

## API Endpoints

### Health Check
```bash
curl http://localhost:5000/health
```
**Response:** `{"status": "healthy", "device": "cpu", "model_loaded": true}`

### Get Shot Classes
```bash
curl http://localhost:5000/classes
```

### Predict from Image
```bash
curl -X POST -F "file=@image.jpg" http://localhost:5000/predict/image
```

### Predict from Video
```bash
curl -X POST -F "file=@video.mp4" http://localhost:5000/predict/video
```

## Configuration

**API URL:** Edit `frontend/src/config.js`
```javascript
export const API_URL = "http://localhost:5000";
```

**Backend Settings:** Edit `backend/config.py`
- `BATCH_SIZE`
- `EPOCHS`
- `LEARNING_RATE`
- `DEVICE` (cpu/cuda)

## Training (Optional)

Extract poses from new videos and retrain:
```bash
python training/process_videos_mediapipe.py --input-dir <path> --output-dir data/processed
python training/run_training.py --data-dir data/processed --epochs 50
```

Then restart backend to load new model.

## Model Files

- **pose_landmarker_lite.task** (~35 MB) - MediaPipe pose detection
- **temporal_model.pt** - LSTM for temporal sequences
- **static_model.pt** - FC layers for single frames
- **best_model.pt** - Ensemble of both models

## Requirements

- Python 3.12+
- PyTorch
- FastAPI & Uvicorn
- MediaPipe 0.10.33
- OpenCV
- React 18+
- Node.js 16+

Install Python packages:
```bash
pip install -r requirements.txt
```

Install React packages:
```bash
cd frontend
npm install
```

## Troubleshooting

**Backend won't start:**
- Check port 5000 is free: `netstat -ano | findstr :5000`
- Kill process: `taskkill /PID <PID> /F`

**Frontend can't connect:**
- Verify backend is running: `curl http://localhost:5000/health`
- Check API_URL in `frontend/src/config.js`

**No pose detection:**
- Verify model exists: `backend/models/pose_landmarker_lite.task`
- Ensure image has visible person in cricket stance

**Low prediction accuracy:**
- Retrain model with fresh videos: See Training section above
- Ensure videos have good lighting and clear cricket shots

## Notes

- All development is **local only** (localhost)
- No external APIs or cloud services needed
- Models run on CPU (GPU support available via config)
- Data never leaves your machine
