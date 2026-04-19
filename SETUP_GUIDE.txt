CRICKET POSE CLASSIFICATION SYSTEM

## Project Structure

backend/
  - app.py: FastAPI server
  - config.py: Configuration settings
  - models/
    - classifier.py: Neural network models
  - utils/
    - pose_extractor.py: Pose detection
    - shot_analyzer.py: Shot analysis

training/
  - dataset.py: Data loading and preprocessing
  - train.py: Model training pipeline
  - export.py: Model export utilities
  - run_training.py: Training script

frontend/
  - src/
    - App.js: Main React component
    - components/: UI components
    - styles/: CSS files
  - package.json: Dependencies

data/
  - raw/: Raw video data
  - processed/: Processed pose data

models/
  - checkpoints/: Trained model weights

## Quick Start

1. Initialize project:
   python setup.py init

2. Install dependencies:
   pip install -r requirements.txt
   cd frontend && npm install && cd ..

3. Prepare training data:
   Place videos in data/raw/{shot_class}/ directories

4. Extract poses:
   python extract_poses.py --batch data/raw --output-dir data/processed

5. Train model:
   python training/run_training.py --mode train

6. Start backend:
   python backend/app.py

7. Start frontend:
   cd frontend && npm start

## API Endpoints

POST /predict/image - Classify single image
POST /predict/video - Classify video file
POST /predict/stream - Real-time stream analysis
GET /health - Health check
GET /classes - Get available shot classes
GET /model/info - Model information

## Shot Classes

- drive
- pull
- cut
- defense
- backfoot_play
- cover_drive
- off_drive
- on_drive
- lofted_shot

## Model Architecture

Temporal Model: LSTM + Attention mechanism
Static Model: Fully connected neural network
Ensemble: Average predictions from both models

## Training

Epochs: 50
Batch Size: 32
Learning Rate: 0.001
Device: CUDA (GPU)
Validation Split: 20%

## Command Line Tools

Extract poses from videos:
  python extract_poses.py --batch data/raw --output-dir data/processed

Predict shot from image:
  python predict.py --image path/to/image.jpg

Predict shot from video:
  python predict.py --video path/to/video.mp4

Train model:
  python training/run_training.py --mode train

Export model:
  python training/run_training.py --mode export --export-format ensemble

## Deployment

Using Docker Compose:
  docker-compose up

Access:
  Frontend: http://localhost:3000
  Backend: http://localhost:8000
  API Docs: http://localhost:8000/docs

## Configuration

Edit backend/config.py to modify:
- Confidence thresholds
- Model paths
- Batch sizes
- Learning rates
- Device selection
