# Initialize
python setup.py init

# Install dependencies
pip install -r requirements.txt
cd frontend && npm install

# Extract poses from your cricket videos
python extract_poses.py --batch data/raw --output-dir data/processed

# Train model
python training/run_training.py --mode train --epochs 50

# Start development environment
python start_dev.py

# Access:
# Frontend: http://localhost:3000
# API: http://localhost:8000
# API Docs: http://localhost:8000/docs