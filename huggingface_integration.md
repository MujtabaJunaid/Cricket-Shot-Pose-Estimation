UPDATED CODEBASE WITH HUGGINGFACE INTEGRATION
==============================================

All files have been updated to support HuggingFace cricket-shot dataset.

QUICK START WITH HUGGINGFACE
=============================

1. Install dependencies:
   pip install -r requirements.txt

2. Inspect dataset:
   python inspect_hf_dataset.py

3. Train model directly:
   python training/run_training.py --mode train --use-huggingface --epochs 50

4. Start API:
   python backend/app.py

5. Access frontend:
   http://localhost:3000

KEY CHANGES MADE
================

1. training/dataset.py
   - Added HuggingFaceDataset class
   - Updated get_data_loaders() with HuggingFace support
   - Automatic landmark extraction from dataset
   - Label mapping handling

2. training/run_training.py
   - Added --use-huggingface flag
   - Added --dataset-name parameter
   - Integrated HuggingFace dataset loading
   - Fallback error handling

3. backend/config.py
   - Added REVERSE_SHOT_CLASSES mapping
   - Added HuggingFace configuration
   - Added USE_HUGGINGFACE environment variable

4. requirements.txt
   - Added datasets==2.16.0
   - Added huggingface-hub==0.20.0

NEW FILES
=========

- inspect_hf_dataset.py: Dataset inspection tool
- HUGGINGFACE_GUIDE.txt: Comprehensive integration guide
- HUGGINGFACE_INTEGRATION.txt: This file

FEATURES ADDED
==============

✓ Load cricket-shot dataset from HuggingFace
✓ Automatic data preprocessing
✓ Label mapping and conversion
✓ Support for both local and HuggingFace datasets
✓ Dataset inspection tool
✓ Error handling and fallbacks
✓ Caching support
✓ Memory-efficient streaming

USAGE EXAMPLES
==============

Example 1: Train with HuggingFace
---------------------------------
python training/run_training.py \
  --mode train \
  --use-huggingface \
  --epochs 50 \
  --batch-size 32

Example 2: Inspect Dataset
--------------------------
python inspect_hf_dataset.py --samples 10

Example 3: Use in Python Code
------------------------------
from training.dataset import HuggingFaceDataset
from backend.config import REVERSE_SHOT_CLASSES

dataset = HuggingFaceDataset(
    dataset_name="rokmr/cricket-shot",
    split="train",
    label_mapping=REVERSE_SHOT_CLASSES
)

train_loader = DataLoader(dataset, batch_size=32)

DATASET STRUCTURE
=================

Expected Format:
- landmarks: List or array of 33 keypoints × 3 coords
- label: Shot class ID or name
- image: (Optional) Video frame

Output:
- 99 features per frame (33 keypoints × 3 coords)
- Variable sequence length (padded/truncated to 10)
- Label ID (0-8 for 9 shot classes)

DEPLOYMENT OPTIONS
==================

Option 1: Local Development
   python start_dev.py

Option 2: Docker
   docker-compose up

Option 3: AMD Cloud
   1. Build: docker build -f Dockerfile.backend -t cricket-pose .
   2. Push: docker push <registry>/cricket-pose
   3. Deploy on AMD Developer Cloud

TROUBLESHOOTING
===============

Dataset Not Found:
  pip install datasets huggingface-hub
  huggingface-cli login (if needed)

Memory Issues:
  Use smaller batch size: --batch-size 16
  Or use CPU: --device cpu

Import Errors:
  Ensure datasets package is installed
  pip install --upgrade datasets

CONNECTION ERRORS:
  Check internet connection for dataset download
  Dataset is cached after first download

COMPLETE FILE LIST
===================

Backend:
  backend/app.py
  backend/config.py (UPDATED)
  backend/models/classifier.py
  backend/utils/pose_extractor.py
  backend/utils/shot_analyzer.py

Training:
  training/dataset.py (UPDATED)
  training/train.py
  training/export.py
  training/run_training.py (UPDATED)

Frontend:
  frontend/src/App.js
  frontend/src/components/*.js
  frontend/package.json

Utilities:
  extract_poses.py
  predict.py
  annotate_video.py
  validate_data.py
  performance_monitor.py
  test_api.py
  setup.py
  start_dev.py
  inspect_hf_dataset.py (NEW)

Configuration:
  requirements.txt (UPDATED)
  docker-compose.yml
  Dockerfile.backend
  .env.example
  .gitignore

Documentation:
  README.txt
  SETUP_GUIDE.txt
  PROJECT_REFERENCE.txt
  HUGGINGFACE_GUIDE.txt (NEW)
  IMPLEMENTATION_COMPLETE.txt

NEXT STEPS
==========

1. Install packages:
   pip install -r requirements.txt

2. Inspect dataset:
   python inspect_hf_dataset.py

3. Start training:
   python training/run_training.py --mode train --use-huggingface

4. Monitor training in logs

5. Export model:
   python training/run_training.py --mode export

6. Deploy:
   python backend/app.py

SUPPORT
=======

For detailed documentation:
  - See HUGGINGFACE_GUIDE.txt for comprehensive guide
  - See PROJECT_REFERENCE.txt for technical details
  - See README.txt for project overview

For issues:
  - Check logs in logs/ directory
  - Run python test_api.py to verify setup
  - Review error messages and troubleshooting section
