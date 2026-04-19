HUGGINGFACE DATASET INTEGRATION GUIDE
=====================================

This project now supports training with the HuggingFace cricket-shot dataset.
Dataset: rokmr/cricket-shot (https://huggingface.co/datasets/rokmr/cricket-shot)

QUICK START
===========

1. Install Required Packages:
   pip install -r requirements.txt

2. Inspect the Dataset:
   python inspect_hf_dataset.py

3. Train Model with HuggingFace Data:
   python training/run_training.py --mode train --use-huggingface --epochs 50

4. For Local Data (Original Method):
   python training/run_training.py --mode train --data-dir data/processed

DETAILED USAGE
==============

Option 1: Train with HuggingFace Dataset (Recommended)
------------------------------------------------------

python training/run_training.py \
  --mode train \
  --use-huggingface \
  --dataset-name rokmr/cricket-shot \
  --epochs 50 \
  --batch-size 32 \
  --learning-rate 0.001 \
  --device cuda

Advantages:
  - No local data collection needed
  - Pre-processed and validated data
  - Easy scaling
  - Community-maintained dataset

Option 2: Train with Local Data
---------------------------------

1. Prepare your cricket videos:
   data/raw/
   ├── drive/
   ├── pull/
   ├── cut/
   ├── defense/
   ├── backfoot_play/
   ├── cover_drive/
   ├── off_drive/
   ├── on_drive/
   └── lofted_shot/

2. Extract poses:
   python extract_poses.py --batch data/raw --output-dir data/processed

3. Validate:
   python validate_data.py

4. Train:
   python training/run_training.py \
     --mode train \
     --data-dir data/processed \
     --epochs 50 \
     --batch-size 32 \
     --learning-rate 0.001 \
     --device cuda

API REFERENCE
=============

Training Script Arguments:

  --mode {train, export, preprocess}
    train: Train model
    export: Export trained model
    preprocess: Prepare local data

  --use-huggingface
    Use HuggingFace dataset (boolean flag)

  --dataset-name <name>
    HuggingFace dataset identifier
    Default: rokmr/cricket-shot

  --data-dir <path>
    Local data directory (for local training)
    Default: data/processed

  --checkpoint-dir <path>
    Directory to save model checkpoints
    Default: models/checkpoints

  --epochs <int>
    Number of training epochs
    Default: 50

  --batch-size <int>
    Training batch size
    Default: 32

  --learning-rate <float>
    Learning rate for optimizer
    Default: 0.001

  --device {cuda, cpu}
    Device for training
    Default: cuda (if available)

  --export-format {onnx, torchscript, quantized, ensemble}
    Format to export model
    Default: ensemble

PYTHON API
==========

Using HuggingFace Dataset in Code:

  from training.dataset import HuggingFaceDataset, get_data_loaders
  from backend.config import REVERSE_SHOT_CLASSES

  # Direct loading
  dataset = HuggingFaceDataset(
    dataset_name="rokmr/cricket-shot",
    split="train",
    label_mapping=REVERSE_SHOT_CLASSES
  )

  # Via get_data_loaders
  train_loader, val_loader, dataset = get_data_loaders(
    label_mapping=REVERSE_SHOT_CLASSES,
    batch_size=32,
    validation_split=0.2,
    use_huggingface=True,
    dataset_name="rokmr/cricket-shot"
  )

SHOT CLASSES
============

The dataset includes these cricket shots:
  0: drive
  1: pull
  2: cut
  3: defense
  4: backfoot_play
  5: cover_drive
  6: off_drive
  7: on_drive
  8: lofted_shot

DATASET FEATURES
================

Data Format:
  - Landmarks: Array of 33 keypoints × 3 coordinates (99 features)
  - Label: Shot class ID or name
  - Sequence: Variable length frame sequences

Processing:
  - Automatic normalization
  - Padding/truncation to fixed sequence length
  - Standard scaling of features

Splits Available:
  - train: Training data
  - validation: Validation data (if available)
  - test: Test data (if available)

DATASET STATISTICS
==================

To view dataset statistics:
  python inspect_hf_dataset.py

Output includes:
  - Total number of samples
  - Sample shape and format
  - Label distribution
  - Shot class breakdown

TROUBLESHOOTING
===============

Dataset Download Issues:

1. Network Error:
   The dataset is large and may take time to download.
   Ensure stable internet connection.
   Dataset is cached in ~/.cache/huggingface/

2. Authentication Error:
   If dataset requires login:
   huggingface-cli login
   
   Enter your HuggingFace token from:
   https://huggingface.co/settings/tokens

3. Memory Issues:
   Reduce batch size:
   --batch-size 16
   
   Or use CPU:
   --device cpu

4. Missing Dependencies:
   Install datasets package:
   pip install datasets huggingface-hub

Dataset Format Issues:

1. Unknown Label Format:
   Check REVERSE_SHOT_CLASSES in backend/config.py
   Ensure labels match expected values

2. Feature Shape Mismatch:
   Expected: (batch, seq_len, 99)
   Verify dataset preprocessing in HuggingFaceDataset._process_dataset()

PERFORMANCE TIPS
================

1. GPU Training:
   Use --device cuda for 10-20x faster training

2. Batch Size:
   Larger batches (64, 128) for better GPU utilization
   Smaller batches (8, 16) if memory limited

3. Data Loading:
   HuggingFace datasets are streamed and cached automatically
   First load may be slow due to downloading

4. Checkpointing:
   Models saved frequently with best validation accuracy
   Resume training from checkpoint if interrupted

EXPORTING MODELS
================

After training, export your model:

1. Ensemble Export:
   python training/run_training.py --mode export --export-format ensemble

2. ONNX Export:
   python training/run_training.py --mode export --export-format onnx

3. TorchScript:
   python training/run_training.py --mode export --export-format torchscript

4. Quantized:
   python training/run_training.py --mode export --export-format quantized

INFERENCE WITH TRAINED MODEL
=============================

Using the API:
  python backend/app.py

Upload image/video for prediction:
  curl -X POST -F "file=@image.jpg" http://localhost:8000/predict/image

Command line prediction:
  python predict.py --image image.jpg
  python predict.py --video video.mp4

COMBINED WORKFLOWS
==================

Scenario 1: Train on HuggingFace, Deploy Locally

  1. Train: python training/run_training.py --use-huggingface
  2. Export: python training/run_training.py --mode export
  3. Start API: python backend/app.py
  4. Use Frontend: http://localhost:3000

Scenario 2: Train on HuggingFace, Deploy on Cloud

  1. Train: python training/run_training.py --use-huggingface
  2. Export: python training/run_training.py --mode export
  3. Push to registry: docker push cricket-pose-backend
  4. Deploy on AMD cloud or other platform

Scenario 3: Use Local Data

  1. Collect videos
  2. Extract poses: python extract_poses.py --batch data/raw
  3. Train: python training/run_training.py
  4. Deploy: docker-compose up

CONFIGURATION
=============

Environment Variables:

  USE_HUGGINGFACE=true
  HUGGINGFACE_DATASET=rokmr/cricket-shot
  DEVICE=cuda
  BATCH_SIZE=32
  EPOCHS=50

Set in .env or shell:
  export USE_HUGGINGFACE=true
  python training/run_training.py --mode train

DATASET LICENSING
=================

Ensure you understand the license of rokmr/cricket-shot
Visit: https://huggingface.co/datasets/rokmr/cricket-shot
for license terms and attribution requirements.

SUPPORT & RESOURCES
===================

HuggingFace Documentation:
  https://huggingface.co/docs/datasets

Cricket Shot Dataset:
  https://huggingface.co/datasets/rokmr/cricket-shot

PyTorch Documentation:
  https://pytorch.org/docs

Project Documentation:
  - README.txt
  - SETUP_GUIDE.txt
  - PROJECT_REFERENCE.txt
