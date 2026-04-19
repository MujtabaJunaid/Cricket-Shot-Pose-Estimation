import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import argparse
import logging
from training.dataset import get_data_loaders, CricketPoseDataset, DataPreprocessor, HuggingFaceDataset
from training.train import ModelTrainer
from training.export import ModelExporter
from backend.models.classifier import PoseClassifier, StaticPoseClassifier, EnsembleClassifier
from backend.config import BATCH_SIZE, EPOCHS, LEARNING_RATE, VALIDATION_SPLIT, SHOT_CLASSES, REVERSE_SHOT_CLASSES

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Train Cricket Pose Classification Model')
    parser.add_argument('--data-dir', default='data/processed', help='Path to processed data directory')
    parser.add_argument('--checkpoint-dir', default='models/checkpoints', help='Path to save checkpoints')
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=LEARNING_RATE, help='Learning rate')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    parser.add_argument('--mode', choices=['train', 'export', 'preprocess'], default='train')
    parser.add_argument('--export-format', choices=['onnx', 'torchscript', 'quantized', 'ensemble'], default='ensemble')
    parser.add_argument('--use-huggingface', action='store_true', help='Use HuggingFace dataset')
    parser.add_argument('--dataset-name', default='rokmr/cricket-shot', help='HuggingFace dataset name')
    
    args = parser.parse_args()
    
    device = args.device
    logger.info(f"Using device: {device}")
    
    if args.mode == 'preprocess':
        logger.info("Preprocessing data...")
        preprocessor = DataPreprocessor('data/raw', args.data_dir)
        logger.info(f"Preprocessor ready at {args.data_dir}")
    
    elif args.mode == 'train':
        logger.info("Starting training...")
        
        try:
            if args.use_huggingface:
                logger.info(f"Loading HuggingFace dataset: {args.dataset_name}")
                train_loader, val_loader, dataset = get_data_loaders(
                    label_mapping=REVERSE_SHOT_CLASSES,
                    batch_size=args.batch_size,
                    validation_split=VALIDATION_SPLIT,
                    use_huggingface=True,
                    dataset_name=args.dataset_name
                )
            else:
                logger.info(f"Loading local data from: {args.data_dir}")
                train_loader, val_loader, dataset = get_data_loaders(
                    data_dir=args.data_dir,
                    label_mapping=SHOT_CLASSES,
                    batch_size=args.batch_size,
                    validation_split=VALIDATION_SPLIT,
                    use_huggingface=False
                )
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            if not args.use_huggingface:
                logger.info("Creating sample data structure...")
                Path(args.data_dir).mkdir(parents=True, exist_ok=True)
                for shot_class in SHOT_CLASSES.values():
                    (Path(args.data_dir) / shot_class).mkdir(exist_ok=True)
                logger.info("Please add training data to data/processed/{shot_class}/ directories")
                return
            else:
                logger.error("Failed to load HuggingFace dataset. Ensure 'datasets' package is installed.")
                logger.error("Install with: pip install datasets")
                return
        
        num_classes = len(SHOT_CLASSES)
        input_size = 99
        
        temporal_model = PoseClassifier(num_classes=num_classes, input_size=input_size)
        static_model = StaticPoseClassifier(num_classes=num_classes, input_size=input_size)
        
        logger.info("Training temporal model...")
        trainer_temporal = ModelTrainer(temporal_model, device=device)
        trainer_temporal.fit(train_loader, val_loader, args.epochs, args.learning_rate, args.checkpoint_dir)
        trainer_temporal.save_checkpoint(Path(args.checkpoint_dir) / 'temporal_model.pt')
        
        logger.info("Training static model...")
        trainer_static = ModelTrainer(static_model, device=device)
        trainer_static.fit(train_loader, val_loader, args.epochs, args.learning_rate, args.checkpoint_dir)
        trainer_static.save_checkpoint(Path(args.checkpoint_dir) / 'static_model.pt')
        
        logger.info("Training completed successfully!")
    
    elif args.mode == 'export':
        logger.info(f"Exporting model to {args.export_format}...")
        
        ensemble = EnsembleClassifier(num_classes=len(SHOT_CLASSES), device=device)
        checkpoint_path = Path(args.checkpoint_dir) / 'best_model.pt'
        
        if checkpoint_path.exists():
            ensemble.load_checkpoint(str(checkpoint_path))
        else:
            logger.warning("No checkpoint found, using untrained models")
        
        temporal_model, static_model = ensemble.get_models()
        
        if args.export_format == 'ensemble':
            ModelExporter.export_ensemble(temporal_model, static_model, f'{args.checkpoint_dir}/ensemble')
        elif args.export_format == 'onnx':
            ModelExporter.export_to_onnx(temporal_model, str(checkpoint_path), 
                                        f'{args.checkpoint_dir}/model.onnx')
        elif args.export_format == 'torchscript':
            ModelExporter.export_to_torchscript(temporal_model, f'{args.checkpoint_dir}/model.pt')
        elif args.export_format == 'quantized':
            ModelExporter.export_to_quantized(temporal_model, f'{args.checkpoint_dir}/model_quantized.pt', device)
        
        logger.info("Export completed successfully!")

if __name__ == '__main__':
    main()
