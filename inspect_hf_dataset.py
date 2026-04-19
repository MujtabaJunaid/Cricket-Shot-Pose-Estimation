#!/usr/bin/env python3

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import logging
import argparse
from training.dataset import HuggingFaceDataset
from backend.config import REVERSE_SHOT_CLASSES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def inspect_dataset(dataset_name: str = "rokmr/cricket-shot", num_samples: int = 5):
    logger.info(f"Inspecting dataset: {dataset_name}")
    logger.info(f"Loading first {num_samples} samples...")
    
    dataset = HuggingFaceDataset(
        dataset_name=dataset_name,
        split="train",
        label_mapping=REVERSE_SHOT_CLASSES
    )
    
    logger.info(f"\nDataset size: {len(dataset)}")
    logger.info(f"Sample shape: {dataset.samples[0][0].shape if dataset.samples else 'N/A'}")
    
    logger.info("\nDataset Structure:")
    logger.info("-" * 60)
    
    label_counts = {}
    for i, (data, label) in enumerate(dataset.samples[:num_samples]):
        shot_name = list(REVERSE_SHOT_CLASSES.keys())[list(REVERSE_SHOT_CLASSES.values()).index(label)] if label in REVERSE_SHOT_CLASSES.values() else f"Class {label}"
        label_counts[shot_name] = label_counts.get(shot_name, 0) + 1
        logger.info(f"Sample {i}: Shape={data.shape}, Label={shot_name} ({label})")
    
    logger.info("\n" + "-" * 60)
    logger.info("Label Distribution (in first samples):")
    for shot_name, count in sorted(label_counts.items()):
        logger.info(f"  {shot_name}: {count}")
    
    logger.info("\n" + "-" * 60)
    logger.info("Total samples by label in full dataset:")
    full_label_counts = {}
    for _, label in dataset.samples:
        full_label_counts[label] = full_label_counts.get(label, 0) + 1
    
    for label in sorted(full_label_counts.keys()):
        shot_name = list(REVERSE_SHOT_CLASSES.keys())[list(REVERSE_SHOT_CLASSES.values()).index(label)] if label in REVERSE_SHOT_CLASSES.values() else f"Class {label}"
        logger.info(f"  {shot_name}: {full_label_counts[label]}")

def main():
    parser = argparse.ArgumentParser(description='Inspect HuggingFace Cricket Dataset')
    parser.add_argument('--dataset', default='rokmr/cricket-shot', help='Dataset name')
    parser.add_argument('--samples', type=int, default=5, help='Number of samples to inspect')
    
    args = parser.parse_args()
    inspect_dataset(args.dataset, args.samples)

if __name__ == '__main__':
    main()
