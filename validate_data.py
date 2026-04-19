#!/usr/bin/env python3

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import json
import logging
from pathlib import Path
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataValidator:
    def __init__(self, data_dir: str = "data/processed"):
        self.data_dir = Path(data_dir)
    
    def validate_structure(self):
        logger.info("Validating data structure...")
        issues = []
        
        if not self.data_dir.exists():
            issues.append(f"Data directory does not exist: {self.data_dir}")
            return issues
        
        shot_classes = [d.name for d in self.data_dir.iterdir() if d.is_dir()]
        logger.info(f"Found shot classes: {shot_classes}")
        
        stats = {}
        for shot_class in shot_classes:
            class_dir = self.data_dir / shot_class
            npy_files = list(class_dir.glob("*.npy"))
            stats[shot_class] = len(npy_files)
            
            if len(npy_files) < 5:
                issues.append(f"Low sample count for {shot_class}: {len(npy_files)}")
        
        logger.info("Class distribution: " + json.dumps(stats, indent=2))
        return issues
    
    def validate_data_shapes(self):
        logger.info("Validating data shapes...")
        issues = []
        
        for shot_class_dir in self.data_dir.iterdir():
            if not shot_class_dir.is_dir():
                continue
            
            for npy_file in shot_class_dir.glob("*.npy"):
                try:
                    data = np.load(npy_file)
                    
                    if data.ndim == 1:
                        expected_size = 99
                        if data.shape[0] != expected_size:
                            issues.append(f"{npy_file}: Expected shape ({expected_size},), got {data.shape}")
                    elif data.ndim == 2:
                        if data.shape[1] != 99:
                            issues.append(f"{npy_file}: Expected last dim 99, got {data.shape[1]}")
                    else:
                        issues.append(f"{npy_file}: Unexpected number of dimensions: {data.ndim}")
                
                except Exception as e:
                    issues.append(f"{npy_file}: Error reading file - {e}")
        
        if not issues:
            logger.info("All data shapes are valid!")
        
        return issues
    
    def print_report(self):
        logger.info("\n" + "="*60)
        logger.info("DATA VALIDATION REPORT")
        logger.info("="*60)
        
        structure_issues = self.validate_structure()
        shape_issues = self.validate_data_shapes()
        
        all_issues = structure_issues + shape_issues
        
        if all_issues:
            logger.warning(f"Found {len(all_issues)} issues:")
            for issue in all_issues:
                logger.warning(f"  - {issue}")
        else:
            logger.info("No issues found! Data is ready for training.")
        
        logger.info("="*60 + "\n")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Data Validation Tool')
    parser.add_argument('--data-dir', type=str, default='data/processed')
    
    args = parser.parse_args()
    
    validator = DataValidator(args.data_dir)
    validator.print_report()

if __name__ == '__main__':
    main()
