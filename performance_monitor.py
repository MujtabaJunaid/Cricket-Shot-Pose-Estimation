#!/usr/bin/env python3

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import json
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceMonitor:
    def __init__(self, log_file: str = "logs/performance.json"):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.metrics = []
    
    def log_inference(self, model_type: str, input_type: str, processing_time: float, 
                      confidence: float, accuracy: bool = None):
        metric = {
            'timestamp': datetime.now().isoformat(),
            'model_type': model_type,
            'input_type': input_type,
            'processing_time_ms': processing_time,
            'confidence': confidence,
            'accuracy': accuracy
        }
        self.metrics.append(metric)
        self._save_metrics()
    
    def _save_metrics(self):
        with open(self.log_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def get_stats(self):
        if not self.metrics:
            return None
        
        inference_times = [m['processing_time_ms'] for m in self.metrics]
        confidences = [m['confidence'] for m in self.metrics]
        
        stats = {
            'total_inferences': len(self.metrics),
            'avg_inference_time_ms': sum(inference_times) / len(inference_times),
            'min_inference_time_ms': min(inference_times),
            'max_inference_time_ms': max(inference_times),
            'avg_confidence': sum(confidences) / len(confidences),
            'min_confidence': min(confidences),
            'max_confidence': max(confidences)
        }
        
        if any(m['accuracy'] is not None for m in self.metrics):
            accuracies = [m['accuracy'] for m in self.metrics if m['accuracy'] is not None]
            stats['accuracy'] = sum(accuracies) / len(accuracies)
        
        return stats
    
    def print_report(self):
        stats = self.get_stats()
        if not stats:
            logger.warning("No metrics available")
            return
        
        logger.info("\n" + "="*60)
        logger.info("PERFORMANCE REPORT")
        logger.info("="*60)
        logger.info(f"Total Inferences: {stats['total_inferences']}")
        logger.info(f"Avg Inference Time: {stats['avg_inference_time_ms']:.2f}ms")
        logger.info(f"Min/Max Inference Time: {stats['min_inference_time_ms']:.2f}ms / {stats['max_inference_time_ms']:.2f}ms")
        logger.info(f"Avg Confidence: {stats['avg_confidence']:.4f}")
        if 'accuracy' in stats:
            logger.info(f"Accuracy: {stats['accuracy']:.4f}")
        logger.info("="*60 + "\n")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Performance Monitoring Tool')
    parser.add_argument('--log-file', type=str, default='logs/performance.json')
    parser.add_argument('--report', action='store_true', help='Print performance report')
    
    args = parser.parse_args()
    
    monitor = PerformanceMonitor(args.log_file)
    
    if args.report:
        monitor.print_report()

if __name__ == '__main__':
    main()
