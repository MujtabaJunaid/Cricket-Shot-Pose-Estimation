import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SHOT_CLASSES = {
    'cover': 0, 'defense': 1, 'flick': 2, 'hook': 3, 
    'late_cut': 4, 'lofted': 5, 'pull': 6, 
    'square_cut': 7, 'straight': 8, 'sweep': 9
}


class CricketPoseDataset(Dataset):
    """Load pre-processed cricket pose landmarks from numpy files."""
    
    def __init__(self, data_dir: str, label_mapping: dict = None, 
                 use_temporal: bool = True, sequence_length: int = 10,
                 use_scaler: bool = False):
        self.data_dir = Path(data_dir)
        self.label_mapping = label_mapping or SHOT_CLASSES
        self.use_temporal = use_temporal
        self.sequence_length = sequence_length
        self.samples = []
        self.scaler = StandardScaler() if use_scaler else None
        
        self._load_data()
        
        if use_scaler and self.samples:
            self.normalize()

    def _load_data(self):
        """Load all .npy files from class subdirectories."""
        for label_name, label_id in self.label_mapping.items():
            label_dir = self.data_dir / label_name
            if not label_dir.exists():
                logger.debug(f"Class directory not found: {label_dir}")
                continue
            
            for npy_file in sorted(label_dir.glob("*.npy")):
                try:
                    data = np.load(npy_file, allow_pickle=True)
                    if isinstance(data, np.ndarray) and data.size > 0:
                        self.samples.append((data, label_id, label_name))
                except Exception as e:
                    logger.warning(f"Error loading {npy_file}: {e}")
        
        if not self.samples:
            logger.warning(f"No samples loaded from {self.data_dir}")
        else:
            logger.info(f"Loaded {len(self.samples)} samples from {len(self.label_mapping)} classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data, label, _ = self.samples[idx]
        
        # Ensure 2D shape (frames, landmarks)
        if len(data.shape) == 1:
            data = data.reshape(1, -1)
        
        # Apply temporal padding/truncation
        if self.use_temporal:
            if data.shape[0] < self.sequence_length:
                pad_length = self.sequence_length - data.shape[0]
                data = np.vstack([data, np.tile(data[-1], (pad_length, 1))])
            elif data.shape[0] > self.sequence_length:
                data = data[:self.sequence_length]
        
        # Normalize if scaler is available
        if self.scaler:
            data = self.scaler.transform(data)
        
        data = torch.FloatTensor(data)
        label = torch.LongTensor([label])
        
        return data, label.squeeze()

    def normalize(self):
        """Fit standardscaler on all data."""
        all_data = []
        for data, _, _ in self.samples:
            if len(data.shape) == 1:
                all_data.append(data)
            else:
                all_data.extend(data)
        
        if all_data:
            all_data = np.array(all_data)
            self.scaler.fit(all_data)
            logger.info(f"Scaler fitted on {len(all_data)} frames")
        else:
            logger.warning("No data to fit scaler")

    def get_class_distribution(self):
        """Get count of samples per class."""
        dist = {}
        for _, label, label_name in self.samples:
            dist[label_name] = dist.get(label_name, 0) + 1
        return dist




def create_dataloaders(data_dir: str, batch_size: int = 32, 
                      num_workers: int = 0, train_split: float = 0.8):
    """Create train and validation dataloaders from local data."""
    
    dataset = CricketPoseDataset(data_dir, use_scaler=True)
    
    if len(dataset) == 0:
        raise ValueError(f"No samples found in {data_dir}")
    
    # Log class distribution
    dist = dataset.get_class_distribution()
    logger.info("Class distribution:")
    for class_name, count in sorted(dist.items()):
        logger.info(f"  {class_name}: {count}")
    
    # Split into train/val
    train_size = int(len(dataset) * train_split)
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    return train_loader, val_loader, dataset


if __name__ == '__main__':
    # Test loading
    dataset = CricketPoseDataset('data/processed')
    logger.info(f"Dataset size: {len(dataset)}")
    if len(dataset) > 0:
        sample_data, sample_label = dataset[0]
        logger.info(f"Sample shape: {sample_data.shape}, Label: {sample_label.item()}")
