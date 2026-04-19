import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import json
import logging
from typing import Optional, Dict, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CricketPoseDataset(Dataset):
    def __init__(self, data_dir: str, label_mapping: dict, use_temporal: bool = True, sequence_length: int = 10):
        self.data_dir = Path(data_dir)
        self.label_mapping = label_mapping
        self.use_temporal = use_temporal
        self.sequence_length = sequence_length
        self.samples = []
        self.scaler = StandardScaler()
        
        self._load_data()

    def _load_data(self):
        for label_name, label_id in self.label_mapping.items():
            label_dir = self.data_dir / label_name
            if not label_dir.exists():
                logger.warning(f"Directory not found: {label_dir}")
                continue
            
            for npy_file in label_dir.glob("*.npy"):
                try:
                    data = np.load(npy_file, allow_pickle=True)
                    if isinstance(data, np.ndarray):
                        if len(data.shape) == 1:
                            self.samples.append((data, label_id))
                        else:
                            self.samples.append((data, label_id))
                except Exception as e:
                    logger.error(f"Error loading {npy_file}: {e}")
        
        if not self.samples:
            logger.warning("No samples loaded from data directory")
        else:
            logger.info(f"Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data, label = self.samples[idx]
        
        if len(data.shape) == 1:
            data = data.reshape(1, -1)
        
        if self.use_temporal and data.shape[0] < self.sequence_length:
            pad_length = self.sequence_length - data.shape[0]
            data = np.vstack([data, np.tile(data[-1], (pad_length, 1))])
        elif self.use_temporal and data.shape[0] > self.sequence_length:
            data = data[:self.sequence_length]
        
        data = torch.FloatTensor(data)
        label = torch.LongTensor([label])
        
        return data, label.squeeze()

    def normalize(self):
        all_data = []
        for data, _ in self.samples:
            if len(data.shape) == 1:
                all_data.append(data)
            else:
                all_data.extend(data)
        
        all_data = np.array(all_data)
        self.scaler.fit(all_data)
        logger.info("Scaler fitted")


class HuggingFaceDataset(Dataset):
    def __init__(self, dataset_name: str = "rokmr/cricket-shot", split: str = "train", 
                 label_mapping: dict = None, sequence_length: int = 10, use_temporal: bool = True):
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Please install datasets: pip install datasets")
        
        self.dataset_name = dataset_name
        self.split = split
        self.sequence_length = sequence_length
        self.use_temporal = use_temporal
        self.label_mapping = label_mapping or {}
        self.scaler = StandardScaler()
        
        logger.info(f"Loading dataset: {dataset_name} (split: {split})")
        self.dataset = load_dataset(dataset_name, split=split)
        
        self.samples = []
        self._process_dataset()

    def _process_dataset(self):
        for idx, example in enumerate(self.dataset):
            try:
                label_name = example.get('label', example.get('shot_type', None))
                
                if label_name is None:
                    logger.warning(f"Sample {idx} has no label")
                    continue
                
                if isinstance(label_name, int):
                    label_id = label_name
                else:
                    label_id = self.label_mapping.get(label_name, None)
                    if label_id is None:
                        logger.debug(f"Unknown label: {label_name}")
                        continue
                
                if 'landmarks' in example:
                    landmarks = example['landmarks']
                    if isinstance(landmarks, list):
                        landmarks = np.array(landmarks)
                    elif isinstance(landmarks, dict):
                        landmarks = np.array(list(landmarks.values()))
                    
                    if landmarks.ndim == 1:
                        landmarks = landmarks.reshape(1, -1)
                    
                    self.samples.append((landmarks, label_id))
                
                elif 'image' in example:
                    logger.debug(f"Sample {idx} contains image (requires pose extraction)")
                    continue
                
                if (idx + 1) % 100 == 0:
                    logger.info(f"Processed {idx + 1} samples")
            
            except Exception as e:
                logger.debug(f"Error processing sample {idx}: {e}")
                continue
        
        logger.info(f"Total samples loaded: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data, label = self.samples[idx]
        
        if len(data.shape) == 1:
            data = data.reshape(1, -1)
        
        if self.use_temporal and data.shape[0] < self.sequence_length:
            pad_length = self.sequence_length - data.shape[0]
            data = np.vstack([data, np.tile(data[-1], (pad_length, 1))])
        elif self.use_temporal and data.shape[0] > self.sequence_length:
            data = data[:self.sequence_length]
        
        data = torch.FloatTensor(data)
        label = torch.LongTensor([label])
        
        return data, label.squeeze()

    def normalize(self):
        all_data = []
        for data, _ in self.samples:
            if data.ndim == 1:
                all_data.append(data)
            else:
                all_data.extend(data)
        
        if all_data:
            all_data = np.array(all_data)
            self.scaler.fit(all_data)
            logger.info("Scaler fitted")
        else:
            logger.warning("No data to fit scaler")


class DataPreprocessor:
    def __init__(self, raw_data_dir: str, processed_data_dir: str):
        self.raw_dir = Path(raw_data_dir)
        self.processed_dir = Path(processed_data_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def preprocess_video_landmarks(self, video_landmarks: np.ndarray, label: str):
        (self.processed_dir / label).mkdir(exist_ok=True)
        
        if len(video_landmarks.shape) == 1:
            video_landmarks = video_landmarks.reshape(1, -1)
        
        output_path = self.processed_dir / label / f"{label}_{len(list(self.processed_dir.glob('*.npy')))}.npy"
        np.save(output_path, video_landmarks)
        
        return output_path

    def create_sequence_data(self, landmarks_list: list, labels: list, sequence_length: int = 10):
        sequences = []
        sequence_labels = []
        
        for landmarks, label in zip(landmarks_list, labels):
            if len(landmarks) < sequence_length:
                continue
            
            for i in range(len(landmarks) - sequence_length + 1):
                seq = landmarks[i:i+sequence_length]
                sequences.append(seq)
                sequence_labels.append(label)
        
        return np.array(sequences), np.array(sequence_labels)


def get_data_loaders(data_dir: str = None, label_mapping: dict = None, batch_size: int = 32, 
                     validation_split: float = 0.2, num_workers: int = 0,
                     use_huggingface: bool = False, dataset_name: str = "rokmr/cricket-shot"):
    
    if use_huggingface:
        logger.info("Loading from HuggingFace...")
        dataset = HuggingFaceDataset(
            dataset_name=dataset_name,
            split="train",
            label_mapping=label_mapping
        )
    else:
        logger.info("Loading from local directory...")
        dataset = CricketPoseDataset(data_dir, label_mapping)
    
    dataset.normalize()
    
    num_samples = len(dataset)
    val_size = int(num_samples * validation_split)
    train_size = num_samples - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, dataset
