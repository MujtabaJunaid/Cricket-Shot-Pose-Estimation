import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import json
import logging
from tqdm import tqdm
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, model: nn.Module, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.model = model.to(device)
        self.device = device
        self.best_accuracy = 0
        self.writer = None
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }

    def train_epoch(self, train_loader, optimizer, criterion):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        progress_bar = tqdm(train_loader, desc="Training")
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            optimizer.zero_grad()
            
            if inputs.dim() == 2:
                outputs = self.model(inputs)
            else:
                outputs = self.model(inputs)
            
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar.set_postfix({'loss': total_loss/(batch_idx+1), 'acc': 100.*correct/total})

        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy

    def validate(self, val_loader, criterion):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc="Validation"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                if inputs.dim() == 2:
                    outputs = self.model(inputs)
                else:
                    outputs = self.model(inputs)
                
                loss = criterion(outputs, targets)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy

    def fit(self, train_loader, val_loader, epochs: int, learning_rate: float, 
            checkpoint_dir: str = "models/checkpoints", patience: int = 10):
        
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
        
        patience_counter = 0
        
        for epoch in range(1, epochs + 1):
            logger.info(f"\nEpoch {epoch}/{epochs}")
            
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)
            val_loss, val_acc = self.validate(val_loader, criterion)
            
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_acc'].append(val_acc)
            
            logger.info(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
            logger.info(f"Val - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
            
            scheduler.step(val_acc)
            
            if val_acc > self.best_accuracy:
                self.best_accuracy = val_acc
                patience_counter = 0
                self.save_checkpoint(Path(checkpoint_dir) / "best_model.pt")
                logger.info(f"New best model saved with accuracy: {val_acc:.2f}%")
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch} epochs")
                break

    def save_checkpoint(self, checkpoint_path: str):
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state': self.model.state_dict(),
            'history': self.training_history,
            'best_accuracy': self.best_accuracy
        }, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.training_history = checkpoint.get('history', {})
        self.best_accuracy = checkpoint.get('best_accuracy', 0)
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
