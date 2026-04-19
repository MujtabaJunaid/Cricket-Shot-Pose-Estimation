import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import pickle

class PoseClassifier(nn.Module):
    def __init__(self, num_classes: int = 10, input_size: int = 99):
        super(PoseClassifier, self).__init__()
        
        self.lstm1 = nn.LSTM(input_size, 256, batch_first=True, dropout=0.3)
        self.lstm2 = nn.LSTM(256, 128, batch_first=True, dropout=0.3)
        
        self.attention = nn.MultiheadAttention(128, 8, batch_first=True, dropout=0.3)
        
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)
        
        self.dropout = nn.Dropout(0.4)
        self.norm1 = nn.LayerNorm(256)
        self.norm2 = nn.LayerNorm(128)

    def forward(self, x):
        lstm1_out, _ = self.lstm1(x)
        lstm1_out = self.norm1(lstm1_out)
        
        lstm2_out, _ = self.lstm2(lstm1_out)
        lstm2_out = self.norm2(lstm2_out)
        
        attn_out, _ = self.attention(lstm2_out, lstm2_out, lstm2_out)
        attn_out = attn_out + lstm2_out
        
        pool_out = torch.mean(attn_out, dim=1)
        
        x = F.relu(self.fc1(pool_out))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


class StaticPoseClassifier(nn.Module):
    def __init__(self, num_classes: int = 10, input_size: int = 99):
        super(StaticPoseClassifier, self).__init__()
        
        self.fc1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, num_classes)
        
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = self.fc4(x)
        
        return x


class EnsembleClassifier:
    def __init__(self, num_classes: int = 10, device: str = "cpu"):
        self.num_classes = num_classes
        self.device = device
        
        self.temporal_model = PoseClassifier(num_classes).to(device)
        self.static_model = StaticPoseClassifier(num_classes).to(device)
        
        self.static_scaler = None
        self.temporal_scaler = None

    def forward(self, landmarks: np.ndarray) -> tuple:
        """Forward pass for ensemble prediction.
        
        Args:
            landmarks: np.ndarray of shape (99,) for single frame or (seq_len, 99) for sequences
            
        Returns:
            (prediction, confidence, probabilities)
        """
        # Ensure models are in eval mode
        self.temporal_model.eval()
        self.static_model.eval()
        
        landmarks_tensor = torch.FloatTensor(landmarks).to(self.device)
        
        # Handle single landmark (1D)
        if landmarks_tensor.dim() == 1:
            landmarks_tensor = landmarks_tensor.unsqueeze(0)  # (1, 99)
        
        try:
            with torch.no_grad():
                # For static model: input is (batch, features)
                static_out = self.static_model(landmarks_tensor)
                
                # For temporal model: input is (batch, seq_len, features)
                # If we have single frame, repeat it to create a sequence
                if landmarks_tensor.shape[0] == 1 and landmarks_tensor.dim() == 2:
                    # Single frame case - create a sequence of repeated frames
                    temporal_input = landmarks_tensor.unsqueeze(1).repeat(1, 10, 1)  # (1, 10, 99)
                else:
                    # Sequence case
                    temporal_input = landmarks_tensor.unsqueeze(1) if landmarks_tensor.dim() == 2 else landmarks_tensor
                
                temporal_out = self.temporal_model(temporal_input)
            
            ensemble_out = (static_out + temporal_out) / 2
            
            probabilities = F.softmax(ensemble_out, dim=1).cpu().numpy()
            prediction = np.argmax(probabilities[0])
            confidence = probabilities[0][prediction]
            
            return prediction, confidence, probabilities[0]
        except Exception as e:
            raise RuntimeError(f"Error in classifier forward pass: {type(e).__name__}: {e}")

    def save_checkpoint(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'temporal_state': self.temporal_model.state_dict(),
            'static_state': self.static_model.state_dict(),
            'num_classes': self.num_classes
        }, path)

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.temporal_model.load_state_dict(checkpoint['temporal_state'])
        self.static_model.load_state_dict(checkpoint['static_state'])
        self.temporal_model.eval()
        self.static_model.eval()

    def get_models(self):
        return self.temporal_model, self.static_model
