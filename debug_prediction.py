import cv2
import numpy as np
import torch
from backend.models.classifier import EnsembleClassifier
from backend.utils.shot_analyzer import ShotAnalyzer
from backend.config import SHOT_CLASSES, DEVICE
import traceback

print("=" * 60)
print("DEBUG: Testing prediction pipeline step by step")
print("=" * 60)

# Create a test image
print("\n1. Creating test image...")
img = np.ones((480, 640, 3), dtype=np.uint8) * 255
cv2.rectangle(img, (100, 100), (200, 200), (0, 255, 0), 2)
print(f"   ✓ Image shape: {img.shape}")

# Simulate landmarks extraction (since MediaPipe not available)
print("\n2. Creating fake landmarks...")
try:
    landmarks = np.random.randn(99).astype(np.float32)
    print(f"   ✓ Landmarks shape: {landmarks.shape}, dtype: {landmarks.dtype}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    traceback.print_exc()

# Test normalization
print("\n3. Normalizing landmarks...")
try:
    normalized = ShotAnalyzer.normalize_landmarks(landmarks)
    print(f"   ✓ Normalized shape: {normalized.shape}, dtype: {normalized.dtype}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    traceback.print_exc()

# Test angle extraction
print("\n4. Extracting angle features...")
try:
    angles = ShotAnalyzer.extract_angle_features(landmarks)
    print(f"   ✓ Angles shape: {angles.shape}")
    print(f"   Angles: {angles}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    traceback.print_exc()

# Test classifier
print("\n5. Loading classifier...")
try:
    classifier = EnsembleClassifier(num_classes=len(SHOT_CLASSES), device=DEVICE)
    print(f"   ✓ Classifier loaded on device: {DEVICE}")
except Exception as e:
    print(f"   ✗ Error: {e}")
    traceback.print_exc()

# Test classifier forward pass
print("\n6. Running classifier forward pass...")
try:
    print(f"   Input normalized landmarks shape: {normalized.shape}")
    print(f"   Input dtype: {normalized.dtype}")
    
    with torch.no_grad():
        prediction, confidence, probabilities = classifier.forward(normalized)
    
    print(f"   ✓ Prediction: {prediction}")
    print(f"   ✓ Confidence: {confidence:.4f}")
    print(f"   ✓ Probabilities shape: {probabilities.shape}")
    print(f"   ✓ Shot class: {SHOT_CLASSES.get(int(prediction), 'unknown')}")
except Exception as e:
    print(f"   ✗ Error in forward pass: {type(e).__name__}: {e}")
    traceback.print_exc()

print("\n" + "=" * 60)
print("END DEBUG")
print("=" * 60)
