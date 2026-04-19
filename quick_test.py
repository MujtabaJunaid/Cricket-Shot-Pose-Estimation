import cv2
import numpy as np
import requests
import os

print("=" * 60)
print("TESTING API ON PORT 8000")
print("=" * 60)

# Create test image
img = np.ones((480, 640, 3), dtype=np.uint8) * 255
cv2.rectangle(img, (100, 100), (200, 200), (0, 255, 0), 2)
cv2.circle(img, (320, 240), 50, (0, 0, 255), 2)
test_img_path = 'test_img.jpg'
cv2.imwrite(test_img_path, img)
print(f"\n✓ Test image created")

# Test health
print("\nTesting health endpoint...")
r = requests.get('http://localhost:8000/health')
print(f"  Status: {r.status_code} - {r.json()}")

# Test prediction
print("\nTesting prediction endpoint...")
with open(test_img_path, 'rb') as f:
    r = requests.post('http://localhost:8000/predict/image', files={'file': f})

print(f"  Status: {r.status_code}")
if r.status_code == 200:
    data = r.json()
    print(f"  ✓ Prediction successful!")
    print(f"    - Shot: {data.get('shot_class')}")
    print(f"    - Confidence: {data.get('confidence'):.2%}")
else:
    print(f"  ✗ Error: {r.text}")

os.remove(test_img_path)
print("\n" + "=" * 60)
