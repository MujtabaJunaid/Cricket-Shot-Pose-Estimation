import cv2
import numpy as np
import requests
import os

# Create a simple test image (480x640 with some shapes)
img = np.ones((480, 640, 3), dtype=np.uint8) * 255
cv2.rectangle(img, (100, 100), (200, 200), (0, 255, 0), 2)
cv2.circle(img, (320, 240), 50, (0, 0, 255), 2)
cv2.putText(img, 'Test', (250, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

# Save test image
test_img_path = 'test_cricket.jpg'
cv2.imwrite(test_img_path, img)
print(f"✓ Created test image: {test_img_path}")

# Upload to API on port 5000
print("\nTesting /predict/image endpoint...")
with open(test_img_path, 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:5000/predict/image', files=files)
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"✓ Success: {result.get('success')}")
        if result.get('pose_found'):
            print(f"  Shot Class: {result.get('shot_class')}")
            print(f"  Confidence: {result.get('confidence'):.4f}")
            print(f"  Sample probabilities: {list(result.get('probabilities', {}).values())[:3]}")
        else:
            print(f"  Message: {result.get('message')}")
    else:
        print(f"✗ Error: {response.status_code}")
        print(f"  Response: {response.text}")

# Cleanup
os.remove(test_img_path)
