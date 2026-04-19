# ============ COPY ALL THIS CODE INTO COLAB ============
# Run this in a single cell

# Install dependencies
!pip install fastapi uvicorn mediapipe opencv-python numpy torch torchvision -q

# Mount Google Drive (optional - if you upload files via Drive)
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# Upload your backend code
from google.colab import files
print("📁 Upload your project folder as ZIP file...")
uploaded = files.upload()

# Extract uploaded file
import os
for filename in uploaded.keys():
    if filename.endswith('.zip'):
        !unzip -q "{filename}"
        print(f"✓ Extracted {filename}")
        break

# Install NGrok
!pip install pyngrok -q

print("\\n🚀 Starting backend setup...\\n")

# ============ BACKEND START ============

import subprocess
import time
from pyngrok import ngrok
import requests

# ⚠️  REPLACE THIS WITH YOUR ACTUAL NGROK TOKEN
NGROK_TOKEN = "YOUR_NGROK_TOKEN_HERE"

if NGROK_TOKEN == "YOUR_NGROK_TOKEN_HERE":
    print("❌ ERROR: Replace NGROK_TOKEN with your actual token!")
    print("   Get it from: https://dashboard.ngrok.com/auth/your-authtoken")
else:
    ngrok.set_auth_token(NGROK_TOKEN)
    
    # Find backend directory
    backend_path = None
    for root, dirs, files in os.walk('/content'):
        if 'app.py' in files and 'backend' in root:
            backend_path = os.path.dirname(root)
            break
    
    if backend_path is None:
        print("❌ Could not find backend/app.py")
        print("   Available folders:")
        !ls -la /content
    else:
        print(f"✓ Found backend at: {backend_path}")
        
        # Change to project directory
        os.chdir(backend_path)
        
        # Start backend
        print("\\n🔧 Starting FastAPI backend...")
        process = subprocess.Popen(
            ['python', '-m', 'uvicorn', 'backend.app:app', 
             '--host', '0.0.0.0', '--port', '8000'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for startup
        time.sleep(5)
        
        # Create public URL
        try:
            public_url = ngrok.connect(8000)
            print(f"\\n{'='*70}")
            print(f"✅ BACKEND IS RUNNING!")
            print(f"{'='*70}")
            print(f"\\n🌐 YOUR PUBLIC API URL (copy this!):")
            print(f"\\n   {public_url}")
            print(f"\\n{'='*70}")
            print(f"\\n📋 Next steps:")
            print(f"   1. Copy the URL above")
            print(f"   2. Update frontend/src/config.js")
            print(f"   3. Start frontend with: npm start")
            print(f"\\n💡 Keep this Colab tab OPEN while using the frontend!")
            print(f"   (If you close it, the URL expires)")
            print(f"{'='*70}")
            
            # Test health
            time.sleep(2)
            try:
                response = requests.get(f"{public_url}/health", timeout=5)
                health = response.json()
                print(f"\\n✓ Health Check:")
                print(f"  Status: {health.get('status', 'unknown')}")
                print(f"  Device: {health.get('device', 'unknown')}")
                print(f"  Model: {health.get('model_loaded', False)}")
            except Exception as e:
                print(f"\\n⚠️  Backend still starting (this is normal): {e}")
                print(f"   Try again in 5 seconds")
                
        except Exception as e:
            print(f"❌ Error creating ngrok tunnel: {e}")
            print(f"   Make sure your NGROK_TOKEN is correct")
