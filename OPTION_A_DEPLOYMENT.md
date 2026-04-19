# 🎯 OPTION A: Frontend Local + Backend on Google Colab

## What You'll Get ✨

```
Your Laptop (Frontend)          Google Colab (Backend)
┌──────────────────┐            ┌────────────────────┐
│ React App        │            │ FastAPI Server     │
│ Port 3000        │ ─ngrok──→  │ Port 8000          │
│ Predictions UI   │ TUNNEL     │ + MediaPipe (GPU)  │
└──────────────────┘            │ + PyTorch Models   │
                                 └────────────────────┘
```

**Why this approach?**
- ✅ MediaPipe automatically downloads models in Colab
- ✅ Free GPU for faster predictions
- ✅ Keep UI on your laptop (fast & responsive)
- ✅ Backend runs 24/7 on Colab (with keep-alive)

---

## 📋 Complete Setup (7 Steps, ~10 minutes)

### Step 1: Get NGrok Token (2 min)

1. Go to: **https://dashboard.ngrok.com/signup**
2. Create free account
3. Go to **Auth Token** section
4. Copy your token (looks like: `2XXXXXX_XXXXXXXXXXXXXXXXXXXXXX`)
5. **Keep it safe** - you'll use it multiple times

---

### Step 2: Prepare Your Code (1 min)

**Option A - Upload as ZIP:**
```bash
# Create ZIP of your project
# (or just drag-drop folder to compress)
```

**Option B - Use GitHub:**
```bash
# Push to GitHub first
git push origin main
```

---

### Step 3: Create Colab Notebook (2 min)

1. Open: **https://colab.research.google.com**
2. Click **"New notebook"**
3. In the first cell, paste **all** the code from:
   `COLAB_COPY_PASTE.py` (in your project folder)
4. Replace this line with your actual token:
   ```python
   NGROK_TOKEN = "YOUR_NGROK_TOKEN_HERE"
   ```
   ↓ becomes ↓
   ```python
   NGROK_TOKEN = "2XXXXXX_XXXXXXXXXXXXXXXXXXXXXX"
   ```

---

### Step 4: Run the Colab Cell (5 min)

1. Press **Ctrl+Enter** (or click ▶ button)
2. Select **GPU runtime** if asked
3. Wait for output...
4. You'll see:
   ```
   ✅ BACKEND IS RUNNING!
   
   🌐 YOUR PUBLIC API URL (copy this!):
   
      https://abc123def456.ngrok.io
   ```
5. **Copy that URL** (you'll need it next)

---

### Step 5: Update Frontend Config (1 min)

1. On your laptop, open:
   ```
   frontend/src/config.js
   ```

2. Find this line:
   ```javascript
   export const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
   ```

3. Replace with your Colab URL:
   ```javascript
   export const API_URL = 'https://abc123def456.ngrok.io';
   ```

4. Save the file

---

### Step 6: Start Frontend (1 min)

On your laptop, open terminal and run:

```bash
cd c:\Users\hp\amd-hackathon\frontend
npm start
```

This opens: **http://localhost:3000**

---

### Step 7: Test It! (1 min)

1. Go to **http://localhost:3000** in your browser
2. Upload a cricket shot image
3. Should see:
   ```
   ✓ Shot detected: Hook
   ✓ Confidence: 75%
   ```

🎉 **Success!** Your frontend is talking to Colab!

---

## 🔧 File Reference

| File | Purpose | Updated? |
|------|---------|----------|
| `frontend/src/config.js` | Centralized API URL config | ✅ Ready |
| `frontend/src/App.js` | Health check & classes | ✅ Updated |
| `frontend/src/components/ImagePredictor.js` | Image prediction | ✅ Updated |
| `frontend/src/components/VideoPredictor.js` | Video prediction | ✅ Updated |
| `frontend/src/components/ModelInfo.js` | Model info display | ✅ Updated |
| `backend/app.py` | FastAPI server | ✅ CORS enabled |
| `COLAB_COPY_PASTE.py` | Ready-to-paste Colab code | ✅ New |

---

## ⚠️ Important Notes

### Keep Colab Tab OPEN
- If you close the Colab tab, ngrok URL expires
- The frontend will show connection error
- Just run the Colab cell again for a new URL

### Colab Session Timeout
- Free Colab shuts down after 12 hours of inactivity
- Your URL will expire
- Run the cell again to restart backend with new URL

### CORS (Already Configured ✅)
- Backend already has CORS enabled
- Requests from localhost:3000 will work

---

## 🆘 Troubleshooting

### ❌ "Failed to fetch from API"

**Check:**
1. Colab notebook is still running (tab is open)
2. API URL in `config.js` is correct and current
3. Browser console (F12) for exact error

**Fix:**
```
1. Go to Colab
2. Check if still running
3. If not, run cell again
4. Get new URL
5. Update config.js
6. Refresh frontend (Ctrl+R)
```

### ❌ "CORS Error" / "No Access-Control-Allow-Origin"

**Backend already has this enabled**, but if you see it:

Add to `backend/app.py`:
```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### ❌ "Backend not starting"

1. Check Colab cell output for errors
2. Verify path to `backend/app.py` is correct
3. Make sure all dependencies installed
4. Try running cell again

### ❌ "Ngrok auth token not recognized"

1. Go to https://dashboard.ngrok.com/auth/your-authtoken
2. Copy fresh token
3. Replace in Colab
4. Run cell again

---

## 🚀 Next Steps

Once everything is working:

1. **Finetune Models on Colab GPU**
   - Colab provides free GPU/TPU
   - Much faster training
   - Keep models in Drive to persist

2. **Real Pose Detection Works Now!**
   - MediaPipe models auto-download in Colab
   - Your predictions should be accurate now
   - See actual cricket shots, not random

3. **Optimize Backend**
   - Add model caching
   - Batch predictions
   - Monitor performance

4. **Production Deployment**
   - Deploy backend to cloud permanently
   - Use production ngrok plan (if needed)
   - Or use Google Cloud Run / AWS Lambda

---

## 📚 Files in Your Project

After setup, you should have:

```
amd-hackathon/
├── frontend/
│   ├── src/
│   │   ├── config.js ⭐ (UPDATED - your API URL here)
│   │   ├── App.js ⭐ (UPDATED - uses API_URL)
│   │   └── components/
│   │       ├── ImagePredictor.js ⭐ (UPDATED)
│   │       ├── VideoPredictor.js ⭐ (UPDATED)
│   │       └── ModelInfo.js ⭐ (UPDATED)
│   └── package.json
├── backend/
│   ├── app.py (CORS enabled ✅)
│   ├── models/
│   ├── utils/
│   └── config.py
├── COLAB_COPY_PASTE.py ⭐ (Copy to Colab)
├── OPTION_A_QUICKSTART.py ⭐ (Quick reference)
└── COLAB_SETUP_GUIDE_OPTION_A.py ⭐ (Detailed guide)
```

---

## ✅ Checklist

Before you start:

- [ ] Ngrok token obtained
- [ ] Colab account ready
- [ ] Backend code prepared (ZIP or GitHub)
- [ ] Laptop has Node.js (for npm start)
- [ ] Browser ready for testing

After setup:

- [ ] Colab cell running
- [ ] Public URL obtained
- [ ] Frontend config.js updated
- [ ] Frontend `npm start` running
- [ ] Image prediction working

---

## 💬 Questions?

1. **Frontend not loading?** → Check npm start output
2. **Predictions failing?** → Check browser console (F12)
3. **Colab erroring?** → Check cell output for stack trace
4. **URL expired?** → Run Colab cell again

---

## 🎓 What You Learned

- How to configure multi-environment deployments
- How to use ngrok for secure tunneling
- How Google Colab can provide free GPU
- How to keep frontend and backend separate
- How to make your code environment-agnostic

**Next:** Try running this setup and get your predictions working with real poses! 🚀
