# 🚀 Best Deployment Strategy for DR Assistant

## Current Problem

- **Docker image size:** 9.3 GB
- **Railway free tier limit:** 4.0 GB
- **Main culprits:** MLflow runs, model checkpoints, large data files

## 🎯 Recommended Strategy: Split Deployment + External Model Storage

### Architecture

```
┌─────────────────┐         ┌─────────────────┐
│                 │         │                 │
│   Netlify       │  ────>  │   Railway API   │
│   (Frontend)    │         │   (Backend)     │
│   (Free)        │         │   (Optimized)   │
│                 │         │                 │
└─────────────────┘         └─────────────────┘
                                       │
                                       ▼
                            ┌─────────────────┐
                            │  Model Storage  │
                            │  (GitHub LFS/   │
                            │   Cloud)        │
                            └─────────────────┘
```

## ✅ Solution 1: Optimize Docker Image + External Model (Recommended)

### Step 1: Create `.dockerignore`

✅ **Already created!** This excludes:
- MLflow runs (`1/`)
- Model checkpoints
- Data files
- Logs and outputs
- Documentation
- Frontend files

**Expected image size:** ~2-3 GB (down from 9.3 GB)

### Step 2: Store Model Externally

**Option A: GitHub Releases (Free)**
1. Upload model checkpoint to GitHub Releases
2. Download at runtime in `src/inference.py`
3. Cache locally after first download

**Option B: Cloud Storage (Free tiers available)**
- Google Drive (15 GB free)
- Dropbox (2 GB free)
- AWS S3 (5 GB free tier)
- Cloudflare R2 (10 GB free)

### Step 3: Download Model at Runtime

Modify `src/inference.py` to download model if not present:

```python
def _download_model_if_needed(self):
    """Download model checkpoint if not present."""
    checkpoint_path = "1/7b2108ed09bf401fa06ff1b7d8c1e949/checkpoints/dr-model-epoch=60-val_qwk=0.853.ckpt"
    
    if not os.path.exists(checkpoint_path):
        logger.info("Model checkpoint not found. Downloading...")
        # Download from GitHub Releases or cloud storage
        # Implementation depends on storage choice
        pass
```

## ✅ Solution 2: Use Render.com Instead (Alternative)

**Render.com** has more flexible limits:
- **Free tier:** 512 MB RAM, but no strict image size limit
- **Paid tier:** $7/month, 512 MB RAM, no image size limit
- Better for ML applications

### Steps:
1. Go to https://render.com
2. Create Web Service
3. Connect GitHub repo
4. Build command: `pip install -r requirements.txt`
5. Start command: `python src/inference.py`
6. Add environment variables

## ✅ Solution 3: Hybrid Approach (Best for Production)

### Frontend: Netlify (Free)
- Static HTML/JS/CSS
- Fast global CDN
- Free tier: 100 GB bandwidth/month

### API: Render.com or Railway (Optimized)
- Use `.dockerignore` to reduce image size
- Store model externally
- Download model at runtime

### Model Storage: GitHub Releases
- Free
- Version control
- Easy to update

## 📋 Step-by-Step: Optimized Railway Deployment

### 1. Prepare Model for External Storage

**Upload to GitHub Releases:**
```bash
# Create a release on GitHub
# Upload: dr-model-epoch=60-val_qwk=0.853.ckpt
# Get download URL
```

### 2. Update Code to Download Model

**Modify `src/inference.py`:**

```python
import os
import requests
from pathlib import Path

def download_model_checkpoint(url: str, save_path: str):
    """Download model checkpoint from URL."""
    if os.path.exists(save_path):
        logger.info(f"Model already exists at {save_path}")
        return
    
    logger.info(f"Downloading model from {url}...")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(save_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    logger.info(f"Model downloaded to {save_path}")

# In DRPredictionService.__init__:
MODEL_URL = os.environ.get(
    'MODEL_CHECKPOINT_URL',
    'https://github.com/pathik1501/DR-assistant/releases/download/v1.0/dr-model-epoch=60-val_qwk=0.853.ckpt'
)
checkpoint_path = "1/7b2108ed09bf401fa06ff1b7d8c1e949/checkpoints/dr-model-epoch=60-val_qwk=0.853.ckpt"
download_model_checkpoint(MODEL_URL, checkpoint_path)
```

### 3. Deploy to Railway

1. **Push `.dockerignore` to GitHub**
2. **Railway will auto-deploy**
3. **Image size should be ~2-3 GB** ✅
4. **Model downloads on first API call**

### 4. Set Environment Variables in Railway

- `OPENAI_API_KEY`: Your OpenAI key
- `MODEL_CHECKPOINT_URL`: (Optional) Custom model URL

## 🎯 Quick Comparison

| Platform | Image Size Limit | Free Tier | Best For |
|----------|-----------------|-----------|----------|
| **Railway** | 4 GB | 500 hours/month | Small apps |
| **Render** | No strict limit | 512 MB RAM | ML apps |
| **Fly.io** | 10 GB | 3 VMs free | Large apps |
| **Google Cloud Run** | 10 GB | 2M requests/month | Enterprise |

## 📊 Expected Results

### Before Optimization:
- ❌ Image size: 9.3 GB
- ❌ Exceeds Railway limit
- ❌ Build fails

### After Optimization:
- ✅ Image size: ~2-3 GB
- ✅ Under Railway limit
- ✅ Build succeeds
- ✅ Model downloads at runtime (~215 MB)

## 🚀 Recommended Path Forward

1. **Immediate:** Use `.dockerignore` + Deploy to Railway
   - Image size: ~2-3 GB ✅
   - Should work on Railway free tier

2. **If still too large:** Store model externally
   - Download at runtime
   - Image size: ~1-2 GB ✅

3. **For production:** Use Render.com
   - No strict size limits
   - Better for ML apps
   - Still free tier available

## ✅ Next Steps

1. ✅ Create `.dockerignore` (Done!)
2. ⏳ Commit and push `.dockerignore`
3. ⏳ Railway will auto-redeploy
4. ⏳ Check new image size
5. ⏳ If still > 4 GB, implement model download

---

**This strategy should get you deployed successfully!** 🎉

