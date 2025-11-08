# Model Checkpoint Deployment Guide

## 📦 Model Checkpoint Location

**Best Model Checkpoint:**
```
1/7b2108ed09bf401fa06ff1b7d8c1e949/checkpoints/dr-model-epoch=60-val_qwk=0.853.ckpt
```

## 🚨 GitHub File Size Limits

- **GitHub Free**: 100MB per file
- **GitHub Pro**: 100MB per file (Git LFS required for larger files)
- **Warning**: Files > 50MB trigger warnings
- **Git LFS**: Required for files > 100MB

## 📊 Options for Model Deployment

### Option 1: Git LFS (Recommended for Large Files)

If checkpoint > 100MB, use Git LFS:

```bash
# Install Git LFS
git lfs install

# Track .ckpt files
git lfs track "*.ckpt"
git lfs track "1/**/*.ckpt"

# Add .gitattributes
git add .gitattributes

# Add checkpoint
git add 1/7b2108ed09bf401fa06ff1b7d8c1e949/checkpoints/dr-model-epoch=60-val_qwk=0.853.ckpt

# Commit and push
git commit -m "Add best model checkpoint via Git LFS"
git push origin main
```

**Pros:**
- ✅ Handles large files automatically
- ✅ Works with GitHub
- ✅ No external dependencies

**Cons:**
- ⚠️ Requires Git LFS installation
- ⚠️ Uses GitHub LFS bandwidth quota

### Option 2: Host Separately (Best for Very Large Files)

Upload checkpoint to cloud storage and download during deployment:

**Cloud Storage Options:**
- Google Drive (public link)
- Dropbox (public link)
- AWS S3
- Google Cloud Storage
- Azure Blob Storage

**Deployment Script:**
```python
# download_model.py
import os
import requests
from pathlib import Path

def download_checkpoint():
    """Download model checkpoint if not exists."""
    checkpoint_path = Path("1/7b2108ed09bf401fa06ff1b7d8c1e949/checkpoints/dr-model-epoch=60-val_qwk=0.853.ckpt")
    
    if checkpoint_path.exists():
        print("Checkpoint already exists")
        return
    
    # Download from cloud storage
    url = os.getenv("MODEL_CHECKPOINT_URL", "https://your-cloud-storage.com/model.ckpt")
    
    print("Downloading checkpoint...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    print("Checkpoint downloaded successfully")

if __name__ == "__main__":
    download_checkpoint()
```

**Pros:**
- ✅ No GitHub size limits
- ✅ Faster deployments (download only when needed)
- ✅ Can version models separately

**Cons:**
- ⚠️ Requires external hosting
- ⚠️ Need download script

### Option 3: Include in Repo (If < 100MB)

If checkpoint < 100MB, you can include it directly:

```bash
# Remove .ckpt from .gitignore temporarily
# Edit .gitignore to allow this specific file

# Add checkpoint
git add -f 1/7b2108ed09bf401fa06ff1b7d8c1e949/checkpoints/dr-model-epoch=60-val_qwk=0.853.ckpt

# Commit
git commit -m "Add best model checkpoint"
git push origin main
```

**Pros:**
- ✅ Simple, no extra steps
- ✅ Always available in repo

**Cons:**
- ⚠️ Only works if < 100MB
- ⚠️ Increases repo size

### Option 4: Compress and Include

Compress the checkpoint before uploading:

```bash
# Compress checkpoint
7z a -mx=9 model-checkpoint.7z 1/7b2108ed09bf401fa06ff1b7d8c1e949/checkpoints/dr-model-epoch=60-val_qwk=0.853.ckpt

# Add compressed file
git add model-checkpoint.7z
git commit -m "Add compressed model checkpoint"
git push origin main
```

**Deployment:**
```bash
# Extract during deployment
7z x model-checkpoint.7z
```

**Pros:**
- ✅ Reduces file size significantly
- ✅ Can fit under 100MB limit

**Cons:**
- ⚠️ Requires compression/decompression
- ⚠️ Extra step in deployment

## 🎯 Recommended Approach

### For GitHub Deployment:

1. **Check file size first**
   ```bash
   # Check size
   ls -lh 1/7b2108ed09bf401fa06ff1b7d8c1e949/checkpoints/dr-model-epoch=60-val_qwk=0.853.ckpt
   ```

2. **If < 100MB**: Include directly (Option 3)
3. **If > 100MB**: Use Git LFS (Option 1) or host separately (Option 2)

### For Production Deployment:

- **Docker**: Include in Docker image or download during build
- **Cloud Platforms**: Download from cloud storage during deployment
- **VPS**: Clone repo and download checkpoint separately

## 📝 Update .gitignore

If including checkpoint, update `.gitignore`:

```gitignore
# Model checkpoints - exclude all except best model
*.ckpt
!1/7b2108ed09bf401fa06ff1b7d8c1e949/checkpoints/dr-model-epoch=60-val_qwk=0.853.ckpt
```

Or use Git LFS and keep `.ckpt` in `.gitignore` (LFS handles it).

## 🔧 Update Deployment Scripts

Update deployment to handle checkpoint:

```bash
# In Dockerfile or deployment script
# Option 1: Already in repo
# No action needed

# Option 2: Download from cloud
python download_model.py

# Option 3: Extract compressed
7z x model-checkpoint.7z
```

## ✅ Checklist

- [ ] Check checkpoint file size
- [ ] Choose deployment option
- [ ] Update `.gitignore` if needed
- [ ] Add checkpoint to repo
- [ ] Test deployment with checkpoint
- [ ] Verify model loads correctly

