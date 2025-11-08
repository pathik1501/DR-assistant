# Launch Frontend - Simple Guide

## 🚀 Start Frontend Server

**Open a NEW terminal window** (keep API server running in Terminal 1)

### Option 1: Use Script (Easiest)
```powershell
.\start_frontend.ps1
```

### Option 2: Run Directly
```powershell
cd "C:\Users\pathi\Documents\DR assistant"
python serve_frontend.py
```

## 🌐 Open Browser

Once the frontend starts, open:
```
http://localhost:3000/index.html
```

## ✅ Check Status

**Frontend should show:**
- 🟢 "API Connected" (green) if API server is running
- 🔴 "API Not Connected" (red) if API server is not running

**If red:**
- Make sure Terminal 1 (API server) is still running
- Check `http://localhost:8080/health` in browser

## 🎯 Using the Interface

1. Upload a retinal fundus image
2. Click "🔍 Analyze Image"
3. View results:
   - Classification (Grade 0-4)
   - Grad-CAM heatmaps
   - Clinical recommendation

---

**That's it! Frontend is now running on port 3000!** ✅



