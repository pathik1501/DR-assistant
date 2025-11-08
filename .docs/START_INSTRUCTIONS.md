# How to Start the DR Assistant System

## 🚀 Quick Start Guide

The system consists of **two separate processes**:
1. **API Server** (Backend) - Port 8080
2. **Web Frontend** (UI) - Port 3000

---

## 📋 Step-by-Step Instructions

### Step 1: Start the API Server (Backend)

**Option A: Use the startup script**
```powershell
.\start_api_server.ps1
```

**Option B: Manual start**
```powershell
cd "C:\Users\pathi\Documents\DR assistant"

# Set OpenAI API key (REQUIRES QUOTES!)
$env:OPENAI_API_KEY="sk-proj-your-key-here"

# Start API server
python src/inference.py
```

**Important PowerShell Syntax:**
```powershell
# ✅ CORRECT (with quotes)
$env:OPENAI_API_KEY="sk-proj-abc123..."

# ❌ WRONG (without quotes - will cause error)
$env:OPENAI_API_KEY=sk-proj-abc123...
```

Wait for: `Uvicorn running on http://127.0.0.1:8080`

### Step 2: Start the Frontend Server (UI)

**Open a NEW terminal window** (keep API server running in Terminal 1)

**Option A: Use the startup script**
```powershell
.\start_frontend.ps1
```

**Option B: Manual start**
```powershell
cd "C:\Users\pathi\Documents\DR assistant"
python serve_frontend.py
```

Wait for: `🌍 Frontend URL: http://localhost:3000`

### Step 3: Open in Browser

Open your web browser and go to:
```
http://localhost:3000/index.html
```

Or just:
```
http://localhost:3000
```

---

## 🔑 Setting OpenAI API Key

### PowerShell (Correct Syntax)

```powershell
# Use quotes around the value!
# Get your key from: https://platform.openai.com/api-keys
$env:OPENAI_API_KEY="sk-proj-your-key-here"
```

**Common Error:**
```powershell
# ❌ This will FAIL
$env:OPENAI_API_KEY=sk-proj-abc...

# ✅ This is CORRECT
$env:OPENAI_API_KEY="sk-proj-abc..."
```

### Verify It's Set

```powershell
# Check if it's set
echo $env:OPENAI_API_KEY

# Should output your key (if set correctly)
```

---

## 🎯 Complete Startup Sequence

### Terminal 1 (API Server):
```powershell
cd "C:\Users\pathi\Documents\DR assistant"
$env:OPENAI_API_KEY="your-key-here"
python src/inference.py
```

### Terminal 2 (Frontend):
```powershell
cd "C:\Users\pathi\Documents\DR assistant"
python serve_frontend.py
```

### Browser:
```
http://localhost:3000/index.html
```

---

## ✅ Verify Everything is Running

1. **API Server**: Check `http://localhost:8080/health`
   - Should return: `{"status": "healthy"}`

2. **Frontend**: Check `http://localhost:3000`
   - Should show: "🟢 API Connected" in green

3. **Upload an image** and click "Analyze Image"

---

## 🐛 Troubleshooting

### Error: "API Not Connected"
- Make sure API server is running on port 8080
- Check Terminal 1 for errors
- Try accessing `http://localhost:8080/health` directly

### Error: PowerShell Command Not Found
- **Problem**: Missing quotes around API key
- **Solution**: Use `$env:OPENAI_API_KEY="your-key-here"` (with quotes!)

### Error: Port Already in Use
- **Problem**: Port 8080 or 3000 already in use
- **Solution**: 
  - Stop other servers using those ports
  - Or change ports in code

### Frontend Shows No Images
- **Problem**: API not returning heatmaps
- **Solution**: Check API logs for Grad-CAM errors
- Verify model checkpoint exists

---

## 📝 Summary

**Two Terminals Needed:**
1. **Terminal 1**: API Server (`python src/inference.py`)
2. **Terminal 2**: Frontend Server (`python serve_frontend.py`)

**Browser:**
- Open: `http://localhost:3000/index.html`

**Important:**
- Always use **quotes** when setting environment variables in PowerShell
- API key is optional (for RAG features)
- Both servers must be running simultaneously

---

**You're all set!** 🎉


