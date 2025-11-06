# Start API Server - Quick Guide

## 🚀 Start the API Server

**Open a terminal** and run:

### Option 1: Use the Startup Script (Recommended)
```powershell
.\start_api_server.ps1
```

This will:
- Check if OpenAI API key is set
- Start the API server
- Show you if there are any issues

### Option 2: Manual Start
```powershell
cd "C:\Users\pathi\Documents\DR assistant"

# Set OpenAI API key (REQUIRES QUOTES!)
# Get your key from: https://platform.openai.com/api-keys
$env:OPENAI_API_KEY="sk-proj-your-key-here"

# Start API server
python src/inference.py
```

## ✅ What You Should See

**Successful startup:**
```
INFO:__main__:Loading trained model from 1/7d0928bb87954a739123ca35fa03cccf/checkpoints\dr-model-epoch=11-val_qwk=0.769.ckpt
INFO:__main__:Successfully loaded trained model from checkpoint
INFO:     Started server process [XXXXX]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8080 (Press CTRL+C to quit)
```

## 🔍 Verify Server is Running

**Option 1: Check in Browser**
```
http://localhost:8080/health
```

Should return: `{"status": "healthy"}`

**Option 2: Check API Documentation**
```
http://localhost:8080/docs
```

Should show the FastAPI interactive documentation.

## 🐛 Troubleshooting

### Port 8080 Already in Use
If you see:
```
ERROR: [Errno 10048] error while attempting to bind on address ('0.0.0.0', 8080)
```

**Fix:**
```powershell
# Stop existing processes
.\kill_port_8080.ps1

# Wait 3 seconds, then start again
python src/inference.py
```

### OpenAI API Key Not Set
If you see warnings about RAG features:
- **OK**: Server will still work, but RAG hints will use templates
- **To enable RAG**: Set `$env:OPENAI_API_KEY="your-key-here"` (with quotes!)

### Model Checkpoint Not Found
If you see:
```
Model not found at ... using untrained model
```

**Fix:**
- Verify checkpoint exists at: `1/7d0928bb87954a739123ca35fa03cccf/checkpoints/dr-model-epoch=11-val_qwk=0.769.ckpt`
- Or train a new model first

## 📋 Complete Startup Sequence

**Terminal 1 (API Server):**
```powershell
cd "C:\Users\pathi\Documents\DR assistant"
$env:OPENAI_API_KEY="sk-proj-your-key-here"
python src/inference.py
```

**Terminal 2 (Frontend - AFTER API is running):**
```powershell
cd "C:\Users\pathi\Documents\DR assistant"
python serve_frontend.py
```

**Browser:**
```
http://localhost:3000/index.html
```

## ✅ Success Indicators

- ✅ Terminal shows: `Uvicorn running on http://0.0.0.0:8080`
- ✅ Browser shows: `http://localhost:8080/health` returns `{"status": "healthy"}`
- ✅ Frontend shows: "🟢 API Connected" (green)

---

**Start the API server now and you're good to go!** 🚀


