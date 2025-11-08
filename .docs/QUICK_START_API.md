# Quick Start API Server

## ⚡ Fast Start

**Run this command:**

```powershell
cd "C:\Users\pathi\Documents\DR assistant"
# Get your key from: https://platform.openai.com/api-keys
$env:OPENAI_API_KEY="sk-proj-your-key-here"
python src/inference.py
```

## ✅ Wait for This Message

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8080 (Press CTRL+C to quit)
```

**Once you see this, the API server is running!** ✅

## 🔍 Verify It's Running

Open browser: `http://localhost:8080/health`

Should show: `{"status": "healthy"}`

---

**That's it! API server is now running on port 8080.** 🎉


