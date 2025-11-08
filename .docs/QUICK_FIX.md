# Quick Fix for Port 8080 Error

## 🚨 Error You're Seeing

```
ERROR: [Errno 10048] error while attempting to bind on address ('0.0.0.0', 8080): 
only one usage of each socket address (protocol/network address/port) is normally permitted
```

## ⚡ Quick Fix (30 seconds)

**Just run this:**

```powershell
.\stop_api_server.ps1
```

Wait 2 seconds, then start the server again:

```powershell
$env:OPENAI_API_KEY="sk-proj-your-key-here"
python src/inference.py
```

**Done!** ✅

---

## 📋 What I Fixed

1. ✅ **Created `stop_api_server.ps1`** - Automatically stops processes on port 8080
2. ✅ **Created `check_port.ps1`** - Check what's using port 8080
3. ✅ **Fixed LangChain warnings** - Updated imports in `src/rag_pipeline.py`
4. ✅ **Note about RAG database** - Missing database is OK (uses templates as fallback)

---

## 🎯 Next Steps

1. **Stop existing server**: `.\stop_api_server.ps1`
2. **Start API server**: `python src/inference.py`
3. **Start frontend** (new terminal): `python serve_frontend.py`
4. **Open browser**: `http://localhost:3000/index.html`

---

**The port error is now easily fixable!** 🚀



