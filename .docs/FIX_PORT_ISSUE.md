# Fix Port 8080 Already in Use Error

## 🔴 Problem

You're seeing this error:
```
ERROR: [Errno 10048] error while attempting to bind on address ('0.0.0.0', 8080): 
only one usage of each socket address (protocol/network address/port) is normally permitted
```

This means **port 8080 is already in use** by another process.

---

## ✅ Solution: Stop Existing API Server

### Option 1: Use the Stop Script (Easiest)

```powershell
.\stop_api_server.ps1
```

This will automatically find and stop any processes using port 8080.

### Option 2: Check What's Using Port 8080

```powershell
.\check_port.ps1
```

This shows you which processes are using port 8080, then you can stop them manually.

### Option 3: Manual PowerShell Commands

```powershell
# Find processes using port 8080
Get-NetTCPConnection -LocalPort 8080 | Select-Object -ExpandProperty OwningProcess -Unique

# Stop the process (replace PID with actual process ID)
Stop-Process -Id <PID> -Force
```

### Option 4: Use Task Manager

1. Open **Task Manager** (Ctrl+Shift+Esc)
2. Go to **Details** tab
3. Find **Python** processes running `src/inference.py`
4. Right-click → **End Task**

---

## 🚀 Complete Startup Sequence

### Step 1: Stop Any Existing Servers

```powershell
.\stop_api_server.ps1
```

Wait 2-3 seconds.

### Step 2: Start API Server

```powershell
$env:OPENAI_API_KEY="sk-proj-your-key-here"
.\start_api_server.ps1
```

Or manually:
```powershell
$env:OPENAI_API_KEY="sk-proj-your-key-here"
python src/inference.py
```

### Step 3: Start Frontend (in a NEW terminal)

```powershell
.\start_frontend.ps1
```

Or manually:
```powershell
python serve_frontend.py
```

### Step 4: Open Browser

```
http://localhost:3000/index.html
```

---

## 🔧 Additional Fixes Applied

### 1. LangChain Deprecation Warnings (Fixed)

Updated `src/rag_pipeline.py` to use:
- `langchain_community.embeddings.OpenAIEmbeddings`
- `langchain_community.vectorstores.FAISS`
- `langchain_community.chat_models.ChatOpenAI` (when available)

These warnings are now fixed.

### 2. RAG Vector Database Missing (Non-Critical)

**Warning you're seeing:**
```
RAG pipeline initialization failed: Error: could not open data\vector_db\index.faiss
```

**This is OK!** The system will:
- Fall back to template-based clinical hints
- Still work perfectly for predictions
- You can create the vector database later if needed

To create it later (optional):
```powershell
python -c "from src.rag_pipeline import RAGPipeline; RAGPipeline()"
```

This will create the vector database.

---

## ✅ Quick Fix Summary

**If port 8080 is in use:**

1. Run: `.\stop_api_server.ps1`
2. Wait 2 seconds
3. Run: `python src/inference.py` again

**That's it!** The server should start successfully.

---

## 🎯 Expected Output After Fix

After stopping existing processes and restarting, you should see:

```
INFO:     Started server process [XXXXX]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8080 (Press CTRL+C to quit)
```

No more port errors! ✅

---

## 💡 Pro Tip

To prevent this in the future:
- Always stop the API server (Ctrl+C) before restarting
- Or use the stop script before starting a new one
- Check for existing processes with `.\check_port.ps1`



