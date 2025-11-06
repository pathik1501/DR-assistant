# Diagnose All Issues - Complete Troubleshooting

## 🔍 Let's Find What's Wrong

Run these diagnostic commands one by one:

### 1. Check Ports
```powershell
# Check port 8080
netstat -ano | findstr :8080

# Check port 3000
netstat -ano | findstr :3000
```

**If ports are in use:** Stop processes with `.\kill_port_8080.ps1`

### 2. Check Files Exist
```powershell
# Check API file
Test-Path "src\inference.py"

# Check frontend file
Test-Path "serve_frontend.py"

# Check model checkpoint
Test-Path "1\7d0928bb87954a739123ca35fa03cccf\checkpoints\dr-model-epoch=11-val_qwk=0.769.ckpt"
```

### 3. Test Python Dependencies
```powershell
python -c "import fastapi; import streamlit; print('OK')"
```

### 4. Test API Manually
```powershell
# Start API server and check for errors
python src/inference.py
```

**Look for:**
- ✅ `INFO:     Uvicorn running on http://0.0.0.0:8080` = SUCCESS
- ❌ `ERROR: [Errno 10048]` = Port in use
- ❌ `ModuleNotFoundError` = Missing dependency
- ❌ `FileNotFoundError` = Missing file

### 5. Test Frontend Manually
```powershell
python serve_frontend.py
```

**Look for:**
- ✅ `🌍 Frontend URL: http://localhost:3000` = SUCCESS
- ❌ `[Errno 10048]` = Port in use
- ❌ `ModuleNotFoundError` = Missing dependency

---

## 🎯 Common Issues & Fixes

### Issue 1: Port Already in Use
**Symptom:** `ERROR: [Errno 10048]`

**Fix:**
```powershell
.\kill_port_8080.ps1
# Wait 3 seconds
python src/inference.py
```

### Issue 2: Module Not Found
**Symptom:** `ModuleNotFoundError: No module named 'X'`

**Fix:**
```powershell
pip install fastapi uvicorn streamlit requests
```

### Issue 3: Model Checkpoint Missing
**Symptom:** `Model not found at ... using untrained model`

**Fix:**
- Check if checkpoint exists
- Or use pretrained model (will work but less accurate)

### Issue 4: API Key Not Set
**Symptom:** RAG warnings (not critical, system still works)

**Fix:**
```powershell
$env:OPENAI_API_KEY="sk-proj-your-key-here"
```

---

## 📋 Complete Working Sequence

### Step 1: Clean Everything
```powershell
# Kill all processes on ports
.\kill_port_8080.ps1
# Kill frontend if needed
Get-Process python | Where-Object {$_.Path -like "*python*"} | Stop-Process -Force
```

### Step 2: Start API Server (Terminal 1)
```powershell
cd "C:\Users\pathi\Documents\DR assistant"
$env:OPENAI_API_KEY="sk-proj-your-key-here"
python src/inference.py
```

**Wait for:** `INFO:     Uvicorn running on http://0.0.0.0:8080`

### Step 3: Verify API
Open browser: `http://localhost:8080/health`
Should show: `{"status": "healthy"}`

### Step 4: Start Frontend (Terminal 2 - NEW)
```powershell
cd "C:\Users\pathi\Documents\DR assistant"
python serve_frontend.py
```

**Wait for:** `🌍 Frontend URL: http://localhost:3000`

### Step 5: Open Browser
```
http://localhost:3000/index.html
```

---

## 🆘 Still Not Working?

**Tell me:**
1. What error message do you see?
2. Which step fails?
3. What does `python src/inference.py` show?
4. What does `python serve_frontend.py` show?

Then I can help you fix the specific issue!


