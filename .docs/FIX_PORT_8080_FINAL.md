# Final Fix for Port 8080 Error

## 🚨 Problem

Even after running `stop_api_server.ps1`, you're still getting:
```
ERROR: [Errno 10048] error while attempting to bind on address ('0.0.0.0', 8080)
```

## ✅ Solutions (Try in Order)

### Solution 1: Use Fixed Stop Script (Recommended)

I've fixed the bug in `stop_api_server.ps1`. Try it again:

```powershell
.\stop_api_server.ps1
```

Wait 3 seconds, then start the server:
```powershell
python src/inference.py
```

### Solution 2: Use Aggressive Kill Script

If Solution 1 doesn't work, use the more aggressive script:

```powershell
.\kill_port_8080.ps1
```

This will:
- Find all processes using port 8080 via netstat
- Kill them forcefully
- Optionally kill all Python processes (if you confirm)

### Solution 3: Manual Windows Commands

**Step 1: Find what's using port 8080**
```powershell
netstat -ano | findstr :8080
```

You'll see output like:
```
TCP    0.0.0.0:8080    0.0.0.0:0    LISTENING    12345
```

The last number (12345) is the Process ID (PID).

**Step 2: Kill the process**
```powershell
taskkill /F /PID 12345
```

Replace `12345` with the actual PID from Step 1.

### Solution 4: Use Task Manager

1. Open **Task Manager** (Ctrl+Shift+Esc)
2. Go to **Details** tab
3. Click **PID** column header to sort by PID
4. Find the PID from `netstat` command above
5. Right-click → **End Task** or **End Process Tree**

### Solution 5: Nuclear Option - Kill All Python

```powershell
# Kill ALL Python processes (be careful!)
Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force
```

**Warning:** This kills ALL Python processes, not just the API server.

### Solution 6: Restart Terminal/Computer

If nothing else works:
1. Close ALL terminal windows
2. Reopen PowerShell
3. Start the server again

Or restart your computer (last resort).

---

## 🔍 Verify Port is Free

After stopping processes, verify port 8080 is free:

```powershell
netstat -ano | findstr :8080
```

**If this shows nothing**, the port is free! ✅

**If this still shows a process**, use Solution 2 or 3 above.

---

## 🎯 Complete Workflow

```powershell
# Step 1: Stop existing processes
.\kill_port_8080.ps1
# OR
.\stop_api_server.ps1

# Step 2: Wait a moment
Start-Sleep -Seconds 3

# Step 3: Verify port is free
netstat -ano | findstr :8080
# Should show nothing

# Step 4: Start server
$env:OPENAI_API_KEY="sk-proj-your-key-here"
python src/inference.py
```

---

## 📝 Why This Happens

1. **Previous server didn't close properly** - Ctrl+C didn't work, process hung
2. **Multiple terminals running server** - You started server in multiple terminals
3. **Process crashed but didn't release port** - Port takes time to free up
4. **Windows port binding issue** - Sometimes Windows holds ports longer than expected

---

## ✅ Expected Success Output

After killing processes and starting server, you should see:

```
INFO:__main__:Successfully loaded trained model from checkpoint
INFO:     Started server process [XXXXX]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8080 (Press CTRL+C to quit)
```

**No port errors!** ✅

---

## 💡 Prevent This in Future

1. **Always stop server with Ctrl+C** before starting a new one
2. **Check port first**: `netstat -ano | findstr :8080`
3. **Use only ONE terminal** for the API server
4. **Use the stop scripts** before restarting

---

**Try Solution 2 (`kill_port_8080.ps1`) - it's the most aggressive and should definitely work!** 💪



