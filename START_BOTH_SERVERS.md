# Start Both Servers - Complete Guide

## 🚀 Quick Start

You need **TWO terminals** running simultaneously:

### Terminal 1: API Server (Port 8080)
Already running! ✅

### Terminal 2: Frontend Server (Port 3000)

Open a **NEW terminal window** and run:

```powershell
cd "C:\Users\pathi\Documents\DR assistant"
python serve_frontend.py
```

Or use the script:
```powershell
.\start_frontend.ps1
```

## ✅ What You Should See

### Terminal 2 Output:
```
========================================
🌐 Frontend Server Started
========================================
📁 Serving from: C:\Users\pathi\Documents\DR assistant\frontend
🌍 Frontend URL: http://localhost:3000
📄 Open in browser: http://localhost:3000/index.html

⚠️  Make sure the API server is running on localhost:8080

Press Ctrl+C to stop the server
========================================
```

## 🌐 Open in Browser

Once the frontend server starts, open your browser and go to:

```
http://localhost:3000/index.html
```

Or just:
```
http://localhost:3000
```

## ✅ Verify Everything is Working

1. **API Server**: Check `http://localhost:8080/health`
   - Should return: `{"status": "healthy"}`

2. **Frontend**: Check `http://localhost:3000`
   - Should show the DR Assistant interface
   - Should show "🟢 API Connected" (green) if API is running

3. **Upload an image** and click "🔍 Analyze Image"

## 📋 Complete Startup Sequence

### Terminal 1 (API - Already Running):
```powershell
cd "C:\Users\pathi\Documents\DR assistant"
$env:OPENAI_API_KEY="sk-proj-your-key-here"
python src/inference.py
```
✅ Should show: `INFO:     Uvicorn running on http://0.0.0.0:8080`

### Terminal 2 (Frontend - New Terminal):
```powershell
cd "C:\Users\pathi\Documents\DR assistant"
python serve_frontend.py
```
✅ Should show: `🌍 Frontend URL: http://localhost:3000`

### Browser:
```
http://localhost:3000/index.html
```

## 🎯 Using the Frontend

1. **Upload Image**: Drag and drop or browse for a retinal fundus image
2. **Configure Options**:
   - ✅ Include Grad-CAM Explanation (default: checked)
   - ✅ Include Clinical Recommendation (default: checked)
3. **Analyze**: Click "🔍 Analyze Image" button
4. **View Results**:
   - Classification card (Grade 0-4)
   - Grad-CAM heatmaps (4 visualizations)
   - Clinical recommendation
   - Download report button

## 🐛 Troubleshooting

### "API Not Connected" in Frontend
- Make sure API server is running on port 8080
- Check Terminal 1 for errors
- Try accessing `http://localhost:8080/health` directly

### Port 3000 Already in Use
- Another process is using port 3000
- Close other servers or change port in `serve_frontend.py`

### Frontend Shows Blank Page
- Check browser console (F12) for errors
- Verify `frontend/index.html` exists
- Check Terminal 2 for server errors

## ✅ Success Indicators

**Terminal 1 (API):**
```
INFO:     Uvicorn running on http://0.0.0.0:8080 (Press CTRL+C to quit)
```

**Terminal 2 (Frontend):**
```
🌍 Frontend URL: http://localhost:3000
```

**Browser:**
- Shows DR Assistant interface
- Shows "🟢 API Connected" in green
- Can upload and analyze images

---

**You're all set! Both servers are running and ready to use!** 🎉


