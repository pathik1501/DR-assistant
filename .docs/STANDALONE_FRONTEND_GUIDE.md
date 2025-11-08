# Standalone Web Frontend Guide

## 🎯 Overview

The frontend is now a **standalone web application** (HTML/CSS/JavaScript) that communicates with the FastAPI backend. This separates concerns:

- **Frontend**: Pure web application served on port 3000
- **Backend API**: FastAPI server on port 8080
- **No dependencies**: Frontend doesn't require Python or Streamlit

## 🏗️ Architecture

```
┌─────────────────┐         HTTP REST API         ┌─────────────────┐
│                 │  ───────────────────────────>  │                 │
│  Web Frontend   │                                │  FastAPI Server │
│  (Port 3000)    │  <──────────────────────────  │  (Port 8080)    │
│  HTML/CSS/JS    │         JSON Responses          │  Python/PyTorch │
└─────────────────┘                                └─────────────────┘
```

## ✨ Features

### 1. **Pure Web Frontend**
- No Python dependencies for the UI
- Works in any modern browser
- Fast and responsive
- Can be hosted anywhere (GitHub Pages, Netlify, etc.)

### 2. **Real-time API Connection**
- Automatically checks API connection status
- Visual indicator (🟢 connected / 🔴 disconnected)
- Automatic reconnection checking

### 3. **Complete Feature Set**
- 📤 Image upload (drag & drop or browse)
- 🎯 Classification display (Grade 0-4, confidence)
- 🔍 Grad-CAM heatmap visualization (4 visualizations)
- 💡 RAG model clinical recommendations
- 📥 Download report as JSON

## 🚀 How to Start

### Step 1: Start the API Server (Backend)

Open **Terminal 1**:
```powershell
cd "C:\Users\pathi\Documents\DR assistant"
$env:OPENAI_API_KEY="YOUR_KEY_HERE"  # Optional for RAG
python src/inference.py
```

Wait for: `Uvicorn running on http://127.0.0.1:8080`

### Step 2: Start the Frontend Server

Open **Terminal 2**:
```powershell
cd "C:\Users\pathi\Documents\DR assistant"
python serve_frontend.py
```

Or use the script:
```powershell
.\start_frontend.ps1
```

You'll see:
```
🌐 Frontend Server Started
🌍 Frontend URL: http://localhost:3000
📄 Open in browser: http://localhost:3000/index.html
```

### Step 3: Open in Browser

Open your browser and go to:
```
http://localhost:3000/index.html
```

Or just:
```
http://localhost:3000
```

## 📁 File Structure

```
frontend/
├── index.html      # Main HTML structure
├── style.css       # All styling
└── app.js          # JavaScript application logic

serve_frontend.py   # Simple HTTP server for frontend
start_frontend.ps1  # Startup script
```

## 🎨 UI Components

### Upload Section
- Drag and drop area
- File browser button
- Image preview with file info

### Analysis Options
- Checkbox: "Include Grad-CAM Explanation"
- Checkbox: "Include Clinical Recommendation"
- Analyze button (disabled until image uploaded)

### Results Display
1. **Classification Card**
   - Color-coded by grade (0-4)
   - Large icon and grade number
   - Confidence percentage
   - Visual gauge

2. **Heatmaps Grid**
   - 4 visualizations (2x2 grid)
   - Grad-CAM heatmap
   - Grad-CAM overlay
   - Grad-CAM++ heatmap
   - Grad-CAM++ overlay

3. **Clinical Recommendation**
   - RAG-generated or template-based hint
   - Medical disclaimer

4. **Download Button**
   - JSON report with all results

## 🔧 Technical Details

### API Communication

The frontend makes HTTP requests to:
```
POST http://localhost:8080/predict_base64
```

Request body:
```json
{
  "image_base64": "base64_encoded_image",
  "include_explanation": true,
  "include_hint": true
}
```

Response:
```json
{
  "prediction": 0,
  "confidence": 0.85,
  "grade_description": "No Diabetic Retinopathy",
  "explanation": {
    "gradcam_heatmap_base64": "...",
    "gradcam_overlay_base64": "...",
    "gradcam_plus_heatmap_base64": "...",
    "gradcam_plus_overlay_base64": "..."
  },
  "clinical_hint": "✅ No diabetic retinopathy detected...",
  "processing_time": 12.34,
  "abstained": false
}
```

### CORS Configuration

The frontend server includes CORS headers to allow API communication:
- `Access-Control-Allow-Origin: *`
- `Access-Control-Allow-Methods: GET, POST, OPTIONS`
- `Access-Control-Allow-Headers: Content-Type`

## 🌐 Deployment Options

Since it's a pure web frontend, you can deploy it anywhere:

### Option 1: Static Hosting
- **GitHub Pages**: Upload `frontend/` folder
- **Netlify**: Drag and drop deployment
- **Vercel**: Connect GitHub repo

### Option 2: Custom Server
- Use `serve_frontend.py` on your server
- Or use nginx/Apache to serve static files
- Or use any static file server

### Option 3: Docker
- Serve frontend in a lightweight nginx container
- Separate from API container

## 🎯 Advantages of Standalone Frontend

1. **Separation of Concerns**
   - Frontend and backend are independent
   - Can update one without affecting the other
   - Easier to maintain

2. **Performance**
   - No Python overhead for UI
   - Fast JavaScript rendering
   - Efficient browser-native features

3. **Deployment Flexibility**
   - Frontend can be hosted separately
   - API can scale independently
   - Can use CDN for static assets

4. **Developer Experience**
   - Standard web development tools work
   - Easy to debug with browser DevTools
   - Familiar HTML/CSS/JS workflow

## 🐛 Troubleshooting

### "API Not Connected"
- Ensure API server is running on `localhost:8080`
- Check if API server started successfully
- Verify port 8080 is not blocked

### Heatmaps Not Showing
- Check browser console for errors
- Verify API returns `*_base64` fields
- Check network tab for API response

### CORS Errors
- Ensure `serve_frontend.py` is used (includes CORS headers)
- Or configure API server to allow CORS from frontend origin

### Images Not Uploading
- Check browser console for errors
- Verify file format (JPG, JPEG, PNG)
- Check file size (not too large)

## 📝 Next Steps

1. **Customize Styling**: Edit `frontend/style.css`
2. **Add Features**: Modify `frontend/app.js`
3. **Deploy**: Choose deployment option above
4. **Host API**: Deploy FastAPI server to cloud

## ✅ Summary

- ✅ Pure web frontend (HTML/CSS/JS)
- ✅ Completely separate from API server
- ✅ No Streamlit or Python for UI
- ✅ Works in any modern browser
- ✅ Easy to deploy and maintain
- ✅ Full feature set (upload, classification, heatmaps, RAG)

---

**The frontend is now completely independent from the API server!** 🎉



