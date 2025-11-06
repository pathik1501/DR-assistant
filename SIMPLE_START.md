# Simple DR Assistant - Quick Start

The simplest way to run the DR Assistant.

## Quick Start (2 Steps)

### Step 1: Start the API Server
Open a terminal and run:
```powershell
python src/inference.py
```

Wait until you see: "Uvicorn running on http://0.0.0.0:8080"

### Step 2: Start the Simple Frontend
Open a **second terminal** and run:
```powershell
.\start_simple.ps1
```

That's it! The browser will open automatically.

## What You Get

- ✅ **Upload** a retinal image
- ✅ **Grade** prediction (0-4)
- ✅ **Confidence** percentage
- ✅ **Clinical recommendation**
- ✅ **Clean, simple interface**

## File Structure

```
DR assistant/
├── src/
│   └── inference.py        # API server (Step 1)
├── simple_frontend.py      # Simple UI
├── start_simple.ps1        # Startup script
└── SIMPLE_START.md         # This file
```

## Troubleshooting

### "API not running" error
Make sure Step 1 is running in a separate terminal window.

### "Import errors"
Install dependencies:
```powershell
pip install -r requirements.txt
```

### Predictions seem wrong
Make sure you're using the preprocessing fixes from earlier. The API now correctly uses 224x224 images (not 512x512).

## Comparison

| Feature | Simple Frontend | Full Frontend |
|---------|----------------|---------------|
| Upload & Analyze | ✅ | ✅ |
| Grade & Confidence | ✅ | ✅ |
| Clinical Hints | ✅ | ✅ |
| Grad-CAM Heatmaps | ❌ | ✅ |
| Advanced Stats | ❌ | ✅ |
| Downloads | ❌ | ✅ |
| Code Lines | 90 | 450 |

The simple frontend is perfect for quick tests and demonstrations!


