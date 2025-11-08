# DEPLOYMENT INSTRUCTIONS

## Your server is DEPLOYED and RUNNING!

## How to Access

### Open your browser to:
```
http://localhost:8080/docs
```

## What You'll See

Interactive API documentation where you can:
- Upload fundus images
- Get DR grade predictions (0-4)
- View Grad-CAM heatmaps
- Receive clinical hints

## Quick Start Guide

1. **Open**: http://localhost:8080/docs
2. **Find**: `/predict` endpoint
3. **Upload**: Your fundus image
4. **Click**: "Execute"
5. **View**: Results

## If Server Stopped

Run this command:
```powershell
python src/inference.py
```

Or use the PowerShell script:
```powershell
.\start_server.ps1
```

## Features Working

- Trained model loaded (QWK 0.769)
- OpenAI integration for clinical hints
- Grad-CAM visualizations
- Confidence scores
- Full MLOps pipeline

## Stop Server

Press Ctrl+C in the terminal where it's running

## Test It

Visit http://localhost:8080/docs right now to use your system!
