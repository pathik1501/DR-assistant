# How to Start Your DR Assistant

## Step-by-Step Instructions

### Method 1: The Server is Already Running!

Your server is already running from the last command. You can access it right now!

**Just open your browser and go to:**
```
http://localhost:8080/docs
```

### Method 2: If You Need to Restart the Server

If the server stopped, here's how to start it again:

#### Step 1: Open PowerShell/Terminal
Open PowerShell or Command Prompt

#### Step 2: Navigate to Your Project
```powershell
cd "C:\Users\pathi\Documents\DR assistant"
```

#### Step 3: Start the Server
```powershell
python src/inference.py
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8080
```

#### Step 4: Open the API Interface
Open your browser and go to:
```
http://localhost:8080/docs
```

### Method 3: Use the Deploy Script (Easiest)

Just run:
```powershell
python deploy.py
```

## Using the System

1. **Open API Interface**: http://localhost:8080/docs
2. **Test with an Image**:
   - Find the `/predict` endpoint
   - Click "Try it out"
   - Click "Choose File" to upload an image
   - Click "Execute"
3. **See the Results**:
   - Prediction (grade 0-4)
   - Confidence score
   - Grad-CAM heatmap visualization
   - Clinical hint (if configured)

## Quick Test

If you want to test if it's working:

```powershell
curl http://localhost:8080/health
```

You should see: `{"status":"healthy"}`

## Stop the Server

Press **Ctrl+C** in the terminal where the server is running.

## Troubleshooting

**If you get "address already in use":**
- Another instance is running
- Press Ctrl+C to stop it
- Or change the port in `src/inference.py`

**If you get import errors:**
- Make sure you're in the project directory
- Run: `pip install -r requirements.txt`

**If localhost doesn't load:**
- Check the server is running: look for "Uvicorn running on http://0.0.0.0:8080"
- Wait a few seconds after starting
- Try refreshing the browser
