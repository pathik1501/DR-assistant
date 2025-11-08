# How to Use Your DR Assistant - Complete Guide

## Your server is running at http://localhost:8080

## Method 1: Browser Interface (Easiest)

### Step 1: Open the API Docs
Open your web browser and go to:
**http://localhost:8080/docs**

### Step 2: Test with the Interactive UI
1. Scroll down to find the `/predict` endpoint
2. Click **"Try it out"**
3. In the request body, you'll see a sample JSON
4. To upload an image, you need to convert it to base64

### Step 3: Upload a Test Image
You can use any retinal fundus image (or any test image):

#### Option A: Quick Test (No Image)
Just click "Execute" to see the response structure

#### Option B: Upload Real Image
1. Use an online base64 encoder: https://www.base64-image.de/
2. Upload your fundus image there
3. Copy the base64 string
4. Paste it in the JSON body where it says "string"
5. Click "Execute"

## Method 2: Python Script (For Developers)

### Create a test script:

```python
import requests
import base64
from PIL import Image
import io

# Load an image
image_path = "path/to/your/fundus_image.jpg"
with open(image_path, "rb") as f:
    img_data = base64.b64encode(f.read()).decode()

# Make prediction
response = requests.post(
    "http://localhost:8080/predict",
    json={"image": img_data}
)

result = response.json()
print(f"Predicted Grade: {result['prediction']}")
print(f"Confidence: {result['confidence']}")
print(f"Description: {result['grade_description']}")
```

## Method 3: Streamlit Web Interface

### Start the web UI:

```bash
streamlit run frontend/app.py
```

Then open http://localhost:8501 in your browser

## Available Endpoints

### 1. GET /health
Check if server is running
```bash
curl http://localhost:8080/health
```

### 2. GET /model/info
Get model information
```bash
curl http://localhost:8080/model/info
```

### 3. POST /predict
Make a prediction

**Request body:**
```json
{
  "image": "base64_encoded_string_here"
}
```

**Response:**
```json
{
  "prediction": 2,
  "confidence": 0.85,
  "grade_description": "Moderate NPDR",
  "all_probabilities": [0.1, 0.2, 0.5, 0.15, 0.05],
  "explanation": {...},
  "clinical_hint": {...},
  "processing_time": 0.12,
  "abstained": false
}
```

## Understanding the Results

- **prediction**: DR grade (0-4)
  - 0: No DR
  - 1: Mild NPDR
  - 2: Moderate NPDR
  - 3: Severe NPDR
  - 4: Proliferative DR

- **confidence**: How confident the model is (0-1)
- **grade_description**: Human-readable description
- **all_probabilities**: Probabilities for each grade
- **abstained**: Whether model abstained due to low confidence

## Troubleshooting

### "Cannot connect to server"
1. Make sure the server is running
2. Check that it says "Uvicorn running on http://0.0.0.0:8080"
3. Try http://localhost:8080/health

### "Model loading failed"
The server will fall back to a pretrained model if your trained model doesn't load

### No images to test?
- You can test the API without an image first
- Or download sample fundus images from:
  - APTOS 2019 dataset
  - EyePACS dataset

## Quick Start Commands

```bash
# 1. Check server is running
curl http://localhost:8080/health

# 2. Get model info
curl http://localhost:8080/model/info

# 3. Test prediction (replace BASE64_STRING with actual image data)
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"image": "BASE64_STRING"}'
```

## Using the Browser Interface

The easiest way is:
1. Go to http://localhost:8080/docs
2. You'll see all endpoints listed
3. Click on any endpoint to see its details
4. Click "Try it out" to test it
5. The interface will show you the request format and response

That's it! Your system is ready to use.
