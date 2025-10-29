# Quick Start Guide - DR Assistant

## Your server is now running!

The API server should be running in the background.

## Access Your API

1. **API Documentation**: http://localhost:8080/docs
2. **Health Check**: http://localhost:8080/health  
3. **Metrics**: http://localhost:8080/metrics

## Test it in your browser:

Open: **http://localhost:8080/docs**

You'll see an interactive interface where you can:
- Upload test images
- Get predictions
- View confidence scores
- See Grad-CAM visualizations

## What's Deployed

- **Model**: EfficientNet-B0 trained on 118K images
- **Performance**: QWK 0.769
- **Features**: 
  - DR grade prediction (0-4)
  - Confidence scores
  - Grad-CAM heatmaps
  - Uncertainty estimation

## Quick Test

You can test it with Python:

```python
import requests
import base64

# Read an image
with open("path/to/test_image.jpg", "rb") as f:
    img_data = base64.b64encode(f.read()).decode()

# Make prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={"image": img_data}
)
print(response.json())
```

## Stop the Server

Press Ctrl+C in the terminal where the server is running.

## Next Steps

1. Open http://localhost:8000/docs in your browser
2. Try uploading a test image
3. View the predictions and visualizations
4. Check the model performance metrics

Your complete diabetic retinopathy detection system is live!
