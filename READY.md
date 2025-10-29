# Your DR Assistant is Now Running with Full Features!

## API Endpoint
**http://localhost:8080/docs**

## What's Enabled Now

- Core model predictions
- Grad-CAM heatmaps and visualizations
- Confidence scores
- Clinical hints (RAG + OpenAI enabled)

## How to Use

### In Browser
1. Open: http://localhost:8080/docs
2. Find `/predict` endpoint
3. Upload your fundus image
4. Click "Execute"
5. View results

### In Python
```python
import requests

files = {'file': open('image.jpg', 'rb')}
response = requests.post(
    'http://localhost:8080/predict?include_explanation=true&include_hint=true',
    files=files
)
result = response.json()

print(f"Grade: {result['prediction']}")
print(f"Confidence: {result['confidence']}")
print(f"Hint: {result['clinical_hint']}")
```

## Features You'll Get

- DR grade (0-4)
- Confidence score (0-1)
- Grad-CAM heatmap visualization
- Clinical hints based on the predicted grade
- Uncertainty estimation

## Example Response

```json
{
  "prediction": 2,
  "confidence": 0.87,
  "grade_description": "Moderate NPDR",
  "explanation": {
    "heatmap": "...",
    "highlighted_regions": "..."
  },
  "clinical_hint": {
    "recommendation": "Schedule follow-up in 3-6 months..."
  },
  "abstained": false,
  "processing_time": 0.15
}
```

## Stop the Server

Press Ctrl+C in the terminal

## Notes

- OpenAI API key is configured for clinical hints
- Model is trained on 118K images
- QWK performance: 0.769
- System fully operational

Your diabetic retinopathy detection system is ready with all features enabled!
