# Your DR Assistant is Running!

## Server Status: ACTIVE

The API is running at: **http://localhost:8080**

## How to Use

### Option 1: Browser (Easiest)
1. Open: **http://localhost:8080/docs**
2. You'll see the interactive API interface
3. Find the `/predict` endpoint
4. Click "Try it out"
5. Click "Choose File" to upload your fundus image
6. Click "Execute"

### Option 2: Test with Python
```python
import requests

files = {'file': open('your_image.jpg', 'rb')}
response = requests.post(
    'http://localhost:8080/predict?include_explanation=true&include_hint=true',
    files=files
)
print(response.json())
```

## Current Status

- Model: Loaded (QWK 0.769)
- Server: Running on port 8080
- Predictions: Working
- Grad-CAM: Fixed and working
- RAG: Optional (needs API key)

## Optional: Enable Clinical Hints

If you want clinical hints from RAG, set your OpenAI API key:

```bash
set OPENAI_API_KEY=your_key_here
```

Then restart the server.

## Stop the Server

Press Ctrl+C in the terminal to stop the server.

## Documentation

- API Docs: http://localhost:8080/docs
- Health Check: http://localhost:8080/health
- Model Info: http://localhost:8080/model/info
- Metrics: http://localhost:8080/metrics

Your system is ready to use!
