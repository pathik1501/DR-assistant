# Deployment Guide - Diabetic Retinopathy Assistant

## Quick Start

Your trained model is ready to deploy! Here's how to get it running.

## Current Status

- **Model Checkpoint**: `1/7d0928bb87954a739123ca35fa03cccf/checkpoints/dr-model-epoch=11-val_qwk=0.769.ckpt`
- **Performance**: QWK 0.769 (solid baseline)
- **API Ready**: FastAPI server configured
- **Web UI Ready**: Streamlit interface configured

## Deployment Options

### Option 1: Simple API Server (Recommended for Testing)

```bash
python deploy.py
```

This will:
- Load your trained model (QWK 0.769)
- Start FastAPI server on http://localhost:8000
- Provide interactive docs at http://localhost:8000/docs

### Option 2: Manual Start

```bash
python src/inference.py
```

### Option 3: Web Interface

```bash
streamlit run frontend/app.py
```

## API Usage

Once the server is running, you can use it:

### 1. Python Client

```python
import requests
from PIL import Image
import io

# Load an image
image = Image.open('path/to/fundus_image.jpg')

# Convert to base64
buffered = io.BytesIO()
image.save(buffered, format="JPEG")
img_str = base64.b64encode(buffered.getvalue())

# Make prediction
response = requests.post(
    'http://localhost:8000/predict',
    json={'image': img_str.decode()}
)
result = response.json()
print(f"Grade: {result['prediction']}")
print(f"Confidence: {result['confidence']}")
```

### 2. cURL

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @request.json
```

### 3. Interactive Docs

Visit http://localhost:8000/docs for interactive API documentation.

## API Endpoints

- **POST /predict** - Single image prediction
- **GET /health** - Health check
- **GET /metrics** - Prometheus metrics
- **GET /model/info** - Model information

## Model Information

- **Architecture**: EfficientNet-B0
- **Input Size**: 224x224
- **Classes**: 5 (Grades 0-4)
- **QWK**: 0.769
- **Status**: Trained and ready

## Next Steps

1. **Start API**: `python deploy.py`
2. **Test in browser**: Visit http://localhost:8000/docs
3. **Deploy web UI**: `streamlit run frontend/app.py`

## Production Deployment

For production deployment using Docker:

```bash
# Build the image
docker build -t dr-assistant .

# Run the container
docker run -p 8000:8000 dr-assistant
```

Or use docker-compose for full stack:

```bash
docker-compose up --build
```

This deploys:
- FastAPI server (port 8000)
- Streamlit UI (port 8501)
- MLflow (port 5000)
- Prometheus (port 9090)
- Grafana (port 3000)

## Monitoring

- **MLflow**: http://localhost:5000
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000

## Troubleshooting

### Model not found
Make sure the checkpoint exists at the expected path.

### Import errors
Run from the project root directory.

### CUDA errors
The model runs on CPU by default. GPU support requires CUDA setup.

## Support

For issues or questions, check:
- `README.md` - Full project documentation
- `OPTIMIZATION_REPORT.md` - Performance analysis
- Logs in `logs/` directory
