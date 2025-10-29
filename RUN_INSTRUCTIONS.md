# DATASET DOWNLOAD INSTRUCTIONS

## Required Datasets:

### 1. APTOS 2019 Blindness Detection
- **URL**: https://www.kaggle.com/competitions/aptos2019-blindness-detection/data
- **Files to download**:
  - `train.csv`
  - `train_images.zip`
- **Extract to**: `data/aptos2019/`

### 2. EyePACS Dataset  
- **URL**: https://www.kaggle.com/datasets/sovitrath/eyepacs-dataset
- **Files to download**:
  - `trainLabels.csv`
  - `train.zip`
- **Extract to**: `data/eyepacs/`

## Final Directory Structure:
```
data/
├── aptos2019/
│   ├── train.csv
│   └── train_images/
│       ├── image1.png
│       ├── image2.png
│       └── ...
└── eyepacs/
    ├── trainLabels.csv
    └── train/
        ├── image1.jpeg
        ├── image2.jpeg
        └── ...
```

## How to Run After Downloading Datasets:

### Option 1: Quick Test (Without Training)
```bash
# Test the model with a sample image
python example_inference.py
```

### Option 2: Full Training Pipeline
```bash
# 1. Train the model
python src/train.py

# 2. Start the API server
python src/inference.py

# 3. Launch the web interface (in another terminal)
streamlit run frontend/app.py
```

### Option 3: Docker Deployment
```bash
# Build and run all services
docker-compose up --build
```

## API Endpoints (after starting inference.py):
- **Health Check**: http://localhost:8080/health
- **Predict**: http://localhost:8080/predict
- **Metrics**: http://localhost:8080/metrics

## Web Interface (after starting streamlit):
- **URL**: http://localhost:8501
