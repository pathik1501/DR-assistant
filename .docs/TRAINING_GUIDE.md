# 🚀 Full Training Setup Guide

## Prerequisites

### 1. Kaggle API Setup
You need a Kaggle account and API key to download the datasets:

1. **Create Kaggle Account**: Go to https://www.kaggle.com/account
2. **Generate API Token**: 
   - Click "Create New API Token"
   - Download `kaggle.json`
3. **Place API Key**:
   - Create directory: `C:\Users\[your-username]\.kaggle\`
   - Place `kaggle.json` in that directory
   - Set permissions: Right-click → Properties → Security → Full Control

### 2. OpenAI API Key (Optional but Recommended)
For the RAG pipeline and clinical hints:

1. **Get OpenAI API Key**: https://platform.openai.com/api-keys
2. **Set Environment Variable**:
   ```bash
   set OPENAI_API_KEY=your-api-key-here
   ```

## Step-by-Step Setup

### Step 1: Download Datasets
```bash
python download_datasets.py
```

This will download:
- **APTOS 2019**: ~3GB (3,661 images)
- **EyePACS**: ~10GB (35,126 images)

### Step 2: Verify Setup
```bash
python test_system.py
```

### Step 3: Start Training
```bash
python src/train.py
```

### Step 4: Start API Server
```bash
python src/inference.py
```

### Step 5: Launch Web Interface
```bash
streamlit run frontend/app.py
```

## Expected Training Time

With your RTX 3070 GPU:
- **EfficientNet-B0**: ~2-3 hours for 100 epochs
- **Batch Size 2**: Conservative for memory
- **224x224 Images**: Memory-optimized resolution

## Monitoring Training

### MLflow Dashboard
- **URL**: http://localhost:5000
- **Track**: Loss, accuracy, QWK, F1 scores
- **View**: Training curves and metrics

### GPU Monitoring
```bash
# Monitor GPU usage
nvidia-smi -l 1
```

## Troubleshooting

### Memory Issues
If you get CUDA out of memory:
1. Reduce batch size to 1
2. Use gradient accumulation
3. Enable mixed precision training

### Dataset Issues
If download fails:
1. Check Kaggle API key
2. Verify internet connection
3. Try downloading manually from Kaggle website

### Training Issues
If training fails:
1. Check GPU memory usage
2. Verify dataset paths
3. Check configuration in `configs/config.yaml`

## Performance Expectations

### Model Performance Targets
- **QWK**: ≥ 0.88
- **Macro F1**: ≥ 0.79
- **ECE**: < 0.05
- **Inference Time**: < 120ms per image

### Hardware Utilization
- **GPU Memory**: ~2-4GB during training
- **GPU Utilization**: 80-95%
- **CPU Usage**: Moderate (data loading)

## Next Steps After Training

1. **Evaluate Model**: `python src/eval.py`
2. **Generate Reports**: Check `outputs/evaluation/`
3. **Deploy API**: Use trained model in production
4. **Monitor Performance**: Use Prometheus/Grafana

## Support

If you encounter issues:
1. Check the logs in `logs/` directory
2. Verify all dependencies are installed
3. Ensure GPU drivers are up to date
4. Check CUDA compatibility
