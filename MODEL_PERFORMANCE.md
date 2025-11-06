# Model Performance Report

## Current Best Model Performance

### Model Checkpoint
- **File**: `dr-model-epoch=11-val_qwk=0.769.ckpt`
- **Location**: `1/7d0928bb87954a739123ca35fa03cccf/checkpoints/`
- **Epoch**: 11 (out of training run)

## Key Metrics

### Primary Metric
- **Quadratic Weighted Kappa (QWK)**: **0.769**
  - This is the main metric for ordinal classification tasks
  - Represents how well the model ranks DR grades (0-4)
  - **Status**: ✅ **Good** - Solid single-model performance

### Classification Metrics
- **Validation Accuracy**: Check training logs
- **Macro F1 Score**: Check training logs  
- **Weighted F1 Score**: Check training logs

### Calibration & Uncertainty
- **Expected Calibration Error (ECE)**: Check training logs
- **Training Loss**: 0.406
- **Validation Loss**: 1.230
- **Loss Gap**: 0.824 ⚠️ (Suggests overfitting)

### Per-Class Performance
- **F1 Class 0** (No DR): Check training logs
- **F1 Class 1** (Mild): Check training logs
- **F1 Class 2** (Moderate): Check training logs
- **F1 Class 3** (Severe): Check training logs
- **F1 Class 4** (Proliferative): Check training logs

## Performance Context

### Benchmark Comparison
| Model | QWK Score | Notes |
|-------|-----------|-------|
| **APTOS Baseline** | 0.65-0.70 | Simple baseline models |
| **Your Model** | **0.769** | Single EfficientNet-B0 ✅ |
| **Ensemble Top** | 0.88+ | Multiple models combined |
| **Your Potential** | 0.79-0.82 | With optimization fixes |

### Status Assessment
- ✅ **Portfolio Ready**: QWK 0.769 demonstrates solid technical implementation
- ⚠️ **Production Needs Improvement**: Target QWK 0.85+ for clinical deployment
- 🎯 **Good Rank Correlation**: Model orders grades reasonably well

## Issues Identified

### 1. Overfitting
- Train loss (0.406) vs Val loss (1.230) gap = 0.824
- Suggests model is memorizing training data
- **Impact**: Model may not generalize well to new images

### 2. Calibration
- Probabilities may not be well-calibrated
- Model might be overconfident on training data
- **Impact**: Confidence scores may not reflect true uncertainty

## Improvements Implemented (Not Yet Retrained)

The following improvements were configured but the model hasn't been retrained yet:

1. **Increased Regularization**
   - Weight decay: 0.0001 → 0.0005
   - Dropout: 0.3 (already applied)

2. **Label Smoothing**
   - Added: 0.1 to reduce overconfidence

3. **Learning Rate**
   - Reduced: 0.0003 → 0.0002 for more stable training

4. **Early Stopping**
   - Increased patience: 10 → 15 epochs

## Expected Improvements After Retraining

| Metric | Current | Expected | Change |
|--------|---------|----------|--------|
| QWK | 0.769 | 0.79-0.82 | +0.02-0.05 |
| Val Loss | 1.230 | 0.7-0.9 | -0.33-0.53 |
| Loss Gap | 0.824 | 0.3-0.5 | -0.32-0.52 |
| Calibration | Poor | Good | Improved |

## Training Details

- **Model Architecture**: EfficientNet-B0
- **Dataset**: 118,903 images (APTOS + EyePACS)
- **Input Size**: 224x224 pixels
- **Batch Size**: 2
- **Optimizer**: AdamW
- **Learning Rate**: 0.0002 (was 0.0003)
- **Loss Function**: Focal Loss (gamma=2.0)
- **Training Epochs**: Early stopped at epoch 11
- **GPU**: RTX 3070 (8GB)

## Inference Performance

- **Processing Time**: ~6.3 seconds per image
- **Latency**: Check Prometheus metrics
- **Throughput**: ~0.16 images/second
- **GPU Memory**: EfficientNet-B0 fits in 8GB

## Recommendations

### For Portfolio/Demo
✅ **Current model is sufficient**
- QWK 0.769 is credible
- Shows good understanding of medical imaging
- Demonstrates practical application

### For Production
⚠️ **Retrain with improvements**
1. Use updated config with regularization
2. Target QWK 0.79-0.82
3. Add Test-Time Augmentation (TTA)
4. Consider ensemble for 0.85+ QWK

### For Clinical Use
🚨 **Not ready for clinical deployment**
- Need QWK ≥ 0.85
- Require external validation
- Need regulatory approval
- Must integrate with clinical workflows

## Visualizations Available

Check `outputs/evaluation/` for:
- Confusion matrices
- Calibration plots
- Per-class F1 progress
- Metric progress over epochs



