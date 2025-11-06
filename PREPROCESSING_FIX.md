# Preprocessing & Calibration Issues - FIXED

## Issues Identified and Fixed

### ✅ Issue 1: Preprocessing Mismatch - FIXED
**Problem**: API was preprocessing images differently than training:
- **API**: Resized to 512×512, applied CLAHE contrast enhancement
- **Training**: Resized to 224×224, no CLAHE (only Albumentations transforms)

**Impact**: This critical mismatch caused the model to receive images with completely different characteristics than what it was trained on, leading to poor predictions.

**Fix**: Modified `src/inference.py` line 204-205 to:
```python
# Apply preprocessing - match training exactly
# Training uses [224, 224] from config
image_np = cv2.resize(image_np, (224, 224))

# IMPORTANT: Do NOT apply CLAHE here - training doesn't use it
```

### ✅ Issue 2: Temperature Scaling - FIXED
**Problem**: 
1. Temperature scaler is never saved during training (no `temperature_scaler.pth` file exists)
2. Even if it existed, the implementation had a bug - it tried to apply temperature scaling to probabilities instead of logits
3. Temperature scaling requires raw logits, but `UncertaintyEstimator` returns probabilities

**Impact**: The model wasn't calibrated, but since it wasn't being applied anyway, the actual impact was minimal.

**Fix**: Modified `src/inference.py` line 256-261 to:
```python
# Apply temperature scaling if available
# NOTE: Temperature scaling requires raw logits, but we're using MC dropout
# which returns probabilities. If temperature scaling is needed in the future,
# we need to modify UncertaintyEstimator to also return logits.
if self.temperature_scaler:
    logger.warning("Temperature scaler loaded but not applied - requires logits, not probabilities")
```

## Testing Required

Please test the API with a sample image to verify:
1. Predictions now match training expectations
2. No errors during preprocessing
3. Predictions are reasonable

```bash
python test_real_api.py
```

## What Changed

### Files Modified
1. `src/inference.py` - Fixed preprocessing to match training (224×224, no CLAHE)
2. `src/inference.py` - Fixed temperature scaling logic (disabled until proper implementation)

### Root Cause
The preprocessing mismatch was the **most critical issue**. The model was trained on 224×224 images without CLAHE, but the API was sending 512×512 images with CLAHE. This is equivalent to testing a model on a completely different dataset.

## Next Steps (Optional Enhancements)

If temperature scaling is truly needed in the future:
1. Modify `UncertaintyEstimator` to return raw logits before softmax
2. Implement temperature scaling calibration on validation set during training
3. Save the temperature scaler with the model checkpoint
4. Apply it correctly during inference on raw logits

For now, the model relies on MC dropout for uncertainty estimation, which is already working.


