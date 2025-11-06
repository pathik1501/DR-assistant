# Implementation Summary - Class Weights & Weighted Sampling

## ✅ What Was Implemented

### 1. Class Weights for Focal Loss ⭐⭐⭐⭐⭐

**File**: `src/train.py`

**Changes**:
- Added `compute_class_weight` import from sklearn
- Modified `DRModelModule.__init__` to accept optional `class_weights` parameter
- Added logic to use class weights in FocalLoss when enabled
- Modified `Trainer.train()` to calculate class weights from training data
- Added class distribution printing for transparency

**How It Works**:
1. When `use_class_weights: true` in config, training labels are analyzed
2. Balanced class weights are calculated using sklearn's `compute_class_weight('balanced')`
3. Weights are passed to FocalLoss via the `alpha` parameter
4. Rare classes (3 & 4) get higher weights, compensating for imbalance

**Expected Impact**:
- Class 3 F1: 0.60 → 0.65-0.70 (+8-17%)
- Class 4 F1: 0.47 → 0.55-0.65 (+17-38%)

---

### 2. Weighted Sampling for Training Data ⭐⭐⭐⭐

**File**: `src/data_processing.py`

**Changes**:
- Added `WeightedRandomSampler` import from torch.utils.data
- Modified `create_data_loaders` to accept `use_weighted_sampling` parameter
- Added weighted sampler logic that samples rare classes more frequently
- Modified `prepare_datasets` to return training labels
- Added class distribution printing for weighted sampling

**How It Works**:
1. When `use_weighted_sampling: true` in config, sampling weights are calculated
2. Weights are inverse of class frequency (rare classes get higher weights)
3. `WeightedRandomSampler` ensures balanced batches during training
4. Each batch has more examples from rare classes

**Expected Impact**:
- Additional +5% F1 for rare classes when combined with class weights
- Better feature learning for Classes 3 & 4

---

### 3. Configuration Updates

**File**: `configs/config.yaml`

**Added**:
```yaml
training:
  use_class_weights: true  # Enable class weights for Focal Loss
  use_weighted_sampling: false  # Enable weighted sampling (optional)
```

**Recommendation**:
- **Start with**: `use_class_weights: true`, `use_weighted_sampling: false`
- **After testing**: Try `use_weighted_sampling: true` for additional boost
- Using both together might over-emphasize rare classes, so test carefully

---

## 📋 Code Changes Summary

### Modified Files:

1. **`src/train.py`**
   - Added class weight calculation
   - Modified DRModelModule to accept class_weights
   - Updated Trainer to calculate and pass weights
   - Added checkpoint loading handling

2. **`src/data_processing.py`**
   - Added weighted sampling support
   - Modified prepare_datasets to return training labels
   - Added class distribution printing

3. **`configs/config.yaml`**
   - Added use_class_weights flag
   - Added use_weighted_sampling flag

---

## 🚀 How to Use

### Basic Usage (Class Weights Only):

1. Ensure `use_class_weights: true` in `configs/config.yaml`
2. Run training:
   ```bash
   python src/train.py
   ```
3. You should see output like:
   ```
   Calculating class weights from training data...
   Class distribution in training set:
     Class 0: 82345 samples, weight: 0.1234
     Class 1:  3700 samples, weight: 2.7891
     Class 2: 10000 samples, weight: 1.0321
     Class 3:  1600 samples, weight: 6.4523
     Class 4:  2400 samples, weight: 4.3012
   Using class weights: tensor([0.1234, 2.7891, 1.0321, 6.4523, 4.3012])
   ```

### With Weighted Sampling:

1. Set both flags in config:
   ```yaml
   use_class_weights: true
   use_weighted_sampling: true
   ```
2. Run training (same command)
3. You'll see additional output about sampling weights

---

## ⚠️ Important Notes

### 1. **Class Weights vs Weighted Sampling**
- **Class weights** affect the loss function (penalizes errors on rare classes more)
- **Weighted sampling** affects which examples are seen more often
- Using both can be powerful but might cause overfitting on rare classes
- **Recommendation**: Start with class weights only, add sampling if needed

### 2. **Checkpoint Compatibility**
- Old checkpoints will still work (class_weights is optional)
- When loading from checkpoint, class_weights=None is used (fine for inference)
- New checkpoints will have hyperparameters saved for proper reconstruction

### 3. **Performance Impact**
- Class weights: Negligible overhead (just tensor operations)
- Weighted sampling: Slight overhead (sampler computation)
- Both are very efficient and shouldn't slow down training significantly

### 4. **Expected Improvements**
- **Immediate**: Class weights should improve rare class performance
- **Best Practice**: Evaluate on validation set after training
- **Monitoring**: Watch for overfitting on rare classes (validation loss increase)

---

## 📊 Monitoring Training

When training with class weights, monitor:

1. **Per-Class F1 Scores**:
   - Should see improvement in Classes 3 & 4
   - Classes 0-2 should remain strong
   - Watch for overfitting (train >> val F1)

2. **Loss Values**:
   - Training loss might increase slightly (weights amplify rare class errors)
   - Validation loss should decrease or stay stable
   - Large gap = overfitting on rare classes

3. **Class Distribution in Batches** (if using weighted sampling):
   - Batches should be more balanced
   - Rare classes should appear more frequently

---

## 🔧 Troubleshooting

### Issue: "class_weights is None but use_class_weights is True"
- **Cause**: Training labels not properly returned from prepare_datasets
- **Fix**: Ensure DataProcessor.prepare_datasets() returns 4 values

### Issue: "TypeError when loading checkpoint"
- **Cause**: Checkpoint saved with different hyperparameters
- **Fix**: The try/except in train.py should handle this, but if not, retrain

### Issue: "Training loss very high"
- **Cause**: Class weights might be too large
- **Fix**: Check weight values - if any >10, consider normalizing

### Issue: "No improvement in rare classes"
- **Cause**: Weights might not be large enough, or insufficient data
- **Fix**: 
  1. Check class distribution (are Classes 3 & 4 really rare?)
  2. Try enabling weighted_sampling as well
  3. Consider getting more data for rare classes

---

## ✅ Next Steps

1. **Train a model** with class weights enabled
2. **Evaluate** on validation set
3. **Compare** with previous results (should see +5-10% F1 for Classes 3 & 4)
4. **If needed**: Enable weighted sampling and retrain
5. **Monitor** for overfitting and adjust if necessary

---

## 📈 Expected Results

With class weights enabled, you should see:

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| **Class 3 F1** | 0.600 | 0.65-0.70 | +8-17% |
| **Class 4 F1** | 0.471 | 0.55-0.65 | +17-38% |
| **Class 3 Recall** | 0.563 | 0.65-0.75 | Better detection |
| **Class 4 Recall** | 0.333 | 0.45-0.60 | Much better! |
| **Overall QWK** | 0.865 | 0.87-0.89 | +0.5-3% |

**Note**: Actual improvements depend on your dataset distribution and model architecture.

---

## 🎉 Summary

**Implemented Features**:
✅ Class weights in Focal Loss
✅ Weighted sampling option
✅ Config flags for easy toggling
✅ Comprehensive logging
✅ Backward compatible with old checkpoints

**Ready to Use**: Yes! Just enable in config and start training.

**Expected Time to Benefit**: Immediate (next training run)

**Risk Level**: Low (both features are well-established techniques)
