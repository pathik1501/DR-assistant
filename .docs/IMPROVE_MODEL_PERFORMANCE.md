# 🚀 Model Performance Improvement Guide

## Current Performance: QWK 0.785 ✅

Your model is already performing excellently! This guide provides actionable strategies to push it even higher.

## 📊 Performance Baseline

| Metric | Current | Target | Potential |
|--------|---------|--------|-----------|
| **QWK** | **0.785** | 0.82-0.85 | +0.04-0.07 |
| **Accuracy** | 74.7% | 77-80% | +2-5% |
| **Macro F1** | 0.651 | 0.70-0.75 | +0.05-0.10 |
| **ECE** | 0.480 | <0.1 | -0.38 |
| **Val Loss** | 0.520 | 0.4-0.5 | -0.02-0.12 |

## 🎯 Quick Wins (Easy, High Impact)

### 1. Test-Time Augmentation (TTA) ⭐⭐⭐⭐⭐
**Impact**: +0.02-0.04 QWK | **Effort**: Easy | **Time**: 30 mins

Test-Time Augmentation averages predictions from multiple augmented versions of each test image.

**Implementation Steps:**
```python
# In src/inference.py, modify predict function:

def predict_with_tta(self, image_tensor, num_augmentations=10):
    """Predict with TTA."""
    self.model.eval()
    predictions = []
    
    with torch.no_grad():
        # Original image
        pred = F.softmax(self.model(image_tensor), dim=1)
        predictions.append(pred)
        
        # Augmented predictions
        tta_transform = A.Compose([
            A.HorizontalFlip(p=1.0),
            A.VerticalFlip(p=1.0),
            A.Rotate90(p=1.0),
            A.RandomBrightnessContrast(p=1.0),
        ])
        
        for i in range(num_augmentations):
            aug_img = tta_transform(image=image_tensor.cpu().numpy().transpose(1,2,0))['image']
            aug_tensor = torch.from_numpy(aug_img).unsqueeze(0)
            pred = F.softmax(self.model(aug_tensor), dim=1)
            predictions.append(pred)
    
    # Average predictions
    final_pred = torch.stack(predictions).mean(dim=0)
    return final_pred
```

**Expected Improvement:**
- QWK: 0.785 → 0.81-0.82
- Accuracy: +1-2%
- Confidence: More stable predictions

### 2. Temperature Scaling ⭐⭐⭐⭐⭐
**Impact**: ECE 0.48 → 0.05 | **Effort**: Easy | **Time**: 15 mins

Temperature scaling calibrates model probabilities by learning a temperature parameter.

**Implementation:**
```python
# Add to src/model.py (already defined but not used)

class TemperatureScaler:
    def __init__(self, model, val_loader):
        self.model = model
        self.val_loader = val_loader
        self.temperature = nn.Parameter(torch.ones(1))
        
    def calibrate(self):
        """Calibrate on validation set."""
        self.model.eval()
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01)
        
        def eval():
            optimizer.zero_grad()
            loss = 0
            for images, targets in self.val_loader:
                logits = self.model(images)
                scaled_logits = logits / self.temperature
                loss += F.cross_entropy(scaled_logits, targets)
            loss.backward()
            return loss
        
        # Fit temperature
        for _ in range(100):
            optimizer.step(eval)
        
        return self.temperature.item()
    
    def predict(self, logits):
        """Return calibrated probabilities."""
        return F.softmax(logits / self.temperature, dim=1)
```

**Usage in inference:**
```python
# After loading model
scaler = TemperatureScaler(model, val_loader)
temp = scaler.calibrate()  # Usually ~0.7-1.2

# In prediction
logits = model(image)
probs = F.softmax(logits / temp, dim=1)
```

**Expected Improvement:**
- ECE: 0.48 → <0.1
- Better confidence estimates
- More reliable probability scores

### 3. Mixup Augmentation (Already Configured! 🔧)
**Impact**: +0.01-0.02 QWK | **Effort**: Easy | **Time**: Already in config**

Your config already has `mixup_alpha: 0.4` but it's not implemented! Add this to training:

**Add to src/train.py training_step:**
```python
def training_step(self, batch, batch_idx):
    images, targets = batch
    
    # Mixup augmentation
    alpha = self.config['training']['mixup_alpha']
    if alpha > 0 and random.random() < 0.5:
        lam = np.random.beta(alpha, alpha)
        batch_size = images.size(0)
        index = torch.randperm(batch_size)
        images = lam * images + (1 - lam) * images[index]
        targets_0, targets_1 = targets, targets[index]
    
    logits = self(images)
    
    if self.training and alpha > 0 and 'lam' in locals():
        loss = lam * self.criterion(logits, targets_0) + \
               (1 - lam) * self.criterion(logits, targets_1)
    else:
        loss = self.criterion(logits, targets)
    
    return loss
```

**Expected Improvement:**
- QWK: +0.01-0.02
- Better generalization
- Reduced overfitting

## 🏗️ Medium Effort, Medium Impact

### 4. Larger Input Resolution ⭐⭐⭐⭐
**Impact**: +0.02-0.03 QWK | **Effort**: Medium | **Time**: 2-3 hours

Current: 224×224 | Try: 384×384 or 512×512

**Trade-offs:**
- ✅ Better detail capture
- ✅ More micro-lesion detection
- ❌ 4x memory usage
- ❌ Slower training/inference

**Implementation:**
```yaml
# configs/config.yaml
data:
  output_size: [384, 384]  # Increase from [224, 224]
```

**Expected Improvement:**
- QWK: 0.785 → 0.81
- Better fine-grained features
- Slower inference (~3x)

### 5. Model Ensembling ⭐⭐⭐⭐
**Impact**: +0.04-0.06 QWK | **Effort**: High | **Time**: 1-2 days

Train 3-5 models with different seeds/initializations, then average predictions.

**Strategy:**
1. Train 5 models with different random seeds
2. Use diverse augmentations for each model
3. Ensemble predictions

**Implementation:**
```python
# Create ensemble.py
class EnsembleModel:
    def __init__(self, model_paths):
        self.models = []
        for path in model_paths:
            model = DRModel.load_from_checkpoint(path)
            model.eval()
            self.models.append(model)
    
    def predict(self, image):
        predictions = []
        for model in self.models:
            with torch.no_grad():
                logits = model(image)
                predictions.append(F.softmax(logits, dim=1))
        
        # Average predictions
        ensemble_pred = torch.stack(predictions).mean(dim=0)
        return ensemble_pred
```

**Expected Improvement:**
- QWK: 0.785 → 0.83-0.85
- Accuracy: +3-5%
- Best single approach

### 6. Class-Weighted Loss ⭐⭐⭐
**Impact**: +0.01-0.02 QWK | **Effort**: Medium | **Time**: 30 mins

Balance rare classes (Grade 3, 4) with weighted loss.

**Implementation:**
```python
# Calculate class weights from training data
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(train_labels),
    y=train_labels
)

# Update FocalLoss in src/model.py
criterion = FocalLoss(
    alpha=torch.tensor(class_weights),
    gamma=2.0
)
```

**Expected Improvement:**
- Better minority class performance
- More balanced predictions
- +0.01-0.02 QWK

## 🔬 Advanced Techniques

### 7. Pseudo-Labeling ⭐⭐⭐
**Impact**: +0.02-0.03 QWK | **Effort**: High | **Time**: Days**

Semi-supervised learning using unlabeled data.

**Process:**
1. Train model on labeled data
2. Predict on unlabeled validation/test splits
3. Retrain with high-confidence pseudo-labels

### 8. CutMix Augmentation ⭐⭐⭐
**Impact**: +0.01-0.02 QWK | **Effort**: Medium | **Time**: 1 hour**

More advanced than Mixup, cuts and patches images.

### 9. Progressive Resizing ⭐⭐⭐
**Impact**: +0.02 QWK | **Effort**: High | **Time**: Days**

Start training at 224×224, gradually increase to 384×384.

### 10. Focal Loss Tuning ⭐⭐
**Impact**: +0.005-0.01 QWK | **Effort**: Low | **Time**: 1 hour**

Experiment with different gamma values:
- Current: gamma=2.0
- Try: gamma=1.5, 2.5, 3.0

## 📋 Recommended Improvement Sequence

### Phase 1: Quick Wins (1-2 days) 🚀
1. ✅ Implement Temperature Scaling → Calibrate probabilities
2. ✅ Add Test-Time Augmentation → +0.02-0.04 QWK
3. ✅ Implement Mixup augmentation → +0.01-0.02 QWK
4. ✅ Add class-weighted loss → Better balance

**Expected Result**: QWK 0.785 → **0.82-0.84** (+0.035-0.055)

### Phase 2: Model Enhancement (1 week) 🏗️
5. Train 3-5 ensemble models → +0.04-0.06 QWK
6. Experiment with larger input size (384×384)
7. Fine-tune hyperparameters

**Expected Result**: QWK 0.84 → **0.85-0.87** (with ensemble)

### Phase 3: Advanced Techniques (2+ weeks) 🔬
8. Pseudo-labeling
9. Progressive resizing
10. External data integration

**Expected Result**: QWK 0.87 → **0.88-0.90** (best-in-class)

## 🎯 Realistic Target Performance

| Approach | QWK | Accuracy | ECE | Effort |
|----------|-----|----------|-----|--------|
| **Current** | 0.785 | 74.7% | 0.48 | - |
| **With TTA** | 0.81 | 76% | 0.48 | ⭐ Easy |
| **+ Temperature** | 0.81 | 76% | **0.05** | ⭐ Easy |
| **+ Mixup** | 0.82 | 77% | 0.05 | ⭐ Easy |
| **+ Ensemble (5)** | **0.85-0.87** | **79-81%** | **0.03** | ⭐⭐⭐ High |
| **Production Ready** | **0.88+** | **82%+** | **<0.1** | ⭐⭐⭐⭐⭐ Very High |

## 💡 Implementation Priority

**Start Here (Highest ROI):**
1. ✅ Temperature Scaling (already implemented in code!)
2. ✅ Test-Time Augmentation  
3. ✅ Mixup augmentation

**Then:**
4. Class-weighted loss
5. Model ensembling
6. Larger input resolution

**Finally (diminishing returns):**
7. Pseudo-labeling
8. CutMix
9. Progressive resizing

## 🎉 Summary

**Your model is already excellent at QWK 0.785!**

With these improvements, you can realistically reach:
- **Easy wins**: QWK **0.82-0.84** in 1-2 days ⭐⭐⭐⭐⭐
- **With ensemble**: QWK **0.85-0.87** in ~1 week ⭐⭐⭐⭐
- **Best-in-class**: QWK **0.88+** with extensive training ⭐⭐⭐

The quickest path to 0.82-0.84: **Temperature Scaling + TTA + Mixup**

Good luck! 🚀



