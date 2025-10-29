"""
Model architecture module for Diabetic Retinopathy detection.
Implements EfficientNet-B3 with custom head and loss functions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Optional, Tuple
import numpy as np


class FocalLoss(nn.Module):
    """Focal Loss implementation for handling class imbalance."""
    
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0, label_smoothing: float = 0.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.label_smoothing > 0:
            # Apply label smoothing
            num_classes = inputs.size(1)
            smooth_targets = torch.zeros_like(inputs)
            smooth_targets.fill_(self.label_smoothing / (num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
            
            # Focal loss with label smoothing
            ce_loss = -torch.sum(smooth_targets * F.log_softmax(inputs, dim=1), dim=1)
        else:
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            if self.alpha.type() != inputs.data.type():
                self.alpha = self.alpha.type_as(inputs.data)
            at = self.alpha.gather(0, targets.data.view(-1))
            focal_loss = at * focal_loss
            
        return focal_loss.mean()


class DRModel(nn.Module):
    """EfficientNet-B3 model for Diabetic Retinopathy classification."""
    
    def __init__(
        self,
        num_classes: int = 5,
        pretrained: bool = True,
        dropout_rate: float = 0.3,
        use_mc_dropout: bool = False
    ):
        super(DRModel, self).__init__()
        
        self.num_classes = num_classes
        self.use_mc_dropout = use_mc_dropout
        
        # Load pretrained EfficientNet-B0 (more memory efficient)
        self.backbone = timm.create_model(
            'efficientnet_b0',
            pretrained=pretrained,
            num_classes=0,  # Remove classifier
            global_pool='avg'
        )
        
        # Get feature dimension
        feature_dim = self.backbone.num_features
        
        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize classifier weights."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits
    
    def forward_with_dropout(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with MC Dropout enabled."""
        if self.training:
            return self.forward(x)
        
        # Enable dropout during inference for uncertainty estimation
        self.train()
        with torch.no_grad():
            logits = self.forward(x)
        self.eval()
        return logits


class TemperatureScaling(nn.Module):
    """Temperature scaling for calibration."""
    
    def __init__(self):
        super(TemperatureScaling, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature


class UncertaintyEstimator:
    """Handles uncertainty estimation using MC Dropout."""
    
    def __init__(self, model: DRModel, num_samples: int = 30):
        self.model = model
        self.num_samples = num_samples
    
    def predict_with_uncertainty(
        self, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict with uncertainty estimation."""
        predictions = []
        
        for _ in range(self.num_samples):
            pred = self.model.forward_with_dropout(x)
            predictions.append(F.softmax(pred, dim=1))
        
        predictions = torch.stack(predictions)
        mean_pred = predictions.mean(dim=0)
        uncertainty = predictions.var(dim=0).sum(dim=1)  # Total variance
        
        return mean_pred, uncertainty


def calculate_qwk(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Quadratic Weighted Kappa."""
    from sklearn.metrics import cohen_kappa_score
    
    # Convert predictions to integers
    y_pred_int = np.round(y_pred).astype(int)
    y_pred_int = np.clip(y_pred_int, 0, 4)
    
    return cohen_kappa_score(y_true, y_pred_int, weights='quadratic')


def calculate_ece(
    y_true: np.ndarray, 
    y_pred_probs: np.ndarray, 
    n_bins: int = 15
) -> float:
    """Calculate Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_pred_probs > bin_lower) & (y_pred_probs <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_pred_probs[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece


class ModelMetrics:
    """Calculate various model metrics."""
    
    @staticmethod
    def calculate_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_probs: np.ndarray
    ) -> dict:
        """Calculate comprehensive metrics."""
        from sklearn.metrics import (
            accuracy_score, precision_recall_fscore_support,
            classification_report, confusion_matrix
        )
        
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['qwk'] = calculate_qwk(y_true, y_pred)
        metrics['ece'] = calculate_ece(y_true, y_pred_probs.max(axis=1))
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None
        )
        
        metrics['macro_f1'] = np.mean(f1)
        metrics['weighted_f1'] = np.average(f1, weights=support)
        
        # Per-class F1 scores (handle missing classes)
        for i in range(5):
            if i < len(f1):
                metrics[f'f1_class_{i}'] = f1[i]
            else:
                metrics[f'f1_class_{i}'] = 0.0
        
        return metrics


def create_model(
    num_classes: int = 5,
    pretrained: bool = True,
    dropout_rate: float = 0.3
) -> DRModel:
    """Create and return DR model."""
    return DRModel(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_rate=dropout_rate
    )


def main():
    """Test model creation and forward pass."""
    model = create_model()
    
    # Test forward pass
    x = torch.randn(2, 3, 512, 512)
    logits = model(x)
    
    print(f"Model created successfully!")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")


if __name__ == "__main__":
    main()
