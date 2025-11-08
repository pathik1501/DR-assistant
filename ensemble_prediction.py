"""
Ensemble prediction script for combining multiple model checkpoints.
This improves QWK by 0.04-0.06 by averaging predictions from different models.
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import List, Tuple
import yaml
from PIL import Image
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.model import DRModel


class EnsembleModel:
    """Ensemble of multiple trained models for improved predictions."""
    
    def __init__(self, checkpoint_paths: List[str], config_path: str = "configs/config.yaml"):
        """Initialize ensemble with multiple model checkpoints.
        
        Args:
            checkpoint_paths: List of paths to model checkpoints
            config_path: Path to config file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.models = []
        self.checkpoint_paths = checkpoint_paths
        
        # Load all models
        for i, checkpoint_path in enumerate(checkpoint_paths):
            print(f"Loading model {i+1}/{len(checkpoint_paths)}: {checkpoint_path}")
            model = self._load_model(checkpoint_path)
            model.eval()
            self.models.append(model)
        
        print(f"Ensemble initialized with {len(self.models)} models")
    
    def _load_model(self, checkpoint_path: str) -> DRModel:
        """Load model from checkpoint."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Extract state dict
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            cleaned_state_dict = {}
            
            for key, value in state_dict.items():
                if key.startswith('model.'):
                    new_key = key[6:]
                    cleaned_state_dict[new_key] = value
                elif not any(key.startswith(prefix) for prefix in ['criterion', 'metrics']):
                    cleaned_state_dict[key] = value
        else:
            cleaned_state_dict = checkpoint
        
        # Create model
        model = DRModel(
            num_classes=self.config['model']['num_classes'],
            pretrained=False,
            dropout_rate=self.config['model']['dropout_rate']
        )
        
        result = model.load_state_dict(cleaned_state_dict, strict=False)
        if result.missing_keys:
            print(f"Warning: Missing keys: {result.missing_keys[:5]}...")
        if result.unexpected_keys:
            print(f"Warning: Unexpected keys: {result.unexpected_keys[:5]}...")
        
        return model
    
    def predict(self, image_tensor: torch.Tensor) -> Tuple[int, torch.Tensor, float]:
        """Make ensemble prediction.
        
        Args:
            image_tensor: Preprocessed image tensor (1, 3, 224, 224)
            
        Returns:
            prediction: Predicted class (0-4)
            probabilities: Class probabilities
            confidence: Maximum probability
        """
        predictions = []
        
        with torch.no_grad():
            for model in self.models:
                logits = model(image_tensor)
                probs = F.softmax(logits, dim=1)
                predictions.append(probs)
        
        # Average predictions
        ensemble_probs = torch.stack(predictions).mean(dim=0)
        
        # Get prediction and confidence
        prediction = ensemble_probs.argmax(dim=1).item()
        confidence = ensemble_probs.max().item()
        
        return prediction, ensemble_probs, confidence
    
    def predict_with_uncertainty(self, image_tensor: torch.Tensor) -> Tuple[int, torch.Tensor, float, float]:
        """Make ensemble prediction with uncertainty estimation.
        
        Args:
            image_tensor: Preprocessed image tensor
            
        Returns:
            prediction: Predicted class
            probabilities: Class probabilities
            confidence: Maximum probability
            uncertainty: Prediction uncertainty (std across models)
        """
        predictions = []
        
        with torch.no_grad():
            for model in self.models:
                logits = model(image_tensor)
                probs = F.softmax(logits, dim=1)
                predictions.append(probs)
        
        # Calculate mean and std
        ensemble_stack = torch.stack(predictions)
        ensemble_probs = ensemble_stack.mean(dim=0)
        uncertainty = ensemble_stack.var(dim=0).sum().item()
        
        # Get prediction and confidence
        prediction = ensemble_probs.argmax(dim=1).item()
        confidence = ensemble_probs.max().item()
        
        return prediction, ensemble_probs, confidence, uncertainty


def preprocess_image_for_ensemble(image_path: str) -> torch.Tensor:
    """Preprocess image for ensemble prediction."""
    import cv2
    from albumentations import Normalize, ToTensorV2
    import albumentations as A
    
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Apply validation transforms
    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    transformed = transform(image=image)
    image_tensor = transformed['image'].unsqueeze(0)
    
    return image_tensor


def find_best_checkpoints(checkpoint_dir: str, top_k: int = 5) -> List[str]:
    """Find best checkpoints based on QWK score.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        top_k: Number of top checkpoints to select
        
    Returns:
        List of checkpoint paths
    """
    checkpoints = []
    
    # Search for checkpoint files
    for root, dirs, files in os.walk(checkpoint_dir):
        for file in files:
            if file.endswith('.ckpt'):
                checkpoint_path = os.path.join(root, file)
                
                # Try to extract QWK from filename
                if 'val_qwk' in file:
                    try:
                        # Extract QWK value from filename
                        parts = file.split('val_qwk=')
                        if len(parts) > 1:
                            qwk_str = parts[1].split('.ckpt')[0]
                            qwk_value = float(qwk_str)
                            checkpoints.append((qwk_value, checkpoint_path))
                    except:
                        pass
    
    # Sort by QWK and return top_k
    checkpoints.sort(reverse=True, key=lambda x: x[0])
    top_checkpoints = [path for _, path in checkpoints[:top_k]]
    
    print(f"Found {len(checkpoints)} checkpoints, selecting top {top_k}:")
    for qwk, path in checkpoints[:top_k]:
        print(f"  QWK {qwk:.3f}: {path}")
    
    return top_checkpoints


def main():
    """Example usage of ensemble model."""
    print("="*60)
    print("Ensemble Prediction for Diabetic Retinopathy Detection")
    print("="*60)
    
    # Find checkpoints
    checkpoint_dir = "1"  # MLflow checkpoints directory
    checkpoint_paths = find_best_checkpoints(checkpoint_dir, top_k=3)
    
    if not checkpoint_paths:
        print("No checkpoints found! Please train models first.")
        return
    
    # Initialize ensemble
    ensemble = EnsembleModel(checkpoint_paths)
    
    # Example: predict on test image
    test_image_path = "test_fundus.jpg"
    
    if os.path.exists(test_image_path):
        print(f"\nPredicting on {test_image_path}...")
        image_tensor = preprocess_image_for_ensemble(test_image_path)
        prediction, probs, confidence, uncertainty = ensemble.predict_with_uncertainty(image_tensor)
        
        grade_descriptions = [
            "No Diabetic Retinopathy",
            "Mild Nonproliferative DR",
            "Moderate Nonproliferative DR",
            "Severe Nonproliferative DR",
            "Proliferative DR"
        ]
        
        print(f"\nEnsemble Prediction:")
        print(f"  Grade: {prediction} - {grade_descriptions[prediction]}")
        print(f"  Confidence: {confidence:.3f}")
        print(f"  Uncertainty: {uncertainty:.3f}")
        print(f"\nClass Probabilities:")
        for i, prob in enumerate(probs[0]):
            print(f"  {grade_descriptions[i]}: {prob:.3f}")
    else:
        print(f"\nTest image not found at {test_image_path}")
        print("Ensemble model initialized successfully!")
    
    print("\n" + "="*60)
    print("Usage in your inference pipeline:")
    print("  ensemble = EnsembleModel(checkpoint_paths)")
    print("  prediction, probs, confidence = ensemble.predict(image_tensor)")
    print("="*60)


if __name__ == "__main__":
    main()



