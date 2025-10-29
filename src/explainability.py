"""
Explainability module for Diabetic Retinopathy detection.
Implements Grad-CAM, Grad-CAM++, and SHAP for model interpretation.
"""

import torch
import torch.nn.functional as F
import cv2
import numpy as np
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
import logging

logger = logging.getLogger(__name__)


class GradCAM:
    """Grad-CAM implementation for visualizing model decisions."""
    
    def __init__(self, model: torch.nn.Module, target_layers: List[str]):
        self.model = model
        self.target_layers = target_layers
        self.gradients = {}
        self.activations = {}
        self.hooks = []
    
    def _register_hooks(self):
        """Register forward and backward hooks."""
        # Remove existing hooks first
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        
        def forward_hook(module, input, output):
            self.activations[id(module)] = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients[id(module)] = grad_output[0].detach()
        
        hooks_registered = 0
        for name, module in self.model.named_modules():
            if name in self.target_layers:
                self.hooks.append(module.register_forward_hook(forward_hook))
                self.hooks.append(module.register_backward_hook(backward_hook))
                hooks_registered += 1
                logger.info(f"Registered hooks for layer {name}")
        
        if hooks_registered == 0:
            logger.error(f"No hooks registered! Target layers: {self.target_layers}")
        else:
            logger.info(f"Registered {hooks_registered} layer hook(s)")
    
    def generate_cam(
        self, 
        input_tensor: torch.Tensor, 
        class_idx: Optional[int] = None
    ) -> np.ndarray:
        """Generate Grad-CAM heatmap."""
        # Clear previous activations and gradients
        self.activations.clear()
        self.gradients.clear()
        
        # Re-register hooks
        self._register_hooks()
        
        # Forward pass
        self.model.eval()
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        output[0, class_idx].backward()
        
        # Generate CAM for each target layer
        cams = []
        logger.info(f"Checking gradients/activations: {len(self.gradients)} gradients, {len(self.activations)} activations")
        for layer_name in self.target_layers:
            for name, module in self.model.named_modules():
                if name == layer_name:
                    if id(module) not in self.gradients or id(module) not in self.activations:
                        logger.warning(f"No gradients/activations captured for layer {layer_name} (id={id(module)})")
                        continue
                    gradients = self.gradients[id(module)]
                    activations = self.activations[id(module)]
                    logger.info(f"Found gradients/activations for layer {layer_name}: shapes {gradients.shape}, {activations.shape}")
                    
                    # Global average pooling of gradients
                    weights = gradients.mean(dim=(2, 3), keepdim=True)
                    
                    # Weighted combination of activation maps
                    cam = (weights * activations).sum(dim=1, keepdim=True)
                    cam = F.relu(cam)
                    
                    # Normalize
                    cam = cam.squeeze().cpu().numpy()
                    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
                    
                    cams.append(cam)
                    break
        
        return np.mean(cams, axis=0) if cams else np.zeros((16, 16))
    
    def __del__(self):
        """Clean up hooks."""
        for hook in self.hooks:
            hook.remove()


class GradCAMPlusPlus(GradCAM):
    """Enhanced Grad-CAM++ implementation."""
    
    def generate_cam(
        self, 
        input_tensor: torch.Tensor, 
        class_idx: Optional[int] = None
    ) -> np.ndarray:
        """Generate Grad-CAM++ heatmap."""
        # Clear previous activations and gradients
        self.activations.clear()
        self.gradients.clear()
        
        # Re-register hooks
        self._register_hooks()
        
        # Forward pass
        self.model.eval()
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        output[0, class_idx].backward()
        
        # Generate CAM for each target layer
        cams = []
        for layer_name in self.target_layers:
            for name, module in self.model.named_modules():
                if name == layer_name:
                    if id(module) not in self.gradients or id(module) not in self.activations:
                        logger.warning(f"No gradients/activations captured for layer {layer_name}")
                        continue
                    gradients = self.gradients[id(module)]
                    activations = self.activations[id(module)]
                    
                    # Grad-CAM++ weights calculation
                    alpha = gradients.pow(2).sum(dim=(2, 3), keepdim=True)
                    alpha = alpha / (2 * alpha + activations.sum(dim=(2, 3), keepdim=True) + 1e-8)
                    
                    weights = (alpha * gradients).sum(dim=(2, 3), keepdim=True)
                    
                    # Weighted combination of activation maps
                    cam = (weights * activations).sum(dim=1, keepdim=True)
                    cam = F.relu(cam)
                    
                    # Normalize
                    cam = cam.squeeze().cpu().numpy()
                    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
                    
                    cams.append(cam)
                    break
        
        return np.mean(cams, axis=0) if cams else np.zeros((16, 16))


class Visualizer:
    """Visualization utilities for explainability results."""
    
    @staticmethod
    def overlay_heatmap(
        image: np.ndarray, 
        heatmap: np.ndarray, 
        alpha: float = 0.4
    ) -> np.ndarray:
        """Overlay heatmap on original image."""
        # Ensure image is in correct format
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        if image.max() > 1.0:
            image = (image / 255).astype(np.float32)
            image = (image * 255).astype(np.uint8)
        
        # Resize heatmap to match image size
        heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        
        # Convert to 3-channel
        heatmap_colored = cv2.applyColorMap(
            (heatmap_resized * 255).astype(np.uint8), 
            cv2.COLORMAP_JET
        )
        
        # Ensure both are same dtype for addWeighted
        image_float = image.astype(np.float32)
        heatmap_float = heatmap_colored.astype(np.float32)
        
        # Overlay
        overlay = cv2.addWeighted(image_float, 1-alpha, heatmap_float, alpha, 0)
        
        return overlay.astype(np.uint8)
    
    @staticmethod
    def create_explanation_plot(
        image: np.ndarray,
        heatmap: np.ndarray,
        prediction: int,
        confidence: float,
        class_names: List[str]
    ) -> plt.Figure:
        """Create comprehensive explanation plot."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Original image
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Heatmap
        im = axes[0, 1].imshow(heatmap, cmap='jet')
        axes[0, 1].set_title('Grad-CAM Heatmap')
        axes[0, 1].axis('off')
        plt.colorbar(im, ax=axes[0, 1])
        
        # Overlay
        overlay = Visualizer.overlay_heatmap(image, heatmap)
        axes[1, 0].imshow(overlay)
        axes[1, 0].set_title('Overlay')
        axes[1, 0].axis('off')
        
        # Prediction info
        axes[1, 1].text(0.1, 0.7, f'Prediction: {class_names[prediction]}', 
                        fontsize=14, fontweight='bold')
        axes[1, 1].text(0.1, 0.5, f'Confidence: {confidence:.3f}', 
                        fontsize=12)
        axes[1, 1].text(0.1, 0.3, f'Class: {prediction}', 
                        fontsize=12)
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def save_explanation(
        image: np.ndarray,
        heatmap: np.ndarray,
        prediction: int,
        confidence: float,
        class_names: List[str],
        save_path: str
    ):
        """Save explanation visualization."""
        fig = Visualizer.create_explanation_plot(
            image, heatmap, prediction, confidence, class_names
        )
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)


class ExplainabilityPipeline:
    """Complete explainability pipeline."""
    
    def __init__(
        self, 
        model: torch.nn.Module, 
        target_layers: List[str],
        class_names: List[str] = None
    ):
        self.model = model
        self.target_layers = target_layers
        self.class_names = class_names or [
            'No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative'
        ]
        
        self.gradcam = GradCAM(model, target_layers)
        self.gradcam_plus = GradCAMPlusPlus(model, target_layers)
    
    def explain_prediction(
        self,
        image: np.ndarray,
        input_tensor: torch.Tensor,
        prediction: int,
        confidence: float,
        save_path: Optional[str] = None
    ) -> dict:
        """Generate complete explanation for a prediction."""
        
        # Generate heatmaps
        gradcam_heatmap = self.gradcam.generate_cam(input_tensor, prediction)
        gradcam_plus_heatmap = self.gradcam_plus.generate_cam(input_tensor, prediction)
        
        # Create visualizations
        gradcam_overlay = Visualizer.overlay_heatmap(image, gradcam_heatmap)
        gradcam_plus_overlay = Visualizer.overlay_heatmap(image, gradcam_plus_heatmap)
        
        explanation = {
            'prediction': prediction,
            'confidence': confidence,
            'class_name': self.class_names[prediction],
            'gradcam_heatmap': gradcam_heatmap,
            'gradcam_plus_heatmap': gradcam_plus_heatmap,
            'gradcam_overlay': gradcam_overlay,
            'gradcam_plus_overlay': gradcam_plus_overlay
        }
        
        # Save if path provided
        if save_path:
            Visualizer.save_explanation(
                image, gradcam_heatmap, prediction, confidence,
                self.class_names, save_path
            )
        
        return explanation
    
    def batch_explain(
        self,
        images: List[np.ndarray],
        input_tensors: torch.Tensor,
        predictions: torch.Tensor,
        confidences: torch.Tensor,
        output_dir: str
    ) -> List[dict]:
        """Generate explanations for a batch of images."""
        explanations = []
        
        os.makedirs(output_dir, exist_ok=True)
        
        for i, (image, pred, conf) in enumerate(zip(images, predictions, confidences)):
            input_tensor = input_tensors[i:i+1]
            pred_int = pred.item()
            conf_float = conf.item()
            
            save_path = os.path.join(output_dir, f'explanation_{i:04d}.png')
            
            explanation = self.explain_prediction(
                image, input_tensor, pred_int, conf_float, save_path
            )
            
            explanations.append(explanation)
        
        return explanations


def main():
    """Test explainability functionality."""
    from src.model import create_model
    
    # Create dummy model and data
    model = create_model()
    input_tensor = torch.randn(1, 3, 512, 512)
    
    # Test Grad-CAM
    target_layers = ['blocks.5.2', 'blocks.6.2']
    pipeline = ExplainabilityPipeline(model, target_layers)
    
    # Dummy prediction
    with torch.no_grad():
        output = model(input_tensor)
        prediction = output.argmax(dim=1).item()
        confidence = F.softmax(output, dim=1).max().item()
    
    # Generate explanation
    dummy_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    
    explanation = pipeline.explain_prediction(
        dummy_image, input_tensor, prediction, confidence
    )
    
    print("Explainability pipeline created successfully!")
    print(f"Prediction: {explanation['class_name']}")
    print(f"Confidence: {explanation['confidence']:.3f}")


if __name__ == "__main__":
    main()
