"""
Example inference script for Diabetic Retinopathy detection.
Demonstrates how to use the trained model for prediction.
"""

import os
import torch
import numpy as np
from PIL import Image
import cv2
import json
from pathlib import Path

from src.model import DRModel, UncertaintyEstimator
from src.explainability import ExplainabilityPipeline
from src.rag_pipeline import RAGPipeline


def preprocess_image(image_path: str) -> tuple:
    """Preprocess image for inference."""
    # Load image
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)
    
    # Resize
    image_np = cv2.resize(image_np, (512, 512))
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    image_np = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    # Normalize
    image_np = image_np.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_np = (image_np - mean) / std
    
    # Convert to tensor
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)
    
    return image_tensor, image_np


def predict_dr(image_path: str, model_path: str = None) -> dict:
    """Predict DR grade for an image."""
    
    # Load model
    if model_path and os.path.exists(model_path):
        model = DRModel(num_classes=5, pretrained=False)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        print(f"Loaded model from {model_path}")
    else:
        model = DRModel(num_classes=5, pretrained=True)
        print("Using pretrained model")
    
    model.eval()
    
    # Initialize uncertainty estimator
    uncertainty_estimator = UncertaintyEstimator(model, num_samples=10)
    
    # Initialize explainability pipeline
    explainability_pipeline = ExplainabilityPipeline(
        model, 
        target_layers=['blocks.5.2', 'blocks.6.2']
    )
    
    # Initialize RAG pipeline (optional)
    try:
        rag_pipeline = RAGPipeline()
        print("RAG pipeline initialized")
    except Exception as e:
        print(f"RAG pipeline not available: {e}")
        rag_pipeline = None
    
    # Preprocess image
    image_tensor, image_np = preprocess_image(image_path)
    
    # Make prediction with uncertainty
    with torch.no_grad():
        mean_pred, uncertainty = uncertainty_estimator.predict_with_uncertainty(image_tensor)
        
        prediction = mean_pred.argmax(dim=1).item()
        confidence = mean_pred.max().item()
        uncertainty_score = uncertainty.item()
    
    # Generate explanation
    explanation = explainability_pipeline.explain_prediction(
        image_np, image_tensor, prediction, confidence
    )
    
    # Generate clinical hint
    clinical_hint = None
    if rag_pipeline:
        try:
            clinical_hint = rag_pipeline.generate_hint(prediction, confidence)
        except Exception as e:
            print(f"Hint generation failed: {e}")
    
    # Prepare results
    grade_descriptions = [
        "No Diabetic Retinopathy",
        "Mild Nonproliferative DR",
        "Moderate Nonproliferative DR",
        "Severe Nonproliferative DR",
        "Proliferative DR"
    ]
    
    results = {
        "image_path": image_path,
        "prediction": prediction,
        "confidence": float(confidence),
        "uncertainty": float(uncertainty_score),
        "grade_description": grade_descriptions[prediction],
        "explanation": {
            "gradcam_heatmap_shape": explanation['gradcam_heatmap'].shape,
            "has_overlay": 'gradcam_overlay' in explanation
        },
        "clinical_hint": clinical_hint['hint'] if clinical_hint else None,
        "abstained": confidence < 0.7
    }
    
    return results


def main():
    """Example usage."""
    print("Diabetic Retinopathy Detection - Example Inference")
    print("=" * 50)
    
    # Example image path (replace with actual image)
    image_path = "example_retinal_image.jpg"
    
    if not os.path.exists(image_path):
        print(f"Example image not found: {image_path}")
        print("Please provide a retinal fundus image for analysis")
        return
    
    # Make prediction
    print(f"Analyzing image: {image_path}")
    results = predict_dr(image_path)
    
    # Display results
    print("\nResults:")
    print(f"DR Grade: {results['prediction']} - {results['grade_description']}")
    print(f"Confidence: {results['confidence']:.3f}")
    print(f"Uncertainty: {results['uncertainty']:.3f}")
    
    if results['abstained']:
        print("⚠️ Low confidence prediction - specialist review recommended")
    
    if results['clinical_hint']:
        print(f"\nClinical Recommendation: {results['clinical_hint']}")
    
    # Save results
    output_path = "inference_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
