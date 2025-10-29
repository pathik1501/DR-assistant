"""
Evaluation script for Diabetic Retinopathy detection model.
Calculates comprehensive metrics and generates evaluation reports.
"""

import os
import yaml
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix, cohen_kappa_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import json
from pathlib import Path

from src.model import DRModel, ModelMetrics, calculate_qwk, calculate_ece
from src.data_processing import DataProcessor
from src.explainability import ExplainabilityPipeline


class ModelEvaluator:
    """Comprehensive model evaluation."""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_config = self.config['model']
        self.paths_config = self.config['paths']
        
        # Load model
        self.model = self._load_model()
        
        # Initialize explainability pipeline
        self.explainability_pipeline = ExplainabilityPipeline(
            self.model,
            self.config['explainability']['grad_cam_layers']
        )
        
        # Class names
        self.class_names = [
            'No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative'
        ]
    
    def _load_model(self) -> DRModel:
        """Load trained model."""
        model_path = os.path.join(self.paths_config['models_dir'], 'best_model.pth')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        model = DRModel(
            num_classes=self.model_config['num_classes'],
            pretrained=False,
            dropout_rate=self.model_config['dropout_rate']
        )
        
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        
        return model
    
    def evaluate_model(self, test_loader) -> Dict:
        """Evaluate model on test set."""
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        all_probabilities = []
        all_confidences = []
        
        with torch.no_grad():
            for images, targets in test_loader:
                logits = self.model(images)
                probabilities = F.softmax(logits, dim=1)
                predictions = logits.argmax(dim=1)
                confidences = probabilities.max(dim=1)[0]
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_confidences.extend(confidences.cpu().numpy())
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        all_probabilities = np.array(all_probabilities)
        all_confidences = np.array(all_confidences)
        
        # Calculate metrics
        metrics = self._calculate_metrics(all_targets, all_predictions, all_probabilities)
        
        # Add confidence analysis
        metrics['confidence_stats'] = {
            'mean': float(np.mean(all_confidences)),
            'std': float(np.std(all_confidences)),
            'min': float(np.min(all_confidences)),
            'max': float(np.max(all_confidences))
        }
        
        # Abstention analysis
        threshold = self.config['uncertainty']['confidence_threshold']
        abstentions = np.sum(all_confidences < threshold)
        metrics['abstention_rate'] = float(abstentions / len(all_confidences))
        
        return {
            'metrics': metrics,
            'predictions': all_predictions,
            'targets': all_targets,
            'probabilities': all_probabilities,
            'confidences': all_confidences
        }
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict:
        """Calculate comprehensive metrics."""
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
        metrics['qwk'] = float(calculate_qwk(y_true, y_pred))
        metrics['ece'] = float(calculate_ece(y_true, y_prob.max(axis=1)))
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        metrics['macro_f1'] = float(np.mean(f1))
        metrics['weighted_f1'] = float(np.average(f1, weights=support))
        
        # Per-class F1 scores
        for i in range(5):
            metrics[f'f1_class_{i}'] = float(f1[i])
            metrics[f'precision_class_{i}'] = float(precision[i])
            metrics[f'recall_class_{i}'] = float(recall[i])
            metrics[f'support_class_{i}'] = int(support[i])
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        return metrics
    
    def generate_visualizations(self, results: Dict, output_dir: str):
        """Generate evaluation visualizations."""
        os.makedirs(output_dir, exist_ok=True)
        
        predictions = results['predictions']
        targets = results['targets']
        confidences = results['confidences']
        probabilities = results['probabilities']
        
        # Confusion Matrix
        plt.figure(figsize=(10, 8))
        cm = np.array(results['metrics']['confusion_matrix'])
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300)
        plt.close()
        
        # Confidence Distribution
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.hist(confidences, bins=30, alpha=0.7, color='skyblue')
        plt.axvline(self.config['uncertainty']['confidence_threshold'], 
                   color='red', linestyle='--', label='Threshold')
        plt.xlabel('Confidence')
        plt.ylabel('Frequency')
        plt.title('Confidence Distribution')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.boxplot([confidences[targets == i] for i in range(5)], 
                   labels=self.class_names)
        plt.xlabel('DR Grade')
        plt.ylabel('Confidence')
        plt.title('Confidence by Grade')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confidence_analysis.png'), dpi=300)
        plt.close()
        
        # Per-class Performance
        f1_scores = [results['metrics'][f'f1_class_{i}'] for i in range(5)]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(self.class_names, f1_scores, color=['green', 'orange', 'red', 'darkred', 'purple'])
        plt.xlabel('DR Grade')
        plt.ylabel('F1 Score')
        plt.title('Per-Class F1 Scores')
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar, score in zip(bars, f1_scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'per_class_f1.png'), dpi=300)
        plt.close()
        
        # Calibration Plot
        self._plot_calibration(targets, probabilities.max(axis=1), output_dir)
    
    def _plot_calibration(self, y_true: np.ndarray, y_prob: np.ndarray, output_dir: str):
        """Plot reliability diagram."""
        from sklearn.calibration import calibration_curve
        
        plt.figure(figsize=(8, 6))
        
        # Convert to binary for calibration curve
        y_true_binary = (y_true > 0).astype(int)
        
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true_binary, y_prob, n_bins=10
        )
        
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", 
                label="Model", color='blue')
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Reliability Diagram')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'calibration_plot.png'), dpi=300)
        plt.close()
    
    def generate_explanations(self, test_loader, output_dir: str, num_samples: int = 10):
        """Generate Grad-CAM explanations for sample images."""
        explanation_dir = os.path.join(output_dir, 'explanations')
        os.makedirs(explanation_dir, exist_ok=True)
        
        self.model.eval()
        sample_count = 0
        
        with torch.no_grad():
            for images, targets in test_loader:
                if sample_count >= num_samples:
                    break
                
                for i in range(images.size(0)):
                    if sample_count >= num_samples:
                        break
                    
                    image = images[i:i+1]
                    target = targets[i].item()
                    
                    # Get prediction
                    logits = self.model(image)
                    prediction = logits.argmax(dim=1).item()
                    confidence = F.softmax(logits, dim=1).max().item()
                    
                    # Generate explanation
                    image_np = image.squeeze().permute(1, 2, 0).cpu().numpy()
                    image_np = (image_np * np.array([0.229, 0.224, 0.225]) + 
                              np.array([0.485, 0.456, 0.406])) * 255
                    image_np = np.clip(image_np, 0, 255).astype(np.uint8)
                    
                    explanation = self.explainability_pipeline.explain_prediction(
                        image_np, image, prediction, confidence,
                        save_path=os.path.join(explanation_dir, f'explanation_{sample_count:03d}.png')
                    )
                    
                    sample_count += 1
    
    def save_results(self, results: Dict, output_dir: str):
        """Save evaluation results to JSON."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare results for JSON serialization
        json_results = {
            'metrics': results['metrics'],
            'summary': {
                'total_samples': len(results['targets']),
                'accuracy': results['metrics']['accuracy'],
                'qwk': results['metrics']['qwk'],
                'macro_f1': results['metrics']['macro_f1'],
                'ece': results['metrics']['ece'],
                'abstention_rate': results['metrics']['abstention_rate']
            }
        }
        
        # Save to JSON
        with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
            json.dump(json_results, f, indent=2)
        
        # Save detailed results
        detailed_results = pd.DataFrame({
            'target': results['targets'],
            'prediction': results['predictions'],
            'confidence': results['confidences']
        })
        
        detailed_results.to_csv(
            os.path.join(output_dir, 'detailed_predictions.csv'), 
            index=False
        )


def main():
    """Main evaluation function."""
    print("Starting model evaluation...")
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Prepare test data
    print("Loading test data...")
    processor = DataProcessor()
    _, _, test_loader = processor.prepare_datasets()
    
    # Evaluate model
    print("Evaluating model...")
    results = evaluator.evaluate_model(test_loader)
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Accuracy: {results['metrics']['accuracy']:.4f}")
    print(f"QWK: {results['metrics']['qwk']:.4f}")
    print(f"Macro F1: {results['metrics']['macro_f1']:.4f}")
    print(f"ECE: {results['metrics']['ece']:.4f}")
    print(f"Abstention Rate: {results['metrics']['abstention_rate']:.4f}")
    
    # Generate visualizations
    print("Generating visualizations...")
    output_dir = "outputs/evaluation"
    evaluator.generate_visualizations(results, output_dir)
    
    # Generate explanations
    print("Generating explanations...")
    evaluator.generate_explanations(test_loader, output_dir, num_samples=20)
    
    # Save results
    print("Saving results...")
    evaluator.save_results(results, output_dir)
    
    print(f"\nEvaluation completed! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
