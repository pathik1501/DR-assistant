"""
Evaluate ensemble model performance on validation set.
Compares single model vs ensemble performance.
"""

import os
import yaml
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix, cohen_kappa_score
)
from typing import Dict, List, Tuple
import json

from src.model import DRModel, ModelMetrics, calculate_qwk, calculate_ece
from src.data_processing import DataProcessor
from ensemble_prediction import EnsembleModel


class EnsembleEvaluator:
    """Evaluate ensemble vs single model performance."""
    
    def __init__(self, ensemble_checkpoints: List[str], single_checkpoint: str = None):
        """Initialize evaluator.
        
        Args:
            ensemble_checkpoints: List of checkpoint paths for ensemble
            single_checkpoint: Optional single model checkpoint for comparison
        """
        print("="*60)
        print("Initializing Ensemble Evaluator")
        print("="*60)
        
        # Load ensemble
        print(f"\nLoading ensemble with {len(ensemble_checkpoints)} models...")
        self.ensemble = EnsembleModel(ensemble_checkpoints)
        
        # Load single model for comparison (optional)
        self.single_model = None
        if single_checkpoint and os.path.exists(single_checkpoint):
            print(f"\nLoading single model for comparison: {single_checkpoint}")
            self.single_model = self._load_single_model(single_checkpoint)
        
        # Class names
        self.class_names = [
            'No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative'
        ]
    
    def _load_single_model(self, checkpoint_path: str) -> DRModel:
        """Load a single model for comparison."""
        # Load config
        with open("configs/config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
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
                    cleaned_state_dict[new_key] = value
        else:
            cleaned_state_dict = checkpoint
        
        # Create model
        model = DRModel(
            num_classes=config['model']['num_classes'],
            pretrained=False,
            dropout_rate=config['model']['dropout_rate']
        )
        
        model.load_state_dict(cleaned_state_dict, strict=False)
        model.eval()
        
        return model
    
    def evaluate_on_validation(self, val_loader) -> Dict:
        """Evaluate both ensemble and single model on validation set.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary with results for both models
        """
        print("\n" + "="*60)
        print("Evaluating on Validation Set")
        print("="*60)
        
        # Collect predictions
        ensemble_predictions = []
        ensemble_probabilities = []
        single_predictions = []
        single_probabilities = []
        all_targets = []
        
        print("\nRunning inference...")
        num_batches = len(val_loader)
        
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(val_loader):
                if (batch_idx + 1) % 10 == 0:
                    print(f"  Progress: {batch_idx+1}/{num_batches} batches ({100*(batch_idx+1)/num_batches:.1f}%)")
                
                # Ensemble predictions - handle batch directly
                predictions_list = []
                for model in self.ensemble.models:
                    logits = model(images)
                    probs = F.softmax(logits, dim=1)
                    predictions_list.append(probs)
                
                # Average ensemble predictions
                ensemble_stack = torch.stack(predictions_list)
                probs_ensemble = ensemble_stack.mean(dim=0)
                pred_ensemble = probs_ensemble.argmax(dim=1)
                
                ensemble_predictions.extend(pred_ensemble.cpu().numpy())
                ensemble_probabilities.extend(probs_ensemble.cpu().numpy())
                
                # Single model predictions (if available)
                if self.single_model:
                    logits_single = self.single_model(images)
                    probs_single = F.softmax(logits_single, dim=1)
                    pred_single = logits_single.argmax(dim=1)
                    
                    single_predictions.extend(pred_single.cpu().numpy())
                    single_probabilities.extend(probs_single.cpu().numpy())
                
                all_targets.extend(targets.numpy())
        
        # Convert to numpy arrays
        ensemble_predictions = np.array(ensemble_predictions)
        ensemble_probabilities = np.array(ensemble_probabilities)
        single_predictions = np.array(single_predictions) if single_predictions else None
        single_probabilities = np.array(single_probabilities) if single_probabilities else None
        all_targets = np.array(all_targets)
        
        # Calculate metrics
        results = {}
        
        # Ensemble results
        results['ensemble'] = self._calculate_metrics(
            all_targets, ensemble_predictions, ensemble_probabilities, "Ensemble"
        )
        
        # Single model results (if available)
        if single_predictions is not None:
            results['single'] = self._calculate_metrics(
                all_targets, single_predictions, single_probabilities, "Single Model"
            )
        
        return results
    
    def _calculate_metrics(self, y_true, y_pred, y_prob, model_name: str) -> Dict:
        """Calculate comprehensive metrics."""
        print(f"\nCalculating metrics for {model_name}...")
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        qwk = calculate_qwk(y_true, y_pred)
        ece = calculate_ece(y_true, y_prob.max(axis=1))
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        macro_f1 = np.mean(f1)
        weighted_f1 = np.average(f1, weights=support)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Per-class breakdown
        per_class = {}
        for i in range(5):
            per_class[f'class_{i}'] = {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1': float(f1[i]),
                'support': int(support[i])
            }
        
        results = {
            'model_name': model_name,
            'accuracy': float(accuracy),
            'qwk': float(qwk),
            'ece': float(ece),
            'macro_f1': float(macro_f1),
            'weighted_f1': float(weighted_f1),
            'per_class': per_class,
            'confusion_matrix': cm.tolist()
        }
        
        return results
    
    def print_results(self, results: Dict):
        """Print evaluation results."""
        print("\n" + "="*80)
        print("EVALUATION RESULTS SUMMARY")
        print("="*80)
        
        # Ensemble results
        ensemble = results['ensemble']
        print(f"\n[ENSEMBLE] {ensemble['model_name']}:")
        print(f"  Accuracy:  {ensemble['accuracy']:.4f} ({100*ensemble['accuracy']:.2f}%)")
        print(f"  QWK:       {ensemble['qwk']:.4f}")
        print(f"  Macro F1:  {ensemble['macro_f1']:.4f}")
        print(f"  Weighted F1: {ensemble['weighted_f1']:.4f}")
        print(f"  ECE:       {ensemble['ece']:.4f}")
        
        # Single model results (if available)
        if 'single' in results:
            single = results['single']
            print(f"\n[SINGLE] {single['model_name']}:")
            print(f"  Accuracy:  {single['accuracy']:.4f} ({100*single['accuracy']:.2f}%)")
            print(f"  QWK:       {single['qwk']:.4f}")
            print(f"  Macro F1:  {single['macro_f1']:.4f}")
            print(f"  Weighted F1: {single['weighted_f1']:.4f}")
            print(f"  ECE:       {single['ece']:.4f}")
            
            # Improvement
            print(f"\n[IMPROVEMENT]:")
            print(f"  Accuracy:  +{ensemble['accuracy'] - single['accuracy']:.4f} ({100*(ensemble['accuracy'] - single['accuracy'])/single['accuracy']:.2f}%)")
            print(f"  QWK:       +{ensemble['qwk'] - single['qwk']:.4f} ({100*(ensemble['qwk'] - single['qwk']):.2f}% relative)")
            print(f"  Macro F1:  +{ensemble['macro_f1'] - single['macro_f1']:.4f} ({100*(ensemble['macro_f1'] - single['macro_f1'])/single['macro_f1']:.2f}%)")
        
        # Per-class results
        print("\n" + "="*80)
        print("PER-CLASS RESULTS (Ensemble)")
        print("="*80)
        
        for i in range(5):
            metrics = ensemble['per_class'][f'class_{i}']
            print(f"\n{self.class_names[i]}:")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1:        {metrics['f1']:.4f}")
            print(f"  Support:   {metrics['support']}")
        
        print("\n" + "="*80)
    
    def save_results(self, results: Dict, output_path: str = "ensemble_evaluation_results.json"):
        """Save results to JSON."""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")


def main():
    """Main evaluation function."""
    print("="*80)
    print("ENSEMBLE MODEL EVALUATION")
    print("="*80)
    
    # Define checkpoints
    ensemble_checkpoints = [
        "1/38054df5c2da4cc6b648ff50fbd36590/checkpoints/dr-model-epoch=10-val_qwk=0.785.ckpt",
        "1/38054df5c2da4cc6b648ff50fbd36590/checkpoints/dr-model-epoch=24-val_qwk=0.767.ckpt",
        "1/38054df5c2da4cc6b648ff50fbd36590/checkpoints/dr-model-epoch=15-val_qwk=0.758.ckpt",
        "1/7d0928bb87954a739123ca35fa03cccf/checkpoints/dr-model-epoch=11-val_qwk=0.769.ckpt",
        "1/7d0928bb87954a739123ca35fa03cccf/checkpoints/dr-model-epoch=07-val_qwk=0.768.ckpt",
    ]
    
    single_checkpoint = "1/38054df5c2da4cc6b648ff50fbd36590/checkpoints/dr-model-epoch=10-val_qwk=0.785.ckpt"
    
    # Initialize evaluator
    evaluator = EnsembleEvaluator(
        ensemble_checkpoints=ensemble_checkpoints,
        single_checkpoint=single_checkpoint
    )
    
    # Load validation data
    print("\nLoading validation data...")
    processor = DataProcessor()
    _, val_loader, _ = processor.prepare_datasets()
    
    print(f"Validation set: {len(val_loader.dataset)} images")
    
    # Use subset for faster evaluation (optional - comment out for full evaluation)
    print("\nNote: Using first 1000 samples for quick evaluation...")
    from torch.utils.data import Subset
    val_subset = Subset(val_loader.dataset, range(min(1000, len(val_loader.dataset))))
    val_loader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=val_loader.batch_size,
        shuffle=False,
        num_workers=val_loader.num_workers
    )
    print(f"Evaluating on {len(val_subset)} images (subset for speed)")
    
    # Evaluate
    results = evaluator.evaluate_on_validation(val_loader)
    
    # Print results
    evaluator.print_results(results)
    
    # Save results
    evaluator.save_results(results, "outputs/ensemble_evaluation_results.json")
    
    print("\n[SUCCESS] Evaluation complete!")


if __name__ == "__main__":
    main()
