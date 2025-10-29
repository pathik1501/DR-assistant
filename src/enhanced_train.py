"""
Enhanced training with comprehensive monitoring and evaluation.
Implements all the metrics and visualizations from the evaluation checklist.
"""

import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import MLFlowLogger, CSVLogger
import mlflow
import mlflow.pytorch
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support,
    classification_report, confusion_matrix, cohen_kappa_score,
    roc_curve, auc
)
import numpy as np
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import time

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import DRModel, FocalLoss, ModelMetrics, calculate_qwk, calculate_ece
from src.data_processing import DataProcessor


class EnhancedDRModelModule(pl.LightningModule):
    """Enhanced PyTorch Lightning module with comprehensive monitoring."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.save_hyperparameters()
        
        self.config = config
        self.model_config = config['model']
        self.training_config = config['training']
        self.uncertainty_config = config['uncertainty']
        
        # Create model
        self.model = DRModel(
            num_classes=self.model_config['num_classes'],
            pretrained=self.model_config['pretrained'],
            dropout_rate=self.model_config['dropout_rate']
        )
        
        # Loss function with label smoothing
        self.criterion = FocalLoss(
            gamma=self.training_config['focal_loss_gamma'],
            label_smoothing=self.training_config.get('label_smoothing', 0.0)
        )
        
        # Metrics
        self.metrics = ModelMetrics()
        
        # Store predictions for epoch-end metrics
        self.train_preds = []
        self.train_targets = []
        self.val_preds = []
        self.val_targets = []
        self.val_probs = []
        self.test_preds = []
        self.test_targets = []
        self.test_probs = []
        
        # Performance tracking
        self.train_losses = []
        self.val_losses = []
        self.qwk_scores = []
        self.macro_f1_scores = []
        self.per_class_f1 = [[] for _ in range(5)]
        self.latencies = []
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        images, targets = batch
        
        # Time inference
        start_time = time.time()
        logits = self(images)
        self.latencies.append((time.time() - start_time) * 1000)  # ms
        
        loss = self.criterion(logits, targets)
        
        # Store predictions
        probs = F.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)
        
        self.train_preds.extend(preds.cpu().numpy())
        self.train_targets.extend(targets.cpu().numpy())
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', accuracy_score(targets.cpu(), preds.cpu()), 
                on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, targets = batch
        logits = self(images)
        loss = self.criterion(logits, targets)
        
        # Store predictions
        probs = F.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)
        
        self.val_preds.extend(preds.cpu().numpy())
        self.val_targets.extend(targets.cpu().numpy())
        self.val_probs.extend(probs.cpu().numpy())
        
        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', accuracy_score(targets.cpu(), preds.cpu()), 
                on_step=False, on_epoch=True)
        
        return loss
    
    def on_validation_epoch_end(self):
        """Comprehensive validation metrics and logging."""
        if not self.val_preds:
            return
        
        # Convert to numpy arrays
        val_preds = np.array(self.val_preds)
        val_targets = np.array(self.val_targets)
        val_probs = np.array(self.val_probs)
        
        # Calculate metrics
        metrics = self.metrics.calculate_metrics(val_targets, val_preds, val_probs)
        
        # Store for plotting
        self.val_losses.append(float(self.trainer.callback_metrics.get('val_loss', 0)))
        self.qwk_scores.append(metrics['qwk'])
        self.macro_f1_scores.append(metrics['macro_f1'])
        
        # Per-class F1
        for i in range(5):
            self.per_class_f1[i].append(metrics.get(f'f1_class_{i}', 0.0))
        
        # Log all metrics
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.log(f'val_{key}', value, on_epoch=True)
        
        # Log latency statistics
        if self.latencies:
            latencies_np = np.array(self.latencies)
            self.log('p50_latency', np.percentile(latencies_np, 50), on_epoch=True)
            self.log('p95_latency', np.percentile(latencies_np, 95), on_epoch=True)
            self.log('p99_latency', np.percentile(latencies_np, 99), on_epoch=True)
            self.latencies = []
        
        # Generate visualizations every 5 epochs
        if self.current_epoch % 5 == 0 or self.current_epoch == 0:
            self._generate_visualizations(val_targets, val_preds, val_probs)
        
        # Clear stored predictions
        self.train_preds.clear()
        self.train_targets.clear()
        self.val_preds.clear()
        self.val_targets.clear()
        self.val_probs.clear()
    
    def _generate_visualizations(self, targets, preds, probs):
        """Generate all required visualizations."""
        
        # 1. Confusion Matrix
        cm = confusion_matrix(targets, preds)
        self._plot_confusion_matrix(cm, f'confusion_matrix_epoch_{self.current_epoch}')
        
        # 2. Per-Class F1 Trend
        self._plot_per_class_f1()
        
        # 3. Calibration Plot
        self._plot_calibration(targets, probs)
        
        # 4. QWK and F1 Progress
        self._plot_metric_progress()
        
    def _plot_confusion_matrix(self, cm, name):
        """Plot normalized confusion matrix."""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Normalize
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(
            cm_normalized, 
            annot=True, 
            fmt='.2f', 
            cmap='Blues',
            xticklabels=['Grade 0', 'Grade 1', 'Grade 2', 'Grade 3', 'Grade 4'],
            yticklabels=['Grade 0', 'Grade 1', 'Grade 2', 'Grade 3', 'Grade 4']
        )
        plt.title(f'Confusion Matrix (Epoch {self.current_epoch})')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig(f'outputs/evaluation/{name}.png', dpi=150)
        plt.close()
        
        # Log to MLflow
        if hasattr(self.logger, 'experiment'):
            mlflow.log_figure(fig, f"{name}.png")
    
    def _plot_per_class_f1(self):
        """Plot per-class F1 scores over time."""
        if len(self.per_class_f1[0]) < 2:
            return
            
        fig, ax = plt.subplots(figsize=(10, 6))
        
        epochs = range(len(self.per_class_f1[0]))
        for i in range(5):
            ax.plot(epochs, self.per_class_f1[i], label=f'Grade {i}', marker='o', markersize=3)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('F1 Score')
        ax.set_title('Per-Class F1 Scores Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'outputs/evaluation/per_class_f1_epoch_{self.current_epoch}.png', dpi=150)
        plt.close()
        
        if hasattr(self.logger, 'experiment'):
            mlflow.log_figure(fig, f"per_class_f1_epoch_{self.current_epoch}.png")
    
    def _plot_calibration(self, targets, probs, suffix=''):
        """Plot reliability diagram."""
        from sklearn.calibration import calibration_curve
        
        # Convert to binary for calibration
        targets_binary = (targets > 0).astype(int)
        probs_binary = probs.max(axis=1)
        
        fraction_of_positives, mean_predicted_value = calibration_curve(
            targets_binary, probs_binary, n_bins=15
        )
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model", linewidth=2)
        ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated", linewidth=2)
        
        ax.set_xlabel('Mean Predicted Probability', fontsize=12)
        ax.set_ylabel('Fraction of Positives', fontsize=12)
        title = f'Reliability Diagram (Epoch {self.current_epoch})' if hasattr(self, 'current_epoch') else 'Reliability Diagram'
        ax.set_title(title, fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Add ECE to legend
        ece = calculate_ece(targets_binary, probs_binary)
        ax.text(0.05, 0.95, f'ECE: {ece:.3f}', transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=12, verticalalignment='top')
        
        plt.tight_layout()
        filename = f'outputs/evaluation/calibration{suffix}.png'
        plt.savefig(filename, dpi=150)
        plt.close()
        
        if hasattr(self.logger, 'experiment'):
            mlflow.log_figure(fig, filename)
    
    def _plot_metric_progress(self):
        """Plot QWK and macro F1 progress."""
        if len(self.qwk_scores) < 2:
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        epochs = range(len(self.qwk_scores))
        
        # QWK
        ax1.plot(epochs, self.qwk_scores, 'b-o', markersize=4, linewidth=2)
        ax1.set_ylabel('QWK', fontsize=12)
        ax1.set_title('QWK Progress', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0.88, color='r', linestyle='--', label='Target (0.88)')
        ax1.legend()
        
        # Macro F1
        ax2.plot(epochs, self.macro_f1_scores, 'g-o', markersize=4, linewidth=2)
        ax2.set_ylabel('Macro F1', fontsize=12)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_title('Macro F1 Progress', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0.79, color='r', linestyle='--', label='Target (0.79)')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(f'outputs/evaluation/metric_progress_epoch_{self.current_epoch}.png', dpi=150)
        plt.close()
        
        if hasattr(self.logger, 'experiment'):
            mlflow.log_figure(fig, f"metric_progress_epoch_{self.current_epoch}.png")
    
    def test_step(self, batch, batch_idx):
        """Test step for final evaluation."""
        images, targets = batch
        logits = self(images)
        loss = self.criterion(logits, targets)
        
        # Store predictions
        probs = F.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)
        
        self.test_preds.extend(preds.cpu().numpy())
        self.test_targets.extend(targets.cpu().numpy())
        self.test_probs.extend(probs.cpu().numpy())
        
        return loss
    
    def on_test_epoch_end(self):
        """Comprehensive test metrics."""
        if not self.test_preds:
            return
        
        test_preds = np.array(self.test_preds)
        test_targets = np.array(self.test_targets)
        test_probs = np.array(self.test_probs)
        
        # Calculate metrics
        metrics = self.metrics.calculate_metrics(test_targets, test_preds, test_probs)
        
        # Generate final visualizations
        self._plot_confusion_matrix(
            confusion_matrix(test_targets, test_preds),
            'confusion_matrix_test_final'
        )
        self._plot_calibration(test_targets, test_probs, 'test')
        
        # Print summary
        print("\n" + "="*60)
        print("TEST SET RESULTS")
        print("="*60)
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"QWK:       {metrics['qwk']:.4f}")
        print(f"Macro F1:  {metrics['macro_f1']:.4f}")
        print(f"ECE:       {metrics['ece']:.4f}")
        print("="*60)
        
        # Log all metrics
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.log(f'test_{key}', value, on_epoch=True)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.training_config['learning_rate'],
            weight_decay=self.training_config['weight_decay']
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.training_config['epochs'],
            eta_min=1e-6
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch'
            }
        }


class EnhancedTrainer:
    """Enhanced trainer with comprehensive monitoring."""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.mlflow_config = self.config['mlflow']
        self.training_config = self.config['training']
        
        # Setup MLflow
        mlflow.set_tracking_uri(self.mlflow_config['tracking_uri'])
        mlflow.set_experiment(self.mlflow_config['experiment_name'])
    
    def train(self):
        """Enhanced training with full monitoring."""
        print("Setting up comprehensive monitoring...")
        
        # Prepare data
        processor = DataProcessor()
        train_loader, val_loader, test_loader = processor.prepare_datasets()
        
        # Create model
        model_module = EnhancedDRModelModule(self.config)
        
        # Callbacks
        checkpoint_callback = ModelCheckpoint(
            monitor='val_qwk',
            mode='max',
            save_top_k=3,
            filename='dr-model-{epoch:02d}-{val_qwk:.3f}',
            save_last=True
        )
        
        early_stopping = EarlyStopping(
            monitor='val_qwk',
            mode='max',
            patience=self.training_config['patience'],
            verbose=True
        )
        
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        
        # Loggers
        mlflow_logger = MLFlowLogger(
            experiment_name=self.mlflow_config['experiment_name'],
            tracking_uri=self.mlflow_config['tracking_uri']
        )
        
        csv_logger = CSVLogger("logs", name="dr_training")
        
        # Create trainer
        trainer = pl.Trainer(
            max_epochs=self.training_config['epochs'],
            callbacks=[checkpoint_callback, early_stopping, lr_monitor],
            logger=[mlflow_logger, csv_logger],
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=1,
            precision='16-mixed' if torch.cuda.is_available() else 32,
            gradient_clip_val=1.0,
            log_every_n_steps=50
        )
        
        # Start training
        print("Starting enhanced training with full monitoring...")
        with mlflow.start_run():
            # Log config
            mlflow.log_params({
                'model': self.config['model']['architecture'],
                'batch_size': self.config['data']['batch_size'],
                'learning_rate': self.training_config['learning_rate'],
                'weight_decay': self.training_config['weight_decay'],
                'label_smoothing': self.training_config.get('label_smoothing', 0.0)
            })
            
            trainer.fit(model_module, train_loader, val_loader)
            
            # Test on best model
            print("Testing on best model...")
            trainer.test(model_module, test_loader)
            
            # Log model
            mlflow.pytorch.log_model(
                model_module.model,
                "model",
                registered_model_name="dr-detection-enhanced"
            )
        
        print("Enhanced training completed!")


def main():
    """Main training function."""
    trainer = EnhancedTrainer()
    trainer.train()


if __name__ == "__main__":
    main()
