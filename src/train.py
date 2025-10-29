"""
Training script for Diabetic Retinopathy detection model.
Implements PyTorch Lightning training with MLflow tracking.
"""

import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import MLFlowLogger
import mlflow
import mlflow.pytorch
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from typing import Dict, Any

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import DRModel, FocalLoss, ModelMetrics, calculate_qwk, calculate_ece
from src.data_processing import DataProcessor


class DRModelModule(pl.LightningModule):
    """PyTorch Lightning module for DR model training."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.save_hyperparameters()
        
        self.config = config
        self.model_config = config['model']
        self.training_config = config['training']
        
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
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        images, targets = batch
        logits = self(images)
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
        """Calculate comprehensive validation metrics."""
        if not self.val_preds:
            return
        
        # Convert to numpy arrays
        val_preds = np.array(self.val_preds)
        val_targets = np.array(self.val_targets)
        val_probs = np.array(self.val_probs)
        
        # Calculate metrics
        metrics = self.metrics.calculate_metrics(val_targets, val_preds, val_probs)
        
        # Log metrics
        for key, value in metrics.items():
            self.log(f'val_{key}', value, on_epoch=True)
        
        # Clear stored predictions
        self.train_preds.clear()
        self.train_targets.clear()
        self.val_preds.clear()
        self.val_targets.clear()
        self.val_probs.clear()
    
    def test_step(self, batch, batch_idx):
        """Test step for final evaluation."""
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
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        self.log('test_acc', accuracy_score(targets.cpu(), preds.cpu()), 
                on_step=False, on_epoch=True)
        
        return loss
    
    def on_test_epoch_end(self):
        """Calculate comprehensive test metrics."""
        if not self.val_preds:
            return
        
        # Convert to numpy arrays
        test_preds = np.array(self.val_preds)
        test_targets = np.array(self.val_targets)
        test_probs = np.array(self.val_probs)
        
        # Calculate metrics
        metrics = self.metrics.calculate_metrics(test_targets, test_preds, test_probs)
        
        # Log metrics
        for key, value in metrics.items():
            self.log(f'test_{key}', value, on_epoch=True)
        
        # Print summary
        print("\n" + "="*50)
        print("TEST RESULTS")
        print("="*50)
        print(f"Test Accuracy: {metrics['accuracy']:.4f}")
        print(f"Test QWK: {metrics['qwk']:.4f}")
        print(f"Test Macro F1: {metrics['macro_f1']:.4f}")
        print(f"Test ECE: {metrics['ece']:.4f}")
        print("="*50)
        
        # Clear stored predictions
        self.val_preds.clear()
        self.val_targets.clear()
        self.val_probs.clear()
    
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


class Trainer:
    """Main trainer class."""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.mlflow_config = self.config['mlflow']
        self.training_config = self.config['training']
        
        # Setup MLflow
        mlflow.set_tracking_uri(self.mlflow_config['tracking_uri'])
        mlflow.set_experiment(self.mlflow_config['experiment_name'])
    
    def train(self):
        """Main training function."""
        # Prepare data
        print("Preparing data...")
        processor = DataProcessor()
        train_loader, val_loader, test_loader = processor.prepare_datasets()
        
        # Create model
        print("Creating model...")
        model_module = DRModelModule(self.config)
        
        # Setup callbacks
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
        
        # Setup logger
        mlflow_logger = MLFlowLogger(
            experiment_name=self.mlflow_config['experiment_name'],
            tracking_uri=self.mlflow_config['tracking_uri']
        )
        
        # Setup trainer
        trainer = pl.Trainer(
            max_epochs=self.training_config['epochs'],
            callbacks=[checkpoint_callback, early_stopping, lr_monitor],
            logger=mlflow_logger,
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            devices=1,
            precision=16 if torch.cuda.is_available() else 32,
            gradient_clip_val=1.0,
            log_every_n_steps=10
        )
        
        # Start training
        print("Starting training...")
        with mlflow.start_run():
            # Log config
            mlflow.log_params(self.config)
            
            trainer.fit(model_module, train_loader, val_loader)
            
            # Test on best model
            print("Testing on best model...")
            best_model_path = checkpoint_callback.best_model_path
            if best_model_path:
                model_module = DRModelModule.load_from_checkpoint(best_model_path)
                trainer.test(model_module, test_loader)
            
            # Log model
            mlflow.pytorch.log_model(
                model_module.model,
                "model",
                registered_model_name="dr-detection-model"
            )
        
        print("Training completed!")


def main():
    """Main training function."""
    trainer = Trainer()
    trainer.train()


if __name__ == "__main__":
    main()
