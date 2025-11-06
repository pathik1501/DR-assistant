"""
Data processing module for Diabetic Retinopathy detection.
Handles APTOS 2019 and EyePACS dataset loading, preprocessing, and augmentation.
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedKFold
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import yaml


class DRDataset(Dataset):
    """Custom dataset for Diabetic Retinopathy images."""
    
    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        transform: Optional[A.Compose] = None,
        is_training: bool = True
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.is_training = is_training
        
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]
        
        return image, label


class DataProcessor:
    """Handles data loading, preprocessing, and augmentation."""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_config = self.config['data']
        self.aug_config = self.config['augmentation']
        
    def load_aptos_data(self, data_path: str) -> Tuple[List[str], List[int]]:
        """Load APTOS 2019 dataset."""
        train_csv = os.path.join(data_path, "train.csv")
        train_images_dir = os.path.join(data_path, "train_images")
        
        if not os.path.exists(train_csv):
            raise FileNotFoundError(f"APTOS train.csv not found at {train_csv}")
        
        df = pd.read_csv(train_csv)
        image_paths = []
        labels = []
        
        for _, row in df.iterrows():
            image_path = os.path.join(train_images_dir, f"{row['id_code']}.png")
            if os.path.exists(image_path):
                image_paths.append(image_path)
                labels.append(row['diagnosis'])
        
        return image_paths, labels
    
    def load_eyepacs_data(self, data_path: str) -> Tuple[List[str], List[int]]:
        """Load EyePACS dataset from augmented_resized_V2 structure."""
        train_dir = os.path.join(data_path, "train")
        
        if not os.path.exists(train_dir):
            raise FileNotFoundError(f"EyePACS train directory not found at {train_dir}")
        
        image_paths = []
        labels = []
        
        # Load images from grade-based subdirectories
        for grade in range(5):  # Grades 0-4
            grade_dir = os.path.join(train_dir, str(grade))
            if os.path.exists(grade_dir):
                for image_file in os.listdir(grade_dir):
                    if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        image_path = os.path.join(grade_dir, image_file)
                        image_paths.append(image_path)
                        labels.append(grade)
        
        return image_paths, labels
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess single image."""
        # Resize to target size
        target_size = self.data_config['output_size']
        image = cv2.resize(image, (target_size[1], target_size[0]))
        
        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        return image
    
    def get_training_transforms(self) -> A.Compose:
        """Get training augmentation transforms."""
        return A.Compose([
            A.Resize(self.data_config['output_size'][0], self.data_config['output_size'][1]),
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=self.aug_config['brightness_limit'],
                contrast_limit=self.aug_config['contrast_limit'],
                p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=self.aug_config['hue_shift_limit'],
                sat_shift_limit=self.aug_config['saturation_shift_limit'],
                val_shift_limit=20,
                p=0.5
            ),
            A.CoarseDropout(
                max_holes=8,
                max_height=32,
                max_width=32,
                p=0.5
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    
    def get_validation_transforms(self) -> A.Compose:
        """Get validation transforms (no augmentation)."""
        return A.Compose([
            A.Resize(self.data_config['output_size'][0], self.data_config['output_size'][1]),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    
    def create_data_loaders(
        self,
        train_paths: List[str],
        train_labels: List[int],
        val_paths: List[str],
        val_labels: List[int],
        test_paths: List[str],
        test_labels: List[int],
        use_weighted_sampling: bool = False
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create data loaders for train, validation, and test sets."""
        
        train_dataset = DRDataset(
            train_paths, train_labels,
            transform=self.get_training_transforms(),
            is_training=True
        )
        
        val_dataset = DRDataset(
            val_paths, val_labels,
            transform=self.get_validation_transforms(),
            is_training=False
        )
        
        test_dataset = DRDataset(
            test_paths, test_labels,
            transform=self.get_validation_transforms(),
            is_training=False
        )
        
        # Setup train loader with optional weighted sampling
        if use_weighted_sampling:
            # Calculate sample weights (inverse frequency)
            train_labels_np = np.array(train_labels)
            class_counts = np.bincount(train_labels_np, minlength=5)
            # Avoid division by zero
            class_counts = np.where(class_counts == 0, 1, class_counts)
            class_weights_sample = 1.0 / class_counts
            sample_weights = [class_weights_sample[label] for label in train_labels]
            
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )
            
            print(f"Using weighted sampling for training:")
            for i in range(5):
                count = class_counts[i]
                weight = class_weights_sample[i]
                print(f"  Class {i}: {count:5d} samples, sampling weight: {weight:.4f}")
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.data_config['batch_size'],
                sampler=sampler,
                num_workers=self.data_config['num_workers'],
                pin_memory=True
            )
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.data_config['batch_size'],
                shuffle=True,
                num_workers=self.data_config['num_workers'],
                pin_memory=True
            )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.data_config['batch_size'],
            shuffle=False,
            num_workers=self.data_config['num_workers'],
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.data_config['batch_size'],
            shuffle=False,
            num_workers=self.data_config['num_workers'],
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader
    
    def prepare_datasets(self) -> Tuple[DataLoader, DataLoader, DataLoader, List[int]]:
        """Prepare complete datasets with train/val/test split.
        
        Returns:
            train_loader, val_loader, test_loader, train_labels
        """
        
        # Load both datasets
        aptos_paths, aptos_labels = self.load_aptos_data(self.data_config['aptos_path'])
        eyepacs_paths, eyepacs_labels = self.load_eyepacs_data(self.data_config['eyepacs_path'])
        
        # Combine datasets
        all_paths = aptos_paths + eyepacs_paths
        all_labels = aptos_labels + eyepacs_labels
        
        print(f"Total images loaded: {len(all_paths)}")
        print(f"APTOS images: {len(aptos_paths)}")
        print(f"EyePACS images: {len(eyepacs_paths)}")
        
        # Stratified split
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        train_idx, temp_idx = next(iter(skf.split(all_paths, all_labels)))
        
        # Further split temp into val and test
        val_labels_temp = [all_labels[i] for i in temp_idx]
        skf_val = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
        val_idx_temp, test_idx_temp = next(iter(skf_val.split(temp_idx, val_labels_temp)))
        
        val_idx = [temp_idx[i] for i in val_idx_temp]
        test_idx = [temp_idx[i] for i in test_idx_temp]
        
        # Create final splits
        train_paths = [all_paths[i] for i in train_idx]
        train_labels = [all_labels[i] for i in train_idx]
        
        val_paths = [all_paths[i] for i in val_idx]
        val_labels = [all_labels[i] for i in val_idx]
        
        test_paths = [all_paths[i] for i in test_idx]
        test_labels = [all_labels[i] for i in test_idx]
        
        print(f"Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")
        
        # Check if weighted sampling is enabled
        use_weighted_sampling = self.config.get('training', {}).get('use_weighted_sampling', False)
        
        train_loader, val_loader, test_loader = self.create_data_loaders(
            train_paths, train_labels,
            val_paths, val_labels,
            test_paths, test_labels,
            use_weighted_sampling=use_weighted_sampling
        )
        
        return train_loader, val_loader, test_loader, train_labels


def main():
    """Test data loading functionality."""
    processor = DataProcessor()
    
    try:
        train_loader, val_loader, test_loader, train_labels = processor.prepare_datasets()
        
        print("Data loaders created successfully!")
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        print(f"Test batches: {len(test_loader)}")
        print(f"Training labels count: {len(train_labels)}")
        
        # Test a batch
        for images, labels in train_loader:
            print(f"Batch shape: {images.shape}")
            print(f"Labels: {labels}")
            break
            
    except FileNotFoundError as e:
        print(f"Dataset not found: {e}")
        print("Please download APTOS 2019 and EyePACS datasets to the data/ directory")


if __name__ == "__main__":
    main()
