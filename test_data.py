#!/usr/bin/env python3
"""
Test script to verify data processing pipeline works.
Run this after downloading the datasets.
"""

import os
from pathlib import Path
from src.data_processing import DataProcessor


def check_dataset_structure():
    """Check if datasets are properly structured."""
    print("🔍 Checking dataset structure...")
    
    # Check APTOS
    aptos_csv = Path("data/aptos2019/train.csv")
    aptos_images = Path("data/aptos2019/train_images")
    
    if aptos_csv.exists() and aptos_images.exists():
        aptos_count = len(list(aptos_images.glob("*.png")))
        print(f"✅ APTOS 2019: {aptos_count} images found")
        aptos_ok = True
    else:
        print("❌ APTOS 2019 dataset not found")
        print("   Expected: data/aptos2019/train.csv")
        print("   Expected: data/aptos2019/train_images/")
        aptos_ok = False
    
    # Check EyePACS
    eyepacs_csv = Path("data/eyepacs/trainLabels.csv")
    eyepacs_images = Path("data/eyepacs/train")
    
    if eyepacs_csv.exists() and eyepacs_images.exists():
        eyepacs_count = len(list(eyepacs_images.glob("*.jpeg")))
        print(f"✅ EyePACS: {eyepacs_count} images found")
        eyepacs_ok = True
    else:
        print("❌ EyePACS dataset not found")
        print("   Expected: data/eyepacs/trainLabels.csv")
        print("   Expected: data/eyepacs/train/")
        eyepacs_ok = False
    
    return aptos_ok and eyepacs_ok


def test_data_processor():
    """Test the data processor."""
    print("\n🔧 Testing data processor...")
    
    try:
        processor = DataProcessor()
        print("✅ Data processor created successfully")
        
        # Test transforms
        train_transforms = processor.get_training_transforms()
        val_transforms = processor.get_validation_transforms()
        print("✅ Transforms created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Data processor test failed: {e}")
        return False


def test_data_loading():
    """Test loading actual data."""
    print("\n📊 Testing data loading...")
    
    try:
        processor = DataProcessor()
        
        # Try to load datasets
        aptos_paths, aptos_labels = processor.load_aptos_data("data/aptos2019")
        eyepacs_paths, eyepacs_labels = processor.load_eyepacs_data("data/eyepacs")
        
        print(f"✅ APTOS loaded: {len(aptos_paths)} images")
        print(f"✅ EyePACS loaded: {len(eyepacs_paths)} images")
        print(f"✅ Total images: {len(aptos_paths) + len(eyepacs_paths)}")
        
        # Check label distribution
        import numpy as np
        all_labels = aptos_labels + eyepacs_labels
        unique, counts = np.unique(all_labels, return_counts=True)
        
        print("\nLabel distribution:")
        for label, count in zip(unique, counts):
            print(f"   Grade {label}: {count} images ({count/len(all_labels)*100:.1f}%)")
        
        return True
        
    except Exception as e:
        print(f"❌ Data loading test failed: {e}")
        return False


def main():
    """Main test function."""
    print("🚀 Diabetic Retinopathy Assistant - Data Test")
    print("=" * 50)
    
    # Check dataset structure
    if not check_dataset_structure():
        print("\n❌ Datasets not found. Please download them first:")
        print("   See MANUAL_DOWNLOAD.md for instructions")
        return
    
    # Test data processor
    if not test_data_processor():
        print("\n❌ Data processor test failed")
        return
    
    # Test data loading
    if not test_data_loading():
        print("\n❌ Data loading test failed")
        return
    
    print("\n" + "=" * 50)
    print("🎉 ALL TESTS PASSED!")
    print("=" * 50)
    print("\nYour datasets are ready for training!")
    print("\nNext steps:")
    print("1. Start training: python src/train.py")
    print("2. Monitor progress: Check MLflow at http://localhost:5000")
    print("3. Start API: python src/inference.py")
    print("4. Launch UI: streamlit run frontend/app.py")


if __name__ == "__main__":
    main()
