#!/usr/bin/env python3
"""
Dataset download script for Diabetic Retinopathy Assistant.
Downloads APTOS 2019 and EyePACS datasets from Kaggle.
"""

import os
import subprocess
import zipfile
from pathlib import Path


def check_kaggle_auth():
    """Check if Kaggle API is authenticated."""
    try:
        result = subprocess.run(['kaggle', 'competitions', 'list'], 
                              capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print("✅ Kaggle API authenticated successfully")
            return True
        else:
            print("❌ Kaggle API authentication failed")
            print("Please set up your Kaggle API key:")
            print("1. Go to https://www.kaggle.com/account")
            print("2. Create API token")
            print("3. Place kaggle.json in ~/.kaggle/ directory")
            return False
    except Exception as e:
        print(f"❌ Error checking Kaggle auth: {e}")
        return False


def download_aptos_dataset():
    """Download APTOS 2019 dataset."""
    print("\n📊 Downloading APTOS 2019 dataset...")
    
    try:
        # Create directory
        aptos_dir = Path("data/aptos2019")
        aptos_dir.mkdir(parents=True, exist_ok=True)
        
        # Download train.csv
        print("  Downloading train.csv...")
        subprocess.run([
            'kaggle', 'competitions', 'download', '-c', 'aptos2019-blindness-detection',
            '-f', 'train.csv', '-p', str(aptos_dir)
        ], check=True)
        
        # Download train_images.zip
        print("  Downloading train_images.zip...")
        subprocess.run([
            'kaggle', 'competitions', 'download', '-c', 'aptos2019-blindness-detection',
            '-f', 'train_images.zip', '-p', str(aptos_dir)
        ], check=True)
        
        # Extract train_images.zip
        print("  Extracting train_images.zip...")
        with zipfile.ZipFile(aptos_dir / "train_images.zip", 'r') as zip_ref:
            zip_ref.extractall(aptos_dir)
        
        # Remove zip file
        (aptos_dir / "train_images.zip").unlink()
        
        print("✅ APTOS 2019 dataset downloaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Failed to download APTOS dataset: {e}")
        return False


def download_eyepacs_dataset():
    """Download EyePACS dataset."""
    print("\n📊 Downloading EyePACS dataset...")
    
    try:
        # Create directory
        eyepacs_dir = Path("data/eyepacs")
        eyepacs_dir.mkdir(parents=True, exist_ok=True)
        
        # Download trainLabels.csv
        print("  Downloading trainLabels.csv...")
        subprocess.run([
            'kaggle', 'datasets', 'download', '-d', 'sovitrath/eyepacs-dataset',
            '-f', 'trainLabels.csv', '-p', str(eyepacs_dir)
        ], check=True)
        
        # Download train.zip
        print("  Downloading train.zip...")
        subprocess.run([
            'kaggle', 'datasets', 'download', '-d', 'sovitrath/eyepacs-dataset',
            '-f', 'train.zip', '-p', str(eyepacs_dir)
        ], check=True)
        
        # Extract train.zip
        print("  Extracting train.zip...")
        with zipfile.ZipFile(eyepacs_dir / "train.zip", 'r') as zip_ref:
            zip_ref.extractall(eyepacs_dir)
        
        # Remove zip file
        (eyepacs_dir / "train.zip").unlink()
        
        print("✅ EyePACS dataset downloaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Failed to download EyePACS dataset: {e}")
        return False


def verify_datasets():
    """Verify that datasets are properly downloaded."""
    print("\n🔍 Verifying datasets...")
    
    # Check APTOS
    aptos_csv = Path("data/aptos2019/train.csv")
    aptos_images = Path("data/aptos2019/train_images")
    
    if aptos_csv.exists() and aptos_images.exists():
        aptos_count = len(list(aptos_images.glob("*.png")))
        print(f"✅ APTOS 2019: {aptos_count} images found")
    else:
        print("❌ APTOS 2019 dataset incomplete")
        return False
    
    # Check EyePACS
    eyepacs_csv = Path("data/eyepacs/trainLabels.csv")
    eyepacs_images = Path("data/eyepacs/train")
    
    if eyepacs_csv.exists() and eyepacs_images.exists():
        eyepacs_count = len(list(eyepacs_images.glob("*.jpeg")))
        print(f"✅ EyePACS: {eyepacs_count} images found")
    else:
        print("❌ EyePACS dataset incomplete")
        return False
    
    print(f"✅ Total images: {aptos_count + eyepacs_count}")
    return True


def main():
    """Main download function."""
    print("🚀 Diabetic Retinopathy Assistant - Dataset Download")
    print("=" * 60)
    
    # Check Kaggle authentication
    if not check_kaggle_auth():
        print("\nPlease set up Kaggle API authentication first:")
        print("1. Go to https://www.kaggle.com/account")
        print("2. Click 'Create New API Token'")
        print("3. Download kaggle.json")
        print("4. Place it in C:\\Users\\[username]\\.kaggle\\ directory")
        print("5. Run this script again")
        return
    
    # Download datasets
    success = True
    
    if not download_aptos_dataset():
        success = False
    
    if not download_eyepacs_dataset():
        success = False
    
    # Verify datasets
    if success and verify_datasets():
        print("\n🎉 All datasets downloaded successfully!")
        print("\nNext steps:")
        print("1. Run training: python src/train.py")
        print("2. Start API: python src/inference.py")
        print("3. Launch UI: streamlit run frontend/app.py")
    else:
        print("\n❌ Dataset download failed. Please check the errors above.")


if __name__ == "__main__":
    main()
