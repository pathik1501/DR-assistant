#!/usr/bin/env python3
"""
Simple setup script for Diabetic Retinopathy Triage Assistant.
"""

import os
import sys
from pathlib import Path


def create_directories():
    """Create necessary directories."""
    directories = [
        "data/aptos2019",
        "data/eyepacs", 
        "data/vector_db",
        "models",
        "logs",
        "outputs",
        "outputs/evaluation",
        "outputs/explanations"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")
    
    return True


def create_env_file():
    """Create environment file."""
    env_content = """# Diabetic Retinopathy Assistant Environment Variables

# OpenAI API Key (required for RAG pipeline)
OPENAI_API_KEY=your-openai-api-key-here

# MLflow Configuration
MLFLOW_TRACKING_URI=sqlite:///mlflow.db
MLFLOW_DEFAULT_ARTIFACT_ROOT=./outputs/mlflow_artifacts

# GPU Configuration (optional)
CUDA_VISIBLE_DEVICES=0

# API Configuration
API_HOST=0.0.0.0
API_PORT=8080
"""
    
    env_file = Path(".env")
    if not env_file.exists():
        with open(env_file, 'w') as f:
            f.write(env_content)
        print("Created .env file - please update with your API keys")
    else:
        print(".env file already exists")
    
    return True


def main():
    """Main setup function."""
    print("Diabetic Retinopathy Triage Assistant Setup")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("Python 3.8+ is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"Python version: {sys.version}")
    
    # Create directories
    print("\nCreating directories...")
    create_directories()
    
    # Setup environment
    print("\nSetting up environment...")
    create_env_file()
    
    print("\n" + "="*60)
    print("SETUP COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("""
Next steps:
1. Install dependencies: pip install -r requirements.txt
2. Download datasets (see instructions below)
3. Set your OpenAI API key in the .env file
4. Run training: python src/train.py
5. Start API: python src/inference.py
6. Launch UI: streamlit run frontend/app.py

DATASET DOWNLOAD INSTRUCTIONS:
1. APTOS 2019: https://www.kaggle.com/competitions/aptos2019-blindness-detection/data
   - Download train.csv and train_images.zip
   - Extract to: data/aptos2019/

2. EyePACS: https://www.kaggle.com/datasets/sovitrath/eyepacs-dataset
   - Download trainLabels.csv and train.zip
   - Extract to: data/eyepacs/

For Docker deployment: docker-compose up --build
""")


if __name__ == "__main__":
    main()
