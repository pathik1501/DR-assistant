#!/usr/bin/env python3
"""
Setup script for Diabetic Retinopathy Triage Assistant.
Automates the initial setup and environment configuration.
"""

import os
import sys
import subprocess
import yaml
from pathlib import Path


def run_command(command: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 10):
        print("❌ Python 3.10+ is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version}")
    return True


def install_dependencies():
    """Install Python dependencies."""
    return run_command("pip install -r requirements.txt", "Installing Python dependencies")


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
        print(f"📁 Created directory: {directory}")
    
    return True


def setup_environment():
    """Setup environment variables."""
    env_file = Path(".env")
    
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

# Monitoring
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
"""
    
    if not env_file.exists():
        with open(env_file, 'w') as f:
            f.write(env_content)
        print("📝 Created .env file - please update with your API keys")
    else:
        print("📝 .env file already exists")
    
    return True


def validate_config():
    """Validate configuration files."""
    config_path = Path("configs/config.yaml")
    
    if not config_path.exists():
        print("❌ Configuration file not found: configs/config.yaml")
        return False
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        required_sections = ['data', 'model', 'training', 'api']
        for section in required_sections:
            if section not in config:
                print(f"❌ Missing configuration section: {section}")
                return False
        
        print("✅ Configuration file is valid")
        return True
        
    except Exception as e:
        print(f"❌ Configuration file error: {e}")
        return False


def check_dataset_instructions():
    """Display dataset download instructions."""
    print("\n" + "="*60)
    print("📊 DATASET DOWNLOAD INSTRUCTIONS")
    print("="*60)
    print("""
To complete the setup, you need to download the datasets:

1. APTOS 2019 Blindness Detection:
   - Visit: https://www.kaggle.com/competitions/aptos2019-blindness-detection/data
   - Download train.csv and train_images.zip
   - Extract to: data/aptos2019/

2. EyePACS Dataset:
   - Visit: https://www.kaggle.com/datasets/sovitrath/eyepacs-dataset
   - Download trainLabels.csv and train.zip
   - Extract to: data/eyepacs/

3. Verify dataset structure:
   data/
   ├── aptos2019/
   │   ├── train.csv
   │   └── train_images/
   └── eyepacs/
       ├── trainLabels.csv
       └── train/
""")


def create_gitignore():
    """Create .gitignore file."""
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Data and models
data/
models/
logs/
outputs/
*.pth
*.pt
*.h5
*.pkl

# Environment variables
.env
.env.local

# MLflow
mlflow.db
mlruns/

# Jupyter
.ipynb_checkpoints/

# OS
.DS_Store
Thumbs.db

# Docker
.dockerignore

# Monitoring
prometheus_data/
grafana_data/
mlflow_data/
"""
    
    gitignore_path = Path(".gitignore")
    if not gitignore_path.exists():
        with open(gitignore_path, 'w') as f:
            f.write(gitignore_content)
        print("📝 Created .gitignore file")
    else:
        print("📝 .gitignore file already exists")
    
    return True


def main():
    """Main setup function."""
    print("🚀 Diabetic Retinopathy Triage Assistant Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    print("\n📁 Creating directories...")
    create_directories()
    
    # Install dependencies
    print("\n📦 Installing dependencies...")
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Setup environment
    print("\n🔧 Setting up environment...")
    setup_environment()
    
    # Create .gitignore
    print("\n📝 Creating .gitignore...")
    create_gitignore()
    
    # Validate configuration
    print("\n✅ Validating configuration...")
    if not validate_config():
        print("❌ Configuration validation failed")
        sys.exit(1)
    
    # Display dataset instructions
    check_dataset_instructions()
    
    print("\n" + "="*60)
    print("🎉 SETUP COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("""
Next steps:
1. Download the datasets as instructed above
2. Set your OpenAI API key in the .env file
3. Run training: python src/train.py
4. Start API: python src/inference.py
5. Launch UI: streamlit run frontend/app.py

For Docker deployment:
- docker-compose up --build

For more information, see README.md
""")


if __name__ == "__main__":
    main()
