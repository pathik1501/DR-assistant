#!/usr/bin/env python3
"""
Quick MLflow dashboard launcher
Opens MLflow UI to monitor training progress
"""

import subprocess
import time
import os

def launch_mlflow():
    """Launch MLflow tracking UI."""
    print("Launching MLflow UI...")
    print("URL: http://localhost:5000")
    print("\nAccess the dashboard in your browser!")
    
    # Start MLflow server
    subprocess.run([
        "mlflow", "ui",
        "--backend-store-uri", "sqlite:///mlflow.db",
        "--port", "5000"
    ])


if __name__ == "__main__":
    try:
        launch_mlflow()
    except KeyboardInterrupt:
        print("\nMLflow server stopped")
