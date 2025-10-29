#!/usr/bin/env python3
"""
Simple deployment script to start the DR Assistant API
"""

import subprocess
import sys
import os

def main():
    """Start the deployment."""
    print("="*60)
    print("Diabetic Retinopathy Triage Assistant - Deployment")
    print("="*60)
    
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)
    
    print("\nStarting deployment...")
    print(f"Working directory: {current_dir}")
    
    # Check if model checkpoint exists
    checkpoint_path = "1/7d0928bb87954a739123ca35fa03cccf/checkpoints/dr-model-epoch=11-val_qwk=0.769.ckpt"
    if os.path.exists(checkpoint_path):
        print("\nModel found: ", checkpoint_path)
        print("Starting API server...")
        print("\nAPI will be available at: http://localhost:8000")
        print("Docs will be available at: http://localhost:8000/docs")
        print("\nPress Ctrl+C to stop the server\n")
        
        # Start the API
        try:
            subprocess.run([sys.executable, "src/inference.py"])
        except KeyboardInterrupt:
            print("\n\nServer stopped.")
    else:
        print("\nModel checkpoint not found!")
        print("Please make sure you have trained the model first.")
        print(f"Expected path: {checkpoint_path}")

if __name__ == "__main__":
    main()
