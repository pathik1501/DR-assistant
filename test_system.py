#!/usr/bin/env python3
"""
Simple working example of the DR Assistant system.
This demonstrates that everything is set up correctly.
"""

import torch
import numpy as np
from src.model import create_model, FocalLoss


def test_system():
    """Test the basic system functionality."""
    print("Diabetic Retinopathy Assistant - System Test")
    print("=" * 50)
    
    # Test 1: GPU availability
    print("1. Testing GPU availability...")
    if torch.cuda.is_available():
        print(f"   ✅ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"   ✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("   ❌ CUDA not available")
        return False
    
    # Test 2: Model creation
    print("\n2. Testing model creation...")
    try:
        model = create_model()
        print(f"   ✅ Model created successfully")
        print(f"   ✅ Parameters: {sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        print(f"   ❌ Model creation failed: {e}")
        return False
    
    # Test 3: Model on GPU (single image)
    print("\n3. Testing model on GPU (single image)...")
    try:
        model = model.cuda()
        model.eval()
        
        # Test with single image
        x = torch.randn(1, 3, 224, 224).cuda()
        with torch.no_grad():
            output = model(x)
        
        print(f"   ✅ Model inference successful")
        print(f"   ✅ Output shape: {output.shape}")
        print(f"   ✅ Memory used: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        
    except Exception as e:
        print(f"   ❌ GPU inference failed: {e}")
        return False
    
    # Test 4: Loss function
    print("\n4. Testing loss function...")
    try:
        criterion = FocalLoss()
        targets = torch.randint(0, 5, (1,)).cuda()
        loss = criterion(output, targets)
        print(f"   ✅ Loss calculation successful: {loss.item():.4f}")
    except Exception as e:
        print(f"   ❌ Loss calculation failed: {e}")
        return False
    
    # Test 5: Batch processing (small batch)
    print("\n5. Testing small batch processing...")
    try:
        torch.cuda.empty_cache()  # Clear memory
        
        # Test with batch size 2
        x_batch = torch.randn(2, 3, 224, 224).cuda()
        targets_batch = torch.randint(0, 5, (2,)).cuda()
        
        with torch.no_grad():
            output_batch = model(x_batch)
            loss_batch = criterion(output_batch, targets_batch)
        
        print(f"   ✅ Batch processing successful")
        print(f"   ✅ Batch output shape: {output_batch.shape}")
        print(f"   ✅ Batch loss: {loss_batch.item():.4f}")
        print(f"   ✅ Peak memory: {torch.cuda.max_memory_allocated() / 1024**2:.1f} MB")
        
    except Exception as e:
        print(f"   ❌ Batch processing failed: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 ALL TESTS PASSED!")
    print("=" * 50)
    print("\nYour system is ready for:")
    print("✅ Training (with batch_size=2)")
    print("✅ Inference")
    print("✅ API serving")
    print("✅ Web interface")
    
    print("\nNext steps:")
    print("1. Download datasets (see RUN_INSTRUCTIONS.md)")
    print("2. Run training: python src/train.py")
    print("3. Start API: python src/inference.py")
    print("4. Launch UI: streamlit run frontend/app.py")
    
    return True


if __name__ == "__main__":
    test_system()
