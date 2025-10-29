#!/usr/bin/env python3
"""
Quick test script to verify GPU functionality and model performance.
"""

import torch
import time
from src.model import create_model, FocalLoss
from src.data_processing import DataProcessor


def test_gpu_setup():
    """Test GPU setup and basic functionality."""
    print("🔍 Testing GPU Setup...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        return True
    else:
        print("❌ CUDA not available")
        return False


def test_model_gpu():
    """Test model on GPU."""
    print("\n🤖 Testing Model on GPU...")
    
    try:
        # Create model
        model = create_model()
        model = model.cuda()
        model.eval()
        
        # Test with small batch
        batch_size = 2
        x = torch.randn(batch_size, 3, 512, 512).cuda()
        
        # Time the inference
        start_time = time.time()
        with torch.no_grad():
            output = model(x)
        inference_time = time.time() - start_time
        
        print(f"✅ Model inference successful!")
        print(f"   Input shape: {x.shape}")
        print(f"   Output shape: {output.shape}")
        print(f"   Inference time: {inference_time:.3f}s")
        print(f"   Time per image: {inference_time/batch_size:.3f}s")
        print(f"   Input device: {x.device}")
        print(f"   Output device: {output.device}")
        
        # Test loss function
        criterion = FocalLoss()
        targets = torch.randint(0, 5, (batch_size,)).cuda()
        loss = criterion(output, targets)
        print(f"   Loss calculation: {loss.item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False


def test_memory_usage():
    """Test GPU memory usage."""
    print("\n💾 Testing GPU Memory Usage...")
    
    try:
        # Clear cache
        torch.cuda.empty_cache()
        
        # Get initial memory
        initial_memory = torch.cuda.memory_allocated() / 1024**2  # MB
        
        # Create model
        model = create_model()
        model = model.cuda()
        
        # Get memory after model loading
        model_memory = torch.cuda.memory_allocated() / 1024**2  # MB
        
        # Test inference
        x = torch.randn(1, 3, 512, 512).cuda()
        output = model(x)
        
        # Get peak memory
        peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
        
        print(f"✅ Memory usage test successful!")
        print(f"   Initial memory: {initial_memory:.1f} MB")
        print(f"   Model memory: {model_memory:.1f} MB")
        print(f"   Peak memory: {peak_memory:.1f} MB")
        print(f"   Available memory: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.max_memory_allocated()) / 1024**2:.1f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ Memory test failed: {e}")
        return False


def test_data_processing():
    """Test data processing pipeline."""
    print("\n📊 Testing Data Processing...")
    
    try:
        processor = DataProcessor()
        print("✅ Data processor created successfully!")
        
        # Test transforms
        train_transforms = processor.get_training_transforms()
        val_transforms = processor.get_validation_transforms()
        print("✅ Transforms created successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Data processing test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 Diabetic Retinopathy Assistant - GPU Test Suite")
    print("=" * 60)
    
    tests = [
        ("GPU Setup", test_gpu_setup),
        ("Model GPU", test_model_gpu),
        ("Memory Usage", test_memory_usage),
        ("Data Processing", test_data_processing)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(results)} tests")
    
    if passed == len(results):
        print("\n🎉 All tests passed! Your system is ready for training.")
        print("\nNext steps:")
        print("1. Download datasets (see RUN_INSTRUCTIONS.md)")
        print("2. Run training: python src/train.py")
        print("3. Start API: python src/inference.py")
        print("4. Launch UI: streamlit run frontend/app.py")
    else:
        print("\n⚠️ Some tests failed. Please check the errors above.")


if __name__ == "__main__":
    main()
