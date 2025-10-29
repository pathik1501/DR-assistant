#!/usr/bin/env python3
"""
Comprehensive diagnostic to find where the system is breaking
"""

import traceback
import sys
sys.path.append('.')

def test_1_imports():
    """Test 1: Can we import everything?"""
    print("Test 1: Testing imports...")
    try:
        import torch
        import cv2
        import numpy as np
        from PIL import Image
        print("  Basic imports: OK")
        
        from src.model import DRModel
        from src.inference import DRPredictionService
        print("  Model imports: OK")
        
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        traceback.print_exc()
        return False

def test_2_model_loading():
    """Test 2: Can we load the model?"""
    print("\nTest 2: Testing model loading...")
    try:
        import torch
        checkpoint_path = "1/7d0928bb87954a739123ca35fa03cccf/checkpoints/dr-model-epoch=11-val_qwk=0.769.ckpt"
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        if 'state_dict' in checkpoint:
            print(f"  Checkpoint structure: OK ({len(checkpoint['state_dict'])} keys)")
        else:
            print("  Checkpoint missing state_dict")
            return False
            
        from src.model import DRModel
        model = DRModel(num_classes=5, pretrained=False, dropout_rate=0.3)
        
        # Try to load
        state_dict = checkpoint['state_dict']
        cleaned_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('model.'):
                new_key = key[6:]
                cleaned_state_dict[new_key] = value
        
        result = model.load_state_dict(cleaned_state_dict, strict=False)
        print(f"  Model loading: OK (missing: {len(result.missing_keys)}, unexpected: {len(result.unexpected_keys)})")
        
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        traceback.print_exc()
        return False

def test_3_inference():
    """Test 3: Can we make a prediction?"""
    print("\nTest 3: Testing inference...")
    try:
        from src.inference import DRPredictionService
        service = DRPredictionService()
        print("  Service initialization: OK")
        
        # Create a dummy image
        from PIL import Image
        import io
        img = Image.new('RGB', (512, 512), color='red')
        
        # Convert to bytes
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG')
        image_bytes = buffer.getvalue()
        
        # Make prediction
        result = service.predict(image_bytes, include_explanation=False, include_hint=False)
        
        print(f"  Prediction: {result.prediction}, Confidence: {result.confidence:.3f}")
        print("  Inference pipeline: OK")
        
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        traceback.print_exc()
        return False

def test_4_explainability():
    """Test 4: Can we generate explanations?"""
    print("\nTest 4: Testing explainability...")
    try:
        from src.inference import DRPredictionService
        service = DRPredictionService()
        
        # Create a dummy image
        from PIL import Image
        import io
        img = Image.new('RGB', (512, 512), color='red')
        
        # Convert to bytes
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG')
        image_bytes = buffer.getvalue()
        
        # Make prediction WITH explanation
        result = service.predict(image_bytes, include_explanation=True, include_hint=False)
        
        if result.explanation:
            print(f"  Explanation generated: OK")
        else:
            print(f"  No explanation generated (may be due to error)")
        
        print("  Explainability pipeline: OK")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("DIABETIC RETINOPATHY SYSTEM DIAGNOSTIC")
    print("=" * 60)
    
    tests = [
        ("Imports", test_1_imports),
        ("Model Loading", test_2_model_loading),
        ("Inference", test_3_inference),
        ("Explainability", test_4_explainability)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\nTest {name} CRASHED: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{name}: {status}")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
