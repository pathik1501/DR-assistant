"""
Script to verify the model checkpoint is loaded correctly.
Run this to check if the model is using trained weights or random weights.
"""

import torch
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import DRModel

def verify_model_loading():
    """Verify model checkpoint is loaded correctly."""
    
    checkpoint_path = "1/38054df5c2da4cc6b648ff50fbd36590/checkpoints/dr-model-epoch=10-val_qwk=0.785.ckpt"
    
    if not os.path.exists(checkpoint_path):
        print(f"[ERROR] Checkpoint not found at: {checkpoint_path}")
        return False
    
        print(f"[OK] Checkpoint found: {checkpoint_path}")
    
    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f"[OK] Checkpoint loaded successfully")
        
        if 'state_dict' not in checkpoint:
            print("[ERROR] No 'state_dict' in checkpoint!")
            print(f"Checkpoint keys: {list(checkpoint.keys())}")
            return False
        
        state_dict = checkpoint['state_dict']
        print(f"[OK] State dict found with {len(state_dict)} keys")
        
        # Check for model keys
        model_keys = [k for k in state_dict.keys() if k.startswith('model.') or not k.startswith(('criterion', 'metrics'))]
        print(f"[OK] Found {len(model_keys)} model parameter keys")
        
        # Clean state dict
        cleaned_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('model.'):
                new_key = key[6:]  # Remove 'model.' prefix
                cleaned_state_dict[new_key] = value
            elif not key.startswith('criterion') and not key.startswith('metrics'):
                cleaned_state_dict[key] = value
        
        print(f"[OK] Cleaned state dict has {len(cleaned_state_dict)} keys")
        
        # Create model and load
        model = DRModel(num_classes=5, pretrained=False, dropout_rate=0.3)
        
        # Get model keys
        model_state_keys = set(model.state_dict().keys())
        checkpoint_keys = set(cleaned_state_dict.keys())
        
        matching_keys = model_state_keys & checkpoint_keys
        missing_keys = model_state_keys - checkpoint_keys
        unexpected_keys = checkpoint_keys - model_state_keys
        
        print(f"\nKey Matching Analysis:")
        print(f"  [OK] Matching keys: {len(matching_keys)}")
        print(f"  [WARNING] Missing keys: {len(missing_keys)}")
        print(f"  [WARNING] Unexpected keys: {len(unexpected_keys)}")
        
        if missing_keys:
            print(f"\n[WARNING] Missing keys (first 10):")
            for key in list(missing_keys)[:10]:
                print(f"    - {key}")
        
        # Try loading
        result = model.load_state_dict(cleaned_state_dict, strict=False)
        
        if len(missing_keys) == 0 and len(result.missing_keys) == 0:
            print(f"\n[OK] Perfect match! All keys loaded successfully!")
            return True
        elif len(matching_keys) > 100:  # Reasonable threshold
            print(f"\n[WARNING] Partial match: {len(matching_keys)} keys loaded")
            print(f"   Model may work but might not be fully loaded")
            return True
        else:
            print(f"\n[ERROR] Critical issue: Only {len(matching_keys)} keys matched!")
            print(f"   Model will not work correctly!")
            return False
            
    except Exception as e:
        print(f"[ERROR] Error loading checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Model Checkpoint Verification")
    print("=" * 60)
    print()
    
    success = verify_model_loading()
    
    print()
    print("=" * 60)
    if success:
        print("[OK] Model checkpoint appears to be loadable")
    else:
        print("[ERROR] Model checkpoint has issues!")
    print("=" * 60)

