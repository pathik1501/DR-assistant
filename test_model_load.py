#!/usr/bin/env python3
"""
Quick test to diagnose model loading issues
"""

import torch
import os

def test_model_loading():
    """Test if the model loads correctly."""
    print("Testing model checkpoint loading...")
    
    checkpoint_path = "1/7d0928bb87954a739123ca35fa03cccf/checkpoints/dr-model-epoch=11-val_qwk=0.769.ckpt"
    
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        return False
    
    try:
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        print("\nCheckpoint keys:")
        for key in checkpoint.keys():
            print(f"  - {key}")
        
        if 'state_dict' in checkpoint:
            print("\nState dict keys (first 10):")
            for i, key in enumerate(list(checkpoint['state_dict'].keys())[:10]):
                print(f"  {i+1}. {key}")
            
            # Count keys
            total_keys = len(checkpoint['state_dict'])
            model_keys = [k for k in checkpoint['state_dict'].keys() if k.startswith('model.')]
            print(f"\nTotal keys in state_dict: {total_keys}")
            print(f"Keys starting with 'model.': {len(model_keys)}")
            print(f"Other keys: {total_keys - len(model_keys)}")
        else:
            print("ERROR: 'state_dict' not found in checkpoint")
            return False
        
        print("\nModel loading structure:")
        print("Checkpoint structure is valid!")
        
        return True
        
    except Exception as e:
        print(f"ERROR loading checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_model_loading()
