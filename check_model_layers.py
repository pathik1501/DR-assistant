"""Check EfficientNet-B0 layer names for Grad-CAM."""
import torch
import timm
import sys
sys.path.append('.')

from src.model import DRModel

# Create model
model = DRModel(num_classes=5, pretrained=False)

# Print all layer names
print("\n=== All Layer Names ===")
for name, module in model.named_modules():
    print(name)

print("\n=== Block Layers (best for Grad-CAM) ===")
for name, module in model.named_modules():
    if 'block' in name.lower() and hasattr(module, 'weight'):
        print(f"{name}: {type(module).__name__}")

print("\n=== Backbone Block Layers ===")
# Check EfficientNet backbone structure
for name, module in model.named_modules():
    if 'blocks' in name:
        print(name)

