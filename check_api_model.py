"""Check loaded model structure in API."""
import sys
sys.path.append('.')

from src.inference import DRPredictionService

# Create service (loads model)
print("Loading model...")
service = DRPredictionService()

# Check explainability pipeline
pipeline = service.explainability_pipeline
print(f"\nGradCAM target layers: {pipeline.gradcam.target_layers}")
print(f"\nModel structure:")

# Check if layers exist
for name, module in service.model.named_modules():
    if name in pipeline.gradcam.target_layers:
        print(f"FOUND: {name} -> {type(module).__name__}")

