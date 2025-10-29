"""Test Grad-CAM directly to debug the issue."""
import torch
import numpy as np
import sys
sys.path.append('.')

from src.model import DRModel
from src.explainability import ExplainabilityPipeline
from PIL import Image
import torchvision.transforms as transforms

# Load model
model = DRModel(num_classes=5, pretrained=False)
checkpoint_path = "1/7d0928bb87954a739123ca35fa03cccf/checkpoints/dr-model-epoch=11-val_qwk=0.769.ckpt"
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# Load state dict
if 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
    # Remove 'model.' prefix
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('model.'):
            new_key = key[6:]
            cleaned_state_dict[new_key] = value
        elif not key.startswith('criterion') and not key.startswith('metrics'):
            cleaned_state_dict[key] = value
    model.load_state_dict(cleaned_state_dict, strict=False)
model.eval()

# Load test image
img_path = "data/eyepacs/augmented_resized_V2/train/0/0abf0c485f66-600.jpg"
img = Image.open(img_path).convert('RGB')

# Preprocess
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
img_tensor = transform(img).unsqueeze(0)

# Create explainability pipeline
target_layers = ["backbone.blocks.5.0", "backbone.blocks.6.0"]
pipeline = ExplainabilityPipeline(model, target_layers)

# Forward pass to get prediction
with torch.no_grad():
    output = model(img_tensor)
    prediction = output.argmax(dim=1).item()
    confidence = torch.softmax(output, dim=1).max().item()

print(f"Prediction: {prediction}, Confidence: {confidence}")

# Generate explanation
img_np = np.array(img.resize((224, 224)))
explanation = pipeline.explain_prediction(img_np, img_tensor, prediction, confidence)

# Check heatmap values
heatmap = explanation['gradcam_heatmap']
print(f"\nHeatmap shape: {heatmap.shape}")
print(f"Heatmap min: {heatmap.min()}, max: {heatmap.max()}")
print(f"Heatmap mean: {heatmap.mean()}")

# Check if hooks are working
print(f"\nChecking hooks...")
print(f"Number of hooks in GradCAM: {len(pipeline.gradcam.hooks)}")
print(f"Number of activations captured: {len(pipeline.gradcam.activations)}")
print(f"Number of gradients captured: {len(pipeline.gradcam.gradients)}")

if pipeline.gradcam.activations:
    for key in pipeline.gradcam.activations.keys():
        act = pipeline.gradcam.activations[key]
        print(f"Activation shape: {act.shape}, mean: {act.mean():.4f}")

if pipeline.gradcam.gradients:
    for key in pipeline.gradcam.gradients.keys():
        grad = pipeline.gradcam.gradients[key]
        print(f"Gradient shape: {grad.shape}, mean: {grad.mean():.4f}")
else:
    print("WARNING: No gradients captured!")

