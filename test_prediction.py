import requests
import base64
from pathlib import Path

# Create a dummy test image
from PIL import Image
import numpy as np

# Create a dummy 224x224 RGB image
test_image = Image.new('RGB', (224, 224), color='green')
test_image.save('test_fundus.jpg')

# Convert to base64
with open('test_fundus.jpg', 'rb') as f:
    img_data = base64.b64encode(f.read()).decode()

print("Testing /predict endpoint...")
# Use multipart/form-data
with open('test_fundus.jpg', 'rb') as f:
    files = {'file': ('test_fundus.jpg', f, 'image/jpeg')}
    response = requests.post(
        "http://localhost:8080/predict?include_explanation=true&include_hint=true",
        files=files
    )
print(f"Status: {response.status_code}")
print(f"Response: {response.json()}")
