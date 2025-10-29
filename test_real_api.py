"""Test the real API with the simplified response."""
import requests
import base64
import json

# Load test image
image_path = "data/eyepacs/augmented_resized_V2/train/0/0abf0c485f66-600.jpg"

with open(image_path, "rb") as f:
    image_bytes = f.read()
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')

# Make API call
print("Testing API response...\n")
response = requests.post(
    "http://localhost:8080/predict_base64",
    json={
        "image_base64": image_base64,
        "include_explanation": True,
        "include_hint": True
    }
)

if response.status_code == 200:
    result = response.json()
    print("[OK] API Response:")
    print(json.dumps(result, indent=2))
else:
    print(f"[ERROR] Error: {response.status_code}")
    print(response.text)

