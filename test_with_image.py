"""Test the API with an actual image."""
import requests
import base64
import os

# Check if we have a test image
test_images = [
    "data/aptos2019/aptos2019-blindness-detection/train_images/0_ffac9055b5e8.png",
    "data/aptos2019/aptos2019-blindness-detection/train_images/0_1c623c3d96.png",
    "data/eyepacs/augmented_resized_V2/train/0/00000_left.jpg",
    "data/eyepacs/augmented_resized_V2/train/1/00001_left.jpg",
]

test_image = None
for path in test_images:
    if os.path.exists(path):
        test_image = path
        break

if not test_image:
    print("[ERROR] No test image found")
    print("Please ensure you have downloaded the datasets")
    exit(1)

print(f"[OK] Testing with image: {test_image}")

# Read and encode image
with open(test_image, "rb") as f:
    image_bytes = f.read()
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')

# Make prediction
print("\nMaking prediction...")
url = "http://localhost:8080/predict_base64"
payload = {
    "image_base64": image_base64,
    "include_explanation": True,
    "include_hint": True
}

try:
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        print("[OK] Prediction successful!")
        result = response.json()
        print(f"\nPrediction: Grade {result['prediction']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Description: {result['grade_description']}")
        print(f"Processing time: {result['processing_time']:.2f}s")
        if result.get('explanation'):
            print(f"\n[OK] Explanation generated with Grad-CAM")
            print(f"  - Explanation keys: {list(result['explanation'].keys())}")
        else:
            print("\n[WARNING] No explanation generated")
        if result.get('clinical_hint'):
            print(f"\n[OK] Clinical hint: {result['clinical_hint']}")
        else:
            print("\n[WARNING] No clinical hint")
    else:
        print(f"[ERROR] Prediction failed: {response.status_code}")
        print(f"Response: {response.text}")
except Exception as e:
    print(f"[ERROR] Request failed: {e}")

print("\n[OK] Test completed!")

