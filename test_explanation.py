"""Test the API explanation generation."""
import requests
import base64
import os
import json

# Find a test image
test_image = "data/eyepacs/augmented_resized_V2/train/0/0abf0c485f66-600.jpg"

if not os.path.exists(test_image):
    print(f"[ERROR] Image not found: {test_image}")
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
    print(f"\nStatus Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print("\n=== RESPONSE ===")
        print(f"Prediction: {result.get('prediction')}")
        print(f"Confidence: {result.get('confidence')}")
        print(f"Description: {result.get('grade_description')}")
        print(f"Abstained: {result.get('abstained')}")
        
        # Check explanation
        explanation = result.get('explanation')
        if explanation:
            print("\n[OK] Explanation exists")
            print(f"Explanation keys: {list(explanation.keys())}")
            
            # Check for base64 images
            for key in explanation.keys():
                if '_base64' in key:
                    print(f"  - {key}: {len(explanation[key])} chars")
                else:
                    print(f"  - {key}: {explanation[key]}")
        else:
            print("\n[ERROR] No explanation in response")
        
        # Check hint
        hint = result.get('clinical_hint')
        if hint:
            print(f"\n[OK] Clinical hint: {hint}")
        else:
            print("\n[WARNING] No clinical hint")
        
        # Save full response for inspection
        with open('api_response.json', 'w') as f:
            json.dump(result, f, indent=2)
        print("\n[OK] Full response saved to api_response.json")
        
    else:
        print(f"\n[ERROR] Request failed: {response.status_code}")
        print(f"Response: {response.text}")
except Exception as e:
    print(f"\n[ERROR] Request exception: {e}")
    import traceback
    traceback.print_exc()

print("\n[OK] Test completed!")

