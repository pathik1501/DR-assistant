"""
Quick test script to verify scan explanation generation is working.
Run this after restarting the API server.
"""

import requests
import base64
import json

# Test image - you can use any retinal image
# For now, we'll just test the API endpoint structure

API_URL = "http://localhost:8080/predict"

def test_scan_explanation():
    """Test if scan explanations are in the API response."""
    print("Testing scan explanation generation...")
    print("=" * 60)
    
    # You'll need to provide an actual image file
    # For now, just check the endpoint structure
    print("\nTo test:")
    print("1. Make sure the API server is running")
    print("2. Upload an image via the frontend or API")
    print("3. Check the response for 'scan_explanation' and 'scan_explanation_doctor' fields")
    print("\nExpected response structure:")
    print(json.dumps({
        "prediction": 2,
        "confidence": 0.85,
        "grade_description": "Moderate Nonproliferative DR",
        "clinical_hint": "...",
        "scan_explanation": "Patient-friendly explanation here...",
        "scan_explanation_doctor": "Detailed clinical explanation here...",
        "explanation": {...},
        "processing_time": 2.5
    }, indent=2))
    
    print("\n" + "=" * 60)
    print("Check the API logs for:")
    print("- 'Generating scan explanations for prediction X'")
    print("- 'Generating patient-friendly scan explanation'")
    print("- 'Generating detailed doctor scan explanation'")
    print("- 'Patient scan explanation generated successfully'")
    print("- 'Doctor scan explanation generated successfully'")

if __name__ == "__main__":
    test_scan_explanation()

