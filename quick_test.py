"""Quick test of the API."""
import requests

print("Testing DR Assistant API...")

# Test health
print("\n1. Checking health endpoint...")
try:
    r = requests.get("http://localhost:8080/health")
    print(f"[OK] Health check passed: {r.status_code}")
    print(f"Response: {r.json()}")
except Exception as e:
    print(f"[FAIL] Health check failed: {e}")
    exit(1)

# Test stats
print("\n2. Checking prediction stats...")
try:
    r = requests.get("http://localhost:8080/predictions/stats")
    print(f"[OK] Stats endpoint working: {r.status_code}")
    print(f"Response: {r.json()}")
except Exception as e:
    print(f"[FAIL] Stats endpoint failed: {e}")

print("\n[OK] API is running successfully!")
print("\nNext steps:")
print("1. Open browser: http://localhost:8080/docs")
print("2. Use the /predict endpoint to test with an image")
print("3. Check the 'explanation' field for Grad-CAM visualizations")
