#!/usr/bin/env python3
"""Quick test to verify the API is running."""

import requests
import time

def test_api():
    """Test if the API is running."""
    print("Testing API server...")
    
    # Wait a moment for server to start
    time.sleep(3)
    
    try:
        # Test health endpoint
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("\nServer is running!")
            print(f"Response: {response.json()}")
            print("\nAPI Documentation: http://localhost:8000/docs")
            return True
        else:
            print(f"Server responded with status: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("\nServer is not running or not responding.")
        print("Make sure you started the server with: python src/inference.py")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    test_api()
