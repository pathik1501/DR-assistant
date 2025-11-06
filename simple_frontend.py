"""
Simple DR Frontend - Minimal, easy to use interface.
"""
import os
import io
import base64
import streamlit as st
import requests
from PIL import Image

# Setup
st.set_page_config(page_title="DR Assistant", page_icon="👁️", layout="wide")

# Header
st.title("👁️ Diabetic Retinopathy Detection")
st.markdown("Upload a retinal image to get instant analysis")

# API connection check
# For Docker: use service name. For local: use localhost
api_url = os.getenv("API_URL", "http://localhost:8080")
try:
    requests.get(f"{api_url}/health", timeout=2)
    st.success("✅ Connected to API")
except:
    st.error("❌ API not running. Start it with: `python src/inference.py`")
    st.stop()

# Upload
uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    # Show image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=400)
    
    # Analyze button
    if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
        with st.spinner("Analyzing..."):
            try:
                # Convert to base64
                image_bytes = uploaded_file.getvalue()
                base64_img = base64.b64encode(image_bytes).decode('utf-8')
                
                # Call API
                response = requests.post(
                    f"{api_url}/predict_base64",
                    json={
                        "image_base64": base64_img,
                        "include_explanation": False,
                        "include_hint": True
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Display results
                    grade = result['prediction']
                    confidence = result['confidence']
                    description = result['grade_description']
                    
                    # Grade colors
                    colors = {
                        0: "🟢", 1: "🟡", 2: "🟠", 3: "🔴", 4: "🔴"
                    }
                    
                    st.markdown("---")
                    st.markdown(f"## {colors.get(grade, '📊')} Result: Grade {grade}")
                    st.markdown(f"**{description}**")
                    st.markdown(f"**Confidence:** {confidence:.1%}")
                    
                    # Clinical hint
                    if result.get('clinical_hint'):
                        st.markdown("---")
                        st.info(result['clinical_hint'])
                    
                else:
                    st.error(f"Error: {response.text}")
                    
            except Exception as e:
                st.error(f"Failed to analyze: {e}")

st.markdown("---")
st.caption("⚠️ For research purposes only. Consult a qualified ophthalmologist for medical advice.")

