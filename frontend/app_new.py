"""
Modern Streamlit frontend for Diabetic Retinopathy detection.
Clean, user-friendly interface with better UX.
"""

import streamlit as st
import requests
import json
import base64
import io
import time
from PIL import Image
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="DR Assistant",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2C3E50;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #7F8C8D;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    .grade-0 { 
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .grade-1 { 
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .grade-2 { 
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .grade-3 { 
        background: linear-gradient(135deg, #ff9a56 0%, #ff6a88 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .grade-4 { 
        background: linear-gradient(135deg, #ff0844 0%, #ffb199 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .hint-box {
        background: #F8F9FA;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
    }
    .metric-box {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .success-badge {
        background: #27AE60;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    .warning-badge {
        background: #F39C12;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    .danger-badge {
        background: #E74C3C;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


class ModernDRFrontend:
    """Modern, user-friendly frontend."""
    
    def __init__(self):
        self.api_url = "http://localhost:8080"
        self.grade_descriptions = [
            "No Diabetic Retinopathy",
            "Mild Nonproliferative DR",
            "Moderate Nonproliferative DR",
            "Severe Nonproliferative DR",
            "Proliferative DR"
        ]
        self.grade_colors = {
            0: '#27AE60',  # Green
            1: '#F39C12',  # Orange
            2: '#E67E22',  # Dark Orange
            3: '#E74C3C',  # Red
            4: '#C0392B'   # Dark Red
        }
        self.grade_icons = {
            0: "✅",
            1: "⚠️",
            2: "🔶",
            3: "🔴",
            4: "🚨"
        }
    
    def check_api_connection(self) -> bool:
        """Check if API is available."""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def predict_image(self, image_bytes: bytes, include_explanation: bool = True, include_hint: bool = True) -> dict:
        """Send prediction request to API."""
        # Convert to base64
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
        payload = {
            "image_base64": image_base64,
            "include_explanation": include_explanation,
            "include_hint": include_hint
        }
        
        response = requests.post(
            f"{self.api_url}/predict_base64",
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"API Error: {response.status_code} - {response.text}")
    
    def display_main_result(self, result: dict):
        """Display main prediction result in a beautiful card."""
        prediction = result['prediction']
        confidence = result['confidence']
        grade_description = result['grade_description']
        processing_time = result.get('processing_time', 0)
        
        icon = self.grade_icons[prediction]
        color = self.grade_colors[prediction]
        
        # Main prediction card
        st.markdown(f"""
        <div class="prediction-card" style="background: linear-gradient(135deg, {color} 0%, {color}dd 100%);">
            <div style="text-align: center;">
                <div style="font-size: 4rem; margin-bottom: 1rem;">{icon}</div>
                <h1 style="margin: 0; color: white;">Grade {prediction}</h1>
                <h2 style="margin: 0.5rem 0; color: white; font-weight: 400;">{grade_description}</h2>
                <div style="margin-top: 1rem;">
                    <span style="font-size: 1.5rem; color: white;">Confidence: {confidence:.1%}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Confidence indicator
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = confidence * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Model Confidence"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': color},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "gray"},
                        {'range': [80, 100], 'color': "darkgray"}
                    ],
                    'threshold': {
                        'line': {'color': "white", 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ))
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)
        
        # Processing info
        st.caption(f"⚡ Processed in {processing_time:.2f} seconds")
    
    def display_clinical_hint(self, hint: str):
        """Display clinical hint in a user-friendly way."""
        if not hint:
            return
        
        st.markdown("### 💡 Clinical Recommendation")
        st.markdown(f"""
        <div class="hint-box">
            <p style="font-size: 1.1rem; margin: 0; line-height: 1.6;">{hint}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.info("ℹ️ **Note**: This is an AI-assisted recommendation. Always consult with a qualified ophthalmologist for professional medical advice and diagnosis.")
    
    def display_explanation(self, explanation: dict):
        """Display explanation if available."""
        if not explanation:
            return
        
        st.markdown("### 🔍 Model Explanation")
        
        if explanation.get('note'):
            st.info(explanation['note'])
        else:
            # If we have actual heatmaps (future implementation)
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Grad-CAM Heatmap**")
                st.info("Heatmap visualization will be displayed here when available.")
            with col2:
                st.write("**Overlay Visualization**")
                st.info("Overlay visualization will be displayed here when available.")


def main():
    """Main Streamlit app."""
    
    # Header
    st.markdown('<h1 class="main-header">👁️ Diabetic Retinopathy Detection Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Retinal Image Analysis for Early Detection</p>', unsafe_allow_html=True)
    
    # Initialize frontend
    frontend = ModernDRFrontend()
    
    # Check API connection
    if not frontend.check_api_connection():
        st.error("❌ **Cannot connect to API server**")
        st.info("Please ensure the API server is running on `localhost:8080`")
        st.code("python src/inference.py", language="bash")
        st.stop()
    else:
        st.success("✅ **API Connected** - Ready for analysis")
    
    st.markdown("---")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a retinal fundus image",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a retinal fundus image for DR analysis. Supported formats: JPG, JPEG, PNG"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Retinal Image", use_container_width=True)
            
            # Options
            st.markdown("### ⚙️ Analysis Options")
            include_explanation = st.checkbox("Include Model Explanation", value=False, help="Show Grad-CAM heatmaps (when available)")
            include_hint = st.checkbox("Include Clinical Recommendation", value=True, help="Get AI-generated clinical guidance")
            
            # Analyze button
            if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
                image_bytes = uploaded_file.getvalue()
                
                with st.spinner("Analyzing image... This may take a few seconds."):
                    try:
                        start_time = time.time()
                        result = frontend.predict_image(
                            image_bytes,
                            include_explanation=include_explanation,
                            include_hint=include_hint
                        )
                        
                        st.session_state['last_result'] = result
                        st.session_state['analysis_time'] = time.time() - start_time
                        
                    except Exception as e:
                        st.error(f"❌ Error analyzing image: {str(e)}")
                        st.info("Please check that the API server is running and try again.")
    
    with col2:
        st.markdown("### 📊 Analysis Results")
        
        if 'last_result' in st.session_state:
            result = st.session_state['last_result']
            
            # Main result
            frontend.display_main_result(result)
            
            # Clinical hint
            clinical_hint = result.get('clinical_hint')
            if clinical_hint:
                frontend.display_clinical_hint(clinical_hint)
            else:
                st.warning("⚠️ Clinical recommendation not available for this analysis.")
            
            # Explanation (if requested)
            if result.get('explanation'):
                frontend.display_explanation(result['explanation'])
            
            # Additional info
            with st.expander("📋 Detailed Information"):
                st.json(result)
            
            # Download report
            report = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "prediction": result['prediction'],
                "grade": result['grade_description'],
                "confidence": result['confidence'],
                "clinical_hint": clinical_hint,
                "processing_time": result.get('processing_time', 0)
            }
            
            st.download_button(
                label="📥 Download Report",
                data=json.dumps(report, indent=2),
                file_name=f"dr_analysis_{int(time.time())}.json",
                mime="application/json",
                use_container_width=True
            )
        else:
            st.info("👆 Upload an image and click 'Analyze Image' to see results here.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #7F8C8D; padding: 2rem;">
        <p><strong>Diabetic Retinopathy Detection Assistant</strong></p>
        <p>Powered by EfficientNet-B0 | QWK: 0.785</p>
        <p style="font-size: 0.8rem;">⚠️ For research and educational purposes only. Not intended for clinical diagnosis.</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

