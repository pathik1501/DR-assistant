"""
Complete Streamlit frontend for Diabetic Retinopathy detection.
Features:
- Image upload
- Classification display
- Grad-CAM heatmap visualization
- RAG model explanations
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
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="DR Assistant - Complete",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #7f8c8d;
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
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%) !important;
    }
    .grade-1 { 
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%) !important;
    }
    .grade-2 { 
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%) !important;
    }
    .grade-3 { 
        background: linear-gradient(135deg, #ff9a56 0%, #ff6a88 100%) !important;
    }
    .grade-4 { 
        background: linear-gradient(135deg, #ff0844 0%, #ffb199 100%) !important;
    }
    .heatmap-section {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 2px solid #e9ecef;
    }
    .hint-box {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #2196f3;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        font-size: 1.1rem;
        font-weight: 600;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #165a8a;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


class DRCompleteFrontend:
    """Complete frontend with all features."""
    
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
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
        payload = {
            "image_base64": image_base64,
            "include_explanation": include_explanation,
            "include_hint": include_hint
        }
        
        response = requests.post(
            f"{self.api_url}/predict_base64",
            json=payload,
            timeout=90  # Longer timeout for explanations
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"API Error: {response.status_code} - {response.text}")
    
    def display_classification(self, result: dict):
        """Display classification results prominently."""
        prediction = result['prediction']
        confidence = result['confidence']
        grade_description = result['grade_description']
        processing_time = result.get('processing_time', 0)
        
        icon = self.grade_icons[prediction]
        color = self.grade_colors[prediction]
        
        # Main prediction card
        st.markdown(f"""
        <div class="prediction-card grade-{prediction}">
            <div style="text-align: center;">
                <div style="font-size: 5rem; margin-bottom: 1rem;">{icon}</div>
                <h1 style="margin: 0; color: white; font-size: 3rem;">Grade {prediction}</h1>
                <h2 style="margin: 0.5rem 0; color: white; font-weight: 400; font-size: 1.5rem;">{grade_description}</h2>
                <div style="margin-top: 1.5rem;">
                    <span style="font-size: 1.8rem; color: white; font-weight: 600;">Confidence: {confidence:.1%}</span>
                </div>
                <div style="margin-top: 0.5rem;">
                    <span style="font-size: 1rem; color: rgba(255,255,255,0.9);">Processing Time: {processing_time:.2f}s</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Confidence gauge
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = confidence * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Model Confidence (%)"},
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
            fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)
    
    def display_heatmaps(self, explanation: dict):
        """Display Grad-CAM heatmaps and overlays."""
        if not explanation:
            st.warning("⚠️ No explanation data available")
            return
        
        st.markdown("### 🔍 Grad-CAM Visualization")
        st.markdown('<div class="heatmap-section">', unsafe_allow_html=True)
        
        # Check for heatmap images
        heatmap_keys = {
            'Grad-CAM Heatmap': 'gradcam_heatmap_base64',
            'Grad-CAM Overlay': 'gradcam_overlay_base64',
            'Grad-CAM++ Heatmap': 'gradcam_plus_heatmap_base64',
            'Grad-CAM++ Overlay': 'gradcam_plus_overlay_base64'
        }
        
        available_heatmaps = {k: v for k, v in heatmap_keys.items() if v in explanation}
        
        if available_heatmaps:
            # Display in grid
            cols = st.columns(2)
            
            idx = 0
            for title, key in available_heatmaps.items():
                col_idx = idx % 2
                with cols[col_idx]:
                    try:
                        # Decode base64 image
                        img_bytes = base64.b64decode(explanation[key])
                        img = Image.open(io.BytesIO(img_bytes))
                        
                        st.markdown(f"**{title}**")
                        st.image(img, use_container_width=True, caption=title)
                    except Exception as e:
                        st.error(f"Error displaying {title}: {e}")
                
                idx += 1
            
            # Additional info
            if 'prediction' in explanation:
                st.info(f"**Predicted Class**: {explanation.get('class_name', 'Unknown')} | **Confidence**: {explanation.get('confidence', 0):.2%}")
        else:
            # Check if there's an error or note
            if 'error' in explanation:
                st.error(f"❌ {explanation['error']}")
            elif 'note' in explanation:
                st.info(f"ℹ️ {explanation['note']}")
            else:
                st.warning("⚠️ Heatmap visualization not available in the response")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def display_rag_explanation(self, clinical_hint: str):
        """Display RAG model explanation."""
        if not clinical_hint:
            st.warning("⚠️ No clinical recommendation available")
            return
        
        st.markdown("### 💡 Clinical Recommendation (RAG)")
        st.markdown(f"""
        <div class="hint-box">
            <p style="font-size: 1.1rem; margin: 0; line-height: 1.8; color: #1976d2;">
                {clinical_hint}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.info("ℹ️ **Note**: This recommendation is generated by our RAG pipeline combining evidence-based guidelines with AI-powered analysis. Always consult with a qualified ophthalmologist for professional medical advice.")


def main():
    """Main Streamlit app."""
    
    # Header
    st.markdown('<h1 class="main-header">👁️ Diabetic Retinopathy Detection Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Complete AI-Powered Analysis: Classification • Grad-CAM Heatmaps • Clinical Recommendations</p>', unsafe_allow_html=True)
    
    # Initialize frontend
    frontend = DRCompleteFrontend()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # API connection check
        if not frontend.check_api_connection():
            st.error("❌ **API Not Connected**")
            st.info("Please start the API server:")
            st.code("python src/inference.py", language="bash")
            st.stop()
        else:
            st.success("✅ **API Connected**")
        
        st.markdown("---")
        
        # Options
        st.subheader("Analysis Options")
        include_explanation = st.checkbox(
            "Include Grad-CAM Explanation", 
            value=True,
            help="Generate heatmaps showing which regions the model focused on"
        )
        include_hint = st.checkbox(
            "Include Clinical Recommendation", 
            value=True,
            help="Get AI-generated clinical guidance from RAG pipeline"
        )
        
        st.markdown("---")
        
        # Info
        st.subheader("ℹ️ About")
        st.markdown("""
        This system analyzes retinal fundus images to detect and grade Diabetic Retinopathy (DR).
        
        **Features:**
        - 🎯 Classification (Grade 0-4)
        - 🔍 Grad-CAM Heatmaps
        - 💡 AI Clinical Recommendations
        
        **Model Performance:**
        - QWK: 0.785
        - Accuracy: 74.7%
        """)
        
        st.markdown("---")
        st.warning("⚠️ **For Research Only** - Not for clinical diagnosis")
    
    # Main content area
    st.markdown("---")
    
    # Upload section
    st.markdown("### 📤 Upload Retinal Fundus Image")
    uploaded_file = st.file_uploader(
        "Choose a retinal fundus image",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a retinal fundus image for complete DR analysis"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)
        
        with col2:
            st.markdown("#### Image Information")
            st.write(f"**Format**: {image.format}")
            st.write(f"**Size**: {image.size[0]} × {image.size[1]} pixels")
            st.write(f"**Mode**: {image.mode}")
        
        # Analyze button
        if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
            image_bytes = uploaded_file.getvalue()
            
            with st.spinner("🔬 Analyzing image... This may take 10-30 seconds for complete analysis with heatmaps."):
                try:
                    start_time = time.time()
                    result = frontend.predict_image(
                        image_bytes,
                        include_explanation=include_explanation,
                        include_hint=include_hint
                    )
                    
                    st.session_state['last_result'] = result
                    st.session_state['analysis_time'] = time.time() - start_time
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error analyzing image: {str(e)}")
                    st.info("Please check that the API server is running and try again.")
    
    # Display results
    if 'last_result' in st.session_state:
        result = st.session_state['last_result']
        
        st.markdown("---")
        st.markdown("## 📊 Analysis Results")
        
        # Classification
        frontend.display_classification(result)
        
        # Heatmaps
        if include_explanation and result.get('explanation'):
            st.markdown("---")
            frontend.display_heatmaps(result['explanation'])
        
        # RAG explanation
        if include_hint and result.get('clinical_hint'):
            st.markdown("---")
            frontend.display_rag_explanation(result['clinical_hint'])
        
        # Download report
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            report = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "prediction": result['prediction'],
                "grade": result['grade_description'],
                "confidence": result['confidence'],
                "clinical_hint": result.get('clinical_hint'),
                "processing_time": result.get('processing_time', 0),
                "explanation_available": result.get('explanation') is not None
            }
            
            st.download_button(
                label="📥 Download Report (JSON)",
                data=json.dumps(report, indent=2),
                file_name=f"dr_analysis_{int(time.time())}.json",
                mime="application/json",
                use_container_width=True
            )
        
        # Expandable detailed info
        with st.expander("📋 View Full API Response"):
            st.json(result)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #7f8c8d; padding: 2rem;">
        <p><strong>Diabetic Retinopathy Detection Assistant</strong></p>
        <p>Powered by EfficientNet-B0 | Grad-CAM | RAG Pipeline</p>
        <p style="font-size: 0.8rem;">⚠️ For research and educational purposes only. Not intended for clinical diagnosis.</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

