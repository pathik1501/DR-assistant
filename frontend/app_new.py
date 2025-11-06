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

# Custom CSS for compact UI
st.markdown("""
<style>
    /* Reduce overall padding and margins */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    
    /* Compact header */
    .main-header {
        font-size: 1.2rem;
        font-weight: 700;
        color: #2C3E50;
        text-align: center;
        margin-bottom: 0.25rem;
        padding: 0.25rem 0;
    }
    .sub-header {
        font-size: 0.75rem;
        color: #7F8C8D;
        text-align: center;
        margin-bottom: 0.5rem;
        padding: 0;
    }
    
    /* Compact status bar */
    .status-bar {
        padding: 0.25rem 0.5rem;
        margin: 0.25rem 0;
        font-size: 0.85rem;
    }
    
    /* Compact prediction card */
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0.75rem;
        border-radius: 8px;
        color: white;
        margin: 0.25rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* Smaller images */
    .compact-image {
        max-width: 250px;
        max-height: 200px;
        margin: 0 auto;
        display: block;
    }
    .stImage img {
        max-width: 250px !important;
        max-height: 200px !important;
        object-fit: contain;
    }
    [data-testid="stImage"] img {
        max-width: 250px !important;
        max-height: 200px !important;
    }
    
    /* Compact grade cards */
    .grade-0, .grade-1, .grade-2, .grade-3, .grade-4 { 
        color: white;
        padding: 0.75rem;
        border-radius: 8px;
        margin: 0.25rem 0;
    }
    .grade-0 { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }
    .grade-1 { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); }
    .grade-2 { background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); }
    .grade-3 { background: linear-gradient(135deg, #ff9a56 0%, #ff6a88 100%); }
    .grade-4 { background: linear-gradient(135deg, #ff0844 0%, #ffb199 100%); }
    
    /* Compact hint box */
    .hint-box {
        background: #F8F9FA;
        padding: 0.5rem;
        border-radius: 6px;
        border-left: 3px solid #667eea;
        margin: 0.25rem 0;
        color: #2C3E50 !important;
        font-size: 0.85rem;
    }
    
    /* Reduce section spacing */
    .section-spacing {
        margin-top: 0.25rem;
        margin-bottom: 0.25rem;
    }
    
    /* Compact metric box */
    .metric-box {
        background: white;
        padding: 0.5rem;
        border-radius: 6px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    /* Smaller badges */
    .success-badge, .warning-badge, .danger-badge {
        padding: 0.25rem 0.5rem;
        border-radius: 12px;
        font-weight: bold;
        display: inline-block;
        font-size: 0.8rem;
    }
    .success-badge { background: #27AE60; color: white; }
    .warning-badge { background: #F39C12; color: white; }
    .danger-badge { background: #E74C3C; color: white; }
    
    /* Reduce Streamlit default spacing */
    div[data-testid="stVerticalBlock"] > div {
        gap: 0.5rem;
    }
    
    /* Compact file uploader */
    .uploadedFile {
        padding: 0.25rem;
        margin: 0.25rem 0;
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
        
        # Compact prediction card
        st.markdown(f"""
        <div class="prediction-card" style="background: linear-gradient(135deg, {color} 0%, {color}dd 100%);">
            <div style="text-align: center;">
                <div style="font-size: 1.5rem; margin-bottom: 0.25rem;">{icon}</div>
                <h2 style="margin: 0; color: white; font-size: 1.1rem;">Grade {prediction}</h2>
                <p style="margin: 0.15rem 0; color: white; font-size: 0.85rem; opacity: 0.95;">{grade_description}</p>
                <div style="margin-top: 0.25rem;">
                    <span style="font-size: 0.95rem; color: white; font-weight: 600;">Confidence: {confidence:.1%}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Smaller confidence indicator
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = confidence * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Confidence", 'font': {'size': 12}},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': color},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "gray"},
                        {'range': [80, 100], 'color': "darkgray"}
                    ],
                    'threshold': {
                        'line': {'color': "white", 'width': 2},
                        'thickness': 0.6,
                        'value': 70
                    }
                }
            ))
            fig.update_layout(height=120, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, use_container_width=True)
    
    def display_clinical_hint(self, hint: str):
        """Display clinical hint in a user-friendly way."""
        if not hint:
            return
        
        st.markdown("### 💡 Clinical Recommendation")
        st.markdown(f"""
        <div class="hint-box">
            <p style="font-size: 0.95rem; margin: 0; line-height: 1.5; color: #2C3E50;">{hint}</p>
        </div>
        """, unsafe_allow_html=True)
        st.caption("ℹ️ AI-assisted recommendation. Consult an ophthalmologist for professional medical advice.")
    
    def display_scan_explanation(self, scan_explanation: str = None, scan_explanation_doctor: str = None):
        """Display scan explanations (patient and doctor versions)."""
        if not scan_explanation and not scan_explanation_doctor:
            return
        
        st.markdown("### 📝 Scan Analysis")
        
        # Use tabs for compact display
        if scan_explanation and scan_explanation_doctor:
            tab1, tab2 = st.tabs(["👤 Patient View", "👨‍⚕️ Clinical View"])
            with tab1:
                st.markdown(f"<p style='font-size: 0.9rem; line-height: 1.6;'>{scan_explanation}</p>", unsafe_allow_html=True)
            with tab2:
                st.markdown(f"<p style='font-size: 0.9rem; line-height: 1.6;'>{scan_explanation_doctor}</p>", unsafe_allow_html=True)
                st.caption("ℹ️ Technical medical terminology for healthcare professionals.")
        elif scan_explanation:
            st.markdown(f"<p style='font-size: 0.9rem; line-height: 1.6;'>{scan_explanation}</p>", unsafe_allow_html=True)
        elif scan_explanation_doctor:
            with st.expander("👨‍⚕️ Clinical Explanation", expanded=False):
                st.markdown(f"<p style='font-size: 0.9rem; line-height: 1.6;'>{scan_explanation_doctor}</p>", unsafe_allow_html=True)
                st.caption("ℹ️ Technical medical terminology for healthcare professionals.")
    
    def display_explanation(self, explanation: dict):
        """Display heatmaps if available - compact version in collapsible section."""
        if not explanation:
            return
        
        # Check if we have base64 images from API
        has_heatmaps = False
        
        # Collect all available heatmaps
        heatmaps = []
        if 'gradcam_heatmap_base64' in explanation:
            has_heatmaps = True
            try:
                img_bytes = base64.b64decode(explanation['gradcam_heatmap_base64'])
                heatmaps.append(("Grad-CAM", img_bytes, "Heatmap"))
            except:
                pass
                
        if 'gradcam_overlay_base64' in explanation:
            has_heatmaps = True
            try:
                img_bytes = base64.b64decode(explanation['gradcam_overlay_base64'])
                heatmaps.append(("Grad-CAM", img_bytes, "Overlay"))
            except:
                pass
                
        if 'gradcam_plus_heatmap_base64' in explanation:
            has_heatmaps = True
            try:
                img_bytes = base64.b64decode(explanation['gradcam_plus_heatmap_base64'])
                heatmaps.append(("Grad-CAM++", img_bytes, "Heatmap"))
            except:
                pass
                
        if 'gradcam_plus_overlay_base64' in explanation:
            has_heatmaps = True
            try:
                img_bytes = base64.b64decode(explanation['gradcam_plus_overlay_base64'])
                heatmaps.append(("Grad-CAM++", img_bytes, "Overlay"))
            except:
                pass
        
        # Display in compact grid (2 columns)
        if has_heatmaps and heatmaps:
            st.caption("Visualization of which areas of the image the model focused on for its prediction.")
            # Group by method
            cols = st.columns(2)
            for idx, (method, img_bytes, view_type) in enumerate(heatmaps):
                col_idx = idx % 2
                with cols[col_idx]:
                    try:
                        img = Image.open(io.BytesIO(img_bytes))
                        # Resize image to be much smaller
                        max_size = (200, 200)
                        img.thumbnail(max_size, Image.Resampling.LANCZOS)
                        st.image(img, caption=f"{method} {view_type}", use_container_width=False)
                    except Exception as e:
                        st.error(f"Error: {e}")
        
        # If no visualizations, show fallback
        if not has_heatmaps:
            if explanation.get('error'):
                st.error(f"❌ {explanation['error']}")
            elif explanation.get('note'):
                st.info(explanation['note'])
            else:
                st.info("ℹ️ Heatmap visualization not available")


def main():
    """Main Streamlit app."""
    
    # Compact header
    st.markdown('<h1 class="main-header">👁️ Diabetic Retinopathy Detection Assistant</h1>', unsafe_allow_html=True)
    
    # Initialize frontend and check API connection (silently)
    frontend = ModernDRFrontend()
    if not frontend.check_api_connection():
        st.error("❌ **Cannot connect to API server**")
        st.info("Please ensure the API server is running on `localhost:8080`")
        st.code("python src/inference.py", language="bash")
        st.stop()
    
    st.markdown("---")
    
    # Main content area - adjust column ratios for better space usage
    col1, col2 = st.columns([1.2, 1.3])
    
    with col1:
        st.markdown("### 📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a retinal fundus image",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a retinal fundus image for DR analysis. Supported formats: JPG, JPEG, PNG"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            # Resize image to be much smaller
            max_size = (250, 200)
            img_resized = image.copy()
            img_resized.thumbnail(max_size, Image.Resampling.LANCZOS)
            st.image(img_resized, caption="Uploaded Image", use_container_width=False)
            
            # Options - more compact
            st.markdown("### ⚙️ Options")
            include_explanation = st.checkbox("Show Heatmaps & Scan Analysis", value=True, help="Includes Grad-CAM visualizations and detailed scan explanations")
            include_hint = st.checkbox("Clinical Advice", value=True, help="AI-generated clinical recommendations")
            
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
            
            # Scan explanations (patient and doctor versions) - always show if available
            scan_explanation = result.get('scan_explanation')
            scan_explanation_doctor = result.get('scan_explanation_doctor')
            if scan_explanation or scan_explanation_doctor:
                st.markdown("---")  # Add separator
                frontend.display_scan_explanation(scan_explanation, scan_explanation_doctor)
            
            # Heatmaps (separate, collapsible section) - less important
            if result.get('explanation'):
                st.markdown("---")  # Add separator
                with st.expander("🔍 Model Attention Heatmaps (Optional)", expanded=False):
                    frontend.display_explanation(result['explanation'])
            
            # Download and details in compact row
            col_dl, col_det = st.columns([1, 1])
            with col_dl:
                report = {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "prediction": result['prediction'],
                    "grade": result['grade_description'],
                    "confidence": result['confidence'],
                    "clinical_hint": clinical_hint,
                    "scan_explanation": result.get('scan_explanation'),
                    "scan_explanation_doctor": result.get('scan_explanation_doctor'),
                    "processing_time": result.get('processing_time', 0)
                }
                st.download_button(
                    label="📥 Download Report",
                    data=json.dumps(report, indent=2),
                    file_name=f"dr_analysis_{int(time.time())}.json",
                    mime="application/json",
                    use_container_width=True
                )
            with col_det:
                with st.expander("📋 Full Details"):
                    st.json(result)
        else:
            st.info("👆 Upload an image and click 'Analyze Image' to see results here.")
    
    # Compact footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #7F8C8D; padding: 0.5rem;">
        <p style="font-size: 0.75rem; margin: 0.1rem 0;"><strong>DR Assistant</strong> | EfficientNet-B0 | ⚠️ Research use only</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()


