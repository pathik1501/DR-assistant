"""
Streamlit frontend for Diabetic Retinopathy detection.
Provides interactive web interface for image upload and visualization.
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
import cv2


# Page configuration
st.set_page_config(
    page_title="DR Assistant",
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
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .prediction-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .grade-0 { color: #2e8b57; }
    .grade-1 { color: #ffa500; }
    .grade-2 { color: #ff8c00; }
    .grade-3 { color: #ff4500; }
    .grade-4 { color: #dc143c; }
</style>
""", unsafe_allow_html=True)


class DRFrontend:
    """Main frontend class."""
    
    def __init__(self):
        self.api_url = "http://localhost:8080"
        self.grade_descriptions = [
            "No Diabetic Retinopathy",
            "Mild Nonproliferative DR",
            "Moderate Nonproliferative DR",
            "Severe Nonproliferative DR",
            "Proliferative DR"
        ]
        self.grade_colors = ['#2e8b57', '#ffa500', '#ff8c00', '#ff4500', '#dc143c']
    
    def check_api_connection(self) -> bool:
        """Check if API is available."""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def predict_image(self, image_bytes: bytes, include_explanation: bool = True, include_hint: bool = True) -> dict:
        """Send prediction request to API."""
        files = {"file": ("image.jpg", image_bytes, "image/jpeg")}
        params = {
            "include_explanation": include_explanation,
            "include_hint": include_hint
        }
        
        response = requests.post(
            f"{self.api_url}/predict",
            files=files,
            params=params,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"API Error: {response.status_code} - {response.text}")
    
    def display_prediction_results(self, result: dict):
        """Display prediction results."""
        prediction = result['prediction']
        confidence = result['confidence']
        grade_description = result['grade_description']
        abstained = result.get('abstained', False)
        processing_time = result['processing_time']
        
        # Main prediction card
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.markdown(f"""
            <div class="prediction-card">
                <h2 class="grade-{prediction}">Grade {prediction}: {grade_description}</h2>
                <h3>Confidence: {confidence:.1%}</h3>
                <p>Processing Time: {processing_time:.2f}s</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Confidence gauge
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = confidence * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Confidence (%)"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': self.grade_colors[prediction]},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ))
            fig.update_layout(height=200)
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            # Grade distribution
            grade_counts = [0] * 5
            grade_counts[prediction] = 1
            
            fig = px.bar(
                x=list(range(5)),
                y=grade_counts,
                color=list(range(5)),
                color_discrete_sequence=self.grade_colors,
                labels={'x': 'DR Grade', 'y': 'Count'},
                title="Predicted Grade"
            )
            fig.update_layout(height=200, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        # Abstention warning
        if abstained:
            st.warning("⚠️ **Low Confidence Prediction**: This prediction has low confidence and may require specialist review.")
        
        return prediction, confidence
    
    def display_explanation(self, explanation: dict):
        """Display Grad-CAM explanation."""
        if not explanation:
            return
        
        st.subheader("🔍 Model Explanation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Grad-CAM Heatmap**")
            # Display heatmap (assuming it's stored as base64 or numpy array)
            if 'gradcam_heatmap' in explanation:
                heatmap = explanation['gradcam_heatmap']
                if isinstance(heatmap, list):
                    heatmap = np.array(heatmap)
                
                # Create heatmap visualization
                fig = px.imshow(
                    heatmap,
                    color_continuous_scale='jet',
                    title="Attention Heatmap"
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.write("**Overlay Visualization**")
            if 'gradcam_overlay' in explanation:
                overlay = explanation['gradcam_overlay']
                if isinstance(overlay, list):
                    overlay = np.array(overlay)
                
                # Display overlay
                fig = px.imshow(
                    overlay,
                    title="Original Image with Heatmap Overlay"
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
    
    def display_clinical_hint(self, hint_data: dict):
        """Display clinical hint."""
        if not hint_data:
            return
        
        st.subheader("💡 Clinical Recommendation")
        
        # Main hint
        st.info(f"**{hint_data['hint']}**")
        
        # Sources
        if 'sources' in hint_data and hint_data['sources']:
            with st.expander("📚 Evidence Sources"):
                for i, source in enumerate(hint_data['sources']):
                    st.write(f"**Source {i+1}**: {source['source']}")
                    st.write(f"*{source['content']}*")
                    st.write("---")
    
    def create_download_report(self, result: dict, image_bytes: bytes) -> str:
        """Create downloadable JSON report."""
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "prediction": result['prediction'],
            "confidence": result['confidence'],
            "grade_description": result['grade_description'],
            "processing_time": result['processing_time'],
            "abstained": result.get('abstained', False),
            "clinical_hint": result.get('clinical_hint'),
            "image_info": {
                "size_bytes": len(image_bytes),
                "format": "JPEG"
            }
        }
        
        return json.dumps(report, indent=2)


def main():
    """Main Streamlit app."""
    
    # Header
    st.markdown('<h1 class="main-header">👁️ Diabetic Retinopathy Assistant</h1>', unsafe_allow_html=True)
    
    # Initialize frontend
    frontend = DRFrontend()
    
    # Sidebar
    st.sidebar.title("Settings")
    
    # API connection check
    if not frontend.check_api_connection():
        st.error("❌ Cannot connect to API server. Please ensure the API is running on localhost:8080")
        st.stop()
    else:
        st.sidebar.success("✅ API Connected")
    
    # Options
    include_explanation = st.sidebar.checkbox("Include Explanation", value=True)
    include_hint = st.sidebar.checkbox("Include Clinical Hint", value=True)
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["🔬 Analysis", "📊 Statistics", "ℹ️ About"])
    
    with tab1:
        st.header("Upload Retinal Image")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose a retinal fundus image",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a retinal fundus image for DR analysis"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Convert to bytes
            image_bytes = uploaded_file.getvalue()
            
            # Predict button
            if st.button("🔍 Analyze Image", type="primary"):
                with st.spinner("Analyzing image..."):
                    try:
                        # Make prediction
                        result = frontend.predict_image(
                            image_bytes, 
                            include_explanation, 
                            include_hint
                        )
                        
                        # Display results
                        prediction, confidence = frontend.display_prediction_results(result)
                        
                        # Display explanation
                        if include_explanation and 'explanation' in result:
                            frontend.display_explanation(result['explanation'])
                        
                        # Display clinical hint
                        if include_hint and 'clinical_hint' in result:
                            frontend.display_clinical_hint(result['clinical_hint'])
                        
                        # Download report
                        report_json = frontend.create_download_report(result, image_bytes)
                        st.download_button(
                            label="📥 Download Report",
                            data=report_json,
                            file_name=f"dr_analysis_{int(time.time())}.json",
                            mime="application/json"
                        )
                        
                    except Exception as e:
                        st.error(f"Analysis failed: {str(e)}")
    
    with tab2:
        st.header("Model Statistics")
        
        # Get model info
        try:
            response = requests.get(f"{frontend.api_url}/model_info")
            if response.status_code == 200:
                model_info = response.json()
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Model Architecture", model_info['model_architecture'])
                
                with col2:
                    st.metric("Input Size", f"{model_info['input_size'][0]}x{model_info['input_size'][1]}")
                
                with col3:
                    st.metric("Confidence Threshold", f"{model_info['confidence_threshold']:.1%}")
                
                # Grade descriptions
                st.subheader("DR Grade Descriptions")
                grades_response = requests.get(f"{frontend.api_url}/grades")
                if grades_response.status_code == 200:
                    grades_data = grades_response.json()
                    
                    for grade_info in grades_data['grades']:
                        grade = grade_info['grade']
                        description = grade_info['description']
                        st.markdown(f"**Grade {grade}**: {description}")
        
        except Exception as e:
            st.error(f"Could not fetch model information: {e}")
    
    with tab3:
        st.header("About DR Assistant")
        
        st.markdown("""
        ### Overview
        This AI-powered system detects and grades Diabetic Retinopathy (DR) from retinal fundus images using:
        
        - **EfficientNet-B3** deep learning model
        - **Grad-CAM** for explainable AI
        - **Uncertainty estimation** with Monte Carlo Dropout
        - **Clinical hints** generated using RAG and GPT-4o-mini
        
        ### DR Grading Scale
        - **Grade 0**: No Diabetic Retinopathy
        - **Grade 1**: Mild Nonproliferative DR
        - **Grade 2**: Moderate Nonproliferative DR
        - **Grade 3**: Severe Nonproliferative DR
        - **Grade 4**: Proliferative DR
        
        ### Features
        - Real-time prediction with confidence scores
        - Visual explanation of model decisions
        - Evidence-based clinical recommendations
        - Uncertainty quantification and abstention
        - Comprehensive analysis reports
        
        ### Disclaimer
        This tool is for research and educational purposes only. 
        Always consult with qualified healthcare professionals for medical decisions.
        """)


if __name__ == "__main__":
    main()
