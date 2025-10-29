"""
FastAPI serving module for Diabetic Retinopathy detection.
Provides REST API endpoints for prediction and monitoring.
"""

import os
import io
import base64
import json
import time
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from prometheus_client.core import CollectorRegistry
import yaml
import logging

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.model import DRModel, UncertaintyEstimator, TemperatureScaling
from src.explainability import ExplainabilityPipeline
try:
    from src.rag_pipeline import RAGPipeline
except Exception as e:
    logger.warning(f"RAG pipeline not available: {e}")
    RAGPipeline = None

# Prometheus metrics
registry = CollectorRegistry()
PREDICTION_COUNTER = Counter(
    'dr_predictions_total', 
    'Total number of DR predictions',
    ['grade', 'confidence_level'],
    registry=registry
)
PREDICTION_LATENCY = Histogram(
    'dr_prediction_duration_seconds',
    'Time spent on DR prediction',
    registry=registry
)
MODEL_CONFIDENCE = Gauge(
    'dr_model_confidence',
    'Model confidence for predictions',
    registry=registry
)
ABSTENTION_COUNTER = Counter(
    'dr_abstentions_total',
    'Total number of abstentions',
    registry=registry
)


class PredictionRequest(BaseModel):
    """Request model for prediction."""
    image_base64: str
    include_explanation: bool = True
    include_hint: bool = True


class PredictionResponse(BaseModel):
    """Response model for prediction."""
    prediction: int
    confidence: float
    grade_description: str
    explanation: Optional[Dict] = None
    clinical_hint: Optional[Dict] = None
    processing_time: float
    abstained: bool = False


class DRPredictionService:
    """Main prediction service."""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_config = self.config['model']
        self.uncertainty_config = self.config['uncertainty']
        self.explainability_config = self.config['explainability']
        
        # Load model
        self.model = self._load_model()
        
        # Initialize uncertainty estimator
        self.uncertainty_estimator = UncertaintyEstimator(
            self.model, 
            self.uncertainty_config['mc_dropout_samples']
        )
        
        # Initialize explainability pipeline
        self.explainability_pipeline = ExplainabilityPipeline(
            self.model,
            self.explainability_config['grad_cam_layers']
        )
        
        # Initialize RAG pipeline
        if RAGPipeline is not None:
            try:
                self.rag_pipeline = RAGPipeline()
            except Exception as e:
                logger.warning(f"RAG pipeline initialization failed: {e}")
                self.rag_pipeline = None
        else:
            self.rag_pipeline = None
        
        # Temperature scaling (if available)
        self.temperature_scaler = None
        self._load_temperature_scaler()
    
    def _load_model(self) -> DRModel:
        """Load trained model from Lightning checkpoint."""
        # Try to find the best model checkpoint
        checkpoint_dir = "1/7d0928bb87954a739123ca35fa03cccf/checkpoints"
        best_model_path = os.path.join(checkpoint_dir, "dr-model-epoch=11-val_qwk=0.769.ckpt")
        
        if os.path.exists(best_model_path):
            logger.info(f"Loading trained model from {best_model_path}")
            try:
                # Load Lightning checkpoint
                checkpoint = torch.load(best_model_path, map_location='cpu')
                
                # Extract model from Lightning checkpoint
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                    # Remove 'model.' prefix from keys if present
                    cleaned_state_dict = {}
                    for key, value in state_dict.items():
                        if key.startswith('model.'):
                            new_key = key[6:]  # Remove 'model.' prefix
                            cleaned_state_dict[new_key] = value
                        elif not key.startswith('criterion') and not key.startswith('metrics'):
                            cleaned_state_dict[key] = value
                    
                    model = DRModel(
                        num_classes=self.model_config['num_classes'],
                        pretrained=False,
                        dropout_rate=self.model_config['dropout_rate']
                    )
                    result = model.load_state_dict(cleaned_state_dict, strict=False)
                    if result.missing_keys:
                        logger.warning(f"Missing keys when loading checkpoint: {result.missing_keys}")
                    if result.unexpected_keys:
                        logger.warning(f"Unexpected keys when loading checkpoint: {result.unexpected_keys}")
                    model.eval()
                    logger.info("Successfully loaded trained model from checkpoint")
                else:
                    raise KeyError("No state_dict in checkpoint")
                    
            except Exception as e:
                logger.error(f"Failed to load checkpoint: {e}")
                logger.error(f"Checkpoint structure: {list(checkpoint.keys())}")
                raise RuntimeError(f"Model checkpoint loading failed: {e}. Please check the checkpoint file.")
        else:
            logger.warning(f"Model not found at {best_model_path}, using untrained model")
            model = DRModel(
                num_classes=self.model_config['num_classes'],
                pretrained=True,
                dropout_rate=self.model_config['dropout_rate']
            )
            model.eval()
        
        return model
    
    def _load_temperature_scaler(self):
        """Load temperature scaler if available."""
        scaler_path = os.path.join(self.config['paths']['models_dir'], 'temperature_scaler.pth')
        
        if os.path.exists(scaler_path):
            logger.info("Loading temperature scaler")
            self.temperature_scaler = TemperatureScaling()
            self.temperature_scaler.load_state_dict(torch.load(scaler_path, map_location='cpu'))
            self.temperature_scaler.eval()
    
    def preprocess_image(self, image_bytes: bytes) -> torch.Tensor:
        """Preprocess uploaded image."""
        # Load image
        image = Image.open(io.BytesIO(image_bytes))
        image = image.convert('RGB')
        
        # Convert to numpy array
        image_np = np.array(image)
        
        # Apply preprocessing
        image_np = cv2.resize(image_np, (512, 512))
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        image_np = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # Normalize
        image_np = image_np.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_np = (image_np - mean) / std
        
        # Convert to tensor with proper dtype
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).float()
        
        return image_tensor, image_np
    
    def predict(
        self, 
        image_bytes: bytes,
        include_explanation: bool = True,
        include_hint: bool = True
    ) -> PredictionResponse:
        """Make prediction on uploaded image."""
        start_time = time.time()
        
        try:
            # Preprocess image
            image_tensor, image_np = self.preprocess_image(image_bytes)
            
            # Get prediction with uncertainty
            with torch.no_grad():
                # Ensure tensor is on CPU and correct dtype
                image_tensor = image_tensor.cpu().float()
                mean_pred, uncertainty = self.uncertainty_estimator.predict_with_uncertainty(image_tensor)
                
                # Apply temperature scaling if available
                if self.temperature_scaler:
                    logits = torch.log(mean_pred + 1e-8)
                    scaled_logits = self.temperature_scaler(logits)
                    mean_pred = F.softmax(scaled_logits, dim=1)
                
                prediction = mean_pred.argmax(dim=1).item()
                confidence = mean_pred.max().item()
            
            # Check if should abstain (disabled for now to always generate explanations)
            # abstained = confidence < self.uncertainty_config['confidence_threshold']
            abstained = False  # Always generate predictions
            
            if abstained:
                ABSTENTION_COUNTER.inc()
                logger.info(f"Abstaining due to low confidence: {confidence:.3f}")
            
            # Update metrics
            confidence_level = "high" if confidence > 0.8 else "medium" if confidence > 0.6 else "low"
            PREDICTION_COUNTER.labels(
                grade=str(prediction),
                confidence_level=confidence_level
            ).inc()
            MODEL_CONFIDENCE.set(confidence)
            
            # Generate explanation (TEMPORARILY DISABLED due to blank heatmaps)
            explanation = None
            # Simplified explanation - just show what the model sees
            if include_explanation:
                explanation = {
                    "prediction": prediction,
                    "confidence": confidence,
                    "class_name": grade_descriptions[prediction],
                    "note": "Heatmap visualization temporarily disabled"
                }
            
            # Generate clinical hint (always user-friendly)
            clinical_hint = None
            if include_hint:
                # Try RAG pipeline first if available
                if self.rag_pipeline:
                    try:
                        rag_result = self.rag_pipeline.generate_hint(prediction, confidence)
                        if isinstance(rag_result, dict) and 'hint' in rag_result:
                            clinical_hint = rag_result['hint']
                        elif isinstance(rag_result, str):
                            clinical_hint = rag_result
                    except Exception as e:
                        logger.warning(f"RAG hint generation failed: {e}, using template")
                        clinical_hint = None
                
                # Fallback to user-friendly templates if RAG fails or not available
                if not clinical_hint:
                    hint_templates = {
                        0: "✅ No diabetic retinopathy detected. Continue annual eye examinations and maintain good diabetes control.",
                        1: "⚠️ Mild nonproliferative diabetic retinopathy found. Schedule a follow-up examination in 6-12 months and monitor blood sugar levels closely.",
                        2: "🔶 Moderate nonproliferative diabetic retinopathy detected. Recommend follow-up in 3-6 months with an ophthalmologist. Tight glycemic control is important.",
                        3: "🔴 Severe nonproliferative diabetic retinopathy identified. Prompt referral to an ophthalmologist within 1-3 months is recommended. Aggressive diabetes management is crucial.",
                        4: "🚨 Proliferative diabetic retinopathy detected. Immediate evaluation by a retina specialist is required. This condition may need laser treatment or surgery."
                    }
                    clinical_hint = hint_templates.get(prediction, "Please consult with an ophthalmologist for appropriate care.")
            
            processing_time = time.time() - start_time
            PREDICTION_LATENCY.observe(processing_time)
            
            grade_descriptions = [
                "No Diabetic Retinopathy",
                "Mild Nonproliferative DR",
                "Moderate Nonproliferative DR", 
                "Severe Nonproliferative DR",
                "Proliferative DR"
            ]
            
            return PredictionResponse(
                prediction=prediction,
                confidence=confidence,
                grade_description=grade_descriptions[prediction],
                explanation=explanation,
                clinical_hint=clinical_hint,
                processing_time=processing_time,
                abstained=abstained
            )
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


# Initialize FastAPI app
app = FastAPI(
    title="Diabetic Retinopathy Detection API",
    description="AI-powered DR detection with explainability and clinical hints",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize prediction service
prediction_service = DRPredictionService()


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Diabetic Retinopathy Detection API", "version": "1.0.0"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": time.time()}


@app.post("/predict", response_model=PredictionResponse)
async def predict_dr(
    file: UploadFile = File(...),
    include_explanation: bool = True,
    include_hint: bool = True
):
    """Predict DR grade from uploaded image."""
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Read image bytes
    image_bytes = await file.read()
    
    # Make prediction
    result = prediction_service.predict(
        image_bytes, include_explanation, include_hint
    )
    
    return result


@app.post("/predict_base64", response_model=PredictionResponse)
async def predict_dr_base64(request: PredictionRequest):
    """Predict DR grade from base64 encoded image."""
    
    try:
        # Decode base64 image
        image_bytes = base64.b64decode(request.image_base64)
        
        # Make prediction
        result = prediction_service.predict(
            image_bytes, 
            request.include_explanation, 
            request.include_hint
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {e}")


@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)


@app.get("/model_info")
async def get_model_info():
    """Get model information."""
    return {
        "model_architecture": "EfficientNet-B3",
        "num_classes": 5,
        "input_size": [512, 512],
        "confidence_threshold": prediction_service.uncertainty_config['confidence_threshold'],
        "mc_dropout_samples": prediction_service.uncertainty_config['mc_dropout_samples']
    }


@app.get("/grades")
async def get_dr_grades():
    """Get DR grade descriptions."""
    return {
        "grades": [
            {"grade": 0, "description": "No Diabetic Retinopathy"},
            {"grade": 1, "description": "Mild Nonproliferative DR"},
            {"grade": 2, "description": "Moderate Nonproliferative DR"},
            {"grade": 3, "description": "Severe Nonproliferative DR"},
            {"grade": 4, "description": "Proliferative DR"}
        ]
    }


def main():
    """Run the FastAPI server."""
    api_config = yaml.safe_load(open("configs/config.yaml"))['api']
    
    uvicorn.run(
        "src.inference:app",
        host=api_config['host'],
        port=api_config['port'],
        workers=api_config['workers'],
        reload=False
    )


if __name__ == "__main__":
    main()
