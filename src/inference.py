"""
FastAPI serving module for Diabetic Retinopathy detection.
Provides REST API endpoints for prediction and monitoring.
"""

import os
# Fix OMP error: Allow duplicate OpenMP libraries (workaround for PyTorch/NumPy conflicts)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import io
import base64
import json
import time
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Load .env file from project root
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(env_path)
    else:
        # Try current directory
        load_dotenv()
except ImportError:
    # dotenv not installed, skip loading .env file
    pass

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
import uvicorn
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from prometheus_client.core import CollectorRegistry
import yaml

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Log if .env file was loaded
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        logger.info(f"Loaded .env file from {env_path}")
    else:
        logger.info(f".env file not found at {env_path}, using environment variables")
except ImportError:
    logger.info("python-dotenv not installed, .env file will not be loaded")
except Exception as e:
    logger.warning(f"Could not check .env file: {e}")

from src.model import DRModel, UncertaintyEstimator, TemperatureScaling
from src.explainability import ExplainabilityPipeline
try:
    from src.rag_pipeline import RAGPipeline
except Exception as e:
    logger.warning(f"RAG pipeline not available: {e}")
    RAGPipeline = None

# Import albumentations for TTA
try:
    import albumentations as A
except ImportError:
    logger.warning("Albumentations not available for TTA")
    A = None

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
    clinical_hint: Optional[str] = None  # Changed from Dict to str for RAG hints
    scan_explanation: Optional[str] = None  # Patient-friendly explanation of what the model sees
    scan_explanation_doctor: Optional[str] = None  # Detailed clinical explanation for doctors
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
        
        # Initialize RAG pipeline (optional - system can run without it)
        if RAGPipeline is not None:
            try:
                logger.info("Initializing RAG pipeline...")
                self.rag_pipeline = RAGPipeline()
                logger.info("RAG pipeline initialized successfully")
            except ValueError as e:
                # Quota/billing errors - provide helpful message
                error_msg = str(e)
                if "quota" in error_msg.lower() or "insufficient_quota" in error_msg.lower() or "429" in error_msg:
                    logger.warning("=" * 80)
                    logger.warning("RAG pipeline initialization failed: OpenAI API quota issue")
                    logger.warning("=" * 80)
                    logger.warning("The server will start without RAG features (scan explanations will be disabled).")
                    logger.warning("")
                    logger.warning("To enable RAG features, please:")
                    logger.warning("1. Go to https://platform.openai.com/account/billing")
                    logger.warning("2. Verify your payment method is active")
                    logger.warning("3. Add credits to your account (not just a payment method)")
                    logger.warning("4. Wait 5-10 minutes for quota to propagate")
                    logger.warning("5. Restart the server")
                    logger.warning("")
                    logger.warning("Note: Adding a payment method alone may not add quota.")
                    logger.warning("You may need to explicitly add credits or set up usage-based billing.")
                    logger.warning("=" * 80)
                else:
                    logger.warning(f"RAG pipeline initialization failed: {e}")
                self.rag_pipeline = None
            except Exception as e:
                logger.warning(f"RAG pipeline initialization failed: {e}")
                import traceback
                logger.error(f"RAG pipeline initialization traceback: {traceback.format_exc()}")
                self.rag_pipeline = None
        else:
            logger.warning("RAGPipeline class is not available (import failed)")
            self.rag_pipeline = None
        
        # Temperature scaling (if available)
        self.temperature_scaler = None
        self._load_temperature_scaler()
    
    def _load_model(self) -> DRModel:
        """Load trained model from Lightning checkpoint."""
        # Try to find the best model checkpoint
        checkpoint_dir = "1/7b2108ed09bf401fa06ff1b7d8c1e949/checkpoints"
        best_model_path = os.path.join(checkpoint_dir, "dr-model-epoch=60-val_qwk=0.853.ckpt")
        
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
    
    def preprocess_image(self, image_bytes: bytes) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
        """Preprocess uploaded image.
        Returns:
            image_tensor: Normalized tensor for model input
            image_np_normalized: Normalized numpy array (for denormalization)
            original_image: Original unnormalized image (for Grad-CAM overlay)
        """
        # Load image
        image = Image.open(io.BytesIO(image_bytes))
        image = image.convert('RGB')
        
        # Convert to numpy array
        image_np = np.array(image)
        
        # Apply preprocessing - match training exactly
        # Training uses [224, 224] from config
        image_np = cv2.resize(image_np, (224, 224))
        
        # IMPORTANT: Do NOT apply CLAHE here - training doesn't use it
        # The Albumentations transform only does resize + normalize
        
        # Store original image before normalization (for Grad-CAM overlay)
        original_image = image_np.copy()
        
        # Normalize - match training exactly
        image_np_normalized = image_np.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_np_normalized = (image_np_normalized - mean) / std
        
        # Convert to tensor with proper dtype
        image_tensor = torch.from_numpy(image_np_normalized).permute(2, 0, 1).unsqueeze(0).float()
        
        return image_tensor, image_np_normalized, original_image
    
    def _denormalize_image(self, image_np: np.ndarray) -> np.ndarray:
        """Convert normalized image back to original scale for visualization."""
        # Reverse normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_np = (image_np * std) + mean
        
        # Clip to valid range and convert to uint8
        image_np = np.clip(image_np, 0, 1) * 255.0
        image_np = image_np.astype(np.uint8)
        
        return image_np
    
    def predict_with_tta(self, image_tensor: torch.Tensor, num_augmentations: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict with Test-Time Augmentation.
        
        Args:
            image_tensor: Preprocessed image tensor
            num_augmentations: Number of augmentations to apply
            
        Returns:
            mean_pred: Averaged prediction probabilities
            uncertainty: Prediction uncertainty
        """
        if A is None:
            logger.warning("Albumentations not available, falling back to standard prediction")
            return self.uncertainty_estimator.predict_with_uncertainty(image_tensor)
        
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            # Original prediction
            mean_pred, _ = self.uncertainty_estimator.predict_with_uncertainty(image_tensor)
            predictions.append(mean_pred.squeeze(0))
            
            # Define TTA transforms
            tta_transforms = [
                A.HorizontalFlip(p=1.0),
                A.VerticalFlip(p=1.0),
                A.RandomRotate90(p=1.0),
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0),
                A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=10, p=1.0),
            ]
            
            # Generate augmented predictions
            for i in range(min(num_augmentations, len(tta_transforms))):
                transform = tta_transforms[i]
                
                # Convert tensor to numpy for augmentation
                img_np = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
                
                # Denormalize for augmentation
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img_np = (img_np * std) + mean
                img_np = np.clip(img_np, 0, 1) * 255.0
                img_np = img_np.astype(np.uint8)
                
                # Apply augmentation
                aug_img = transform(image=img_np)['image']
                
                # Normalize again
                aug_img = aug_img.astype(np.float32) / 255.0
                aug_img = (aug_img - mean) / std
                
                # Convert back to tensor
                aug_tensor = torch.from_numpy(aug_img).permute(2, 0, 1).unsqueeze(0).float()
                
                # Predict
                aug_pred, _ = self.uncertainty_estimator.predict_with_uncertainty(aug_tensor)
                predictions.append(aug_pred.squeeze(0))
        
        # Average all predictions
        stacked_preds = torch.stack(predictions)
        mean_prediction = stacked_preds.mean(dim=0, keepdim=True)
        
        # Calculate uncertainty as variance
        uncertainty = stacked_preds.var(dim=0).sum()
        
        return mean_prediction, uncertainty
    
    def predict(
        self, 
        image_bytes: bytes,
        include_explanation: bool = True,
        include_hint: bool = True,
        use_tta: bool = True
    ) -> PredictionResponse:
        """Make prediction on uploaded image.
        
        Args:
            image_bytes: Image bytes
            include_explanation: Whether to include Grad-CAM explanation
            include_hint: Whether to include clinical hints
            use_tta: Whether to use Test-Time Augmentation (default: True)
        """
        start_time = time.time()
        
        try:
            # Preprocess image
            image_tensor, image_np_normalized, original_image = self.preprocess_image(image_bytes)
            
            # Get prediction with uncertainty
            with torch.no_grad():
                # Ensure tensor is on CPU and correct dtype
                image_tensor = image_tensor.cpu().float()
                
                # Use TTA if enabled, otherwise standard prediction
                if use_tta:
                    mean_pred, uncertainty = self.predict_with_tta(image_tensor, num_augmentations=5)
                else:
                    mean_pred, uncertainty = self.uncertainty_estimator.predict_with_uncertainty(image_tensor)
                
                # Apply temperature scaling if available
                # NOTE: Temperature scaling requires raw logits, but we're using MC dropout
                # which returns probabilities. If temperature scaling is needed in the future,
                # we need to modify UncertaintyEstimator to also return logits.
                if self.temperature_scaler:
                    logger.warning("Temperature scaler loaded but not applied - requires logits, not probabilities")
                
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
            
            # Generate explanation with Grad-CAM heatmaps
            explanation = None
            heatmap_for_scan_explanation = None  # Store heatmap for scan explanation
            if include_explanation and not abstained:
                try:
                    logger.info(f"Generating Grad-CAM explanation for prediction {prediction} with confidence {confidence:.3f}")
                    # Ensure tensor is correct dtype for Grad-CAM
                    image_tensor_for_cam = image_tensor.float()
                    
                    # Use original image (already unnormalized) for overlay
                    # Generate explanation with heatmaps
                    explanation_raw = self.explainability_pipeline.explain_prediction(
                        original_image, image_tensor_for_cam, prediction, confidence
                    )
                    
                    logger.info(f"Explanation generated: {list(explanation_raw.keys())}")
                    
                    # Store heatmap for scan explanation generation (before base64 conversion)
                    if 'gradcam_heatmap' in explanation_raw:
                        heatmap_for_scan_explanation = explanation_raw['gradcam_heatmap']
                    
                    # Convert numpy arrays and images to base64 for JSON serialization
                    explanation = {}
                    for key, value in explanation_raw.items():
                        if isinstance(value, np.ndarray):
                            # Convert image arrays to base64
                            if len(value.shape) == 3 and value.shape[2] == 3:  # RGB image (overlay)
                                # Ensure uint8
                                img = value.astype(np.uint8)
                                img_pil = Image.fromarray(img)
                                buffer = io.BytesIO()
                                img_pil.save(buffer, format='PNG')
                                img_str = base64.b64encode(buffer.getvalue()).decode()
                                explanation[key + '_base64'] = img_str
                            elif len(value.shape) == 2:  # 2D heatmap
                                # Convert heatmap to image
                                heatmap_normalized = (value - value.min()) / (value.max() - value.min() + 1e-8)
                                heatmap_uint8 = (heatmap_normalized * 255).astype(np.uint8)
                                heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
                                heatmap_pil = Image.fromarray(heatmap_colored)
                                buffer = io.BytesIO()
                                heatmap_pil.save(buffer, format='PNG')
                                heatmap_str = base64.b64encode(buffer.getvalue()).decode()
                                explanation[key + '_base64'] = heatmap_str
                        elif isinstance(value, (int, float, str, bool, type(None))):
                            explanation[key] = value
                        elif isinstance(value, list):
                            explanation[key] = value
                        else:
                            explanation[key] = str(value)
                    
                    logger.info(f"Explanation serialized successfully")
                except Exception as e:
                    logger.warning(f"Explanation generation failed: {e}")
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    # Fallback explanation
                    explanation = {
                        "prediction": prediction,
                        "confidence": confidence,
                        "class_name": grade_descriptions[prediction],
                        "error": f"Explanation generation failed: {str(e)}"
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
            
            # Generate detailed scan explanations (what the model sees)
            # Patient version (simpler, accessible)
            scan_explanation = None
            # Doctor version (detailed, technical)
            scan_explanation_doctor = None
            
            # Check if RAG pipeline is available
            if include_explanation and not abstained:
                if self.rag_pipeline is None:
                    logger.warning("RAG pipeline is not initialized. Scan explanations will not be generated.")
                    logger.warning("To enable scan explanations, ensure OPENAI_API_KEY is set and RAG pipeline initializes correctly.")
                elif self.rag_pipeline:
                    try:
                        logger.info(f"Generating scan explanations for prediction {prediction}")
                        
                        # Get image shape from original_image
                        image_shape = (original_image.shape[0], original_image.shape[1])
                        
                        # Use stored heatmap (before base64 conversion)
                        heatmap = heatmap_for_scan_explanation
                        
                        # Generate patient-friendly explanation
                        try:
                            logger.info("Generating patient-friendly scan explanation")
                            logger.info(f"Parameters: dr_grade={prediction}, confidence={confidence:.3f}, heatmap={heatmap is not None}, image_shape={image_shape}")
                            scan_result_patient = self.rag_pipeline.generate_scan_explanation(
                                dr_grade=prediction,
                                confidence=confidence,
                                heatmap=heatmap,
                                image_shape=image_shape,
                                for_patient=True
                            )
                            
                            logger.info(f"Scan result type: {type(scan_result_patient)}")
                            logger.info(f"Scan result keys: {scan_result_patient.keys() if isinstance(scan_result_patient, dict) else 'Not a dict'}")
                            
                            if isinstance(scan_result_patient, dict) and 'explanation' in scan_result_patient:
                                scan_explanation = scan_result_patient['explanation']
                                logger.info(f"Patient explanation extracted: {scan_explanation[:100] if scan_explanation else 'None'}...")
                            elif isinstance(scan_result_patient, str):
                                scan_explanation = scan_result_patient
                                logger.info(f"Patient explanation (string): {scan_explanation[:100] if scan_explanation else 'None'}...")
                            else:
                                logger.warning(f"Unexpected scan result format: {type(scan_result_patient)}")
                                scan_explanation = None
                            
                            logger.info("Patient scan explanation generated successfully")
                        except Exception as e:
                            logger.warning(f"Patient scan explanation generation failed: {e}")
                            import traceback
                            logger.error(f"Patient scan explanation traceback: {traceback.format_exc()}")
                            scan_explanation = None
                        
                        # Generate detailed doctor explanation
                        try:
                            logger.info("Generating detailed doctor scan explanation")
                            scan_result_doctor = self.rag_pipeline.generate_scan_explanation(
                                dr_grade=prediction,
                                confidence=confidence,
                                heatmap=heatmap,
                                image_shape=image_shape,
                                for_patient=False
                            )
                            
                            if isinstance(scan_result_doctor, dict) and 'explanation' in scan_result_doctor:
                                scan_explanation_doctor = scan_result_doctor['explanation']
                            elif isinstance(scan_result_doctor, str):
                                scan_explanation_doctor = scan_result_doctor
                            
                            logger.info("Doctor scan explanation generated successfully")
                        except Exception as e:
                            logger.warning(f"Doctor scan explanation generation failed: {e}")
                            import traceback
                            logger.error(f"Traceback: {traceback.format_exc()}")
                            scan_explanation_doctor = None
                    
                    except Exception as e:
                        logger.warning(f"Scan explanation generation failed: {e}")
                        import traceback
                        logger.error(f"Traceback: {traceback.format_exc()}")
                        scan_explanation = None
                        scan_explanation_doctor = None
            
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
                scan_explanation=scan_explanation,
                scan_explanation_doctor=scan_explanation_doctor,
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
    
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image bytes
        image_bytes = await file.read()
        
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty image file")
        
        # Make prediction
        result = prediction_service.predict(
            image_bytes, include_explanation, include_hint
        )
        
        return result
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error in predict endpoint: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500, 
            detail=f"Internal server error during prediction: {str(e)}"
        )


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
    
    # Use PORT environment variable if set (for Railway/Render deployment)
    port = int(os.environ.get('PORT', api_config['port']))
    
    uvicorn.run(
        "src.inference:app",
        host=api_config['host'],
        port=port,
        workers=api_config['workers'],
        reload=False
    )


if __name__ == "__main__":
    main()
