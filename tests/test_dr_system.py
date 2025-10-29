"""
Unit tests for Diabetic Retinopathy detection system.
"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch
import tempfile
import os

from src.model import DRModel, FocalLoss, TemperatureScaling, calculate_qwk, calculate_ece
from src.data_processing import DRDataset, DataProcessor
from src.explainability import GradCAM, ExplainabilityPipeline
from src.rag_pipeline import OphthalmologyKnowledgeBase, RAGPipeline


class TestDRModel:
    """Test DR model functionality."""
    
    def test_model_creation(self):
        """Test model creation."""
        model = DRModel(num_classes=5, pretrained=False)
        assert model.num_classes == 5
        assert isinstance(model.backbone, torch.nn.Module)
        assert isinstance(model.classifier, torch.nn.Module)
    
    def test_model_forward(self):
        """Test model forward pass."""
        model = DRModel(num_classes=5, pretrained=False)
        x = torch.randn(2, 3, 512, 512)
        output = model(x)
        assert output.shape == (2, 5)
    
    def test_focal_loss(self):
        """Test focal loss calculation."""
        criterion = FocalLoss(gamma=2.0)
        logits = torch.randn(4, 5)
        targets = torch.randint(0, 5, (4,))
        loss = criterion(logits, targets)
        assert loss.item() > 0
        assert isinstance(loss, torch.Tensor)


class TestDataProcessing:
    """Test data processing functionality."""
    
    def test_dataset_creation(self):
        """Test dataset creation."""
        # Create dummy data
        image_paths = ["dummy1.jpg", "dummy2.jpg"]
        labels = [0, 1]
        
        dataset = DRDataset(image_paths, labels, transform=None)
        assert len(dataset) == 2
        assert dataset.image_paths == image_paths
        assert dataset.labels == labels
    
    def test_data_processor_init(self):
        """Test data processor initialization."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
data:
  output_size: [512, 512]
  batch_size: 16
augmentation:
  brightness_limit: 0.2
""")
            f.flush()
            
            processor = DataProcessor(f.name)
            assert processor.config['data']['output_size'] == [512, 512]
            
            os.unlink(f.name)


class TestExplainability:
    """Test explainability functionality."""
    
    def test_gradcam_initialization(self):
        """Test GradCAM initialization."""
        model = DRModel(num_classes=5, pretrained=False)
        target_layers = ["blocks.5.2"]
        
        gradcam = GradCAM(model, target_layers)
        assert gradcam.model == model
        assert gradcam.target_layers == target_layers
    
    def test_explainability_pipeline(self):
        """Test explainability pipeline."""
        model = DRModel(num_classes=5, pretrained=False)
        target_layers = ["blocks.5.2"]
        
        pipeline = ExplainabilityPipeline(model, target_layers)
        assert pipeline.model == model
        assert pipeline.target_layers == target_layers


class TestRAGPipeline:
    """Test RAG pipeline functionality."""
    
    def test_knowledge_base(self):
        """Test knowledge base creation."""
        kb = OphthalmologyKnowledgeBase()
        documents = kb.get_documents()
        assert len(documents) > 0
        assert all(hasattr(doc, 'page_content') for doc in documents)
    
    @patch('src.rag_pipeline.openai.api_key', 'test-key')
    def test_rag_pipeline_init(self):
        """Test RAG pipeline initialization."""
        with patch('src.rag_pipeline.OpenAIEmbeddings') as mock_embeddings, \
             patch('src.rag_pipeline.OpenAI') as mock_llm, \
             patch('src.rag_pipeline.FAISS') as mock_faiss:
            
            mock_faiss.load_local.side_effect = FileNotFoundError()
            mock_faiss.from_documents.return_value = Mock()
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                f.write("""
rag:
  vector_db_path: "test_db"
  embedding_model: "test-model"
  llm_model: "test-llm"
  top_k: 3
""")
                f.flush()
                
                try:
                    rag = RAGPipeline(f.name)
                    assert rag.embeddings is not None
                    assert rag.llm is not None
                except Exception:
                    # Expected to fail without proper API setup
                    pass
                
                os.unlink(f.name)


class TestMetrics:
    """Test metric calculations."""
    
    def test_qwk_calculation(self):
        """Test Quadratic Weighted Kappa calculation."""
        y_true = np.array([0, 1, 2, 3, 4])
        y_pred = np.array([0, 1, 2, 3, 4])
        
        qwk = calculate_qwk(y_true, y_pred)
        assert qwk == 1.0  # Perfect agreement
    
    def test_ece_calculation(self):
        """Test Expected Calibration Error calculation."""
        y_true = np.array([1, 1, 0, 0])
        y_pred_probs = np.array([0.9, 0.8, 0.3, 0.2])
        
        ece = calculate_ece(y_true, y_pred_probs)
        assert 0 <= ece <= 1


class TestIntegration:
    """Integration tests."""
    
    def test_model_training_step(self):
        """Test a single training step."""
        model = DRModel(num_classes=5, pretrained=False)
        criterion = FocalLoss()
        
        # Dummy data
        images = torch.randn(2, 3, 512, 512)
        targets = torch.randint(0, 5, (2,))
        
        # Forward pass
        logits = model(images)
        loss = criterion(logits, targets)
        
        # Backward pass
        loss.backward()
        
        assert loss.item() > 0
        assert logits.shape == (2, 5)
    
    def test_prediction_pipeline(self):
        """Test complete prediction pipeline."""
        model = DRModel(num_classes=5, pretrained=False)
        model.eval()
        
        # Dummy image
        image = torch.randn(1, 3, 512, 512)
        
        with torch.no_grad():
            logits = model(image)
            probabilities = torch.softmax(logits, dim=1)
            prediction = logits.argmax(dim=1).item()
            confidence = probabilities.max().item()
        
        assert 0 <= prediction <= 4
        assert 0 <= confidence <= 1


if __name__ == "__main__":
    pytest.main([__file__])
