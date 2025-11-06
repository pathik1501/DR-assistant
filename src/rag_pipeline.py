"""
RAG (Retrieval-Augmented Generation) module for generating clinical hints.
Integrates with OpenAI GPT-4o-mini and FAISS vector database.
"""

import os
import json
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import openai
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
# Updated imports for LangChain compatibility
try:
    from langchain_community.embeddings import OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS
except ImportError:
    # Fallback for older LangChain versions
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.vectorstores import FAISS

try:
    from langchain_openai import ChatOpenAI
    USE_CHAT_MODEL = True
except ImportError:
    try:
        from langchain_community.chat_models import ChatOpenAI
        USE_CHAT_MODEL = True
    except ImportError:
        from langchain.llms import OpenAI
        USE_CHAT_MODEL = False

from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import yaml


class OphthalmologyKnowledgeBase:
    """Knowledge base for ophthalmology guidelines and DR management."""
    
    def __init__(self):
        self.guidelines = [
            {
                "content": "No Diabetic Retinopathy (DR): Annual comprehensive eye examination recommended. Maintain good glycemic control with HbA1c < 7%. Regular monitoring of blood pressure and cholesterol levels.",
                "metadata": {"source": "ADA Guidelines 2023", "grade": 0}
            },
            {
                "content": "Mild Nonproliferative DR: Annual comprehensive eye examination. Tight glycemic control with HbA1c < 7%. Consider more frequent monitoring if other risk factors present.",
                "metadata": {"source": "ADA Guidelines 2023", "grade": 1}
            },
            {
                "content": "Moderate Nonproliferative DR: Comprehensive eye examination every 6-12 months. Intensive glycemic control. Consider referral to retina specialist if progression observed.",
                "metadata": {"source": "ADA Guidelines 2023", "grade": 2}
            },
            {
                "content": "Severe Nonproliferative DR: Comprehensive eye examination every 3-6 months. Immediate referral to retina specialist. Consider panretinal photocoagulation if high-risk characteristics present.",
                "metadata": {"source": "ADA Guidelines 2023", "grade": 3}
            },
            {
                "content": "Proliferative DR: Immediate referral to retina specialist. Panretinal photocoagulation or anti-VEGF therapy may be indicated. Frequent follow-up every 1-3 months.",
                "metadata": {"source": "ADA Guidelines 2023", "grade": 4}
            },
            {
                "content": "Diabetic macular edema (DME) may occur at any stage of DR. Anti-VEGF therapy is first-line treatment for center-involving DME. Focal laser may be considered for non-center-involving DME.",
                "metadata": {"source": "DRCR.net Protocol T", "grade": "any"}
            },
            {
                "content": "Risk factors for DR progression include poor glycemic control, hypertension, dyslipidemia, longer duration of diabetes, and pregnancy. Address modifiable risk factors aggressively.",
                "metadata": {"source": "WESDR Study", "grade": "any"}
            },
            {
                "content": "Screening intervals: Type 1 diabetes - annual screening starting 5 years after diagnosis. Type 2 diabetes - screening at diagnosis and annually thereafter.",
                "metadata": {"source": "ADA Guidelines 2023", "grade": "screening"}
            },
            {
                "content": "Pregnancy increases risk of DR progression. Preconception counseling and frequent monitoring during pregnancy recommended. Consider more frequent examinations.",
                "metadata": {"source": "ADA Guidelines 2023", "grade": "pregnancy"}
            },
            {
                "content": "Laser photocoagulation reduces risk of severe vision loss by 50% in high-risk proliferative DR. Early treatment is crucial for preserving vision.",
                "metadata": {"source": "ETDRS Study", "grade": 4}
            }
        ]
    
    def get_lesion_knowledge(self) -> Dict[int, Dict]:
        """Get lesion type knowledge for each DR grade."""
        return {
            0: {
                "lesions": [],
                "features": ["normal retinal vasculature", "clear optic disc", "normal macula"],
                "description": "No diabetic retinopathy lesions detected. The retina appears normal with intact vasculature."
            },
            1: {
                "lesions": ["microaneurysms"],
                "features": ["few microaneurysms", "mild retinal changes", "normal macula"],
                "description": "Mild nonproliferative changes with few microaneurysms visible."
            },
            2: {
                "lesions": ["microaneurysms", "hemorrhages", "hard exudates"],
                "features": ["moderate microaneurysms", "dot and blot hemorrhages", "hard exudates", "retinal thickening"],
                "description": "Moderate nonproliferative changes with multiple microaneurysms, hemorrhages, and hard exudates."
            },
            3: {
                "lesions": ["microaneurysms", "hemorrhages", "hard exudates", "soft exudates", "cotton wool spots"],
                "features": ["severe microaneurysms", "extensive hemorrhages", "cotton wool spots", "venous beading", "intraretinal microvascular abnormalities"],
                "description": "Severe nonproliferative changes with extensive hemorrhages, cotton wool spots, and venous abnormalities."
            },
            4: {
                "lesions": ["microaneurysms", "hemorrhages", "hard exudates", "soft exudates", "neovascularization", "vitreous hemorrhage"],
                "features": ["neovascularization", "fibrovascular proliferation", "vitreous hemorrhage", "preretinal hemorrhage", "severe retinal changes"],
                "description": "Proliferative changes with neovascularization, fibrovascular proliferation, and potential vitreous hemorrhage."
            }
        }
    
    def get_documents(self) -> List[Document]:
        """Convert guidelines to LangChain documents."""
        documents = []
        for guideline in self.guidelines:
            doc = Document(
                page_content=guideline["content"],
                metadata=guideline["metadata"]
            )
            documents.append(doc)
        return documents


class RAGPipeline:
    """RAG pipeline for generating clinical hints."""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.rag_config = self.config['rag']
        
        # Initialize OpenAI
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        # Set for legacy openai library (if used)
        openai.api_key = api_key
        
        # Log API key info (first 10 chars only for security)
        print(f"Using OpenAI API key: {api_key[:10]}...{api_key[-4:] if len(api_key) > 14 else '***'}")
        
        # Initialize embeddings and LLM
        # Try langchain_openai first (latest), fallback to langchain_community
        try:
            from langchain_openai import OpenAIEmbeddings
            embeddings_class = OpenAIEmbeddings
        except ImportError:
            # Fallback to langchain_community or langchain
            try:
                from langchain_community.embeddings import OpenAIEmbeddings
                embeddings_class = OpenAIEmbeddings
            except ImportError:
                from langchain.embeddings import OpenAIEmbeddings
                embeddings_class = OpenAIEmbeddings
        
        # LangChain classes automatically read OPENAI_API_KEY from environment
        # No need to pass explicitly - they use os.getenv('OPENAI_API_KEY') internally
        
        # Initialize embeddings - will use OPENAI_API_KEY from environment
        print(f"Initializing embeddings with model: {self.rag_config['embedding_model']}")
        self.embeddings = embeddings_class(
            model=self.rag_config['embedding_model']
        )
        
        # Initialize LLM with proper chat model support
        # Try langchain_openai first (latest), fallback to langchain_community
        try:
            from langchain_openai import ChatOpenAI as OpenAI_ChatOpenAI
            llm_class = OpenAI_ChatOpenAI
            use_chat_model = True
        except ImportError:
            try:
                from langchain_community.chat_models import ChatOpenAI as OpenAI_ChatOpenAI
                llm_class = OpenAI_ChatOpenAI
                use_chat_model = True
            except ImportError:
                # Fallback to completion model
                from langchain.llms import OpenAI
                llm_class = OpenAI
                use_chat_model = False
        
        if use_chat_model:
            self.llm = llm_class(
                model_name=self.rag_config['llm_model'],
                temperature=0.3,
                max_tokens=300  # Increased for longer scan explanations
            )
            # Create a separate LLM instance for scan explanations with longer max_tokens
            self.llm_long = llm_class(
                model_name=self.rag_config['llm_model'],
                temperature=0.3,
                max_tokens=400  # Longer for detailed scan explanations
            )
        else:
            self.llm = llm_class(
                model_name=self.rag_config['llm_model'],
                temperature=0.3,
                max_tokens=300
            )
            self.llm_long = self.llm  # Use same for completion models
        
        # Initialize knowledge base
        self.kb = OphthalmologyKnowledgeBase()
        
        # Create or load vector database
        self.vector_db = self._create_vector_db()
        
        # Create QA chain
        self.qa_chain = self._create_qa_chain()
    
    def _create_vector_db(self) -> FAISS:
        """Create or load FAISS vector database."""
        db_path = self.rag_config['vector_db_path']
        
        # Check if both required files exist (FAISS saves index.faiss and index.pkl)
        db_path_obj = Path(db_path)
        index_faiss = db_path_obj / "index.faiss"
        index_pkl = db_path_obj / "index.pkl"
        
        if index_faiss.exists() and index_pkl.exists():
            print("Loading existing vector database...")
            # For security, only allow if the db was created by us
            # Since we trust our own data, we enable deserialization
            try:
                return FAISS.load_local(
                    str(db_path), 
                    self.embeddings, 
                    allow_dangerous_deserialization=True
                )
            except Exception as e:
                print(f"Warning: Could not load existing vector database: {e}")
                print("Will attempt to create a new one...")
                # Fall through to create new database
        
        # Create new vector database
        print("Creating new vector database...")
        try:
            documents = self.kb.get_documents()
            print(f"Generating embeddings for {len(documents)} documents...")
            
            # Add retry logic with exponential backoff for rate limits
            import time
            max_retries = 5
            initial_delay = 5  # Wait 5 seconds before first attempt (in case quota just updated)
            retry_delay = 2  # Base delay for retries
            
            print(f"Waiting {initial_delay} seconds before first attempt (allows quota to propagate if payment plan was just added)...")
            time.sleep(initial_delay)
            
            for attempt in range(max_retries):
                try:
                    print(f"Attempt {attempt + 1}/{max_retries}: Creating vector database...")
                    vector_db = FAISS.from_documents(documents, self.embeddings)
                    print(f"Successfully created vector database on attempt {attempt + 1}")
                    break  # Success, exit retry loop
                except Exception as e:
                    error_msg = str(e)
                    error_type = type(e).__name__
                    
                    # Check if it's a rate limit/quota error
                    is_rate_limit = (
                        "429" in error_msg or 
                        "quota" in error_msg.lower() or 
                        "insufficient_quota" in error_msg.lower() or 
                        "rate_limit" in error_msg.lower() or
                        error_type == "RateLimitError"
                    )
                    
                    if is_rate_limit:
                        if attempt < max_retries - 1:
                            wait_time = retry_delay * (2 ** attempt)  # Exponential backoff: 2s, 4s, 8s, 16s
                            print(f"Rate limit/quota error detected (attempt {attempt + 1}/{max_retries}).")
                            print(f"Waiting {wait_time} seconds before retry...")
                            print("Note: If you just added a payment plan, it may take 5-10 minutes to propagate.")
                            time.sleep(wait_time)
                            continue
                        else:
                            # Final attempt failed
                            raise ValueError(
                                f"OpenAI API quota/rate limit error after {max_retries} attempts.\n"
                                f"Error: {error_msg}\n"
                                f"Please verify:\n"
                                f"1. Your OpenAI account has active billing/credits at https://platform.openai.com/account/billing\n"
                                f"2. If you just added a payment plan, wait 5-10 minutes for it to propagate\n"
                                f"3. Check your usage limits at https://platform.openai.com/account/limits\n"
                                f"Then restart the server to try again."
                            )
                    else:
                        # Not a rate limit error, re-raise immediately
                        print(f"Non-rate-limit error occurred: {error_type}")
                        raise
            
            # Save database
            os.makedirs(str(db_path_obj), exist_ok=True)
            vector_db.save_local(str(db_path))
            
            print(f"Vector database created and saved to {db_path}")
            return vector_db
            
        except ValueError:
            # Re-raise ValueError as-is (already has good error message)
            raise
        except Exception as e:
            error_msg = str(e)
            raise ValueError(
                f"Failed to create vector database: {error_msg}\n"
                f"This is required for RAG features. Please check your OpenAI API key and quota."
            )
    
    def _create_qa_chain(self) -> RetrievalQA:
        """Create QA chain with custom prompt."""
        prompt_template = """
You are a medical assistant specializing in diabetic retinopathy. Based on the provided context and the patient's DR grade, generate a concise one-sentence follow-up recommendation.

Context: {context}

DR Grade: {grade}
Grade Description: {grade_description}

Generate a single, actionable sentence that references the provided evidence and gives specific guidance for this patient's condition.

Recommendation:"""

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "grade", "grade_description"]
        )
        
        retriever = self.vector_db.as_retriever(
            search_kwargs={"k": self.rag_config['top_k']}
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": PROMPT}
        )
        
        return qa_chain
    
    def generate_hint(
        self, 
        dr_grade: int, 
        confidence: float,
        additional_context: Optional[str] = None
    ) -> Dict[str, str]:
        """Generate clinical hint for given DR grade."""
        
        grade_descriptions = {
            0: "No Diabetic Retinopathy",
            1: "Mild Nonproliferative Diabetic Retinopathy",
            2: "Moderate Nonproliferative Diabetic Retinopathy", 
            3: "Severe Nonproliferative Diabetic Retinopathy",
            4: "Proliferative Diabetic Retinopathy"
        }
        
        # Create query based on grade and confidence
        query = f"diabetic retinopathy grade {dr_grade} {grade_descriptions[dr_grade]}"
        if additional_context:
            query += f" {additional_context}"
        
        # Retrieve relevant documents
        relevant_docs = self.vector_db.similarity_search(
            query, k=self.rag_config['top_k']
        )
        
        # Generate hint using QA chain
        try:
            result = self.qa_chain.run(
                query=query,
                grade=dr_grade,
                grade_description=grade_descriptions[dr_grade]
            )
            
            hint = result.strip()
            
        except Exception as e:
            print(f"Error generating hint: {e}")
            # Fallback hint
            hint = f"Based on {grade_descriptions[dr_grade]}, consult with an ophthalmologist for appropriate follow-up care."
        
        # Prepare sources
        sources = []
        for doc in relevant_docs:
            source_info = {
                "content": doc.page_content,
                "source": doc.metadata.get("source", "Unknown"),
                "grade": doc.metadata.get("grade", "any")
            }
            sources.append(source_info)
        
        return {
            "hint": hint,
            "sources": sources,
            "confidence": confidence,
            "grade": dr_grade,
            "grade_description": grade_descriptions[dr_grade]
        }
    
    def batch_generate_hints(
        self, 
        predictions: List[Tuple[int, float]]
    ) -> List[Dict[str, str]]:
        """Generate hints for a batch of predictions."""
        hints = []
        
        for dr_grade, confidence in predictions:
            hint = self.generate_hint(dr_grade, confidence)
            hints.append(hint)
        
        return hints
    
    def analyze_heatmap_regions(self, heatmap: np.ndarray, image_shape: Tuple[int, int]) -> Dict[str, Any]:
        """Analyze Grad-CAM heatmap to identify regions of interest.
        
        Args:
            heatmap: 2D numpy array of heatmap values
            image_shape: (height, width) of original image
            
        Returns:
            Dictionary with region analysis
        """
        # Resize heatmap to match image if needed
        if heatmap.shape != image_shape[:2]:
            heatmap_resized = cv2.resize(heatmap, (image_shape[1], image_shape[0]))
        else:
            heatmap_resized = heatmap
        
        # Threshold to find high activation regions
        threshold = np.percentile(heatmap_resized, 85)  # Top 15% of activations
        high_activation_mask = heatmap_resized > threshold
        
        # Calculate region statistics
        total_pixels = heatmap_resized.size
        high_activation_pixels = np.sum(high_activation_mask)
        activation_ratio = high_activation_pixels / total_pixels
        
        # Find center of mass of high activation regions
        y_coords, x_coords = np.where(high_activation_mask)
        if len(y_coords) > 0:
            center_y = int(np.mean(y_coords))
            center_x = int(np.mean(x_coords))
            
            # Determine region (macula, optic disc, peripheral)
            height, width = image_shape[:2]
            macula_center = (height // 2, width // 2)
            distance_from_center = np.sqrt((center_y - macula_center[0])**2 + (center_x - macula_center[1])**2)
            
            if distance_from_center < min(height, width) * 0.15:
                region = "macula"
            elif distance_from_center < min(height, width) * 0.25:
                region = "perimacular"
            else:
                region = "peripheral"
        else:
            center_y, center_x = image_shape[0] // 2, image_shape[1] // 2
            region = "diffuse"
        
        # Calculate average activation intensity
        avg_intensity = np.mean(heatmap_resized[high_activation_mask]) if high_activation_pixels > 0 else 0
        
        return {
            "activation_ratio": float(activation_ratio),
            "center": (center_y, center_x),
            "region": region,
            "avg_intensity": float(avg_intensity),
            "max_intensity": float(np.max(heatmap_resized)),
            "distribution": "focal" if activation_ratio < 0.1 else "diffuse" if activation_ratio > 0.3 else "moderate"
        }
    
    def generate_scan_explanation(
        self,
        dr_grade: int,
        confidence: float,
        heatmap: Optional[np.ndarray] = None,
        image_shape: Optional[Tuple[int, int]] = None,
        additional_context: Optional[str] = None,
        for_patient: bool = True
    ) -> Dict[str, str]:
        """Generate detailed explanation of what the model sees in the scan.
        
        Args:
            dr_grade: DR grade (0-4)
            confidence: Model confidence score
            heatmap: Optional Grad-CAM heatmap for region analysis
            image_shape: Optional (height, width) of original image
            additional_context: Optional additional context
            for_patient: If True, generates patient-friendly explanation; if False, generates detailed doctor version
            
        Returns:
            Dictionary with detailed scan explanation
        """
        grade_descriptions = {
            0: "No Diabetic Retinopathy",
            1: "Mild Nonproliferative Diabetic Retinopathy",
            2: "Moderate Nonproliferative Diabetic Retinopathy",
            3: "Severe Nonproliferative Diabetic Retinopathy",
            4: "Proliferative Diabetic Retinopathy"
        }
        
        # Get lesion knowledge for this grade
        lesion_kb = self.kb.get_lesion_knowledge()
        grade_info = lesion_kb.get(dr_grade, lesion_kb[0])
        
        # Analyze heatmap if provided
        heatmap_analysis = None
        if heatmap is not None and image_shape is not None:
            heatmap_analysis = self.analyze_heatmap_regions(heatmap, image_shape)
        
        # Build context for LLM
        context_parts = []
        
        # Add lesion knowledge
        if grade_info["lesions"]:
            context_parts.append(f"Typical lesions for {grade_descriptions[dr_grade]}: {', '.join(grade_info['lesions'])}")
            context_parts.append(f"Clinical features: {', '.join(grade_info['features'])}")
        else:
            context_parts.append("No diabetic retinopathy lesions typically present in this grade.")
        
        # Add heatmap analysis if available
        if heatmap_analysis:
            context_parts.append(
                f"Model attention analysis: The model's attention is {'focused' if heatmap_analysis['distribution'] == 'focal' else 'diffused'} "
                f"in the {heatmap_analysis['region']} region, with {heatmap_analysis['activation_ratio']:.1%} of the image showing high activation."
            )
        
        context = " ".join(context_parts)
        
        # Create different prompts for patient vs doctor
        if for_patient:
            # Patient-friendly prompt (simpler, less technical)
            prompt_template = """You are a medical imaging specialist explaining retinal fundus findings to a patient in clear, accessible language.

Based on the following information, provide a clear, patient-friendly 3-4 sentence explanation of what the scan shows:

DR Grade: {grade} ({grade_description})
Confidence: {confidence:.1%}
Typical Lesions: {lesions}
Clinical Features: {features}
Model Attention: {heatmap_info}
Additional Context: {additional_context}

Generate a clear, accessible explanation for the patient describing:
1. What the scan shows in simple terms
2. What changes or signs were found (if any)
3. What this means for their eye health
4. Where in the retina these changes are located (in simple terms)

Use simple, non-technical language. Avoid complex medical jargon. Be reassuring but accurate. Focus on what the patient needs to know.

Patient-Friendly Scan Description:"""
        else:
            # Doctor/clinical prompt (detailed, technical)
            prompt_template = """You are a medical imaging specialist providing a comprehensive clinical report on a retinal fundus photograph for diabetic retinopathy.

Based on the following information, provide a detailed 6-8 sentence clinical explanation of the retinal findings:

DR Grade: {grade} ({grade_description})
Confidence: {confidence:.1%}
Typical Lesions: {lesions}
Clinical Features: {features}
Model Attention: {heatmap_info}
Additional Context: {additional_context}

Generate a comprehensive clinical explanation describing:
1. Specific lesions present with detailed descriptions (microaneurysms, hemorrhages, exudates, cotton wool spots, neovascularization, etc.)
2. Precise anatomical location and distribution (macula, perimacular, peripheral, quadrants)
3. Severity and extent of findings (mild, moderate, severe, extent of involvement)
4. Associated clinical features (retinal thickening, vascular changes, venous beading, IRMA)
5. Clinical significance and implications for patient management
6. Comparison to typical findings for this DR grade
7. Any notable patterns or characteristics (focal vs diffuse, symmetry, etc.)
8. Recommendations for additional imaging or clinical correlation if needed

Use precise medical terminology. Include specific lesion types, anatomical locations, and clinical correlations. Reference imaging characteristics and clinical significance. Be thorough and detailed.

Detailed Clinical Scan Description:"""

        # Format prompt
        lesions_str = ", ".join(grade_info["lesions"]) if grade_info["lesions"] else "None"
        features_str = ", ".join(grade_info["features"])
        
        heatmap_info = "No heatmap analysis available"
        if heatmap_analysis:
            heatmap_info = (
                f"Model attention is {heatmap_analysis['distribution']} in the {heatmap_analysis['region']} region "
                f"({heatmap_analysis['activation_ratio']:.1%} of image with high activation)"
            )
        
        prompt = prompt_template.format(
            grade=dr_grade,
            grade_description=grade_descriptions[dr_grade],
            confidence=confidence,
            lesions=lesions_str,
            features=features_str,
            heatmap_info=heatmap_info,
            additional_context=additional_context or "None"
        )
        
        # Generate explanation using LLM
        try:
            # Use LangChain LLM with enhanced prompt
            full_prompt = (
                "You are a medical imaging specialist providing detailed explanations of retinal fundus photographs for diabetic retinopathy detection. "
                + prompt
            )
            
            # Use different max_tokens for patient vs doctor versions
            if for_patient:
                llm_to_use = getattr(self, 'llm_long', self.llm)  # 400 tokens for patient
                max_tokens = 400
            else:
                # Create or use a longer LLM instance for doctor version
                if not hasattr(self, 'llm_doctor'):
                    # Create doctor LLM with higher max_tokens
                    try:
                        from langchain_openai import ChatOpenAI as OpenAI_ChatOpenAI
                        self.llm_doctor = OpenAI_ChatOpenAI(
                            model_name=self.rag_config['llm_model'],
                            temperature=0.3,
                            max_tokens=800  # Much longer for detailed doctor version
                        )
                    except ImportError:
                        if USE_CHAT_MODEL:
                            self.llm_doctor = ChatOpenAI(
                                model_name=self.rag_config['llm_model'],
                                temperature=0.3,
                                max_tokens=800
                            )
                        else:
                            self.llm_doctor = self.llm_long
                llm_to_use = self.llm_doctor
                max_tokens = 800
            
            # Try to use ChatOpenAI if available
            if hasattr(llm_to_use, 'invoke') or hasattr(llm_to_use, '__call__'):
                # For chat models, use messages format
                try:
                    from langchain.schema import HumanMessage, SystemMessage
                    system_content = (
                        "You are a medical imaging specialist providing detailed explanations of retinal fundus photographs for diabetic retinopathy detection."
                        if not for_patient else
                        "You are a medical imaging specialist explaining retinal findings to patients in clear, accessible language."
                    )
                    messages = [
                        SystemMessage(content=system_content),
                        HumanMessage(content=prompt)
                    ]
                    if hasattr(llm_to_use, 'invoke'):
                        response = llm_to_use.invoke(messages)
                    else:
                        response = llm_to_use(messages)
                    
                    if hasattr(response, 'content'):
                        explanation = response.content.strip()
                    else:
                        explanation = str(response).strip()
                except:
                    # Fallback to direct call
                    explanation = str(llm_to_use(full_prompt)).strip()
            else:
                # For completion models
                explanation = str(llm_to_use(full_prompt)).strip()
            
        except Exception as e:
            print(f"Error generating scan explanation: {e}")
            import traceback
            traceback.print_exc()
            # Fallback explanation
            explanation = (
                f"This retinal scan shows {grade_descriptions[dr_grade].lower()}. "
                f"{grade_info['description']} "
                f"Based on the model's analysis with {confidence:.1%} confidence, "
                f"{'the following lesions are typically present: ' + ', '.join(grade_info['lesions']) + '.' if grade_info['lesions'] else 'no diabetic retinopathy lesions are detected.'}"
            )
        
        # Prepare response
        result = {
            "explanation": explanation,
            "grade": dr_grade,
            "grade_description": grade_descriptions[dr_grade],
            "confidence": confidence,
            "typical_lesions": grade_info["lesions"],
            "clinical_features": grade_info["features"]
        }
        
        if heatmap_analysis:
            result["heatmap_analysis"] = heatmap_analysis
        
        return result


class HintEvaluator:
    """Evaluate quality of generated hints."""
    
    @staticmethod
    def evaluate_hint_relevance(hint: str, grade: int) -> float:
        """Simple evaluation of hint relevance (0-1 scale)."""
        grade_keywords = {
            0: ["annual", "routine", "monitor", "control"],
            1: ["mild", "annual", "monitor", "control"],
            2: ["moderate", "6-12", "referral", "specialist"],
            3: ["severe", "immediate", "referral", "laser"],
            4: ["proliferative", "immediate", "laser", "anti-vegf"]
        }
        
        keywords = grade_keywords.get(grade, [])
        hint_lower = hint.lower()
        
        relevance_score = 0
        for keyword in keywords:
            if keyword in hint_lower:
                relevance_score += 1
        
        return min(relevance_score / len(keywords), 1.0)
    
    @staticmethod
    def evaluate_hint_clarity(hint: str) -> float:
        """Evaluate hint clarity based on length and structure."""
        # Ideal length: 15-25 words
        word_count = len(hint.split())
        
        if 55 <= word_count <= 65:
            length_score = 1.0
        elif 20 <= word_count < 55 or 65 < word_count <= 80:
            length_score = 0.8
        else:
            length_score = 0.6
        
        # Check for actionable terms
        actionable_terms = ["refer", "schedule", "monitor", "control", "examine"]
        hint_lower = hint.lower()
        
        action_score = 0
        for term in actionable_terms:
            if term in hint_lower:
                action_score += 1
        
        action_score = min(action_score / 2, 1.0)  # Normalize
        
        return (length_score + action_score) / 2


def main():
    """Test RAG pipeline functionality."""
    try:
        # Initialize RAG pipeline
        rag = RAGPipeline()
        
        # Test hint generation
        test_cases = [
            (0, 0.95),  # No DR, high confidence
            (2, 0.78),  # Moderate DR, medium confidence
            (4, 0.65),  # Proliferative DR, lower confidence
        ]
        
        print("Testing RAG pipeline...")
        
        for grade, confidence in test_cases:
            hint_data = rag.generate_hint(grade, confidence)
            
            print(f"\nDR Grade: {grade} ({hint_data['grade_description']})")
            print(f"Confidence: {confidence:.3f}")
            print(f"Hint: {hint_data['hint']}")
            print(f"Sources: {len(hint_data['sources'])} documents")
            
            # Evaluate hint quality
            relevance = HintEvaluator.evaluate_hint_relevance(hint_data['hint'], grade)
            clarity = HintEvaluator.evaluate_hint_clarity(hint_data['hint'])
            
            print(f"Relevance Score: {relevance:.3f}")
            print(f"Clarity Score: {clarity:.3f}")
        
        print("\nRAG pipeline test completed successfully!")
        
    except Exception as e:
        print(f"Error testing RAG pipeline: {e}")
        print("Make sure OPENAI_API_KEY environment variable is set")


if __name__ == "__main__":
    main()
