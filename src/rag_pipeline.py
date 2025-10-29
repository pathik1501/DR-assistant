"""
RAG (Retrieval-Augmented Generation) module for generating clinical hints.
Integrates with OpenAI GPT-4o-mini and FAISS vector database.
"""

import os
import json
import numpy as np
from typing import List, Dict, Optional, Tuple
import openai
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.schema import Document
from langchain.vectorstores import FAISS
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
        openai.api_key = os.getenv('OPENAI_API_KEY')
        if not openai.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        # Initialize embeddings and LLM
        self.embeddings = OpenAIEmbeddings(
            model=self.rag_config['embedding_model']
        )
        self.llm = OpenAI(
            model_name=self.rag_config['llm_model'],
            temperature=0.3,
            max_tokens=150
        )
        
        # Initialize knowledge base
        self.kb = OphthalmologyKnowledgeBase()
        
        # Create or load vector database
        self.vector_db = self._create_vector_db()
        
        # Create QA chain
        self.qa_chain = self._create_qa_chain()
    
    def _create_vector_db(self) -> FAISS:
        """Create or load FAISS vector database."""
        db_path = self.rag_config['vector_db_path']
        
        if os.path.exists(db_path):
            print("Loading existing vector database...")
            # For security, only allow if the db was created by us
            # Since we trust our own data, we enable deserialization
            return FAISS.load_local(
                db_path, 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
        else:
            print("Creating new vector database...")
            documents = self.kb.get_documents()
            vector_db = FAISS.from_documents(documents, self.embeddings)
            
            # Save database
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            vector_db.save_local(db_path)
            
            return vector_db
    
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
        
        if 15 <= word_count <= 25:
            length_score = 1.0
        elif 10 <= word_count < 15 or 25 < word_count <= 30:
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
