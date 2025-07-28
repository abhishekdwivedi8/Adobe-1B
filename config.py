"""
Configuration module for Document Intelligence System
Centralizes all settings and model configurations
"""

import os
from pathlib import Path

class Config:
    """Centralized configuration for the document intelligence system"""
    
    # Directory structure
    BASE_DIR = Path(__file__).parent
    INPUT_DIR = BASE_DIR / "input"
    OUTPUT_DIR = BASE_DIR / "output" 
    MODELS_DIR = BASE_DIR / "models"
    LOGS_DIR = BASE_DIR / "logs"
    
    # Model configurations (optimized for sample accuracy)
    MODELS = {
        "sentence_transformer": {
            "name": "all-MiniLM-L6-v2",
            "size_mb": 90,
            "purpose": "semantic similarity and embeddings"
        },
        "distilbert": {
            "name": "distilbert-base-uncased", 
            "size_mb": 268,
            "purpose": "text understanding and classification"
        }
    }
    
    # Processing constraints
    MAX_PROCESSING_TIME = 60  # seconds
    MAX_DOCUMENTS = 10
    MIN_DOCUMENTS = 3
    MAX_MODEL_SIZE_GB = 1.0
    
    # PDF processing settings
    PDF_SETTINGS = {
        "max_pages_per_doc": 50,
        "min_section_length": 50,
        "max_section_length": 2000,
        "min_words_per_section": 10
    }
    
    # Ranking weights for relevance scoring (OPTIMIZED for sample-like accuracy)
    RANKING_WEIGHTS = {
        "persona_job_alignment": 0.5,   # INCREASED - Perfect phrase matching priority
        "content_actionability": 0.3,   # INCREASED - Instruction-focused content
        "section_specificity": 0.15,    # REDUCED - Less emphasis on broad specificity
        "content_quality": 0.05         # REDUCED - Focus on relevance over quality
    }
    
    # Output formatting
    MAX_EXTRACTED_SECTIONS = 15
    MAX_SUBSECTION_ANALYSIS = 20
    
    @classmethod
    def ensure_directories(cls):
        """Ensure all required directories exist"""
        for dir_path in [cls.INPUT_DIR, cls.OUTPUT_DIR, cls.MODELS_DIR, cls.LOGS_DIR]:
            dir_path.mkdir(exist_ok=True)
    
    @classmethod
    def get_total_model_size_mb(cls):
        """Calculate total model size"""
        return sum(model["size_mb"] for model in cls.MODELS.values())
    
    @classmethod
    def validate_constraints(cls):
        """Validate that configuration meets constraints"""
        total_size_gb = cls.get_total_model_size_mb() / 1024
        if total_size_gb > cls.MAX_MODEL_SIZE_GB:
            raise ValueError(f"Total model size {total_size_gb:.1f}GB exceeds {cls.MAX_MODEL_SIZE_GB}GB limit")
        return True
