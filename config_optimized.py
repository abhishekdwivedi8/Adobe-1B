"""
Optimized Configuration for Challenge 1B Compliance
CPU-only, <1GB models, <60s processing
"""

import os
from pathlib import Path

class OptimizedConfig:
    """Optimized configuration meeting challenge constraints"""
    
    # Directory structure
    BASE_DIR = Path(__file__).parent
    INPUT_DIR = BASE_DIR / "input"
    OUTPUT_DIR = BASE_DIR / "output" 
    MODELS_DIR = BASE_DIR / "models"
    LOGS_DIR = BASE_DIR / "logs"
    
    # CHALLENGE CONSTRAINT: Model size ≤ 1GB total
    MODELS = {
        "sentence_transformer": {
            "name": "all-MiniLM-L6-v2",  # 90MB - lightweight sentence embeddings
            "size_mb": 90,
            "purpose": "semantic similarity and embeddings",
            "cpu_optimized": True
        },
        "text_classifier": {
            "name": "distilbert-base-uncased",  # 268MB - lightweight classification
            "size_mb": 268,
            "purpose": "text classification and understanding", 
            "cpu_optimized": True
        }
        # Total: 358MB (well under 1GB limit)
    }
    
    # CHALLENGE CONSTRAINT: Processing time ≤ 60 seconds
    PERFORMANCE_SETTINGS = {
        "max_processing_time": 60,  # seconds
        "cpu_threads": 4,           # CPU-only optimization
        "batch_size": 8,            # Optimized for CPU inference
        "max_sequence_length": 512, # Reduced for speed
        "enable_parallel": True,    # Parallel document processing
        "cache_embeddings": True    # Cache for speed
    }
    
    # CHALLENGE CONSTRAINT: 3-10 documents
    DOCUMENT_CONSTRAINTS = {
        "min_documents": 3,
        "max_documents": 10,
        "max_pages_per_doc": 50,    # Limit for performance
        "max_file_size_mb": 10      # Individual file size limit
    }
    
    # CPU-only inference settings
    INFERENCE_SETTINGS = {
        "device": "cpu",            # CHALLENGE CONSTRAINT: CPU only
        "use_gpu": False,
        "fp16": False,              # No mixed precision on CPU
        "torch_compile": False,     # Avoid compilation overhead
        "num_workers": 1            # Single worker for CPU
    }
    
    # Optimized ranking weights for accuracy
    RANKING_WEIGHTS = {
        "persona_alignment": 0.30,   # Primary factor
        "job_relevance": 0.25,       # Task alignment  
        "content_actionability": 0.20, # Practical value
        "content_quality": 0.15,     # Quality assurance
        "domain_specificity": 0.10   # Domain match
    }
    
    # Output limits for performance
    OUTPUT_LIMITS = {
        "max_extracted_sections": 15,     # Challenge requirement
        "max_subsection_analysis": 10,    # Focused analysis
        "min_section_confidence": 0.6,    # Quality threshold
        "min_content_length": 50          # Minimum viable content
    }
    
    # Quality thresholds for 90%+ accuracy
    QUALITY_THRESHOLDS = {
        "section_detection_confidence": 0.7,
        "title_completeness_score": 0.8,
        "ranking_consistency_score": 0.85,
        "overall_accuracy_target": 0.90
    }
    
    @classmethod
    def ensure_directories(cls):
        """Ensure all required directories exist"""
        for dir_path in [cls.INPUT_DIR, cls.OUTPUT_DIR, cls.MODELS_DIR, cls.LOGS_DIR]:
            dir_path.mkdir(exist_ok=True)
    
    @classmethod
    def get_total_model_size_mb(cls):
        """Calculate total model size - must be under 1GB"""
        total = sum(model["size_mb"] for model in cls.MODELS.values())
        assert total < 1024, f"Total model size {total}MB exceeds 1GB limit"
        return total
    
    @classmethod
    def validate_challenge_constraints(cls):
        """Validate all challenge constraints are met"""
        constraints = {
            "model_size_under_1gb": cls.get_total_model_size_mb() < 1024,
            "cpu_only_inference": cls.INFERENCE_SETTINGS["device"] == "cpu",
            "processing_time_limit": cls.PERFORMANCE_SETTINGS["max_processing_time"] <= 60,
            "document_count_valid": 3 <= cls.DOCUMENT_CONSTRAINTS["max_documents"] <= 10
        }
        
        all_valid = all(constraints.values())
        if not all_valid:
            failed = [k for k, v in constraints.items() if not v]
            raise ValueError(f"Challenge constraints failed: {failed}")
        
        return True, constraints

    @classmethod
    def get_challenge_output_format(cls):
        """Return the exact JSON format expected by challenge"""
        return {
            "metadata": {
                "input_documents": [],
                "persona": "",
                "job_to_be_done": "",
                "processing_timestamp": ""
            },
            "extracted_sections": [
                {
                    "document": "",
                    "page_number": 0,
                    "section_title": "",
                    "importance_rank": 0
                }
            ],
            "subsection_analysis": [
                {
                    "document": "",
                    "refined_text": "",
                    "page_number": 0
                }
            ]
        }

def main():
    """Test optimized configuration"""
    config = OptimizedConfig()
    
    # Validate challenge constraints
    try:
        is_valid, constraints = config.validate_challenge_constraints()
        print("[SUCCESS] Challenge Constraints Validation:")
        for constraint, status in constraints.items():
            print(f"   {constraint}: {'[SUCCESS]' if status else '[ERROR]'}")
        
        print(f"\n[REPORT] Model Size: {config.get_total_model_size_mb()}MB / 1024MB")
        print(f"[STOPWATCH]  Max Processing Time: {config.PERFORMANCE_SETTINGS['max_processing_time']}s")
        print(f"[COMPUTER] Device: {config.INFERENCE_SETTINGS['device'].upper()}")
        
    except Exception as e:
        print(f"[ERROR] Constraint validation failed: {e}")

if __name__ == "__main__":
    main()
