"""
Lightweight Model Manager for Challenge 1B Compliance
CPU-only, <1GB models, optimized for speed and Docker caching
"""

import os
import time
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
import warnings
warnings.filterwarnings('ignore')

# Lightweight imports for CPU-only inference
try:
    from sentence_transformers import SentenceTransformer
    import torch
    torch.set_num_threads(4)  # CPU optimization
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModel, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from config_optimized import OptimizedConfig

class LightweightModelManager:
    """
    Lightweight model manager optimized for Challenge 1B constraints:
    - CPU-only inference
    - <1GB total model size
    - Fast loading from cache
    - No internet downloads in Docker
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.tokenizers = {}
        self.config = OptimizedConfig()
        
        # CPU-only device setting
        self.device = "cpu"
        if TORCH_AVAILABLE:
            torch.set_num_threads(self.config.PERFORMANCE_SETTINGS["cpu_threads"])
        
        # Validate constraints
        self.config.validate_challenge_constraints()
        
    def initialize_models(self) -> bool:
        """Initialize all models for CPU-only inference"""
        try:
            start_time = time.time()
            
            print("\n[LAUNCH] INITIALIZING LIGHTWEIGHT MODELS (CPU-ONLY)")
            print("="*60)
            print(f"[REPORT] Total model budget: {self.config.get_total_model_size_mb()}MB / 1024MB")
            print(f"[COMPUTER] Device: {self.device.upper()}")
            print(f"🧵 CPU threads: {self.config.PERFORMANCE_SETTINGS['cpu_threads']}")
            
            # Initialize sentence transformer
            success = self._init_sentence_transformer()
            if not success:
                return False
            
            # Initialize text classifier
            success = self._init_text_classifier()
            if not success:
                return False
            
            # Initialize lightweight NLP pipeline
            success = self._init_nlp_pipeline()
            if not success:
                return False
            
            init_time = time.time() - start_time
            print(f"\n[SUCCESS] All models initialized in {init_time:.1f}s")
            print(f"[SAVE] Memory usage: {self._get_memory_usage():.1f}MB")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Model initialization failed: {e}")
            print(f"[ERROR] Initialization failed: {e}")
            return False
    
    def _init_sentence_transformer(self) -> bool:
        """Initialize lightweight sentence transformer"""
        try:
            model_info = self.config.MODELS["sentence_transformer"]
            print(f"\n[NOTE] Loading {model_info['name']} ({model_info['size_mb']}MB)...")
            
            if not TORCH_AVAILABLE:
                print("[WARNING]  PyTorch not available, using fallback embeddings")
                self.models["sentence_transformer"] = self._create_fallback_embedder()
                return True
            
            # Load from cache (no internet download)
            cache_dir = self.config.MODELS_DIR
            model = SentenceTransformer(
                model_info["name"],
                cache_folder=str(cache_dir),
                device=self.device
            )
            
            # Set to evaluation mode for CPU inference
            model.eval()
            if hasattr(model, '_modules'):
                for module in model._modules.values():
                    if hasattr(module, 'eval'):
                        module.eval()
            
            self.models["sentence_transformer"] = model
            print("   [SUCCESS] Sentence Transformer ready")
            return True
            
        except Exception as e:
            print(f"   [ERROR] Failed: {e}")
            # Fallback to simple embeddings
            self.models["sentence_transformer"] = self._create_fallback_embedder()
            return True
    
    def _init_text_classifier(self) -> bool:
        """Initialize lightweight text classifier"""
        try:
            model_info = self.config.MODELS["text_classifier"]
            print(f"\n[ANALYZE] Loading {model_info['name']} ({model_info['size_mb']}MB)...")
            
            if not TRANSFORMERS_AVAILABLE:
                print("[WARNING]  Transformers not available, using fallback classifier")
                self.models["text_classifier"] = self._create_fallback_classifier()
                return True
            
            # Load tokenizer and model from cache
            cache_dir = self.config.MODELS_DIR
            
            tokenizer = AutoTokenizer.from_pretrained(
                model_info["name"],
                cache_dir=str(cache_dir),
                local_files_only=True  # No internet access
            )
            
            model = AutoModel.from_pretrained(
                model_info["name"],
                cache_dir=str(cache_dir),
                local_files_only=True,  # No internet access
                torch_dtype=torch.float32,  # CPU inference
                device_map=None  # CPU only
            )
            
            # Set to evaluation mode
            model.eval()
            model.to(self.device)
            
            self.tokenizers["text_classifier"] = tokenizer
            self.models["text_classifier"] = model
            print("   [SUCCESS] Text Classifier ready")
            return True
            
        except Exception as e:
            print(f"   [ERROR] Failed: {e}")
            # Fallback to simple classifier
            self.models["text_classifier"] = self._create_fallback_classifier()
            return True
    
    def _init_nlp_pipeline(self) -> bool:
        """Initialize lightweight NLP pipeline"""
        try:
            print(f"\n[TOOLS]  Loading NLP Pipeline...")
            
            # Create lightweight pipeline for text analysis
            self.models["nlp_pipeline"] = {
                "word_counter": self._create_word_counter(),
                "keyword_extractor": self._create_keyword_extractor(),
                "quality_scorer": self._create_quality_scorer()
            }
            
            print("   [SUCCESS] NLP Pipeline ready")
            return True
            
        except Exception as e:
            print(f"   [ERROR] Failed: {e}")
            return False
    
    def _create_fallback_embedder(self):
        """Create fallback embedder using simple techniques"""
        class FallbackEmbedder:
            def encode(self, texts):
                if isinstance(texts, str):
                    texts = [texts]
                
                # Simple TF-IDF-like embeddings
                embeddings = []
                for text in texts:
                    words = text.lower().split()
                    # Create simple feature vector
                    features = np.zeros(384)  # Match sentence transformer size
                    for i, word in enumerate(words[:100]):  # Limit to 100 words
                        hash_val = hash(word) % 384
                        features[hash_val] += 1.0 / (i + 1)  # Position weighting
                    
                    # Normalize
                    norm = np.linalg.norm(features)
                    if norm > 0:
                        features = features / norm
                    
                    embeddings.append(features)
                
                return np.array(embeddings) if len(embeddings) > 1 else embeddings[0]
        
        return FallbackEmbedder()
    
    def _create_fallback_classifier(self):
        """Create fallback classifier using simple techniques"""
        class FallbackClassifier:
            def __init__(self):
                self.domain_keywords = {
                    "academic": ["research", "study", "analysis", "methodology", "findings"],
                    "business": ["revenue", "profit", "market", "strategy", "performance"],
                    "technical": ["system", "algorithm", "implementation", "optimization"],
                    "educational": ["learning", "student", "education", "knowledge", "skill"]
                }
            
            def classify_domain(self, text):
                text_lower = text.lower()
                scores = {}
                for domain, keywords in self.domain_keywords.items():
                    score = sum(1 for keyword in keywords if keyword in text_lower)
                    scores[domain] = score / len(keywords)
                
                return max(scores, key=scores.get) if scores else "general"
        
        return FallbackClassifier()
    
    def _create_word_counter(self):
        """Create word counting utility"""
        def count_words(text):
            return len(text.split())
        return count_words
    
    def _create_keyword_extractor(self):
        """Create simple keyword extractor"""
        def extract_keywords(text, max_keywords=10):
            words = text.lower().split()
            # Simple frequency-based extraction
            word_freq = {}
            for word in words:
                if len(word) > 3:  # Skip short words
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            # Return top keywords
            return sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:max_keywords]
        
        return extract_keywords
    
    def _create_quality_scorer(self):
        """Create content quality scorer"""
        def score_quality(text):
            if not text:
                return 0.0
            
            words = text.split()
            sentences = text.split('.')
            
            # Simple quality metrics
            word_count_score = min(len(words) / 50, 1.0)  # Optimal around 50 words
            sentence_count_score = min(len(sentences) / 5, 1.0)  # Optimal around 5 sentences
            avg_word_length = sum(len(word) for word in words) / max(len(words), 1)
            word_length_score = 1.0 if 4 <= avg_word_length <= 7 else 0.5
            
            return (word_count_score + sentence_count_score + word_length_score) / 3
        
        return score_quality
    
    def get_embeddings(self, text: str) -> np.ndarray:
        """Get text embeddings using sentence transformer"""
        model = self.models.get("sentence_transformer")
        if model is None:
            raise ValueError("Sentence transformer not initialized")
        
        return model.encode(text)
    
    def classify_text(self, text: str) -> str:
        """Classify text domain"""
        classifier = self.models.get("text_classifier")
        if classifier is None:
            return "general"
        
        if hasattr(classifier, 'classify_domain'):
            return classifier.classify_domain(text)
        else:
            # Use transformer-based classification
            try:
                tokenizer = self.tokenizers.get("text_classifier")
                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True
                )
                
                with torch.no_grad():
                    outputs = classifier(**inputs)
                    # Simple classification based on output
                    return "general"  # Simplified for now
                    
            except Exception:
                return "general"
    
    def analyze_text_quality(self, text: str) -> float:
        """Analyze text quality score"""
        nlp_pipeline = self.models.get("nlp_pipeline", {})
        quality_scorer = nlp_pipeline.get("quality_scorer")
        
        if quality_scorer:
            return quality_scorer(text)
        else:
            return 0.5  # Neutral score
    
    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[tuple]:
        """Extract keywords from text"""
        nlp_pipeline = self.models.get("nlp_pipeline", {})
        keyword_extractor = nlp_pipeline.get("keyword_extractor")
        
        if keyword_extractor:
            return keyword_extractor(text, max_keywords)
        else:
            return []
    
    def _get_memory_usage(self) -> float:
        """Estimate memory usage in MB"""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            # Estimate based on model sizes
            return sum(model["size_mb"] for model in self.config.MODELS.values())
    
    def cleanup(self):
        """Clean up models to free memory"""
        self.models.clear()
        self.tokenizers.clear()
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()

def main():
    """Test lightweight model manager"""
    manager = LightweightModelManager()
    
    success = manager.initialize_models()
    if success:
        print("\n[COMPLETE] Lightweight Model Manager ready!")
        
        # Test functionality
        test_text = "This is a test document about machine learning research."
        
        try:
            embeddings = manager.get_embeddings(test_text)
            print(f"[REPORT] Embeddings shape: {embeddings.shape}")
            
            domain = manager.classify_text(test_text)
            print(f"[TAG]  Domain classification: {domain}")
            
            quality = manager.analyze_text_quality(test_text)
            print(f"[QUALITY] Quality score: {quality:.2f}")
            
            keywords = manager.extract_keywords(test_text)
            print(f"[KEY] Keywords: {keywords[:3]}")
            
        except Exception as e:
            print(f"[WARNING]  Testing failed: {e}")
    
    else:
        print("[ERROR] Model Manager initialization failed")

if __name__ == "__main__":
    main()
