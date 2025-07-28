"""
Advanced Ranking System with Edge Case Handling
Implements sophisticated algorithms for section prioritization and edge case management
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import logging
from collections import defaultdict, Counter
import re
import warnings
warnings.filterwarnings('ignore')

@dataclass
class RankingFeatures:
    """Comprehensive features for ranking sections"""
    semantic_relevance: float
    keyword_density: float
    position_bias: float
    content_uniqueness: float
    information_novelty: float
    domain_specificity: float
    readability_score: float
    structural_importance: float
    cross_document_consistency: float
    temporal_relevance: float

@dataclass
class EdgeCaseHandler:
    """Handles various edge cases in document analysis"""
    min_content_length: int = 30
    max_content_length: int = 5000
    duplicate_threshold: float = 0.85
    low_quality_threshold: float = 0.2
    sparse_document_threshold: int = 3

class AdvancedRankingSystem:
    """
    Sophisticated ranking system that handles edge cases and ensures
    robust performance across diverse document types and scenarios
    """
    
    def __init__(self):
        self.setup_logging()
        self.edge_case_handler = EdgeCaseHandler()
        self._initialize_ranking_components()
        
    def setup_logging(self):
        """Setup comprehensive logging"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _initialize_ranking_components(self):
        """Initialize ranking system components"""
        # Weights for different ranking factors
        self.ranking_weights = {
            "semantic_relevance": 0.25,
            "keyword_density": 0.20,
            "content_uniqueness": 0.15,
            "structural_importance": 0.15,
            "information_novelty": 0.10,
            "domain_specificity": 0.08,
            "readability_score": 0.04,
            "position_bias": 0.03
        }
        
        # Edge case detection patterns
        self.edge_case_patterns = {
            "table_of_contents": [r"table of contents", r"contents", r"chapter \d+", r"section \d+"],
            "bibliography": [r"references", r"bibliography", r"works cited", r"citations"],
            "appendix": [r"appendix", r"supplementary", r"additional materials"],
            "header_footer": [r"page \d+", r"chapter \d+", r"copyright", r"©"],
            "figure_captions": [r"figure \d+", r"fig\. \d+", r"table \d+"],
            "mathematical_content": [r"\$.*\$", r"\\[a-zA-Z]+{", r"equation \d+"]
        }
        
        # Quality indicators
        self.quality_indicators = {
            "high_quality": ["methodology", "analysis", "results", "findings", "conclusion", "evidence"],
            "medium_quality": ["description", "overview", "summary", "introduction", "background"],
            "low_quality": ["unclear", "incomplete", "fragmented", "corrupted", "unreadable"]
        }
    
    def rank_sections_advanced(self, 
                              sections_with_scores: List[Tuple[Any, str, Any]],
                              persona_profile: Any,
                              job_requirements: Any) -> List[Tuple[Any, str, Any, Dict[str, float]]]:
        """
        Advanced ranking with comprehensive edge case handling
        """
        try:
            self.logger.info(f"Starting advanced ranking of {len(sections_with_scores)} sections")
            
            # Handle edge cases first
            filtered_sections = self._handle_edge_cases(sections_with_scores)
            
            if not filtered_sections:
                self.logger.warning("No valid sections after edge case filtering")
                return []
            
            # Calculate advanced ranking features for each section
            ranked_sections = []
            
            for section, doc_title, base_score in filtered_sections:
                try:
                    # Calculate comprehensive ranking features
                    features = self._calculate_ranking_features(
                        section, doc_title, base_score, 
                        filtered_sections, persona_profile, job_requirements
                    )
                    
                    # Calculate final ranking score
                    final_score = self._calculate_final_ranking_score(features, base_score)
                    
                    # Add to ranked list with features for transparency
                    ranked_sections.append((
                        section, doc_title, base_score, {
                            "final_score": final_score,
                            "ranking_features": features.__dict__,
                            "edge_case_flags": self._detect_edge_case_flags(section)
                        }
                    ))
                    
                except Exception as e:
                    self.logger.warning(f"Failed to rank section '{section.section_title}': {e}")
                    # Include with minimal score to avoid losing content
                    ranked_sections.append((
                        section, doc_title, base_score, {
                            "final_score": base_score.overall_score * 0.5,
                            "ranking_features": {},
                            "edge_case_flags": ["ranking_error"]
                        }
                    ))
            
            # Sort by final ranking score
            ranked_sections.sort(key=lambda x: x[3]["final_score"], reverse=True)
            
            # Apply diversity filter to ensure variety in top results
            ranked_sections = self._apply_diversity_filter(ranked_sections)
            
            self.logger.info(f"Advanced ranking completed: {len(ranked_sections)} sections ranked")
            return ranked_sections
            
        except Exception as e:
            self.logger.error(f"Advanced ranking failed: {e}")
            # Fallback to basic ranking
            return [(s, d, score, {"final_score": score.overall_score, "ranking_features": {}, "edge_case_flags": ["fallback"]}) 
                   for s, d, score in sections_with_scores]
    
    def _handle_edge_cases(self, sections_with_scores: List[Tuple[Any, str, Any]]) -> List[Tuple[Any, str, Any]]:
        """Comprehensive edge case handling"""
        filtered_sections = []
        edge_case_stats = defaultdict(int)
        
        for section, doc_title, score in sections_with_scores:
            try:
                # Skip if content is too short or too long
                if not self._validate_content_length(section.content):
                    edge_case_stats["invalid_length"] += 1
                    continue
                
                # Skip if content quality is too low
                if not self._validate_content_quality(section.content):
                    edge_case_stats["low_quality"] += 1
                    continue
                
                # Skip if section is detected as metadata/navigation
                if self._is_metadata_section(section):
                    edge_case_stats["metadata"] += 1
                    continue
                
                # Skip if section is largely mathematical/formulaic without context
                if self._is_pure_mathematical_content(section.content):
                    edge_case_stats["mathematical_only"] += 1
                    continue
                
                # Skip if section appears to be corrupted/garbled text
                if self._is_corrupted_text(section.content):
                    edge_case_stats["corrupted"] += 1
                    continue
                
                filtered_sections.append((section, doc_title, score))
                
            except Exception as e:
                self.logger.warning(f"Error handling edge case for section: {e}")
                # Include in results with warning
                filtered_sections.append((section, doc_title, score))
        
        # Log edge case statistics
        if edge_case_stats:
            self.logger.info(f"Edge cases handled: {dict(edge_case_stats)}")
        
        return filtered_sections
    
    def _validate_content_length(self, content: str) -> bool:
        """Validate content length is within acceptable bounds"""
        length = len(content.strip())
        return self.edge_case_handler.min_content_length <= length <= self.edge_case_handler.max_content_length
    
    def _validate_content_quality(self, content: str) -> bool:
        """Validate content quality using multiple heuristics"""
        if not content or not content.strip():
            return False
        
        # Check for minimum word count
        words = content.split()
        if len(words) < 5:
            return False
        
        # Check for reasonable character-to-word ratio
        avg_word_length = len(content) / len(words)
        if avg_word_length < 2 or avg_word_length > 20:
            return False
        
        # Check for excessive special characters
        special_char_ratio = len(re.findall(r'[^\w\s\.,;:!?\-\(\)]', content)) / len(content)
        if special_char_ratio > 0.3:
            return False
        
        # Check for readability indicators
        readable_chars = len(re.findall(r'[a-zA-Z0-9\s\.,;:!?\-\(\)]', content))
        readability_ratio = readable_chars / len(content)
        if readability_ratio < 0.7:
            return False
        
        return True
    
    def _is_metadata_section(self, section: Any) -> bool:
        """Detect if section is metadata, navigation, or administrative content"""
        title_lower = section.section_title.lower()
        content_lower = section.content.lower()
        
        # Check against known metadata patterns
        for pattern_type, patterns in self.edge_case_patterns.items():
            for pattern in patterns:
                if re.search(pattern, title_lower) or re.search(pattern, content_lower):
                    return True
        
        # Additional heuristics
        if len(section.content.split()) < 20 and any(word in title_lower for word in ["page", "chapter", "section"]):
            return True
        
        return False
    
    def _is_pure_mathematical_content(self, content: str) -> bool:
        """Detect sections that are purely mathematical without explanatory context"""
        # Count mathematical notation
        math_patterns = [r'\$.*?\$', r'\\[a-zA-Z]+\{', r'[=<>≤≥±∑∏∫]', r'\d+[\+\-\*/]\d+']
        math_count = sum(len(re.findall(pattern, content)) for pattern in math_patterns)
        
        # Count regular words
        words = re.findall(r'\b[a-zA-Z]{3,}\b', content)
        word_count = len(words)
        
        # If more than 50% mathematical content and very few explanatory words
        if word_count > 0:
            math_ratio = math_count / (math_count + word_count)
            return math_ratio > 0.5 and word_count < 20
        
        return False
    
    def _is_corrupted_text(self, content: str) -> bool:
        """Detect corrupted or garbled text"""
        # Check for excessive repetitive characters
        repetitive_pattern = r'(.)\1{5,}'
        if re.search(repetitive_pattern, content):
            return True
        
        # Check for excessive random characters
        words = content.split()
        if len(words) > 10:
            unrecognizable_words = sum(1 for word in words if not re.match(r'^[a-zA-Z0-9\.,;:!?\-\(\)]+$', word))
            if unrecognizable_words / len(words) > 0.3:
                return True
        
        return False
    
    def _calculate_ranking_features(self, 
                                  section: Any, 
                                  doc_title: str, 
                                  base_score: Any,
                                  all_sections: List[Tuple[Any, str, Any]],
                                  persona_profile: Any,
                                  job_requirements: Any) -> RankingFeatures:
        """Calculate comprehensive ranking features"""
        
        return RankingFeatures(
            semantic_relevance=base_score.overall_score,
            keyword_density=self._calculate_keyword_density(section, persona_profile, job_requirements),
            position_bias=self._calculate_position_bias(section),
            content_uniqueness=self._calculate_content_uniqueness(section, all_sections),
            information_novelty=self._calculate_information_novelty(section, persona_profile),
            domain_specificity=self._calculate_domain_specificity(section, persona_profile),
            readability_score=self._calculate_readability_score(section.content),
            structural_importance=self._calculate_structural_importance(section),
            cross_document_consistency=self._calculate_cross_document_consistency(section, all_sections),
            temporal_relevance=self._calculate_temporal_relevance(section, job_requirements)
        )
    
    def _calculate_keyword_density(self, section: Any, persona_profile: Any, job_requirements: Any) -> float:
        """Calculate keyword density score"""
        content_lower = section.content.lower()
        title_lower = section.section_title.lower()
        
        # Combine persona keywords and job keywords
        all_keywords = getattr(persona_profile, 'focus_keywords', []) + getattr(job_requirements, 'priority_aspects', [])
        
        if not all_keywords:
            return 0.5  # Neutral score if no keywords available
        
        # Count keyword occurrences
        keyword_count = sum(1 for keyword in all_keywords if keyword in content_lower or keyword in title_lower)
        
        # Normalize by content length and keyword count
        content_words = len(content_lower.split())
        density = (keyword_count / len(all_keywords)) * (100 / max(content_words, 100))
        
        return min(density, 1.0)
    
    def _calculate_position_bias(self, section: Any) -> float:
        """Calculate position-based importance bias"""
        # Sections at beginning or end of documents often more important
        page_num = getattr(section, 'page_number', 1)
        
        if page_num <= 3:  # Early pages
            return 0.8
        elif page_num >= 10:  # Later pages (might be conclusions)
            return 0.6
        else:
            return 0.5  # Middle pages
    
    def _calculate_content_uniqueness(self, section: Any, all_sections: List[Tuple[Any, str, Any]]) -> float:
        """Calculate how unique this section's content is"""
        section_words = set(section.content.lower().split())
        
        if not section_words:
            return 0.0
        
        # Compare with other sections
        max_overlap = 0.0
        for other_section, _, _ in all_sections:
            if other_section == section:
                continue
            
            other_words = set(other_section.content.lower().split())
            if other_words:
                overlap = len(section_words.intersection(other_words)) / len(section_words.union(other_words))
                max_overlap = max(max_overlap, overlap)
        
        return 1.0 - max_overlap
    
    def _calculate_information_novelty(self, section: Any, persona_profile: Any) -> float:
        """Calculate how novel the information is for the persona"""
        content_lower = section.content.lower()
        
        # Look for indicators of novel information
        novelty_indicators = [
            "new", "novel", "recent", "latest", "breakthrough", "discovery",
            "innovative", "advancement", "emerging", "cutting-edge"
        ]
        
        novelty_count = sum(1 for indicator in novelty_indicators if indicator in content_lower)
        
        # Normalize by content length
        content_words = len(content_lower.split())
        novelty_score = novelty_count / max(content_words / 100, 1)
        
        return min(novelty_score, 1.0)
    
    def _calculate_domain_specificity(self, section: Any, persona_profile: Any) -> float:
        """Calculate domain-specific relevance"""
        content_lower = section.content.lower()
        
        # Get domain-specific terms based on persona
        domain_terms = getattr(persona_profile, 'domain_knowledge', [])
        
        if not domain_terms:
            return 0.5  # Neutral if no domain info
        
        # Count domain-specific term occurrences
        domain_count = sum(1 for term in domain_terms if term.lower() in content_lower)
        
        return min(domain_count / len(domain_terms), 1.0)
    
    def _calculate_readability_score(self, content: str) -> float:
        """Calculate content readability score"""
        if not content:
            return 0.0
        
        words = content.split()
        sentences = re.split(r'[.!?]+', content)
        
        if not words or not sentences:
            return 0.0
        
        # Average words per sentence (complexity indicator)
        avg_words_per_sentence = len(words) / len([s for s in sentences if s.strip()])
        
        # Optimal range is 15-20 words per sentence
        if 15 <= avg_words_per_sentence <= 20:
            sentence_score = 1.0
        elif avg_words_per_sentence < 15:
            sentence_score = avg_words_per_sentence / 15
        else:
            sentence_score = max(0.3, 20 / avg_words_per_sentence)
        
        # Average word length (complexity indicator)
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # Optimal range is 4-7 characters per word
        if 4 <= avg_word_length <= 7:
            word_score = 1.0
        elif avg_word_length < 4:
            word_score = avg_word_length / 4
        else:
            word_score = max(0.3, 7 / avg_word_length)
        
        return (sentence_score + word_score) / 2
    
    def _calculate_structural_importance(self, section: Any) -> float:
        """Calculate structural importance based on section characteristics"""
        title = section.section_title.lower()
        
        # High importance indicators
        high_importance = ["abstract", "summary", "conclusion", "results", "findings"]
        medium_importance = ["introduction", "methodology", "discussion", "analysis"]
        low_importance = ["references", "appendix", "acknowledgments"]
        
        if any(keyword in title for keyword in high_importance):
            return 1.0
        elif any(keyword in title for keyword in medium_importance):
            return 0.7
        elif any(keyword in title for keyword in low_importance):
            return 0.3
        else:
            return 0.5  # Neutral for unknown section types
    
    def _calculate_cross_document_consistency(self, section: Any, all_sections: List[Tuple[Any, str, Any]]) -> float:
        """Calculate consistency across documents (higher for concepts appearing in multiple docs)"""
        section_doc = section.document_path
        section_keywords = set(re.findall(r'\b\w{4,}\b', section.content.lower()))
        
        if not section_keywords:
            return 0.0
        
        # Count how many other documents contain similar keywords
        other_docs = set()
        for other_section, _, _ in all_sections:
            if other_section.document_path != section_doc:
                other_keywords = set(re.findall(r'\b\w{4,}\b', other_section.content.lower()))
                overlap = len(section_keywords.intersection(other_keywords))
                if overlap > len(section_keywords) * 0.2:  # 20% keyword overlap threshold
                    other_docs.add(other_section.document_path)
        
        # Normalize by total number of other documents
        total_other_docs = len(set(s[0].document_path for s in all_sections if s[0].document_path != section_doc))
        
        if total_other_docs == 0:
            return 1.0  # Single document case
        
        return len(other_docs) / total_other_docs
    
    def _calculate_temporal_relevance(self, section: Any, job_requirements: Any) -> float:
        """Calculate temporal relevance based on job urgency and content freshness"""
        content_lower = section.content.lower()
        
        # Look for temporal indicators
        recent_indicators = ["2023", "2024", "recent", "current", "latest", "new"]
        dated_indicators = ["2020", "2019", "2018", "old", "previous", "former"]
        
        recent_count = sum(1 for indicator in recent_indicators if indicator in content_lower)
        dated_count = sum(1 for indicator in dated_indicators if indicator in content_lower)
        
        # Base score
        if recent_count > dated_count:
            temporal_score = 0.8
        elif dated_count > recent_count:
            temporal_score = 0.4
        else:
            temporal_score = 0.6
        
        # Adjust based on job urgency
        urgency = getattr(job_requirements, 'urgency_level', 'normal')
        if urgency == 'high' and recent_count > 0:
            temporal_score += 0.2
        
        return min(temporal_score, 1.0)
    
    def _calculate_final_ranking_score(self, features: RankingFeatures, base_score: Any) -> float:
        """Calculate final ranking score using weighted combination"""
        feature_scores = {
            "semantic_relevance": features.semantic_relevance,
            "keyword_density": features.keyword_density,
            "content_uniqueness": features.content_uniqueness,
            "structural_importance": features.structural_importance,
            "information_novelty": features.information_novelty,
            "domain_specificity": features.domain_specificity,
            "readability_score": features.readability_score,
            "position_bias": features.position_bias
        }
        
        # Calculate weighted sum
        final_score = sum(
            self.ranking_weights[feature] * score 
            for feature, score in feature_scores.items()
            if feature in self.ranking_weights
        )
        
        # Boost based on confidence from base score
        confidence_boost = getattr(base_score, 'confidence_level', 0.5) * 0.1
        final_score += confidence_boost
        
        return min(final_score, 1.0)
    
    def _detect_edge_case_flags(self, section: Any) -> List[str]:
        """Detect and flag various edge cases"""
        flags = []
        
        content = section.content
        title = section.section_title
        
        # Content length flags
        if len(content) < 100:
            flags.append("short_content")
        elif len(content) > 3000:
            flags.append("long_content")
        
        # Quality flags
        if not self._validate_content_quality(content):
            flags.append("low_quality")
        
        # Special content type flags
        if self._is_metadata_section(section):
            flags.append("metadata")
        
        if self._is_pure_mathematical_content(content):
            flags.append("mathematical")
        
        # Structure flags
        if not title or len(title.strip()) < 3:
            flags.append("poor_title")
        
        return flags
    
    def _apply_diversity_filter(self, ranked_sections: List[Tuple[Any, str, Any, Dict[str, float]]]) -> List[Tuple[Any, str, Any, Dict[str, float]]]:
        """Apply diversity filter to ensure variety in top results"""
        if len(ranked_sections) <= 10:
            return ranked_sections
        
        diverse_sections = []
        seen_documents = set()
        seen_section_types = set()
        
        # First pass: ensure document diversity
        for section_data in ranked_sections:
            section, doc_title, base_score, ranking_info = section_data
            doc_name = getattr(section, 'document_path', doc_title)
            
            if len(diverse_sections) < 15:  # Take top 15 with diversity
                if doc_name not in seen_documents or len(seen_documents) < 3:
                    diverse_sections.append(section_data)
                    seen_documents.add(doc_name)
                    seen_section_types.add(section.section_title.lower())
            else:
                break
        
        # Second pass: fill remaining slots with highest scoring sections
        remaining_slots = 20 - len(diverse_sections)
        for section_data in ranked_sections:
            if remaining_slots <= 0:
                break
            if section_data not in diverse_sections:
                diverse_sections.append(section_data)
                remaining_slots -= 1
        
        return diverse_sections

def main():
    """Test the advanced ranking system"""
    ranking_system = AdvancedRankingSystem()
    print("Advanced Ranking System initialized successfully")

if __name__ == "__main__":
    main()
