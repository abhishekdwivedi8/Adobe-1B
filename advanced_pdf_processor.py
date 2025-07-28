"""
Advanced PDF Processing Pipeline for Maximum Accuracy
Handles all types of PDFs from simple to highly complex
"""

import fitz  # PyMuPDF
import pdfplumber
import re
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from pathlib import Path

@dataclass
class DocumentSection:
    """Comprehensive document section representation"""
    document_path: str
    page_number: int
    section_title: str
    content: str
    start_pos: Tuple[int, int]
    end_pos: Tuple[int, int]
    font_info: Dict[str, Any]
    section_level: int
    subsections: List['DocumentSection']
    metadata: Dict[str, Any]

@dataclass
class ProcessedDocument:
    """Complete processed document with all extracted information"""
    path: str
    title: str
    sections: List[DocumentSection]
    tables: List[Dict[str, Any]]
    figures: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    text_stats: Dict[str, Any]

class AdvancedPDFProcessor:
    """
    Highly advanced PDF processor that handles all types of documents
    with maximum accuracy and robustness
    """
    
    def __init__(self):
        self.setup_logging()
        self.section_patterns = self._compile_section_patterns()
        
    def setup_logging(self):
        """Setup comprehensive logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _compile_section_patterns(self) -> List[re.Pattern]:
        """Compile comprehensive section detection patterns"""
        patterns = [
            # Academic paper patterns
            r'^(?:Abstract|Introduction|Literature Review|Methodology|Results|Discussion|Conclusion|References)(?:\s*[:.]?\s*$)',
            r'^\d+\.?\s+[A-Z][^.]*$',  # Numbered sections
            r'^[A-Z][A-Z\s]+$',  # ALL CAPS headers
            
            # Business document patterns
            r'^(?:Executive Summary|Overview|Financial|Revenue|Expenses|Analysis|Recommendations)(?:\s*[:.]?\s*$)',
            r'^(?:Q[1-4]|Quarter|Annual|Year)\s+\d{4}',  # Financial periods
            
            # Technical document patterns
            r'^(?:Algorithm|Implementation|Architecture|Design|System|Performance|Evaluation)(?:\s*[:.]?\s*$)',
            
            # Educational content patterns
            r'^(?:Chapter|Section|Unit|Lesson|Exercise|Problem)\s+\d+',
            r'^(?:Definition|Theorem|Proof|Example|Solution)(?:\s*[:.]?\s*$)',
            
            # Legal document patterns
            r'^(?:Article|Section|Clause|Paragraph)\s+[IVX\d]+',
            
            # General patterns
            r'^\s*[A-Z][a-z]*(?:\s+[A-Z][a-z]*)*\s*$',  # Title case
            r'^\d+(?:\.\d+)*\s+[A-Z]',  # Decimal numbering
            r'^[IVX]+\.\s+[A-Z]',  # Roman numerals
        ]
        
        return [re.compile(pattern, re.MULTILINE | re.IGNORECASE) for pattern in patterns]
    
    def process_document(self, pdf_path: str) -> ProcessedDocument:
        """
        Process a single PDF document with maximum extraction accuracy
        """
        try:
            self.logger.info(f"Processing document: {pdf_path}")
            
            # Multi-library approach for maximum text extraction
            pymupdf_data = self._extract_with_pymupdf(pdf_path)
            pdfplumber_data = self._extract_with_pdfplumber(pdf_path)
            
            # Combine and validate extracted data
            combined_text = self._combine_extractions(pymupdf_data, pdfplumber_data)
            
            # Extract document structure
            sections = self._extract_sections(combined_text, pdf_path)
            tables = self._extract_tables(pdfplumber_data)
            figures = self._extract_figures(pymupdf_data)
            
            # Generate comprehensive metadata
            metadata = self._generate_metadata(pdf_path, combined_text)
            text_stats = self._calculate_text_statistics(combined_text)
            
            # Extract document title
            title = self._extract_document_title(combined_text, metadata)
            
            processed_doc = ProcessedDocument(
                path=pdf_path,
                title=title,
                sections=sections,
                tables=tables,
                figures=figures,
                metadata=metadata,
                text_stats=text_stats
            )
            
            self.logger.info(f"Successfully processed {pdf_path}: {len(sections)} sections found")
            return processed_doc
            
        except Exception as e:
            self.logger.error(f"Error processing {pdf_path}: {str(e)}")
            raise
    
    def _extract_with_pymupdf(self, pdf_path: str) -> Dict[str, Any]:
        """Extract data using PyMuPDF with advanced settings"""
        doc = fitz.open(pdf_path)
        pages_data = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            
            # Extract text with detailed formatting
            blocks = page.get_text("dict")
            text = page.get_text()
            
            # Extract fonts and styles
            fonts = set()
            for block in blocks["blocks"]:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            fonts.add((span["font"], span["size"], span["flags"]))
            
            pages_data.append({
                "page_num": page_num + 1,
                "text": text,
                "blocks": blocks,
                "fonts": list(fonts)
            })
        
        doc.close()
        return {"pages": pages_data}
    
    def _extract_with_pdfplumber(self, pdf_path: str) -> Dict[str, Any]:
        """Extract data using pdfplumber for advanced table/layout detection"""
        pages_data = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                # Extract text with precise positioning
                text = page.extract_text(x_tolerance=2, y_tolerance=2)
                
                # Extract tables with advanced settings
                tables = page.extract_tables(
                    table_settings={
                        "vertical_strategy": "lines_strict",
                        "horizontal_strategy": "lines_strict",
                        "snap_tolerance": 3,
                        "join_tolerance": 3
                    }
                )
                
                # Extract characters for detailed analysis
                chars = page.chars
                
                pages_data.append({
                    "page_num": page_num + 1,
                    "text": text,
                    "tables": tables,
                    "chars": chars,
                    "width": page.width,
                    "height": page.height
                })
        
        return {"pages": pages_data}
    
    def _combine_extractions(self, pymupdf_data: Dict, pdfplumber_data: Dict) -> str:
        """Intelligently combine text from multiple extraction methods"""
        combined_text = ""
        
        for i, (pymupdf_page, pdfplumber_page) in enumerate(zip(
            pymupdf_data["pages"], pdfplumber_data["pages"]
        )):
            # Choose best extraction based on content quality
            pymupdf_text = pymupdf_page["text"].strip()
            pdfplumber_text = pdfplumber_page["text"].strip() if pdfplumber_page["text"] else ""
            
            # Quality metrics
            pymupdf_score = self._calculate_text_quality(pymupdf_text)
            pdfplumber_score = self._calculate_text_quality(pdfplumber_text)
            
            # Use higher quality extraction
            if pdfplumber_score > pymupdf_score and pdfplumber_text:
                page_text = pdfplumber_text
            else:
                page_text = pymupdf_text
            
            # Add page marker
            combined_text += f"\n\n--- PAGE {i + 1} ---\n\n{page_text}"
        
        return combined_text
    
    def _calculate_text_quality(self, text: str) -> float:
        """Calculate text extraction quality score"""
        if not text:
            return 0.0
        
        # Count readable characters vs total
        readable_chars = len(re.findall(r'[a-zA-Z0-9\s\.,;:!?\-\(\)]', text))
        total_chars = len(text)
        readability = readable_chars / total_chars if total_chars > 0 else 0
        
        # Penalize excessive whitespace or strange characters
        whitespace_ratio = len(re.findall(r'\s', text)) / total_chars if total_chars > 0 else 0
        strange_chars = len(re.findall(r'[^\w\s\.,;:!?\-\(\)]', text))
        
        score = readability - (whitespace_ratio * 0.3) - (strange_chars / total_chars * 0.5)
        return max(0, score)
    
    def _extract_sections(self, text: str, pdf_path: str) -> List[DocumentSection]:
        """Extract document sections with advanced pattern matching"""
        sections = []
        lines = text.split('\n')
        current_section = None
        section_content = []
        page_num = 1
        
        for i, line in enumerate(lines):
            # Track page numbers
            page_match = re.match(r'--- PAGE (\d+) ---', line)
            if page_match:
                page_num = int(page_match.group(1))
                continue
            
            # Check if line is a section header
            is_header, level = self._is_section_header(line, i, lines)
            
            if is_header:
                # Save previous section
                if current_section and section_content:
                    current_section.content = '\n'.join(section_content).strip()
                    sections.append(current_section)
                
                # Start new section
                current_section = DocumentSection(
                    document_path=pdf_path,
                    page_number=page_num,
                    section_title=line.strip(),
                    content="",
                    start_pos=(i, 0),
                    end_pos=(i, len(line)),
                    font_info={},
                    section_level=level,
                    subsections=[],
                    metadata={}
                )
                section_content = []
            else:
                # Add to current section content
                if line.strip():
                    section_content.append(line)
        
        # Save final section
        if current_section and section_content:
            current_section.content = '\n'.join(section_content).strip()
            sections.append(current_section)
        
        # If no sections found, create one for entire document
        if not sections:
            sections.append(DocumentSection(
                document_path=pdf_path,
                page_number=1,
                section_title="Document Content",
                content=text,
                start_pos=(0, 0),
                end_pos=(len(lines), 0),
                font_info={},
                section_level=1,
                subsections=[],
                metadata={"auto_generated": True}
            ))
        
        return sections
    
    def _is_section_header(self, line: str, line_idx: int, all_lines: List[str]) -> Tuple[bool, int]:
        """Determine if a line is a section header using multiple heuristics"""
        line = line.strip()
        
        if not line or len(line) < 3:
            return False, 0
        
        # Check against compiled patterns
        for pattern in self.section_patterns:
            if pattern.match(line):
                return True, self._determine_section_level(line)
        
        # Additional heuristics
        
        # Check if line is all caps and short
        if line.isupper() and len(line) < 100:
            return True, 1
        
        # Check if line ends without punctuation (typical header)
        if not line.endswith(('.', '!', '?', ';', ',')):
            # Check if next line is empty or starts with content
            if line_idx + 1 < len(all_lines):
                next_line = all_lines[line_idx + 1].strip()
                if not next_line or (next_line and not next_line[0].isupper()):
                    # Additional checks for header characteristics
                    if len(line.split()) <= 10 and any(word[0].isupper() for word in line.split()):
                        return True, self._determine_section_level(line)
        
        return False, 0
    
    def _determine_section_level(self, header: str) -> int:
        """Determine the hierarchical level of a section header"""
        # Check for numbering patterns
        if re.match(r'^\d+\.', header):
            return 1
        elif re.match(r'^\d+\.\d+', header):
            return 2
        elif re.match(r'^\d+\.\d+\.\d+', header):
            return 3
        
        # Check for Roman numerals
        if re.match(r'^[IVX]+\.', header):
            return 1
        
        # Check for lettered sections
        if re.match(r'^[A-Z]\.', header):
            return 2
        
        # Default based on formatting
        if header.isupper():
            return 1
        elif header.istitle():
            return 2
        else:
            return 3
    
    def _extract_tables(self, pdfplumber_data: Dict) -> List[Dict[str, Any]]:
        """Extract and process tables from pdfplumber data"""
        tables = []
        
        for page_data in pdfplumber_data["pages"]:
            if page_data["tables"]:
                for table_idx, table in enumerate(page_data["tables"]):
                    if table and len(table) > 0:
                        tables.append({
                            "page_number": page_data["page_num"],
                            "table_index": table_idx,
                            "data": table,
                            "rows": len(table),
                            "columns": len(table[0]) if table[0] else 0
                        })
        
        return tables
    
    def _extract_figures(self, pymupdf_data: Dict) -> List[Dict[str, Any]]:
        """Extract figure information from PyMuPDF data"""
        figures = []
        
        for page_data in pymupdf_data["pages"]:
            # Count image blocks as figures
            image_count = 0
            for block in page_data["blocks"]["blocks"]:
                if block.get("type") == 1:  # Image block
                    image_count += 1
                    figures.append({
                        "page_number": page_data["page_num"],
                        "figure_index": image_count,
                        "type": "image",
                        "bbox": block.get("bbox", [])
                    })
        
        return figures
    
    def _generate_metadata(self, pdf_path: str, text: str) -> Dict[str, Any]:
        """Generate comprehensive document metadata"""
        return {
            "file_path": pdf_path,
            "file_name": Path(pdf_path).name,
            "word_count": len(text.split()),
            "character_count": len(text),
            "estimated_pages": len(re.findall(r'--- PAGE \d+ ---', text)),
            "language": self._detect_language(text),
            "document_type": self._classify_document_type(text)
        }
    
    def _detect_language(self, text: str) -> str:
        """Simple language detection based on common words"""
        # Basic English detection
        english_words = ['the', 'and', 'or', 'of', 'to', 'in', 'a', 'is', 'that', 'for']
        word_count = len(text.split())
        english_count = sum(1 for word in english_words if word in text.lower())
        
        if word_count > 0 and english_count / len(english_words) > 0.5:
            return "english"
        return "unknown"
    
    def _classify_document_type(self, text: str) -> str:
        """Classify document type based on content patterns"""
        text_lower = text.lower()
        
        # Academic patterns
        academic_keywords = ['abstract', 'methodology', 'references', 'citation', 'hypothesis']
        if sum(1 for kw in academic_keywords if kw in text_lower) >= 2:
            return "academic"
        
        # Business patterns
        business_keywords = ['revenue', 'profit', 'financial', 'quarterly', 'annual report']
        if sum(1 for kw in business_keywords if kw in text_lower) >= 2:
            return "business"
        
        # Technical patterns
        tech_keywords = ['algorithm', 'implementation', 'system', 'architecture', 'performance']
        if sum(1 for kw in tech_keywords if kw in text_lower) >= 2:
            return "technical"
        
        # Educational patterns
        edu_keywords = ['chapter', 'exercise', 'problem', 'solution', 'example']
        if sum(1 for kw in edu_keywords if kw in text_lower) >= 2:
            return "educational"
        
        return "general"
    
    def _calculate_text_statistics(self, text: str) -> Dict[str, Any]:
        """Calculate comprehensive text statistics"""
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        
        return {
            "word_count": len(words),
            "sentence_count": len([s for s in sentences if s.strip()]),
            "average_words_per_sentence": len(words) / max(1, len(sentences)),
            "character_count": len(text),
            "unique_words": len(set(word.lower() for word in words)),
            "vocabulary_diversity": len(set(word.lower() for word in words)) / max(1, len(words))
        }
    
    def _extract_document_title(self, text: str, metadata: Dict) -> str:
        """Extract document title using multiple heuristics"""
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        # Remove page markers
        lines = [line for line in lines if not re.match(r'--- PAGE \d+ ---', line)]
        
        if not lines:
            return metadata.get("file_name", "Untitled Document")
        
        # First substantial line is often the title
        for line in lines[:10]:  # Check first 10 lines
            if len(line) > 10 and len(line) < 200:
                # Check if it looks like a title
                if not line.endswith('.') and len(line.split()) >= 3:
                    return line
        
        # Fallback to first line
        return lines[0] if lines else metadata.get("file_name", "Untitled Document")
