#!/usr/bin/env python3
"""
Main Entry Point for Enhanced Document Intelligence System
Single file execution that runs the complete enhanced analysis
"""

import sys
import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import functions from run_enhanced_analysis
try:
    from run_enhanced_analysis import (
        analyze_current_issues,
        demonstrate_enhanced_components,
        generate_enhanced_output_example,
        create_implementation_summary,
        main as run_enhanced_main
    )
except ImportError as e:
    print(f"Error importing run_enhanced_analysis: {e}")
    sys.exit(1)

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('logs/system.log', mode='w')
        ]
    )
    return logging.getLogger(__name__)

def ensure_directories():
    """Ensure required directories exist"""
    directories = ['input', 'output', 'logs', 'models']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)

def run_complete_analysis():
    """Run the complete enhanced analysis pipeline"""
    logger = setup_logging()
    
    print("=" * 80)
    print("ENHANCED DOCUMENT INTELLIGENCE SYSTEM")
    print("Single Entry Point - Complete Analysis Pipeline")
    print("=" * 80)
    
    start_time = time.time()
    
    try:
        # Ensure directories exist
        ensure_directories()
        logger.info("Directory structure verified")
        
        # Step 1: Check if current analysis exists, if not create sample
        current_output_path = "output/analysis_results.json"
        if not Path(current_output_path).exists():
            logger.info("Creating sample analysis results for demonstration")
            sample_data = create_sample_analysis_data()
            with open(current_output_path, 'w', encoding='utf-8') as f:
                json.dump(sample_data, f, indent=2, ensure_ascii=False)
        
        # Step 2: Run the enhanced analysis demonstration
        logger.info("Starting enhanced analysis demonstration")
        
        # Analyze current issues
        print("\n[STEP 1] Analyzing current system issues...")
        current_analysis = analyze_current_issues(current_output_path)
        
        # Demonstrate enhanced components
        print("\n[STEP 2] Demonstrating enhanced components...")
        enhancement_results = demonstrate_enhanced_components()
        
        # Generate enhanced output example
        print("\n[STEP 3] Generating enhanced output example...")
        enhanced_example = generate_enhanced_output_example()
        
        # Create implementation summary
        print("\n[STEP 4] Creating implementation summary...")
        create_implementation_summary()
        
        # Step 3: Save results
        results = {
            "execution_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "processing_time_seconds": time.time() - start_time,
            "current_analysis": current_analysis,
            "enhancement_results": enhancement_results,
            "enhanced_example": enhanced_example,
            "status": "completed_successfully"
        }
        
        # Save comprehensive results
        output_file = "output/complete_enhanced_analysis.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Save enhanced example separately
        enhanced_output_file = "output/enhanced_analysis_example.json"
        with open(enhanced_output_file, 'w', encoding='utf-8') as f:
            json.dump(enhanced_example, f, indent=2, ensure_ascii=False)
        
        # Final summary
        print("\n" + "=" * 80)
        print("EXECUTION COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"Total processing time: {time.time() - start_time:.2f} seconds")
        print(f"Results saved to: {output_file}")
        print(f"Enhanced example saved to: {enhanced_output_file}")
        
        # Display key metrics
        if 'overall_improvement' in enhancement_results:
            accuracy = enhancement_results['overall_improvement']['overall_accuracy']
            print(f"Enhanced system accuracy: {accuracy:.1%}")
            if accuracy >= 0.9:
                print("[SUCCESS] 90%+ accuracy target achieved!")
            else:
                print("[INFO] Significant improvements demonstrated")
        
        print("=" * 80)
        
        return results
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        print(f"\n[ERROR] Analysis failed: {e}")
        return {"status": "failed", "error": str(e)}

def create_sample_analysis_data():
    """Create sample analysis data for demonstration"""
    return {
        "metadata": {
            "input_documents": [
                "Learn Acrobat - Fill and Sign.pdf",
                "Learn Acrobat - Create and Convert_1.pdf",
                "Learn Acrobat - Request e-signatures_1.pdf"
            ],
            "persona": "HR professional",
            "job_to_be_done": "Create and manage fillable forms for onboarding and compliance",
            "processing_timestamp": "2025-01-28T10:00:00.000000"
        },
        "extracted_sections": [
            {
                "document": "Learn Acrobat - Create and Convert_1.pdf",
                "section_title": "You can create multiple PDFs from multiple native files, including files of different supported",
                "importance_rank": 1,
                "page_number": 12,
                "relevance_score": 0.85,
                "confidence": 0.78
            },
            {
                "document": "Learn Acrobat - Fill and Sign.pdf",
                "section_title": "4. The form fields are detected automatically. Hover the mouse over a field to display a",
                "importance_rank": 2,
                "page_number": 2,
                "relevance_score": 0.72,
                "confidence": 0.65
            },
            {
                "document": "Learn Acrobat - Fill and Sign.pdf",
                "section_title": "other form fields. The suggestions appear in a pop-up menu, from which you can select a",
                "importance_rank": 3,
                "page_number": 8,
                "relevance_score": 0.68,
                "confidence": 0.61
            },
            {
                "document": "Learn Acrobat - Request e-signatures_1.pdf",
                "section_title": "1. From the left panel, select Fill in form fields",
                "importance_rank": 4,
                "page_number": 5,
                "relevance_score": 0.64,
                "confidence": 0.58
            },
            {
                "document": "Learn Acrobat - Fill and Sign.pdf",
                "section_title": "hasn't been optimized for form filling. The Fill & Sign tool automatically detects the",
                "importance_rank": 5,
                "page_number": 12,
                "relevance_score": 0.59,
                "confidence": 0.52
            }
        ],
        "subsection_analysis": []
    }

def main():
    """Main entry point"""
    try:
        # Run the complete analysis
        results = run_complete_analysis()
        
        # Exit with appropriate code
        if results.get("status") == "completed_successfully":
            sys.exit(0)
        else:
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n[WARNING] Execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[FATAL ERROR] {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()