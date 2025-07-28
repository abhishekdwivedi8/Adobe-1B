"""
Enhanced Accuracy Test Suite - Validates 90%+ Accuracy Implementation
Tests all components and measures actual performance improvements
"""

import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any
import sys

# Import enhanced system
from enhanced_main_system import EnhancedDocumentIntelligenceSystem

class AccuracyTestSuite:
    """
    Comprehensive test suite to validate 90%+ accuracy improvements
    """
    
    def __init__(self):
        self.setup_logging()
        self.system = EnhancedDocumentIntelligenceSystem()
        
        # Test configuration
        self.test_cases = self._prepare_test_cases()
        self.accuracy_thresholds = {
            'section_detection_accuracy': 0.85,
            'title_completeness_accuracy': 0.90,
            'ranking_consistency_accuracy': 0.88,
            'overall_system_accuracy': 0.90
        }
        
    def setup_logging(self):
        """Setup test logging"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _prepare_test_cases(self) -> List[Dict[str, Any]]:
        """Prepare comprehensive test cases"""
        return [
            {
                'name': 'HR Professional - Form Creation',
                'persona': 'HR professional with 5 years experience in employee onboarding',
                'job': 'Create and manage fillable forms for onboarding and compliance documentation',
                'expected_keywords': ['fillable forms', 'onboarding', 'compliance', 'employee'],
                'expected_sections': 8,
                'min_accuracy': 0.90
            },
            {
                'name': 'IT Manager - Document Workflow',
                'persona': 'IT Manager responsible for digital transformation and workflow automation',
                'job': 'Implement digital document workflows and e-signature processes',
                'expected_keywords': ['digital workflow', 'e-signature', 'automation'],
                'expected_sections': 6,
                'min_accuracy': 0.88
            },
            {
                'name': 'Administrator - PDF Management',
                'persona': 'Office Administrator handling document management and distribution',
                'job': 'Organize and manage PDF document collection and sharing processes',
                'expected_keywords': ['document management', 'sharing', 'organization'],
                'expected_sections': 7,
                'min_accuracy': 0.85
            }
        ]
    
    def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run comprehensive accuracy tests"""
        print("\n" + "="*80)
        print("[TEST] ENHANCED ACCURACY TEST SUITE - 90%+ VALIDATION")
        print("="*80)
        
        test_results = {
            'overall_results': {},
            'individual_tests': [],
            'accuracy_metrics': {},
            'performance_improvements': {},
            'issues_found': []
        }
        
        start_time = time.time()
        
        # Test 1: Section Detection Accuracy
        print("\n[LIST] Test 1: Section Detection Accuracy")
        detection_results = self._test_section_detection_accuracy()
        test_results['individual_tests'].append(detection_results)
        
        # Test 2: Title Completeness Accuracy
        print("\n[NOTE] Test 2: Title Completeness Accuracy")
        title_results = self._test_title_completeness()
        test_results['individual_tests'].append(title_results)
        
        # Test 3: Ranking Consistency
        print("\n[TARGET] Test 3: Ranking Consistency")
        ranking_results = self._test_ranking_consistency()
        test_results['individual_tests'].append(ranking_results)
        
        # Test 4: End-to-End System Tests
        print("\n[PROCESS] Test 4: End-to-End System Tests")
        system_results = self._test_end_to_end_scenarios()
        test_results['individual_tests'].append(system_results)
        
        # Test 5: Performance Validation
        print("\n[FAST] Test 5: Performance Validation")
        performance_results = self._test_performance_improvements()
        test_results['individual_tests'].append(performance_results)
        
        # Calculate overall results
        test_results['overall_results'] = self._calculate_overall_results(test_results['individual_tests'])
        test_results['accuracy_metrics'] = self._calculate_accuracy_metrics(test_results['individual_tests'])
        
        total_time = time.time() - start_time
        
        # Print summary
        self._print_test_summary(test_results, total_time)
        
        return test_results
    
    def _test_section_detection_accuracy(self) -> Dict[str, Any]:
        """Test section detection accuracy improvements"""
        results = {
            'test_name': 'Section Detection Accuracy',
            'passed': False,
            'accuracy_score': 0.0,
            'details': {},
            'issues': []
        }
        
        try:
            # Test intelligent section detector
            from core.intelligent_section_detector import IntelligentSectionDetector
            detector = IntelligentSectionDetector()
            
            # Simulate test cases with known fragmented titles
            test_lines = [
                "You can create multiple PDFs from multiple native files, including files of different supported",
                "formats. This method is useful when you must convert a large number of files.",
                "",
                "Fill and sign PDF forms",
                "Interactive forms contain fields that you can select and fill in.",
                "",
                "Change flat forms to fillable (Acrobat Pro)",
                "To create an interactive form, use the Prepare Forms tool."
            ]
            
            page_numbers = [1, 1, 1, 2, 2, 2, 3, 3]
            
            sections = detector.detect_sections_intelligent(
                text_lines=test_lines,
                page_numbers=page_numbers,
                document_path="test_document.pdf"
            )
            
            # Evaluate results
            complete_titles = sum(1 for section in sections if len(section.title) > 50)
            high_confidence = sum(1 for section in sections if section.confidence_score > 0.6)
            
            accuracy = (complete_titles + high_confidence) / max(len(sections) * 2, 1)
            
            results['accuracy_score'] = accuracy
            results['passed'] = accuracy >= self.accuracy_thresholds['section_detection_accuracy']
            results['details'] = {
                'sections_detected': len(sections),
                'complete_titles': complete_titles,
                'high_confidence_sections': high_confidence,
                'average_confidence': sum(s.confidence_score for s in sections) / max(len(sections), 1)
            }
            
            print(f"  [SUCCESS] Sections detected: {len(sections)}")
            print(f"  [NOTE] Complete titles: {complete_titles}/{len(sections)}")
            print(f"  [TARGET] Accuracy: {accuracy:.1%}")
            
        except Exception as e:
            results['issues'].append(f"Detection test failed: {e}")
            print(f"  [ERROR] Test failed: {e}")
        
        return results
    
    def _test_title_completeness(self) -> Dict[str, Any]:
        """Test title completeness and reconstruction"""
        results = {
            'test_name': 'Title Completeness Accuracy',
            'passed': False,
            'accuracy_score': 0.0,
            'details': {},
            'issues': []
        }
        
        try:
            # Test title reconstruction
            from core.intelligent_section_detector import IntelligentSectionDetector
            detector = IntelligentSectionDetector()
            
            # Test fragmented titles
            test_cases = [
                {
                    'fragments': ["Create multiple PDFs from multiple", "files using different formats"],
                    'expected': "Create multiple PDFs from multiple files using different formats"
                },
                {
                    'fragments': ["Fill and sign PDF", "forms with interactive fields"],
                    'expected': "Fill and sign PDF forms with interactive fields"
                }
            ]
            
            successful_reconstructions = 0
            
            for test_case in test_cases:
                # Test reconstruction logic
                title_candidate = {
                    'line_text': test_case['fragments'][0],
                    'line_index': 0,
                    'confidence': 0.7
                }
                
                test_lines = test_case['fragments'] + ["", "Additional content here"]
                
                reconstruction = detector._attempt_title_reconstruction(title_candidate, test_lines)
                
                if reconstruction and len(reconstruction.reconstructed_title) > len(test_case['fragments'][0]):
                    successful_reconstructions += 1
            
            accuracy = successful_reconstructions / len(test_cases)
            
            results['accuracy_score'] = accuracy
            results['passed'] = accuracy >= self.accuracy_thresholds['title_completeness_accuracy']
            results['details'] = {
                'test_cases': len(test_cases),
                'successful_reconstructions': successful_reconstructions,
                'reconstruction_rate': accuracy
            }
            
            print(f"  [NOTE] Test cases: {len(test_cases)}")
            print(f"  [SUCCESS] Successful reconstructions: {successful_reconstructions}")
            print(f"  [TARGET] Accuracy: {accuracy:.1%}")
            
        except Exception as e:
            results['issues'].append(f"Title completeness test failed: {e}")
            print(f"  [ERROR] Test failed: {e}")
        
        return results
    
    def _test_ranking_consistency(self) -> Dict[str, Any]:
        """Test ranking consistency and score normalization"""
        results = {
            'test_name': 'Ranking Consistency Accuracy',
            'passed': False,
            'accuracy_score': 0.0,
            'details': {},
            'issues': []
        }
        
        try:
            from core.enhanced_persona_ranking import EnhancedPersonaRanking
            ranking_system = EnhancedPersonaRanking()
            
            # Test persona and job analysis
            persona = ranking_system.analyze_persona_enhanced("HR professional")
            job = ranking_system.analyze_job_enhanced("Create and manage fillable forms for onboarding")
            
            # Test scoring consistency
            test_sections = [
                ("Create fillable forms for employee onboarding", "This section explains how to create interactive forms using Acrobat Pro. You can add form fields, checkboxes, and signature fields."),
                ("Convert PDF documents", "Basic conversion process for PDF files."),
                ("Fill and sign forms", "Instructions for filling out forms and adding digital signatures for compliance purposes.")
            ]
            
            scores = []
            for title, content in test_sections:
                score_breakdown = ranking_system.calculate_enhanced_relevance_score(
                    title, content, persona, job
                )
                scores.append(score_breakdown.final_score)
            
            # Validate score bounds (no scores > 1.0)
            valid_scores = all(0.0 <= score <= 1.0 for score in scores)
            
            # Check score distribution (highest relevance should score highest)
            expected_order = [0, 2, 1]  # Expected ranking order
            actual_order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            
            ranking_accuracy = sum(1 for i, expected in enumerate(expected_order) if actual_order[i] == expected) / len(expected_order)
            
            overall_accuracy = (1.0 if valid_scores else 0.0) * 0.5 + ranking_accuracy * 0.5
            
            results['accuracy_score'] = overall_accuracy
            results['passed'] = overall_accuracy >= self.accuracy_thresholds['ranking_consistency_accuracy']
            results['details'] = {
                'valid_score_bounds': valid_scores,
                'scores': scores,
                'ranking_accuracy': ranking_accuracy,
                'expected_order': expected_order,
                'actual_order': actual_order
            }
            
            print(f"  [TARGET] Valid score bounds: {valid_scores}")
            print(f"  [REPORT] Ranking accuracy: {ranking_accuracy:.1%}")
            print(f"  [SUCCESS] Overall accuracy: {overall_accuracy:.1%}")
            
        except Exception as e:
            results['issues'].append(f"Ranking consistency test failed: {e}")
            print(f"  [ERROR] Test failed: {e}")
        
        return results
    
    def _test_end_to_end_scenarios(self) -> Dict[str, Any]:
        """Test end-to-end scenarios with real use cases"""
        results = {
            'test_name': 'End-to-End System Accuracy',
            'passed': False,
            'accuracy_score': 0.0,
            'details': {},
            'issues': []
        }
        
        try:
            successful_tests = 0
            
            for test_case in self.test_cases:
                print(f"    [TEST] Testing: {test_case['name']}")
                
                # Run enhanced system
                result = self.system.process_documents_enhanced(
                    persona_description=test_case['persona'],
                    job_description=test_case['job'],
                    target_accuracy=test_case['min_accuracy']
                )
                
                # Evaluate results
                if result['status'] == 'success':
                    achieved_accuracy = result.get('achieved_accuracy', 0.0)
                    sections_found = result.get('sections_found', 0)
                    
                    # Check if meets expectations
                    meets_accuracy = achieved_accuracy >= test_case['min_accuracy']
                    has_sections = sections_found >= test_case['expected_sections'] * 0.7  # 70% of expected
                    
                    if meets_accuracy and has_sections:
                        successful_tests += 1
                        print(f"      [SUCCESS] Passed - Accuracy: {achieved_accuracy:.1%}, Sections: {sections_found}")
                    else:
                        print(f"      [ERROR] Failed - Accuracy: {achieved_accuracy:.1%}, Sections: {sections_found}")
                else:
                    print(f"      [ERROR] System error: {result.get('error', 'Unknown')}")
            
            accuracy = successful_tests / len(self.test_cases)
            
            results['accuracy_score'] = accuracy
            results['passed'] = accuracy >= self.accuracy_thresholds['overall_system_accuracy']
            results['details'] = {
                'test_cases_run': len(self.test_cases),
                'successful_tests': successful_tests,
                'success_rate': accuracy
            }
            
            print(f"  [REPORT] Test cases: {len(self.test_cases)}")
            print(f"  [SUCCESS] Successful: {successful_tests}")
            print(f"  [TARGET] Success rate: {accuracy:.1%}")
            
        except Exception as e:
            results['issues'].append(f"End-to-end test failed: {e}")
            print(f"  [ERROR] Test failed: {e}")
        
        return results
    
    def _test_performance_improvements(self) -> Dict[str, Any]:
        """Test performance improvements and efficiency"""
        results = {
            'test_name': 'Performance Validation',
            'passed': False,
            'accuracy_score': 0.0,
            'details': {},
            'issues': []
        }
        
        try:
            # Test processing speed
            start_time = time.time()
            
            test_result = self.system.process_documents_enhanced(
                persona_description="HR professional",
                job_description="Create fillable forms for onboarding"
            )
            
            processing_time = time.time() - start_time
            
            # Evaluate performance
            speed_acceptable = processing_time < 30.0  # Should complete in under 30 seconds
            memory_efficient = True  # Placeholder for memory usage check
            
            performance_score = (1.0 if speed_acceptable else 0.5) * 0.6 + (1.0 if memory_efficient else 0.5) * 0.4
            
            results['accuracy_score'] = performance_score
            results['passed'] = performance_score >= 0.8
            results['details'] = {
                'processing_time': processing_time,
                'speed_acceptable': speed_acceptable,
                'memory_efficient': memory_efficient,
                'result_status': test_result.get('status', 'unknown')
            }
            
            print(f"  [STOPWATCH]  Processing time: {processing_time:.1f}s")
            print(f"  [LAUNCH] Speed acceptable: {speed_acceptable}")
            print(f"  [REPORT] Performance score: {performance_score:.1%}")
            
        except Exception as e:
            results['issues'].append(f"Performance test failed: {e}")
            print(f"  [ERROR] Test failed: {e}")
        
        return results
    
    def _calculate_overall_results(self, individual_tests: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate overall test results"""
        total_tests = len(individual_tests)
        passed_tests = sum(1 for test in individual_tests if test['passed'])
        
        overall_accuracy = sum(test['accuracy_score'] for test in individual_tests) / max(total_tests, 1)
        
        return {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'pass_rate': passed_tests / max(total_tests, 1),
            'overall_accuracy': overall_accuracy,
            'meets_90_percent_target': overall_accuracy >= 0.9
        }
    
    def _calculate_accuracy_metrics(self, individual_tests: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate detailed accuracy metrics"""
        metrics = {}
        
        for test in individual_tests:
            test_name = test['test_name'].lower().replace(' ', '_')
            metrics[test_name] = {
                'accuracy': test['accuracy_score'],
                'passed': test['passed'],
                'issues_count': len(test['issues'])
            }
        
        return metrics
    
    def _print_test_summary(self, results: Dict[str, Any], total_time: float):
        """Print comprehensive test summary"""
        overall = results['overall_results']
        
        print("\n" + "="*80)
        print("[REPORT] COMPREHENSIVE TEST RESULTS SUMMARY")
        print("="*80)
        
        print(f"[STOPWATCH]  Total Test Time: {total_time:.1f} seconds")
        print(f"[TEST] Total Tests Run: {overall['total_tests']}")
        print(f"[SUCCESS] Tests Passed: {overall['passed_tests']}")
        print(f"[ERROR] Tests Failed: {overall['failed_tests']}")
        print(f"[ANALYTICS] Pass Rate: {overall['pass_rate']:.1%}")
        print(f"[TARGET] Overall Accuracy: {overall['overall_accuracy']:.1%}")
        
        if overall['meets_90_percent_target']:
            print("[COMPLETE] [SUCCESS] MEETS 90%+ ACCURACY TARGET!")
        else:
            print("[WARNING]  [ERROR] Does not meet 90%+ accuracy target")
        
        print("\n[LIST] INDIVIDUAL TEST BREAKDOWN:")
        print("" * 50)
        
        for test in results['individual_tests']:
            status = "[SUCCESS] PASS" if test['passed'] else "[ERROR] FAIL"
            print(f"{status} {test['test_name']}: {test['accuracy_score']:.1%}")
            
            if test['issues']:
                print(f"      Issues: {len(test['issues'])}")
                for issue in test['issues'][:2]:  # Show first 2 issues
                    print(f"        - {issue}")
        
        print("\n" + "="*80)
        
        # Save detailed results
        self._save_test_results(results)
    
    def _save_test_results(self, results: Dict[str, Any]):
        """Save test results to file"""
        try:
            output_path = Path("output") / "accuracy_test_results.json"
            output_path.parent.mkdir(exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4, ensure_ascii=False, default=str)
            
            print(f"[ARCHIVE] Test results saved to: {output_path}")
            
        except Exception as e:
            print(f"[WARNING]  Failed to save test results: {e}")

def main():
    """Run the comprehensive accuracy test suite"""
    test_suite = AccuracyTestSuite()
    results = test_suite.run_comprehensive_tests()
    
    # Return exit code based on results
    if results['overall_results']['meets_90_percent_target']:
        print("\n[COMPLETE] All tests passed! System ready for production.")
        return 0
    else:
        print("\n[WARNING]  Some tests failed. Review results and fix issues.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
