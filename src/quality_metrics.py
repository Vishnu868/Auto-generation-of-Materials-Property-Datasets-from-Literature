import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import re
from difflib import SequenceMatcher

class QualityMetricsCalculator:
    def __init__(self):
        self.metrics_config = {
            'completeness_weight': 0.3,
            'accuracy_weight': 0.4,
            'consistency_weight': 0.2,
            'structure_weight': 0.1
        }
    
    def calculate_comprehensive_metrics(self, 
                                      original_matrix: List[List[str]],
                                      refined_result: Dict[str, Any],
                                      ground_truth: List[List[str]] = None) -> Dict[str, Any]:
        
        headers = refined_result.get('headers', [])
        data = refined_result.get('data', [])
        validation = refined_result.get('validation', {})
        
        metrics = {
            'completeness': self.calculate_completeness_metrics(original_matrix, headers, data),
            'accuracy': self.calculate_accuracy_metrics(original_matrix, headers, data, ground_truth),
            'consistency': self.calculate_consistency_metrics(headers, data),
            'structure': self.calculate_structure_metrics(headers, data),
            'improvement': self.calculate_improvement_metrics(original_matrix, headers, data, validation),
            'confidence': validation.get('final_confidence', 0.0)
        }
        
        metrics['overall_score'] = self.calculate_overall_score(metrics)
        
        return metrics
    
    def calculate_completeness_metrics(self, 
                                     original_matrix: List[List[str]], 
                                     headers: List[str], 
                                     data: List[List[str]]) -> Dict[str, float]:
        """Measure how complete the extraction is"""
        
        original_cells = sum(len(row) for row in original_matrix) if original_matrix else 0
        extracted_cells = len(headers) + sum(len(row) for row in data)
        
        recovery_rate = min(extracted_cells / original_cells, 1.0) if original_cells > 0 else 0.0
        
        has_headers = 1.0 if headers else 0.0
        
        total_data_cells = sum(len(row) for row in data)
        non_empty_cells = sum(1 for row in data for cell in row if str(cell).strip())
        data_density = non_empty_cells / total_data_cells if total_data_cells > 0 else 0.0
        
        return {
            'cell_recovery_rate': round(recovery_rate, 3),
            'header_presence': has_headers,
            'data_density': round(data_density, 3),
            'total_cells_extracted': extracted_cells
        }
    
    def calculate_accuracy_metrics(self, 
                                 original_matrix: List[List[str]], 
                                 headers: List[str], 
                                 data: List[List[str]],
                                 ground_truth: List[List[str]] = None) -> Dict[str, float]:
        """Measure accuracy of extraction"""
        
        metrics = {}
        
        if original_matrix:
            original_text = self.matrix_to_text(original_matrix)
            refined_text = self.matrix_to_text([headers] + data if headers else data)
            
            similarity = SequenceMatcher(None, original_text.lower(), refined_text.lower()).ratio()
            metrics['text_similarity'] = round(similarity, 3)
        
        if ground_truth:
            gt_text = self.matrix_to_text(ground_truth)
            refined_text = self.matrix_to_text([headers] + data if headers else data)
            
            gt_similarity = SequenceMatcher(None, gt_text.lower(), refined_text.lower()).ratio()
            metrics['ground_truth_similarity'] = round(gt_similarity, 3)
            
            cell_matches = 0
            total_cells = 0
            
            combined_data = [headers] + data if headers else data
            max_rows = min(len(ground_truth), len(combined_data))
            
            for i in range(max_rows):
                gt_row = ground_truth[i] if i < len(ground_truth) else []
                extracted_row = combined_data[i] if i < len(combined_data) else []
                
                max_cols = min(len(gt_row), len(extracted_row))
                for j in range(max_cols):
                    total_cells += 1
                    if str(gt_row[j]).strip().lower() == str(extracted_row[j]).strip().lower():
                        cell_matches += 1
            
            metrics['cell_accuracy'] = round(cell_matches / total_cells, 3) if total_cells > 0 else 0.0
        
        format_accuracy = self.calculate_format_accuracy(data)
        metrics.update(format_accuracy)
        
        return metrics
    
    def calculate_format_accuracy(self, data: List[List[str]]) -> Dict[str, float]:
        if not data:
            return {'format_consistency': 0.0}
        
        columns = list(zip(*data)) if data else []
        format_scores = []
        
        for column in columns:
            column_score = self.calculate_column_format_score(column)
            format_scores.append(column_score)
        
        avg_format_score = np.mean(format_scores) if format_scores else 0.0
        
        return {
            'format_consistency': round(avg_format_score, 3),
            'columns_analyzed': len(format_scores)
        }
    
    def calculate_column_format_score(self, column: List[str]) -> float:
        if not column:
            return 0.0
        
        format_counts = {
            'currency': 0,
            'percentage': 0,
            'date': 0,
            'number': 0,
            'text': 0
        }
        
        for cell in column:
            cell_str = str(cell).strip()
            if not cell_str:
                continue
            
            if re.match(r'\$[\d,]+\.?\d*', cell_str):
                format_counts['currency'] += 1
            elif re.match(r'\d+\.?\d*%', cell_str):
                format_counts['percentage'] += 1
            elif re.match(r'\d{4}-\d{1,2}-\d{1,2}', cell_str):
                format_counts['date'] += 1
            elif re.match(r'^\d+\.?\d*$', cell_str):
                format_counts['number'] += 1
            else:
                format_counts['text'] += 1
        
        total_cells = sum(format_counts.values())
        if total_cells == 0:
            return 0.0
        
        dominant_count = max(format_counts.values())
        return dominant_count / total_cells
    
    def calculate_consistency_metrics(self, headers: List[str], data: List[List[str]]) -> Dict[str, float]:
        
        metrics = {}
        
        if data:
            row_lengths = [len(row) for row in data]
            expected_length = len(headers) if headers else (max(row_lengths) if row_lengths else 0)
            
            consistent_rows = sum(1 for length in row_lengths if length == expected_length)
            row_consistency = consistent_rows / len(data) if data else 0.0
            
            metrics['row_length_consistency'] = round(row_consistency, 3)
            metrics['expected_columns'] = expected_length
            metrics['actual_row_lengths'] = list(set(row_lengths))
        
        if headers and data:
            header_count = len(headers)
            avg_row_length = np.mean([len(row) for row in data]) if data else 0
            alignment_score = 1.0 - abs(header_count - avg_row_length) / max(header_count, avg_row_length) if max(header_count, avg_row_length) > 0 else 0.0
            
            metrics['header_data_alignment'] = round(max(0.0, alignment_score), 3)
        
        return metrics
    
    def calculate_structure_metrics(self, headers: List[str], data: List[List[str]]) -> Dict[str, float]:
        
        metrics = {}
        
        if data:
            row_lengths = [len(row) for row in data]
            shape_regularity = 1.0 - (np.std(row_lengths) / np.mean(row_lengths)) if np.mean(row_lengths) > 0 else 0.0
            metrics['shape_regularity'] = round(max(0.0, min(1.0, shape_regularity)), 3)
        
        if headers:
            meaningful_headers = sum(1 for h in headers if not re.match(r'^(column_\d+|col_\d+|unnamed).*', str(h).lower()))
            header_quality = meaningful_headers / len(headers) if headers else 0.0
            metrics['header_quality'] = round(header_quality, 3)
        
        structure_score = np.mean([v for v in metrics.values() if isinstance(v, (int, float))])
        metrics['overall_structure'] = round(structure_score, 3)
        
        return metrics
    
    def calculate_improvement_metrics(self, 
                                    original_matrix: List[List[str]], 
                                    headers: List[str], 
                                    data: List[List[str]],
                                    validation: Dict[str, Any]) -> Dict[str, Any]:
        
        metrics = {}
        
        fixes_applied = validation.get('fixes_applied', [])
        metrics['total_fixes'] = len(fixes_applied)
        
        fix_categories = {
            'header_fixes': len([f for f in fixes_applied if 'header' in f]),
            'format_fixes': len([f for f in fixes_applied if any(fmt in f for fmt in ['currency', 'percentage', 'date', 'number'])]),
            'structure_fixes': len([f for f in fixes_applied if any(struct in f for struct in ['row', 'column', 'empty'])]),
            'content_fixes': len([f for f in fixes_applied if 'standardized' in f or 'cleaned' in f])
        }
        metrics.update(fix_categories)
        
        original_issues = self.count_original_issues(original_matrix)
        remaining_warnings = len(validation.get('warnings', []))
        
        improvement_ratio = max(0.0, (original_issues - remaining_warnings) / original_issues) if original_issues > 0 else 0.0
        metrics['quality_improvement'] = round(improvement_ratio, 3)
        
        return metrics
    
    def count_original_issues(self, original_matrix: List[List[str]]) -> int:
        if not original_matrix:
            return 0
        
        issues = 0
        
        row_lengths = [len(row) for row in original_matrix]
        if len(set(row_lengths)) > 1:
            issues += len(row_lengths) - row_lengths.count(max(row_lengths, key=row_lengths.count))
        
        for row in original_matrix:
            issues += sum(1 for cell in row if not str(cell).strip())
        
        for row in original_matrix:
            for cell in row:
                cell_str = str(cell)
                if re.search(r'\s{2,}', cell_str): 
                    issues += 1
                if re.search(r'[^\w\s\.\-\$%,]', cell_str): 
                    issues += 1
        
        return issues
    
    def calculate_overall_score(self, metrics: Dict[str, Any]) -> float:
        
        completeness = metrics.get('completeness', {})
        accuracy = metrics.get('accuracy', {})
        consistency = metrics.get('consistency', {})
        structure = metrics.get('structure', {})
        
        completeness_score = np.mean([
            completeness.get('cell_recovery_rate', 0.0),
            completeness.get('data_density', 0.0),
            completeness.get('header_presence', 0.0)
        ])
        
        accuracy_score = np.mean([v for k, v in accuracy.items() 
                                if isinstance(v, (int, float)) and 'similarity' in k or 'accuracy' in k])
        if not accuracy_score:
            accuracy_score = accuracy.get('format_consistency', 0.0)
        
        consistency_score = np.mean([v for v in consistency.values() if isinstance(v, (int, float))])
        
        structure_score = structure.get('overall_structure', 0.0)
        
        weighted_score = (
            completeness_score * self.metrics_config['completeness_weight'] +
            accuracy_score * self.metrics_config['accuracy_weight'] +
            consistency_score * self.metrics_config['consistency_weight'] +
            structure_score * self.metrics_config['structure_weight']
        )
        
        return round(weighted_score, 3)
    
    def matrix_to_text(self, matrix: List[List[str]]) -> str:
        return ' '.join(' '.join(str(cell) for cell in row) for row in matrix)
    
    def generate_quality_report(self, metrics: Dict[str, Any]) -> str:
        
        report = []
        report.append("=== TABLE EXTRACTION QUALITY REPORT ===\n")
        
        # Overall score
        overall_score = metrics.get('overall_score', 0.0)
        confidence = metrics.get('confidence', 0.0)
        
        report.append(f"Overall Quality Score: {overall_score:.3f}/1.000")
        report.append(f"Confidence Level: {confidence:.3f}/1.000")
        
        # Grade assignment
        if overall_score >= 0.9:
            grade = "A (Excellent)"
        elif overall_score >= 0.8:
            grade = "B (Good)"
        elif overall_score >= 0.7:
            grade = "C (Acceptable)"
        elif overall_score >= 0.6:
            grade = "D (Needs Improvement)"
        else:
            grade = "F (Poor)"
        
        report.append(f"Quality Grade: {grade}\n")
        
        completeness = metrics.get('completeness', {})
        report.append("COMPLETENESS METRICS:")
        report.append(f"  Cell Recovery Rate: {completeness.get('cell_recovery_rate', 0):.1%}")
        report.append(f"  Data Density: {completeness.get('data_density', 0):.1%}")
        report.append(f"  Headers Present: {'Yes' if completeness.get('header_presence') else 'No'}")
        report.append("")
        
        accuracy = metrics.get('accuracy', {})
        if accuracy:
            report.append("ACCURACY METRICS:")
            for key, value in accuracy.items():
                if isinstance(value, (int, float)):
                    if 'similarity' in key or 'accuracy' in key:
                        report.append(f"  {key.replace('_', ' ').title()}: {value:.1%}")
                    else:
                        report.append(f"  {key.replace('_', ' ').title()}: {value}")
            report.append("")
        
        improvement = metrics.get('improvement', {})
        if improvement:
            report.append("IMPROVEMENTS MADE:")
            report.append(f"  Total Fixes Applied: {improvement.get('total_fixes', 0)}")
            report.append(f"  Header Fixes: {improvement.get('header_fixes', 0)}")
            report.append(f"  Format Fixes: {improvement.get('format_fixes', 0)}")
            report.append(f"  Structure Fixes: {improvement.get('structure_fixes', 0)}")
            report.append(f"  Quality Improvement: {improvement.get('quality_improvement', 0):.1%}")
            report.append("")
        
        report.append("RECOMMENDATIONS:")
        if overall_score < 0.7:
            report.append("  • Consider manual review of extracted data")
            if completeness.get('data_density', 0) < 0.5:
                report.append("  • Many cells appear empty - check original image quality")
            if confidence < 0.6:
                report.append("  • Low confidence - verify critical data points")
        elif overall_score < 0.9:
            report.append("  • Good quality extraction - spot check recommended")
        else:
            report.append("  • Excellent extraction quality - ready for use")
        
        return "\n".join(report)



