import pandas as pd
import re
from datetime import datetime
from typing import List, Dict, Any, Tuple
import numpy as np

class RuleBasedValidator:
    def __init__(self):
        self.date_patterns = [
            r'\d{1,2}/\d{1,2}/\d{4}',  
            r'\d{1,2}-\d{1,2}-\d{4}',  
            r'\d{4}-\d{1,2}-\d{1,2}', 
        ]
        
        self.number_patterns = {
            'currency': r'\$[\d,]+\.?\d*',
            'percentage': r'\d+\.?\d*%',
            'decimal': r'\d+\.\d+',
            'integer': r'^\d+$'
        }
    
    def validate_and_fix(self, llm_result: Dict[str, Any]) -> Dict[str, Any]:
        
        headers = llm_result.get('headers', [])
        data = llm_result.get('data', [])
        
        validation_results = {
            'original_confidence': llm_result.get('confidence', 0.5),
            'fixes_applied': [],
            'warnings': [],
            'final_confidence': 0.0
        }
        
        headers, data, header_fixes = self.fix_headers(headers, data)
        validation_results['fixes_applied'].extend(header_fixes)
        
        data, type_fixes = self.standardize_column_types(data, headers)
        validation_results['fixes_applied'].extend(type_fixes)
        
        data, missing_fixes = self.handle_missing_data(data, headers)
        validation_results['fixes_applied'].extend(missing_fixes)
        
        warnings = self.validate_consistency(data, headers)
        validation_results['warnings'].extend(warnings)
        
        data, structure_fixes = self.fix_structure_issues(data, headers)
        validation_results['fixes_applied'].extend(structure_fixes)
        
        validation_results['final_confidence'] = self.calculate_confidence_score(
            llm_result.get('confidence', 0.5),
            len(validation_results['fixes_applied']),
            len(validation_results['warnings'])
        )
        
        return {
            'headers': headers,
            'data': data,
            'validation': validation_results
        }
    
    def fix_headers(self, headers: List[str], data: List[List[str]]) -> Tuple[List[str], List[List[str]], List[str]]:
        """Detect and fix header issues"""
        fixes = []
        
        if not headers and data:
            first_row = data[0]
            if self.looks_like_header_row(first_row):
                headers = first_row
                data = data[1:]
                fixes.append("moved_first_data_row_to_headers")
        
        if not headers and data:
            headers = [f"Column_{i+1}" for i in range(len(data[0]) if data else 0)]
            fixes.append("generated_default_headers")
        
        clean_headers = []
        for i, header in enumerate(headers):
            clean_header = self.clean_header_name(str(header))
            if clean_header != str(header):
                fixes.append(f"cleaned_header_{i}")
            clean_headers.append(clean_header)
        
        return clean_headers, data, fixes
    
    def looks_like_header_row(self, row: List[str]) -> bool:
        """Heuristic to determine if row looks like headers"""
        if not row:
            return False
        

        text_count = 0
        number_count = 0
        
        for cell in row:
            cell_str = str(cell).strip()
            if re.match(r'^\d+\.?\d*$', cell_str):
                number_count += 1
            elif len(cell_str) > 0:
                text_count += 1
        
        return text_count / (text_count + number_count) > 0.7 if (text_count + number_count) > 0 else False
    
    def clean_header_name(self, header: str) -> str:
        """Clean and standardize header names"""
        clean = re.sub(r'[^\w\s]', '', str(header))
        clean = re.sub(r'\s+', '_', clean.strip())
        clean = clean.lower()
        
        if not clean:
            clean = "unnamed_column"
        
        return clean
    
    def standardize_column_types(self, data: List[List[str]], headers: List[str]) -> Tuple[List[List[str]], List[str]]:
        fixes = []
        
        if not data:
            return data, fixes
        
        columns = list(zip(*data)) if data else []
        standardized_columns = []
        
        for col_idx, column in enumerate(columns):
            col_type = self.detect_column_type(column)
            standardized_col, col_fixes = self.standardize_column(column, col_type)
            standardized_columns.append(standardized_col)
            
            if col_fixes:
                header_name = headers[col_idx] if col_idx < len(headers) else f"col_{col_idx}"
                fixes.extend([f"{header_name}_{fix}" for fix in col_fixes])
        
        if standardized_columns:
            standardized_data = list(zip(*standardized_columns))
            standardized_data = [list(row) for row in standardized_data]
        else:
            standardized_data = data
        
        return standardized_data, fixes
    
    def detect_column_type(self, column: List[str]) -> str:
        type_scores = {
            'currency': 0,
            'percentage': 0,
            'date': 0,
            'decimal': 0,
            'integer': 0,
            'text': 0
        }
        
        for cell in column:
            cell_str = str(cell).strip()
            if not cell_str:
                continue
            
            for type_name, pattern in self.number_patterns.items():
                if re.match(pattern, cell_str):
                    type_scores[type_name] += 1
                    break
            else:
                is_date = any(re.match(pattern, cell_str) for pattern in self.date_patterns)
                if is_date:
                    type_scores['date'] += 1
                else:
                    type_scores['text'] += 1
        
        return max(type_scores.items(), key=lambda x: x[1])[0]
    
    def standardize_column(self, column: List[str], col_type: str) -> Tuple[List[str], List[str]]:
        fixes = []
        standardized = []
        
        for cell in column:
            cell_str = str(cell).strip()
            
            if col_type == 'currency':
                clean_cell, was_fixed = self.clean_currency(cell_str)
            elif col_type == 'percentage':
                clean_cell, was_fixed = self.clean_percentage(cell_str)
            elif col_type == 'date':
                clean_cell, was_fixed = self.clean_date(cell_str)
            elif col_type in ['decimal', 'integer']:
                clean_cell, was_fixed = self.clean_number(cell_str)
            else:
                clean_cell, was_fixed = self.clean_text(cell_str)
            
            if was_fixed:
                fixes.append(f"standardized_{col_type}")
            
            standardized.append(clean_cell)
        
        return standardized, list(set(fixes)) 
    
    def clean_currency(self, value: str) -> Tuple[str, bool]:
        if not value:
            return value, False
        
        original = value
        cleaned = re.sub(r'\s+', '', value)
        
        if re.match(r'\d', cleaned) and '$' not in cleaned:
            cleaned = '$' + cleaned
        
        if '$' in cleaned and '.' not in cleaned:
            cleaned += '.00'
        
        return cleaned, cleaned != original
    
    def clean_percentage(self, value: str) -> Tuple[str, bool]:
        if not value:
            return value, False
        
        original = value
        cleaned = re.sub(r'\s+', '', value)
        
        if re.match(r'\d+\.?\d*$', cleaned):
            cleaned += '%'
        
        return cleaned, cleaned != original
    
    def clean_date(self, value: str) -> Tuple[str, bool]:
        if not value:
            return value, False
        
        original = value
        
        date_formats = ['%m/%d/%Y', '%m-%d-%Y', '%Y-%m-%d', '%d/%m/%Y']
        
        for fmt in date_formats:
            try:
                parsed_date = datetime.strptime(value.strip(), fmt)
                standardized = parsed_date.strftime('%Y-%m-%d')
                return standardized, standardized != original
            except ValueError:
                continue
        
        return value, False
    
    def clean_number(self, value: str) -> Tuple[str, bool]:
        if not value:
            return value, False
        
        original = value
        
        cleaned = re.sub(r'[,\s]+', '', value)
        
        try:
            float(cleaned)
            return cleaned, cleaned != original
        except ValueError:
            return original, False
    
    def clean_text(self, value: str) -> Tuple[str, bool]:
        if not value:
            return value, False
        
        original = value
        
        cleaned = re.sub(r'\s+', ' ', value.strip())
        
        return cleaned, cleaned != original
    
    def handle_missing_data(self, data: List[List[str]], headers: List[str]) -> Tuple[List[List[str]], List[str]]:
        fixes = []
        
        if not data:
            return data, fixes
        
        max_cols = max(len(row) for row in data) if data else 0
        
        fixed_data = []
        for row_idx, row in enumerate(data):
            fixed_row = list(row)  
            
            while len(fixed_row) < max_cols:
                fixed_row.append("")
                fixes.append(f"padded_row_{row_idx}")
            
            for col_idx, cell in enumerate(fixed_row):
                if cell is None or str(cell).lower() in ['none', 'null', 'nan']:
                    fixed_row[col_idx] = ""
                    fixes.append(f"cleaned_null_value_{row_idx}_{col_idx}")
            
            fixed_data.append(fixed_row)
        
        return fixed_data, list(set(fixes))
    
    def validate_consistency(self, data: List[List[str]], headers: List[str]) -> List[str]:
        warnings = []
        
        if not data:
            return warnings
        
        row_lengths = [len(row) for row in data]
        if len(set(row_lengths)) > 1:
            warnings.append(f"inconsistent_row_lengths: {set(row_lengths)}")
        
        if data:
            num_cols = len(data[0])
            for col_idx in range(num_cols):
                column_values = [row[col_idx] if col_idx < len(row) else "" for row in data]
                non_empty = [v for v in column_values if str(v).strip()]
                
                if len(non_empty) / len(column_values) < 0.3:  # Less than 30% filled
                    col_name = headers[col_idx] if col_idx < len(headers) else f"col_{col_idx}"
                    warnings.append(f"mostly_empty_column: {col_name}")
        
        return warnings
    
    def fix_structure_issues(self, data: List[List[str]], headers: List[str]) -> Tuple[List[List[str]], List[str]]:
        fixes = []
        
        if not data:
            return data, fixes
        
        non_empty_data = []
        for row_idx, row in enumerate(data):
            if any(str(cell).strip() for cell in row):
                non_empty_data.append(row)
            else:
                fixes.append(f"removed_empty_row_{row_idx}")
        
        return non_empty_data, fixes
    
    def calculate_confidence_score(self, llm_confidence: float, num_fixes: int, num_warnings: int) -> float:
        
        base_confidence = llm_confidence
        
        warning_penalty = min(num_warnings * 0.1, 0.3)
        
        fix_bonus = min(num_fixes * 0.02, 0.1)
        
        final_confidence = max(0.0, min(1.0, base_confidence - warning_penalty + fix_bonus))
        
        return round(final_confidence, 3)