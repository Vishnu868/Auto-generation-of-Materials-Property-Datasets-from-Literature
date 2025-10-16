import openai
import json
import pandas as pd
from typing import List, Dict, Any
import re

class LLMTableRefiner:
    def __init__(self, model="gpt-3.5-turbo", api_key=None):
        self.model = model
        if api_key:
            openai.api_key = api_key
        
        self.use_local = api_key is None
        if self.use_local:
            self.setup_local_model()
    
    def setup_local_model(self):
        from transformers import pipeline
        self.local_llm = pipeline(
            "text-generation",
            model="microsoft/DialoGPT-medium", 
            device_map="auto"
        )
    
    def refine_table_content(self, raw_matrix: List[List[str]]) -> Dict[str, Any]:
        
        table_text = self.matrix_to_text(raw_matrix)
        
        prompt = self.create_refinement_prompt(table_text)
        
        if self.use_local:
            refined_content = self.query_local_model(prompt)
        else:
            refined_content = self.query_openai(prompt)
        
        return self.parse_llm_response(refined_content, raw_matrix)
    
    def create_refinement_prompt(self, table_text: str) -> str:
        return f"""
You are a table data cleaning expert. Clean and improve this extracted table data:

ORIGINAL TABLE DATA:
{table_text}

TASKS:
1. Fix OCR errors in text (common mistakes: 0/O, 1/l, 5/S)
2. Identify and separate merged cell content
3. Align data with appropriate headers
4. Standardize number formats (remove extra spaces, fix decimals)
5. Fix date formats to be consistent
6. Identify the header row(s)

OUTPUT FORMAT (JSON):
{{
  "headers": ["col1", "col2", "col3"],
  "data": [
    ["row1_col1", "row1_col2", "row1_col3"],
    ["row2_col1", "row2_col2", "row2_col3"]
  ],
  "issues_fixed": ["list of problems corrected"],
  "confidence": 0.85
}}

Be conservative - only fix obvious errors. Return the JSON only.
"""
    
    def query_openai(self, prompt: str) -> str:
        """Query OpenAI API"""
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=2000
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return self.fallback_cleaning(prompt)
    
    def query_local_model(self, prompt: str) -> str:
        try:
            result = self.local_llm(prompt, max_length=1000, temperature=0.1)
            return result[0]['generated_text']
        except Exception as e:
            print(f"Local model error: {e}")
            return self.fallback_cleaning(prompt)
    
    def fallback_cleaning(self, prompt: str) -> str:
        # Extract original table from prompt
        table_start = prompt.find("ORIGINAL TABLE DATA:") + len("ORIGINAL TABLE DATA:")
        table_end = prompt.find("TASKS:")
        table_text = prompt[table_start:table_end].strip()
        
        lines = table_text.split('\n')
        cleaned_lines = []
        for line in lines:
            if line.strip():
                cleaned = line.replace(' 0 ', ' O ')  
                cleaned = re.sub(r'\s+', ' ', cleaned)  
                cleaned_lines.append(cleaned.strip())
        
        return json.dumps({
            "headers": cleaned_lines[0].split() if cleaned_lines else [],
            "data": [line.split() for line in cleaned_lines[1:]] if len(cleaned_lines) > 1 else [],
            "issues_fixed": ["basic_cleaning"],
            "confidence": 0.5
        })
    
    def matrix_to_text(self, matrix: List[List[str]]) -> str:
        text_lines = []
        for row in matrix:
            text_lines.append(" | ".join(str(cell) for cell in row))
        return "\n".join(text_lines)
    
    def parse_llm_response(self, response: str, original_matrix: List[List[str]]) -> Dict[str, Any]:
        try:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            json_str = response[json_start:json_end]
            
            result = json.loads(json_str)
            
            if 'headers' not in result or 'data' not in result:
                raise ValueError("Missing required fields")
            
            return result
            
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Failed to parse LLM response: {e}")
            return {
                "headers": original_matrix[0] if original_matrix else [],
                "data": original_matrix[1:] if len(original_matrix) > 1 else [],
                "issues_fixed": [],
                "confidence": 0.3
            }