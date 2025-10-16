import os
import pandas as pd
import json
from src.preprocessing import DocumentPreprocessor
from src.ocr_engine import OCREngine
from src.table_detector import TableDetector
from src.integration import TextTableIntegrator

class TableExtractionPipeline:
    def __init__(self, min_table_confidence=0.7):
        self.preprocessor = DocumentPreprocessor()
        self.ocr = OCREngine()
        self.table_detector = TableDetector()
        self.integrator = TextTableIntegrator()
        self.min_table_confidence = min_table_confidence
    
    def process_document(self, input_path, output_dir="output"):
        os.makedirs(output_dir, exist_ok=True)
        results = {}
        
        print("Step 1: Converting PDF to images...")
        if input_path.lower().endswith('.pdf'):
            image_paths = self.preprocessor.pdf_to_images(input_path)
        else:
            image_paths = [input_path]
        
        total_tables = 0
        pages_with_tables = 0
        
        for i, img_path in enumerate(image_paths):
            print(f"\n{'='*60}")
            print(f"Processing page {i+1}/{len(image_paths)}...")
            print(f"{'='*60}")
            
            print("  - Preprocessing image...")
            processed_path, processed_img = self.preprocessor.preprocess_image(img_path)
            
            print("  - Extracting text with OCR...")
            ocr_data = self.ocr.extract_text_with_boxes(processed_path)
            
            if not ocr_data:
                print(f"   No text extracted from page {i+1}, skipping...")
                results[f'page_{i+1}'] = {
                    'image_path': processed_path,
                    'ocr_data': [],
                    'tables': [],
                    'skipped': True,
                    'reason': 'No text detected'
                }
                continue
            
            ocr_viz_path = os.path.join(output_dir, f"page_{i+1}_ocr_viz.png")
            self.ocr.visualize_ocr_results(processed_path, ocr_data, ocr_viz_path)
            
            print("  - Detecting tables...")
            tables = self.table_detector.detect_tables(
                processed_path, 
                confidence_threshold=self.min_table_confidence
            )
            
            high_confidence_tables = [
                t for t in tables 
                if t.get('confidence', 0) >= self.min_table_confidence
            ]
            
            if not high_confidence_tables:
                print(f"  No high-confidence tables found on page {i+1} (threshold: {self.min_table_confidence})")
                print(f"     Skipping this page...")
                results[f'page_{i+1}'] = {
                    'image_path': processed_path,
                    'ocr_data': ocr_data,
                    'tables': [],
                    'skipped': True,
                    'reason': f'No tables with confidence >= {self.min_table_confidence}'
                }
                continue
            
            print(f"   Found {len(high_confidence_tables)} high-confidence table(s)")
            pages_with_tables += 1
            
            page_results = []
            for j, table in enumerate(high_confidence_tables):
                print(f"\n    Processing table {j+1}/{len(high_confidence_tables)}...")
                print(f"    Confidence: {table['confidence']:.2f}")
                
                structure = self.table_detector.detect_table_structure(
                    processed_path, table['bbox']
                )
                
                has_real_structure = (
                    len(structure.get('rows', [])) > 0 and 
                    len(structure.get('columns', [])) > 0
                )
                
                if not has_real_structure:
                    print(f"     No real table structure detected, skipping this table...")
                    continue
                
                print(f"    Structure: {len(structure['rows'])} rows, {len(structure['columns'])} cols")
                
                cell_mapping, cells = self.integrator.map_text_to_cells(
                    ocr_data, structure
                )
                
                table_matrix = self.integrator.build_table_matrix(cell_mapping, cells)
                
                if not table_matrix or all(not any(cell.strip() for cell in row) for row in table_matrix):
                    print(f"     Table matrix is empty, skipping...")
                    continue
                
                df = pd.DataFrame(table_matrix)
                csv_path = os.path.join(output_dir, f"page_{i+1}_table_{j+1}.csv")
                df.to_csv(csv_path, index=False, header=False)
                print(f"     Saved to: {csv_path}")
                
                page_results.append({
                    'table_id': j+1,
                    'bbox': table['bbox'],
                    'confidence': table['confidence'],
                    'csv_path': csv_path,
                    'matrix': table_matrix
                })
                total_tables += 1
            
            results[f'page_{i+1}'] = {
                'image_path': processed_path,
                'ocr_data': ocr_data,
                'tables': page_results,
                'skipped': False
            }
        
        results_path = os.path.join(output_dir, "extraction_results.json")
        
        json_results = {}
        for page, data in results.items():
            json_results[page] = {
                'image_path': data['image_path'],
                'tables': [
                    {
                        'table_id': t['table_id'],
                        'bbox': t['bbox'],
                        'confidence': t['confidence'],
                        'csv_path': t['csv_path']
                    } for t in data.get('tables', [])
                ],
                'skipped': data.get('skipped', False),
                'reason': data.get('reason', '')
            }
        
        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"EXTRACTION SUMMARY")
        print(f"{'='*60}")
        print(f"Total pages: {len(image_paths)}")
        print(f"Pages with tables: {pages_with_tables}")
        print(f"Pages skipped: {len(image_paths) - pages_with_tables}")
        print(f"Total tables extracted: {total_tables}")
        print(f"Confidence threshold: {self.min_table_confidence}")
        print(f"{'='*60}\n")
        
        return results

if __name__ == "__main__":

    pipeline = TableExtractionPipeline(min_table_confidence=0.7)
    
    input_file = "input/sample_document.pdf"  
    results = pipeline.process_document(input_file)
    
    print("\nExtraction completed!")
    print(f"Results saved in: output/")
    
    for page, data in results.items():
        if data.get('skipped'):
            print(f"{page}: Skipped - {data.get('reason', 'No tables found')}")
        else:
            print(f"{page}: Found {len(data['tables'])} table(s)")