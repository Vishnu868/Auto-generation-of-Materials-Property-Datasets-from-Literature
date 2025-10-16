
import cv2
import numpy as np
from PIL import Image
import os

class TableDetector:
    def __init__(self):
        self.models_loaded = False
        self.processor = None
        self.model = None
        self.structure_processor = None
        self.structure_model = None
        self._init_models()
    
    def _init_models(self):
        try:
            from transformers import AutoImageProcessor, TableTransformerForObjectDetection
            import torch
            
            print("Loading table detection models...")
            
            self.processor = AutoImageProcessor.from_pretrained(
                "microsoft/table-transformer-detection"
            )
            self.model = TableTransformerForObjectDetection.from_pretrained(
                "microsoft/table-transformer-detection"
            )
            
            self.structure_processor = AutoImageProcessor.from_pretrained(
                "microsoft/table-transformer-structure-recognition-v1.1-all",
                size={"height": 800, "width": 800}
            )
            self.structure_model = TableTransformerForObjectDetection.from_pretrained(
                "microsoft/table-transformer-structure-recognition-v1.1-all"
            )
            
            self.models_loaded = True
            print(" Table Transformer models loaded successfully")
            
        except ImportError as e:
            print(f" Transformers library not available: {e}")
            print("   Install with: pip install transformers torch")
            print(" Will use fallback table detection")
            self.models_loaded = False
            
        except Exception as e:
            print(f" Error loading Table Transformer models: {e}")
            print(" Will use fallback table detection")
            self.models_loaded = False
    
    def detect_tables(self, image_path, confidence_threshold=0.7):
        if not self.models_loaded:
            return self._fallback_table_detection(image_path)
        
        try:
            import torch
            
            image = Image.open(image_path).convert("RGB")
            width, height = image.size
            
            inputs = self.processor(images=image, return_tensors="pt")
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            target_sizes = torch.tensor([[height, width]])
            results = self.processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=confidence_threshold
            )[0]

            tables = []
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                box = [round(i, 2) for i in box.tolist()]
                tables.append({
                    'bbox': box,  # [x1, y1, x2, y2]
                    'confidence': score.item(),
                    'label': self.model.config.id2label[label.item()]
                })
            
            if not tables:
                print(f"    No tables detected with confidence > {confidence_threshold}")
                return []
            
            print(f"✅ Detected {len(tables)} tables with confidence > {confidence_threshold}")
            return tables
            
        except Exception as e:
            print(f" Table detection error: {e}")
            return self._fallback_table_detection(image_path)
    
    def _fallback_table_detection(self, image_path):
        print(" Using fallback table detection...")
        
        try:
            img = cv2.imread(image_path)
            if img is None:
                pil_img = Image.open(image_path)
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            
            if img is None:
                raise ValueError("Could not load image")
            
            h, w = img.shape[:2]
            
            tables = []
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
            
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
            vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
            
            table_mask = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0.0)
            
            contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                x, y, w_rect, h_rect = cv2.boundingRect(contour)
                

                if (w_rect > w * 0.3 and h_rect > h * 0.2 and 
                    w_rect > 100 and h_rect > 100):
                    tables.append({
                        'bbox': [x, y, x + w_rect, y + h_rect],
                        'confidence': 0.6,
                        'label': 'table'
                    })
            
            if not tables:
                print("    No table-like structures detected")
                return []
            
            print(f"✅ Fallback detected {len(tables)} potential table regions")
            return tables
            
        except Exception as e:
            print(f" Fallback table detection error: {e}")
            return []
    
    def detect_table_structure(self, image_path, table_bbox):
        if not self.models_loaded:
            return self._fallback_structure_detection(table_bbox)
        
        try:
            import torch
            
            image = Image.open(image_path).convert("RGB")
            x1, y1, x2, y2 = table_bbox
            table_crop = image.crop((x1, y1, x2, y2))
            
            if table_crop.size[0] == 0 or table_crop.size[1] == 0:
                print(" Invalid table crop dimensions")
                return self._fallback_structure_detection(table_bbox)
            
            inputs = self.structure_processor(images=table_crop, return_tensors="pt")
            
            with torch.no_grad():
                outputs = self.structure_model(**inputs)
            
            crop_height, crop_width = table_crop.size[1], table_crop.size[0]
            target_sizes = torch.tensor([[crop_height, crop_width]])
            results = self.structure_processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=0.6
            )[0]
            
            structure = {
                'rows': [],
                'columns': [],
                'cells': []
            }
            
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                box = [round(i, 2) for i in box.tolist()]
                adjusted_box = [
                    box[0] + x1, box[1] + y1,
                    box[2] + x1, box[3] + y1
                ]
                
                element = {
                    'bbox': adjusted_box,
                    'confidence': score.item(),
                    'type': self.structure_model.config.id2label[label.item()]
                }
                
                element_type = element['type'].lower()
                if 'row' in element_type:
                    structure['rows'].append(element)
                elif 'column' in element_type or 'col' in element_type:
                    structure['columns'].append(element)
                else:
                    structure['cells'].append(element)
            
            if not structure['rows'] and not structure['columns'] and not structure['cells']:
                print(" No structure detected by model, using fallback")
                return self._fallback_structure_detection(table_bbox)
            
            print(f"Structure detection: {len(structure['rows'])} rows, {len(structure['columns'])} columns, {len(structure['cells'])} cells")
            return structure
            
        except Exception as e:
            print(f"Structure detection error: {e}")
            return self._fallback_structure_detection(table_bbox)
    
    def _fallback_structure_detection(self, table_bbox):
        print(" Using fallback structure detection...")
        
        try:
            x1, y1, x2, y2 = table_bbox
            width = x2 - x1
            height = y2 - y1
            
            if width <= 0 or height <= 0:
                raise ValueError("Invalid table dimensions")
            
            num_rows = 4  
            num_cols = 3  
            
            rows = []
            columns = []
            cells = []
            
            for i in range(num_rows):
                row_y1 = y1 + (height * i / num_rows)
                row_y2 = y1 + (height * (i + 1) / num_rows)
                rows.append({
                    'bbox': [x1, row_y1, x2, row_y2],
                    'confidence': 0.5,
                    'type': 'table row'
                })
            
            for i in range(num_cols):
                col_x1 = x1 + (width * i / num_cols)
                col_x2 = x1 + (width * (i + 1) / num_cols)
                columns.append({
                    'bbox': [col_x1, y1, col_x2, y2],
                    'confidence': 0.5,
                    'type': 'table column'
                })
            
            for i in range(num_rows):
                for j in range(num_cols):
                    cell_x1 = x1 + (width * j / num_cols)
                    cell_x2 = x1 + (width * (j + 1) / num_cols)
                    cell_y1 = y1 + (height * i / num_rows)
                    cell_y2 = y1 + (height * (i + 1) / num_rows)
                    
                    cells.append({
                        'bbox': [cell_x1, cell_y1, cell_x2, cell_y2],
                        'confidence': 0.5,
                        'type': 'table cell'
                    })
            
            structure = {
                'rows': rows,
                'columns': columns,
                'cells': cells
            }
            
            print(f" Fallback structure: {len(rows)} rows, {len(columns)} columns, {len(cells)} cells")
            return structure
            
        except Exception as e:
            print(f" Fallback structure detection error: {e}")
            return {
                'rows': [{
                    'bbox': table_bbox,
                    'confidence': 0.3,
                    'type': 'table row'
                }],
                'columns': [{
                    'bbox': table_bbox,
                    'confidence': 0.3,
                    'type': 'table column'
                }],
                'cells': [{
                    'bbox': table_bbox,
                    'confidence': 0.3,
                    'type': 'table cell'
                }]
            }
    
    def visualize_table_detection(self, image_path, tables, output_path):
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                pil_img = Image.open(image_path)
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            
            if img is None:
                print(f" Could not read image: {image_path}")
                return None
            
            for i, table in enumerate(tables):
                bbox = table['bbox']
                x1, y1, x2, y2 = [int(coord) for coord in bbox]
                
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)  # Blue box
                
                confidence = table.get('confidence', 0)
                label = f"Table {i+1} ({confidence:.2f})"
                cv2.putText(img, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            success = cv2.imwrite(output_path, img)
            if success:
                print(f" Table visualization saved: {output_path}")
                return output_path
            else:
                print(f" Failed to save table visualization")
                return None
                
        except Exception as e:
            print(f" Table visualization error: {e}")
            return None