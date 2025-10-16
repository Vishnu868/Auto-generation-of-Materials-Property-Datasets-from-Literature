import pandas as pd
import numpy as np
from collections import defaultdict

class TableProcessor:
    def __init__(self):
        pass
    
    def process_table(self, structure, text_data):
        try:
            rows = structure.get('rows', [])
            columns = structure.get('columns', [])
            cells = structure.get('cells', [])
            
            if not rows and not columns and not cells:
                print(" No structure elements found")
                return None
            
            rows = sorted(rows, key=lambda x: x['bbox'][1])
            columns = sorted(columns, key=lambda x: x['bbox'][0])
            
            num_rows = len(rows) if rows else 1
            num_cols = len(columns) if columns else 1
            
            print(f"DEBUG: Processing table with {num_rows} rows and {num_cols} columns")
            
            if cells:
                return self._process_with_cells(cells, text_data, num_rows, num_cols)
            else:
                return self._process_with_grid(rows, columns, text_data)
            
        except Exception as e:
            print(f" Error processing table: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _process_with_cells(self, cells, text_data, num_rows, num_cols):
        try:
            cells = sorted(cells, key=lambda c: (c['bbox'][1], c['bbox'][0]))
            
            for cell in cells:
                cell['text'] = self._assign_text_to_cell(cell['bbox'], text_data)
            
            matrix = self._create_matrix_from_cells(cells, num_rows, num_cols)
            
            return {
                'matrix': matrix,
                'rows': num_rows,
                'columns': num_cols,
                'cells': cells
            }
            
        except Exception as e:
            print(f" Error in _process_with_cells: {e}")
            return None
    
    def _process_with_grid(self, rows, columns, text_data):
        try:
            num_rows = len(rows)
            num_cols = len(columns)
            
            matrix = [['' for _ in range(num_cols)] for _ in range(num_rows)]
            
            for i, row in enumerate(rows):
                for j, col in enumerate(columns):
                    cell_bbox = [
                        col['bbox'][0], 
                        row['bbox'][1], 
                        col['bbox'][2], 
                        row['bbox'][3]  
                    ]
                    
                    cell_text = self._assign_text_to_cell(cell_bbox, text_data)
                    matrix[i][j] = cell_text
            
            return {
                'matrix': matrix,
                'rows': num_rows,
                'columns': num_cols,
                'cells': []
            }
            
        except Exception as e:
            print(f" Error in _process_with_grid: {e}")
            return None
    
    def _assign_text_to_cell(self, cell_bbox, text_data):
        try:
            x1, y1, x2, y2 = cell_bbox
            cell_texts = []
            
            for text_item in text_data:
                tx1 = text_item['x1']
                ty1 = text_item['y1']
                tx2 = text_item['x2']
                ty2 = text_item['y2']
                
                text_center_x = (tx1 + tx2) / 2
                text_center_y = (ty1 + ty2) / 2
                
                if x1 <= text_center_x <= x2 and y1 <= text_center_y <= y2:
                    cell_texts.append({
                        'text': text_item['text'],
                        'y': text_center_y,
                        'x': text_center_x
                    })
            
            cell_texts.sort(key=lambda t: (t['y'], t['x']))
            
            combined_text = ' '.join([t['text'] for t in cell_texts])
            
            return combined_text.strip()
            
        except Exception as e:
            print(f" Error assigning text to cell: {e}")
            return ''
    
    def _create_matrix_from_cells(self, cells, num_rows, num_cols):
        try:
            matrix = [['' for _ in range(num_cols)] for _ in range(num_rows)]
            
            rows_dict = defaultdict(list)
            for cell in cells:
                y_center = (cell['bbox'][1] + cell['bbox'][3]) / 2
                rows_dict[y_center].append(cell)
            
            sorted_rows = sorted(rows_dict.keys())
            
            for row_idx, y_center in enumerate(sorted_rows):
                if row_idx >= num_rows:
                    break
                
                row_cells = sorted(rows_dict[y_center], key=lambda c: c['bbox'][0])
                
                for col_idx, cell in enumerate(row_cells):
                    if col_idx >= num_cols:
                        break
                    matrix[row_idx][col_idx] = cell.get('text', '')
            
            return matrix
            
        except Exception as e:
            print(f" Error creating matrix: {e}")
            return [['' for _ in range(num_cols)] for _ in range(num_rows)]
    
    def save_to_csv(self, table_data, output_path):
        try:
            if not table_data or not table_data.get('matrix'):
                print(" No table data to save")
                return None
            
            matrix = table_data['matrix']
            
            df = pd.DataFrame(matrix)
            
            df.to_csv(output_path, index=False, header=False, encoding='utf-8-sig')
            
            print(f" Table saved to: {output_path}")
            return output_path
            
        except Exception as e:
            print(f" Error saving CSV: {e}")
            return None
    
    def visualize_table_structure(self, image_path, structure, output_path):
        try:
            import cv2
            from PIL import Image
            
            img = cv2.imread(image_path)
            if img is None:
                pil_img = Image.open(image_path)
                img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            
            if img is None:
                print(f" Could not read image: {image_path}")
                return None
            
            for row in structure.get('rows', []):
                bbox = row['bbox']
                x1, y1, x2, y2 = [int(c) for c in bbox]
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            for col in structure.get('columns', []):
                bbox = col['bbox']
                x1, y1, x2, y2 = [int(c) for c in bbox]
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            for cell in structure.get('cells', []):
                bbox = cell['bbox']
                x1, y1, x2, y2 = [int(c) for c in bbox]
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 1)
            
            cv2.imwrite(output_path, img)
            print(f" Structure visualization saved: {output_path}")
            return output_path
            
        except Exception as e:
            print(f" Structure visualization error: {e}")
            return None