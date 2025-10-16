import numpy as np
import pandas as pd
import os

class CellMapper:
    
    def __init__(self, iou_threshold=0.3):
        self.iou_threshold = iou_threshold
    
    def calculate_iou(self, box1, box2):
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        if union_area == 0:
            return 0.0
        
        return inter_area / union_area
    
    def calculate_overlap_percentage(self, text_box, cell_box):
        x1_min, y1_min, x1_max, y1_max = text_box
        x2_min, y2_min, x2_max, y2_max = cell_box
        
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        text_area = (x1_max - x1_min) * (y1_max - y1_min)
        
        if text_area == 0:
            return 0.0
        
        return inter_area / text_area
    
    def map_text_to_cells(self, text_data, table_structure):
        cells = table_structure.get('cells', [])
        
        if not cells:
            print(" No cells in table structure, attempting grid-based mapping")
            return self._fallback_grid_mapping(text_data, table_structure)
        
        cell_text_map = {i: [] for i in range(len(cells))}
        
        for text_item in text_data:
            text_box = [text_item['x1'], text_item['y1'], 
                       text_item['x2'], text_item['y2']]
            
            best_cell_idx = -1
            best_overlap = 0.0
            
            for cell_idx, cell in enumerate(cells):
                cell_box = cell['bbox']
                
                overlap = self.calculate_overlap_percentage(text_box, cell_box)
                
                if overlap > best_overlap and overlap > 0.3: 
                    best_overlap = overlap
                    best_cell_idx = cell_idx
            
            if best_cell_idx >= 0:
                cell_text_map[best_cell_idx].append({
                    'text': text_item['text'],
                    'confidence': text_item['confidence'],
                    'overlap': best_overlap
                })
        
        cell_contents = {}
        for cell_idx, texts in cell_text_map.items():
            if texts:
                texts.sort(key=lambda x: x['confidence'], reverse=True)
                combined_text = ' '.join([t['text'] for t in texts])
                cell_contents[cell_idx] = combined_text
            else:
                cell_contents[cell_idx] = ''
        
        return cell_contents, cells
    
    def _fallback_grid_mapping(self, text_data, table_structure):
        rows = table_structure.get('rows', [])
        columns = table_structure.get('columns', [])
        
        if not rows or not columns:
            print(" No rows/columns found, using text positions directly")
            return self._direct_text_mapping(text_data)
        
        rows.sort(key=lambda r: r['bbox'][1]) 
        columns.sort(key=lambda c: c['bbox'][0]) 
        
        print(f"Creating grid: {len(rows)} rows x {len(columns)} columns")
        
        cells = []
        for row in rows:
            for col in columns:
                cell_box = [
                    col['bbox'][0],  
                    row['bbox'][1],
                    col['bbox'][2],
                    row['bbox'][3]
                ]
                cells.append({'bbox': cell_box})
        
        cell_text_map = {i: [] for i in range(len(cells))}
        
        for text_item in text_data:
            text_box = [text_item['x1'], text_item['y1'], 
                       text_item['x2'], text_item['y2']]
            
            best_cell_idx = -1
            best_overlap = 0.0
            
            for cell_idx, cell in enumerate(cells):
                overlap = self.calculate_overlap_percentage(text_box, cell['bbox'])
                
                if overlap > best_overlap and overlap > 0.2:
                    best_overlap = overlap
                    best_cell_idx = cell_idx
            
            if best_cell_idx >= 0:
                cell_text_map[best_cell_idx].append(text_item['text'])
        
        cell_contents = {}
        for cell_idx, texts in cell_text_map.items():
            cell_contents[cell_idx] = ' '.join(texts) if texts else ''
        
        return cell_contents, cells
    
    def _direct_text_mapping(self, text_data):
        cells = []
        cell_contents = {}
        
        for idx, text_item in enumerate(text_data):
            cells.append({
                'bbox': [text_item['x1'], text_item['y1'], 
                        text_item['x2'], text_item['y2']]
            })
            cell_contents[idx] = text_item['text']
        
        return cell_contents, cells
    
    def create_table_matrix(self, cell_contents, cells, num_rows=None, num_cols=None):
        if not cells:
            return [[]]
        
        if num_rows is None or num_cols is None:
            num_rows, num_cols = self._detect_grid_size(cells)
        
        print(f"Creating {num_rows}x{num_cols} table matrix")
        
        matrix = [['' for _ in range(num_cols)] for _ in range(num_rows)]
        
        sorted_cells = sorted(enumerate(cells), 
                             key=lambda x: (x[1]['bbox'][1], x[1]['bbox'][0]))
        
        for matrix_idx, (cell_idx, cell) in enumerate(sorted_cells):
            row = matrix_idx // num_cols
            col = matrix_idx % num_cols
            
            if row < num_rows and col < num_cols:
                text = cell_contents.get(cell_idx, '')
                matrix[row][col] = text
        
        return matrix
    
    def _detect_grid_size(self, cells):
        if not cells:
            return 1, 1
        
        y_positions = sorted(set(cell['bbox'][1] for cell in cells))
        x_positions = sorted(set(cell['bbox'][0] for cell in cells))
        
        def group_positions(positions, threshold=10):
            if not positions:
                return []
            groups = [[positions[0]]]
            for pos in positions[1:]:
                if pos - groups[-1][-1] < threshold:
                    groups[-1].append(pos)
                else:
                    groups.append([pos])
            return groups
        
        y_groups = group_positions(y_positions)
        x_groups = group_positions(x_positions)
        
        num_rows = len(y_groups)
        num_cols = len(x_groups)
        
        if num_rows == 0:
            num_rows = int(np.sqrt(len(cells)))
        if num_cols == 0:
            num_cols = len(cells) // num_rows if num_rows > 0 else 1
        
        return max(1, num_rows), max(1, num_cols)
    
    def save_table_to_csv(self, matrix, output_path):
        try:
            if not matrix or not matrix[0]:
                print("Empty table matrix, creating placeholder CSV")
                df = pd.DataFrame([['No data extracted']])
            else:
                df = pd.DataFrame(matrix)
            
            df.to_csv(output_path, index=False, header=False, encoding='utf-8-sig')
            print(f"Table saved to: {output_path}")
            return True
        except Exception as e:
            print(f" Error saving CSV: {e}")
            return False