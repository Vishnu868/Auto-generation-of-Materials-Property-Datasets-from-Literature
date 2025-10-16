import numpy as np
from collections import defaultdict

class TextTableIntegrator:
    def __init__(self):
        pass
    
    def calculate_overlap(self, box1, box2):
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        
        return intersection / area1 if area1 > 0 else 0.0
    
    def map_text_to_cells(self, ocr_data, table_structure, overlap_threshold=0.3):
        cell_text_mapping = defaultdict(list)
        
        cells = table_structure.get('cells', [])
        
        if not cells and table_structure.get('rows') and table_structure.get('columns'):
            print("    Creating cell grid from rows and columns...")
            cells = self.create_cell_grid(
                table_structure['rows'], 
                table_structure['columns']
            )
        
        if not cells:
            print("    ⚠️ No cells found, creating default grid...")
            cells = self._create_default_grid(ocr_data, table_structure)
        
        print(f"    Mapping {len(ocr_data)} text items to {len(cells)} cells...")
        
        mapped_count = 0
        for text_item in ocr_data:
            text_bbox = [text_item['x1'], text_item['y1'], text_item['x2'], text_item['y2']]
            best_match = None
            best_overlap = 0
            
            for i, cell in enumerate(cells):
                overlap = self.calculate_overlap(text_bbox, cell['bbox'])
                if overlap > overlap_threshold and overlap > best_overlap:
                    best_overlap = overlap
                    best_match = i
            
            if best_match is not None:
                cell_text_mapping[best_match].append(text_item)
                mapped_count += 1
        
        print(f"    mapped {mapped_count}/{len(ocr_data)} text items to cells")
        
        return cell_text_mapping, cells
    
    def create_cell_grid(self, rows, columns):
        if not rows or not columns:
            return []
        
        cells = []
        
        rows_sorted = sorted(rows, key=lambda x: x['bbox'][1]) 
        cols_sorted = sorted(columns, key=lambda x: x['bbox'][0]) 
        
        for i, row in enumerate(rows_sorted):
            for j, col in enumerate(cols_sorted):
                cell_bbox = [
                    col['bbox'][0], 
                    row['bbox'][1],
                    col['bbox'][2],  
                    row['bbox'][3]  
                ]
                
                if cell_bbox[2] <= cell_bbox[0] or cell_bbox[3] <= cell_bbox[1]:
                    continue
                
                cells.append({
                    'bbox': cell_bbox,
                    'row': i,
                    'col': j,
                    'type': 'table cell'
                })
        
        print(f"    Created {len(cells)} cells from {len(rows_sorted)} rows × {len(cols_sorted)} columns")
        return cells
    
    def _create_default_grid(self, ocr_data, table_structure):
        if not ocr_data:
            return []
        
        rows = table_structure.get('rows', [])
        cols = table_structure.get('columns', [])
        
        if rows:
            y_min = min(r['bbox'][1] for r in rows)
            y_max = max(r['bbox'][3] for r in rows)
        else:
            y_min = min(t['y1'] for t in ocr_data)
            y_max = max(t['y2'] for t in ocr_data)
        
        if cols:
            x_min = min(c['bbox'][0] for c in cols)
            x_max = max(c['bbox'][2] for c in cols)
        else:
            x_min = min(t['x1'] for t in ocr_data)
            x_max = max(t['x2'] for t in ocr_data)
        
        num_rows = len(rows) if rows else 5
        num_cols = len(cols) if cols else 4
        
        cells = []
        row_height = (y_max - y_min) / num_rows
        col_width = (x_max - x_min) / num_cols
        
        for i in range(num_rows):
            for j in range(num_cols):
                cell_bbox = [
                    x_min + j * col_width,
                    y_min + i * row_height,
                    x_min + (j + 1) * col_width,
                    y_min + (i + 1) * row_height
                ]
                
                cells.append({
                    'bbox': cell_bbox,
                    'row': i,
                    'col': j,
                    'type': 'table cell'
                })
        
        print(f"    Created default {num_rows}×{num_cols} grid with {len(cells)} cells")
        return cells
    
    def build_table_matrix(self, cell_text_mapping, cells):
        if not cells:
            print("     No cells to build matrix from")
            return []
        
        rows_set = set(cell.get('row', 0) for cell in cells)
        cols_set = set(cell.get('col', 0) for cell in cells)
        
        if not rows_set or not cols_set:
            print("    Cells missing row/col indices")
            matrix = []
            for cell_idx in sorted(cell_text_mapping.keys()):
                text_items = cell_text_mapping[cell_idx]
                cell_text = " ".join([item['text'] for item in text_items])
                matrix.append([cell_text.strip()])
            return matrix
        
        max_row = max(rows_set)
        max_col = max(cols_set)
        
        matrix = [["" for _ in range(max_col + 1)] for _ in range(max_row + 1)]
        
        for cell_idx, text_items in cell_text_mapping.items():
            if cell_idx < len(cells):
                cell = cells[cell_idx]
                row = cell.get('row', 0)
                col = cell.get('col', 0)
                
                if not text_items:
                    continue
                
                sorted_items = sorted(text_items, key=lambda x: (x['y1'], x['x1']))
                
                if len(sorted_items) > 1:
                    lines = []
                    current_line = [sorted_items[0]]
                    
                    for item in sorted_items[1:]:
                        if abs(item['y1'] - current_line[0]['y1']) < 10:
                            current_line.append(item)
                        else:
                            lines.append(current_line)
                            current_line = [item]
                    lines.append(current_line)
                    
                    cell_text = '\n'.join([' '.join([t['text'] for t in line]) for line in lines])
                else:
                    cell_text = sorted_items[0]['text']
                
                matrix[row][col] = cell_text.strip()
        
        matrix = [row for row in matrix if any(cell.strip() for cell in row)]
        
        print(f"    Built {len(matrix)}×{len(matrix[0]) if matrix else 0} table matrix")
        
        if matrix:
            print(f"    Sample cells: {matrix[0][:3]}")
        
        return matrix