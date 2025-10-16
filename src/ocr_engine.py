import cv2
import numpy as np
import os

class OCREngine:
    def __init__(self, lang='en'):
        self.ocr = None
        self.lang = lang
        self.ocr_type = None
    
    def _init_ocr(self):
        if self.ocr is None:
            try:
                from paddleocr import PaddleOCR
                self.ocr = PaddleOCR(
                    use_angle_cls=True,
                    lang=self.lang,
                    use_gpu=False,
                    show_log=False,
                    det_db_thresh=0.3, 
                    det_db_box_thresh=0.5,  
                    rec_batch_num=10  
                )
                self.ocr_type = "paddleocr"
                print("Enhanced PaddleOCR initialized")
            except:
                try:
                    from paddlex import create_pipeline
                    self.ocr = create_pipeline(pipeline="OCR")
                    self.ocr_type = "paddlex"
                    print("✅ PaddleX pipeline initialized")
                except Exception as e:
                    print(f"❌ Failed to initialize OCR: {e}")
                    self.ocr = "fallback"
                    self.ocr_type = "fallback"
    
    def extract_text_with_boxes(self, image_path):
        self._init_ocr()
        
        if self.ocr == "fallback":
            return self._fallback_ocr(image_path)
        
        try:
            preprocessed_path = self._preprocess_image(image_path)
            
            if self.ocr_type == "paddlex":
                result = self.ocr.predict(preprocessed_path)
            else:
                result = self.ocr.ocr(preprocessed_path, cls=True)
            
            if hasattr(result, '__iter__') and hasattr(result, '__next__'):
                result = list(result)
            
            if not result or len(result) == 0:
                print("⚠️ OCR returned empty results, trying fallback...")
                return self._fallback_ocr(image_path)
            
            first_result = result[0]
            
            if hasattr(first_result, '__dict__') and not isinstance(first_result, (list, tuple, str)):
                print("✅ Detected PaddleX OCRResult object")
                
                if hasattr(first_result, 'json'):
                    json_val = first_result.json
                    
                    if isinstance(json_val, dict):
                        result_parsed = self._parse_from_dict(json_val)
                        if result_parsed:
                            return self._sort_text_data(result_parsed)
                    elif callable(json_val):
                        json_data = json_val()
                        if isinstance(json_data, dict):
                            result_parsed = self._parse_from_dict(json_data)
                            if result_parsed:
                                return self._sort_text_data(result_parsed)
                
                parsed = self._parse_paddlex_result(first_result)
                if parsed:
                    return self._sort_text_data(parsed)
                
                if hasattr(first_result, 'items'):
                    try:
                        dict_data = dict(first_result.items())
                        result_parsed = self._parse_from_dict(dict_data)
                        if result_parsed:
                            return self._sort_text_data(result_parsed)
                    except:
                        pass
                
                print(" All PaddleX parsing failed, using fallback OCR")
                return self._fallback_ocr(image_path)
            
            elif isinstance(first_result, list):
                print(" Detected standard PaddleOCR list format")
                parsed = self._parse_standard_result(first_result)
                return self._sort_text_data(parsed)
            
            else:
                print(f" Unknown result format: {type(first_result)}")
                return self._fallback_ocr(image_path)
            
        except Exception as e:
            print(f"❌ OCR Error: {e}")
            import traceback
            traceback.print_exc()
            return self._fallback_ocr(image_path)
    
    def _preprocess_image(self, image_path):
        try:
            img = cv2.imread(image_path)
            if img is None:
                return image_path
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
            
            binary = cv2.adaptiveThreshold(
                denoised, 255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 
                11, 2
            )
            
            preprocessed_path = image_path.replace('.', '_preprocessed.')
            cv2.imwrite(preprocessed_path, binary)
            
            return preprocessed_path
            
        except Exception as e:
            print(f"⚠️ Preprocessing failed: {e}, using original image")
            return image_path
    
    def _sort_text_data(self, text_data):
        if not text_data:
            return text_data
        
        sorted_data = sorted(text_data, key=lambda x: (x['y1'], x['x1']))
        
        print(f"Sorted {len(sorted_data)} text regions by position")
        return sorted_data
    
    def _parse_paddlex_result(self, ocr_result):
        try:
            text_data = []
            
            dt_polys = None
            rec_text = []
            rec_score = []

            for poly_attr in ['rec_polys', 'dt_polys', 'boxes', 'bboxes', 'polygons']:
                if hasattr(ocr_result, poly_attr):
                    val = getattr(ocr_result, poly_attr)
                    if val is not None and hasattr(val, '__len__') and len(val) > 0:
                        dt_polys = val
                        print(f"DEBUG: Found polygons in '{poly_attr}' ({len(dt_polys)} items)")
                        break

            for text_attr in ['rec_texts', 'rec_text', 'texts', 'text', 'ocr_text']:
                if hasattr(ocr_result, text_attr):
                    val = getattr(ocr_result, text_attr)
                    if val is not None and hasattr(val, '__len__') and len(val) > 0:
                        rec_text = val
                        print(f"DEBUG: Found text in '{text_attr}' ({len(rec_text)} items)")
                        break

            for score_attr in ['rec_scores', 'rec_score', 'scores', 'confidences']:
                if hasattr(ocr_result, score_attr):
                    val = getattr(ocr_result, score_attr)
                    if val is not None and hasattr(val, '__len__') and len(val) > 0:
                        rec_score = val
                        print(f"DEBUG: Found scores in '{score_attr}' ({len(rec_score)} items)")
                        break

            if not dt_polys or len(dt_polys) == 0:
                print("⚠️ No bounding boxes found in OCRResult attributes")
                return []
            
            if len(rec_text) == 0 and hasattr(ocr_result, 'rec_boxes'):
                print("DEBUG: No rec_texts found, trying rec_boxes")
                rec_boxes = getattr(ocr_result, 'rec_boxes')
                if rec_boxes and len(rec_boxes) > 0:
                    for box_data in rec_boxes:
                        if isinstance(box_data, dict):
                            text = box_data.get('text', box_data.get('transcription', ''))
                            if text:
                                rec_text.append(text)
                                rec_score.append(box_data.get('score', box_data.get('confidence', 0.9)))
                        elif isinstance(box_data, (list, tuple)) and len(box_data) >= 2:
                            rec_text.append(box_data[0])
                            rec_score.append(box_data[1])
                    print(f"DEBUG: Extracted {len(rec_text)} texts from rec_boxes")
                
            print(f"DEBUG: Processing {len(dt_polys)} detected regions")
            
            for idx in range(len(dt_polys)):
                try:
                    coords = dt_polys[idx]
                    text = rec_text[idx] if idx < len(rec_text) else f"[Region_{idx+1}]"
                    confidence = rec_score[idx] if idx < len(rec_score) else 0.9
                    
                    if not text or not str(text).strip():
                        continue
                    
                    if hasattr(coords, 'tolist'):
                        coords = coords.tolist()
                    
                    x_coords = [point[0] for point in coords]
                    y_coords = [point[1] for point in coords]
                    
                    text_data.append({
                        'text': str(text).strip(),
                        'confidence': float(confidence),
                        'x1': min(x_coords),
                        'y1': min(y_coords),
                        'x2': max(x_coords),
                        'y2': max(y_coords),
                        'coords': coords
                    })
                    
                except Exception as e:
                    continue
            
            if text_data:
                print(f"Extracted {len(text_data)} text regions from PaddleX")
            
            return text_data
            
        except Exception as e:
            print(f" Error in _parse_paddlex_result: {e}")
            return []

    def _parse_from_dict(self, data_dict):
        try:
            text_data = []
            
            print(f"DEBUG: Dictionary keys: {list(data_dict.keys())}")
            
            ocr_data = None
            
            for key in data_dict.keys():
                val = data_dict[key]
                if isinstance(val, dict):
                    print(f"DEBUG: Found nested dict under key '{key}': {list(val.keys())}")
                    ocr_data = val
                    break
                elif isinstance(val, list) and len(val) > 0:
                    print(f"DEBUG: Found list under key '{key}' with {len(val)} items")
                    if isinstance(val[0], dict):
                        ocr_data = val
                        break
            
            if ocr_data is None:
                ocr_data = data_dict
            
            if isinstance(ocr_data, dict):
                dt_polys = (ocr_data.get('dt_polys') or ocr_data.get('rec_polys') or 
                           ocr_data.get('boxes') or ocr_data.get('bboxes') or 
                           ocr_data.get('polygons') or [])
                rec_text = (ocr_data.get('rec_texts') or ocr_data.get('rec_text') or 
                           ocr_data.get('texts') or ocr_data.get('text') or 
                           ocr_data.get('ocr_text') or [])
                rec_score = (ocr_data.get('rec_scores') or ocr_data.get('rec_score') or 
                            ocr_data.get('scores') or ocr_data.get('confidences') or [])
            elif isinstance(ocr_data, list):
                print(f"DEBUG: Processing list of {len(ocr_data)} OCR results")
                for item in ocr_data:
                    if isinstance(item, dict):
                        coords = item.get('bbox') or item.get('box') or item.get('polygon')
                        text = item.get('text') or item.get('transcription') or ""
                        conf = item.get('score') or item.get('confidence') or 0.9
                        
                        if coords and text and str(text).strip():
                            if hasattr(coords, 'tolist'):
                                coords = coords.tolist()
                            
                            x_coords = [point[0] for point in coords]
                            y_coords = [point[1] for point in coords]
                            
                            text_data.append({
                                'text': str(text).strip(),
                                'confidence': float(conf),
                                'x1': min(x_coords),
                                'y1': min(y_coords),
                                'x2': max(x_coords),
                                'y2': max(y_coords),
                                'coords': coords
                            })
                
                if text_data:
                    print(f" Parsed {len(text_data)} text regions from list")
                    return text_data
                else:
                    return []
            else:
                print(f" Unknown ocr_data type: {type(ocr_data)}")
                return []
            
            if not dt_polys or len(dt_polys) == 0:
                print(" No polygons found in dictionary")
                return []
            
            if len(rec_text) == 0:
                print("Found polygons but no texts - trying rec_boxes format")
                rec_boxes = ocr_data.get('rec_boxes', [])
                
                for idx in range(min(len(dt_polys), max(len(rec_boxes), len(dt_polys)))):
                    try:
                        coords = dt_polys[idx] if idx < len(dt_polys) else None
                        
                        if coords is None:
                            continue
                        
                        text = ""
                        confidence = 0.9
                        
                        if idx < len(rec_boxes):
                            box_data = rec_boxes[idx]
                            if isinstance(box_data, dict):
                                text = box_data.get('text', box_data.get('transcription', ''))
                                confidence = box_data.get('score', box_data.get('confidence', 0.9))
                            elif isinstance(box_data, (list, tuple)) and len(box_data) >= 2:
                                text = str(box_data[0])
                                confidence = float(box_data[1])
                        
                        if not text or not str(text).strip():
                            continue
                        
                        if hasattr(coords, 'tolist'):
                            coords = coords.tolist()
                        
                        x_coords = [point[0] for point in coords]
                        y_coords = [point[1] for point in coords]
                        
                        text_data.append({
                            'text': str(text).strip(),
                            'confidence': float(confidence),
                            'x1': min(x_coords),
                            'y1': min(y_coords),
                            'x2': max(x_coords),
                            'y2': max(y_coords),
                            'coords': coords
                        })
                    except Exception as e:
                        continue
                
                if text_data:
                    print(f" Parsed {len(text_data)} text regions from rec_boxes format")
                    return text_data
            
            print(f"DEBUG: Found {len(dt_polys)} polygons, {len(rec_text)} texts, {len(rec_score)} scores")
            
            for idx in range(len(dt_polys)):
                try:
                    coords = dt_polys[idx]
                    text = rec_text[idx] if idx < len(rec_text) else ""
                    confidence = rec_score[idx] if idx < len(rec_score) else 0.9
                    
                    if not text or not str(text).strip():
                        continue
                    
                    if hasattr(coords, 'tolist'):
                        coords = coords.tolist()
                    
                    x_coords = [point[0] for point in coords]
                    y_coords = [point[1] for point in coords]
                    
                    text_data.append({
                        'text': str(text).strip(),
                        'confidence': float(confidence),
                        'x1': min(x_coords),
                        'y1': min(y_coords),
                        'x2': max(x_coords),
                        'y2': max(y_coords),
                        'coords': coords
                    })
                except Exception as e:
                    continue
            
            if text_data:
                print(f" Parsed {len(text_data)} text regions from dict")
            
            return text_data
            
        except Exception as e:
            print(f" Error in _parse_from_dict: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _parse_standard_result(self, result_list):
        text_data = []
        
        for idx, line in enumerate(result_list):
            try:
                if not line or not isinstance(line, (list, tuple)) or len(line) < 2:
                    continue
                
                coords = line[0]
                text_info = line[1]
                
                if not coords or len(coords) < 4:
                    continue
                
                if not text_info or len(text_info) < 1:
                    continue
                
                text = str(text_info[0]) if text_info[0] else ""
                confidence = float(text_info[1]) if len(text_info) > 1 else 0.9
                
                if not text.strip():
                    continue
                
                x_coords = [point[0] for point in coords]
                y_coords = [point[1] for point in coords]
                
                text_data.append({
                    'text': text.strip(),
                    'confidence': confidence,
                    'x1': min(x_coords),
                    'y1': min(y_coords),
                    'x2': max(x_coords),
                    'y2': max(y_coords),
                    'coords': coords
                })
                
            except Exception as e:
                continue
        
        if text_data:
            print(f"Extracted {len(text_data)} text regions")
        
        return text_data
    
    def _fallback_ocr(self, image_path):
        print("Using enhanced fallback OCR...")
        
        try:
            if not os.path.isabs(image_path):
                image_path = os.path.abspath(image_path)

            img = cv2.imread(image_path)
            if img is None:
                print(f" Could not read image: {image_path}")
                return []
            
            h, w = img.shape[:2]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            
            binary1 = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 11, 2
            )
            
            _, binary2 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            binary = cv2.bitwise_or(binary1, binary2)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            contours, _ = cv2.findContours(
                binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            text_data = []
            min_area = 50  
            
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if area < min_area:
                    continue
                
                x, y, w_box, h_box = cv2.boundingRect(contour)
                
                if 10 < w_box < w*0.95 and 8 < h_box < h*0.6:
                    text_data.append({
                        'text': f'[Text_{i+1}]',
                        'confidence': 0.5,
                        'x1': x,
                        'y1': y,
                        'x2': x + w_box,
                        'y2': y + h_box,
                        'coords': [[x, y], [x + w_box, y], 
                                  [x + w_box, y + h_box], [x, y + h_box]]
                    })
            
            print(f" Fallback detected {len(text_data)} text regions")
            return text_data
            
        except Exception as e:
            print(f" Fallback OCR error: {e}")
            return []
    
    def visualize_ocr_results(self, image_path, text_data, output_path):
        try:
            if not os.path.isabs(image_path):
                image_path = os.path.abspath(image_path)

            img = cv2.imread(image_path)
            
            if img is None:
                print(f" Could not read image: {image_path}")
                return None
            
            if not text_data:
                print(f"No text data to visualize")
                cv2.imwrite(output_path, img)
                return output_path
            
            colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
            
            for idx, item in enumerate(text_data):
                x1, y1 = int(item['x1']), int(item['y1'])
                x2, y2 = int(item['x2']), int(item['y2'])
                
                color = colors[idx % len(colors)]
                
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                
                label = f"{idx+1}: {item['text'][:15]}... ({item['confidence']:.2f})"
                
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                cv2.rectangle(img, (x1, max(y1-20, 0)), (x1 + label_size[0], max(y1, 20)), color, -1)
                
                cv2.putText(img, label, (x1, max(y1-5, 15)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            cv2.imwrite(output_path, img)
            print(f" Visualization saved to: {output_path}")
            return output_path
        except Exception as e:
            print(f" Visualization error: {e}")
            return None