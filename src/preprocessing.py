import cv2
import numpy as np
from PIL import Image
import os
import tempfile

class DocumentPreprocessor:
    def __init__(self, output_dir="temp"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def pdf_to_images(self, pdf_path, dpi=300):
        
        try:
            import fitz  # PyMuPDF
            print("Converting PDF using PyMuPDF...")
            
            doc = fitz.open(pdf_path)
            image_paths = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                zoom = dpi / 72
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat)
                
                img_path = os.path.join(self.output_dir, f"page_{page_num+1}.png")
                pix.save(img_path)
                image_paths.append(img_path)
            
            doc.close()
            print(f"Successfully converted {len(image_paths)} pages")
            return image_paths
            
        except ImportError:
            print("PyMuPDF not available, trying pdf2image...")
        except Exception as e:
            print(f"PyMuPDF conversion failed: {e}, trying pdf2image...")
        
        try:
            from pdf2image import convert_from_path
            print("Converting PDF using pdf2image...")
            
            images = convert_from_path(pdf_path, dpi=dpi)
            image_paths = []
            
            for i, image in enumerate(images):
                img_path = os.path.join(self.output_dir, f"page_{i+1}.png")
                image.save(img_path, 'PNG')
                image_paths.append(img_path)
            
            print(f"Successfully converted {len(image_paths)} pages")
            return image_paths
            
        except ImportError:
            raise Exception(
                "No PDF conversion library available. Please install one:\n"
                "  pip install PyMuPDF  (recommended, no external dependencies)\n"
                "  OR\n"
                "  pip install pdf2image  (requires poppler binary)"
            )
        except Exception as e:
            raise Exception(
                f"Unable to convert PDF. Error: {e}\n\n"
                "If using pdf2image, poppler must be installed:\n"
                "  Windows: Download from https://github.com/oschwartz10612/poppler-windows/releases/\n"
                "  Linux: sudo apt-get install poppler-utils\n"
                "  Mac: brew install poppler\n\n"
                "OR install PyMuPDF (no poppler needed):\n"
                "  pip install PyMuPDF"
            )
    
    def preprocess_image(self, image_path, target_size=(1024, 1024)):
        img = cv2.imread(image_path)
        if img is None:
            pil_img = Image.open(image_path)
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        denoised = cv2.fastNlMeansDenoising(gray)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        processed = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        
        h, w = processed.shape[:2]
        if w > h:
            new_w = target_size[0]
            new_h = int(h * (target_size[0] / w))
        else:
            new_h = target_size[1]
            new_w = int(w * (target_size[1] / h))
        
        resized = cv2.resize(processed, (new_w, new_h))
        
        processed_path = image_path.replace('.png', '_processed.png')
        cv2.imwrite(processed_path, resized)
        
        return processed_path, resized