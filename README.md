# Auto-generation-of-Materials-Property-Datasets-from-Literature
# 🧠 TABLE_EXTRACTION — ML Pipeline for Automated Table Detection & Extraction

> End-to-end Machine Learning & OCR pipeline that extracts structured tables from PDFs and images.  
> Built using **PaddleOCR**, **HuggingFace Table Transformer**, and optional **LLM post-processing** for structure refinement and accuracy improvement.

---

## 🔖 Overview

`TABLE_EXTRACTION` is a **data-driven ML pipeline** that performs:
- **Table detection** inside PDF or image documents.
- **OCR text extraction** with cell-wise mapping.
- **Table reconstruction** into structured formats (CSV/JSON).
- **Quality validation** and **LLM-based post-correction** for refined outputs.

It combines **Computer Vision**, **Deep Learning**, and **Language Models** to transform unstructured tables into usable data for analysis or automation.

---

## 🧩 Features

- 📄 **PDF to Image Conversion** using `pdf2image` and Poppler  
- 🔍 **Table Detection** with HuggingFace Table Transformer  
- 🔠 **Text Extraction** via PaddleOCR (Tesseract fallback)  
- 🧮 **Cell Mapping** to reconstruct true table structure  
- 🧠 **LLM Refinement** for OCR correction & logical alignment  
- 📊 **Output Formats:** CSV, JSON, and annotated table images  
- 🧾 **Quality Metrics & Rule-based Validation**

---

## 🧱 Project Structure

TABLE_EXTRACTION/
├── input/ # Input PDFs or images
├── output/ # Extracted tables (CSV/JSON)
├── src/
│ ├── preprocessing.py # PDF→image conversion & enhancement
│ ├── ocr_engine.py # PaddleOCR / Tesseract text extraction
│ ├── table_detector.py # HuggingFace model for table detection
│ ├── integration.py # Merge OCR & table bounding boxes
│ ├── cell_mapper.py # Map text to cell grid structure
│ ├── table_processor.py # Main orchestrator
│ ├── llm_refiner.py # Optional LLM-based cleanup
│ ├── quality_metrics.py # Quality scoring / reporting
│ ├── rule_validator.py # Domain rules for table validation
│ └── validate.py # General consistency checks
├── temp/ # Temporary or intermediate files
├── demo.py # Example entry point
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md

yaml
Copy code

---

## ⚙️ Installation

### 🧪 Prerequisites
- Python **3.9+**
- Poppler (for PDF to image conversion)

### 🧰 Steps

```bash
# 1️⃣ Clone the repository
git clone https://github.com/<yourusername>/TABLE_EXTRACTION.git
cd TABLE_EXTRACTION

# 2️⃣ Create a virtual environment
python -m venv .venv
source .venv/bin/activate       # Linux/macOS
.venv\Scripts\activate          # Windows PowerShell

# 3️⃣ Install dependencies
pip install -r requirements.txt

# 4️⃣ Verify poppler installation
# (required for pdf2image)
