import os
from main import TableExtractionPipeline
import pandas as pd

def run_demo():
    pipeline = TableExtractionPipeline(min_table_confidence=0.7)
    
    base_dir = r"E:\SEM_5\DATADRIVEN\Project\table_extraction"
    input_folder = os.path.join(base_dir, "input")
    
    
    output_folder = os.path.join(base_dir, "output")

    os.makedirs(input_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)
    
    print("=== Table Extraction Pipeline Demo ===\n")
    print(f"Looking for files in: {input_folder}")
    print(f"Results will be saved in: {output_folder}\n")
    
    input_files = [f for f in os.listdir(input_folder) if f.endswith(('.pdf', '.png', '.jpg', '.jpeg'))]
    
    if not input_files:
        print(f" Please add PDF or image files to: {input_folder}")
        return
    
    for filename in input_files:
        print(f"Processing: {filename}")
        input_path = os.path.join(input_folder, filename)
        
        try:
            results = pipeline.process_document(input_path, output_dir=output_folder)
            
            print(f"\n--- Results for {filename} ---")
            for page, data in results.items():
                print(f"{page}: {len(data['tables'])} tables extracted")
                
                for table in data['tables']:
                    csv_path = os.path.join(output_folder, os.path.basename(table['csv_path']))
                    print(f"  Table {table['table_id']}: saved to {csv_path}")
                    
                    if table['matrix']:
                        df = pd.DataFrame(table['matrix'])
                        print("  Preview (first 3 rows):")
                        print(df.head(3).to_string(index=False))
                        print()
            
        except Exception as e:
            print(f" Error processing {filename}: {e}")
    
    print("\n Demo completed! Check the 'output/' directory for results.")

if __name__ == "__main__":
    run_demo()
