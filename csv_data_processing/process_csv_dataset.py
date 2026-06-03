import os
import sys
import csv
import time

# Add project root to sys.path to import style_transfer
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(PROJECT_ROOT)

from style_transfer import download_if_missing, stylize_image, STYLE_MODEL_URL

def main():
    print("==================================================")
    print("  Neural Style Transfer - CSV Dataset Processing  ")
    print("==================================================")
    
    csv_path = os.path.join(SCRIPT_DIR, "image_dataset.csv")
    output_dir = os.path.join(PROJECT_ROOT, "dataset", "csv_output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if CSV exists
    if not os.path.exists(csv_path):
        print(f"Error: Could not find {csv_path}")
        return
        
    print(f"Reading dataset from: {csv_path}")
    
    start_time = time.time()
    successful = 0
    failed = 0
    total_processed = 0
    
    with open(csv_path, mode='r', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)
        
        # Validate CSV has required columns
        required_cols = {'image_id', 'image_url', 'description', 'style_model'}
        if not required_cols.issubset(set(csv_reader.fieldnames or [])):
            missing = required_cols - set(csv_reader.fieldnames or [])
            print(f"Error: CSV is missing required columns: {missing}")
            return
        
        for row in csv_reader:
            total_processed += 1
            image_id = row['image_id']
            url = row['image_url']
            desc = row['description']
            style = row['style_model']  # E.g., starry_night.t7
            
            print(f"\nProcessing ID {image_id} ({desc})...")
            
            # 1. Download the specific style model if we don't have it
            model_path = os.path.join(PROJECT_ROOT, style)
            if not os.path.exists(model_path):
                print(f"  Warning: Model {style} not found. Using starry_night.t7 instead.")
                model_path = os.path.join(PROJECT_ROOT, "starry_night.t7")
                download_if_missing(model_path, STYLE_MODEL_URL)
            
            # 2. Setup Input Image
            input_to_process = os.path.join(PROJECT_ROOT, url)
            if not os.path.exists(input_to_process):
                print(f"  Error: Local file not found at {input_to_process}")
                failed += 1
                continue
                
            # 3. Apply Style Transfer
            output_filename = f"art_{image_id}_{desc.replace(' ', '_')}.jpg"
            output_path = os.path.join(output_dir, output_filename)
            print(f"  Applying Neural Style Transfer...", end=" ", flush=True)
            
            t0 = time.time()
            try:
                if stylize_image(input_to_process, model_path, output_path):
                    t1 = time.time()
                    print(f"OK ({t1-t0:.2f}s)")
                    successful += 1
                else:
                    print("FAILED")
                    failed += 1
            except Exception as e:
                print(f"ERROR: {e}")
                failed += 1
                
    total_time = time.time() - start_time
    
    print("\n==================================================")
    print("  CSV Evaluation Results ")
    print("==================================================")
    print(f"Total Images Processed: {total_processed}")
    print(f"Successful:             {successful}")
    print(f"Failed:                 {failed}")
    print(f"Total Processing Time:  {total_time:.2f} seconds")
    if successful > 0:
        print(f"Average Time per Image: {total_time/successful:.2f} seconds")
    print(f"Results saved in:       {output_dir}")
    print("==================================================")

if __name__ == "__main__":
    main()
