import os
import sys
import csv
import time

#this gets the direcotry containing the name
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
# tis searches for the main file in the main project
sys.path.append(PROJECT_ROOT)

# Import our style transfer function from the main project folder
from style_transfer1 import download_if_missing , stylize_image

def main():
    print("==================================================")
    print("  Neural Style Transfer - CSV Dataset Processing  ")
    print("==================================================")

    csv_path = os.path.join(SCRIPT_DIR,"image_dataset.csv")
    output_dir = os.path.join(PROJECT_ROOT,"dataset","csv_output")
    os.makedirs(output_dir, exist_ok=True)

    #chack if csv exists
    if not os.path.exists(csv_path):
        print(f"Error: could not find {csv_path}")
        return
    
    print(f"Reading dataset from: {csv_path}")

    start_time = time.time()
    successful = 0
    total_processed = 0

    with open(csv_path,mode='r',encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)

        for row in csv_reader:
            total_processed +=1
            image_id = row['image_id']
            url = row['image_url']
            desc = row['description']
            style = row['style_model']

            print(f"\nProcessing ID{image_id} ({desc})...")

            #1. download style model if missing
            model_path = os.path.join(PROJECT_ROOT,style)
            if not os.path.exists(model_path):
                print(f" warning: model {style} not found.")
                model_path = os.path.join(PROJECT_ROOT,"starry_night.t7")
                download_if_missing(model_path, "https://cs.stanford.edu/people/jcjohns/fast-neural-style/models/instance_norm/starry_night.t7")

            #setup input image
            input_to_process = os.path.join(PROJECT_ROOT, url)
            if not os.path.exists(input_to_process):
                print(f"Error: Local fine not found at {input_to_process}")
                continue
                
            #apply style transfer
            output_filename = f"art_{image_id}_{desc.replace(' ' , '_')}.jpg"
            output_path = os.path.join(output_dir,output_filename)
            print(f"Applying Neural Style Transfer...", end=" ", flush=True)
            
            t0 = time.time()
            try:
                if stylize_image(input_to_process,model_path,output_path):
