import os
import argparse
import json
import csv
import re

def extract_original_name(filename):
    """
    Extracts the original name from a filename.
    Handles the JAI pattern: JAI_<og_name>_<quality>.<ext>
    Otherwise returns the filename without extension.
    """
    name, ext = os.path.splitext(filename)
    # Match JAI_<og_name>_<quality> pattern
    match = re.match(r'^JAI_(.+)_(012|025|050|075|100)$', name)
    if match:
        return match.group(1)
    return name

def main():
    parser = argparse.ArgumentParser(description="Extract ground-truth labels for a subset of images.")
    parser.add_argument("--input_dir", required=True, help="Directory containing the image subset")
    parser.add_argument("--output_csv", required=True, help="Path to write the output CSV")
    parser.add_argument("--json", required=True, nargs='+', help="Path(s) to the label JSON files (Test-Dev, Train, Val)")
    
    args = parser.parse_args()
    
    # 1. Scan input_dir and extract unique original names
    valid_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tif', '.tiff'}
    unique_og_names = set()
    
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory {args.input_dir} not found.")
        return

    for filename in os.listdir(args.input_dir):
        ext = os.path.splitext(filename)[1].lower()
        if ext in valid_exts:
            og_name = extract_original_name(filename)
            unique_og_names.add(og_name)
            
    if not unique_og_names:
        print(f"No valid images found in {args.input_dir}.")
        return
        
    print(f"Found {len(unique_og_names)} unique original images in {args.input_dir}.")

    # 2. Build lookup from JSON files
    # mapping: basename -> {'id': image_id, 'category_id': cat_id, 'file_name': original_json_filename}
    lookup = {}
    
    for json_path in args.json:
        if not os.path.isfile(json_path):
            print(f"Warning: JSON file {json_path} not found. Skipping.")
            continue
            
        print(f"Loading JSON: {json_path} ...")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Map image_id -> category_id (from annotations)
        id_to_category = {}
        for ann in data.get('annotations', []):
            img_id = ann['image_id']
            if img_id not in id_to_category:
                id_to_category[img_id] = ann['category_id']
                
        # Map basename -> lookup dict
        for img in data.get('images', []):
            file_name = img['file_name']
            img_id = img['id']
            basename = os.path.splitext(os.path.basename(file_name))[0]
            
            if img_id in id_to_category:
                lookup[basename] = {
                    'id': img_id,
                    'category_id': id_to_category[img_id],
                    'file_name': os.path.basename(file_name) # keep only the basename with extension for OG_IMG
                }

    print(f"Built lookup table for {len(lookup)} total images from JSON files.")

    # 3. Match and write CSV
    rows_written = 0
    missing = 0
    
    with open(args.output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['OG_IMG', 'ID_IMG', 'IS_SYNTH'])
        
        for og_name in unique_og_names:
            if og_name in lookup:
                info = lookup[og_name]
                writer.writerow([info['file_name'], info['id'], info['category_id']])
                rows_written += 1
            else:
                print(f"Warning: Image '{og_name}' not found in any provided JSON file. Skipping.")
                missing += 1
                
    print(f"\nExtraction complete!")
    print(f"CSV saved to: {args.output_csv}")
    print(f"Rows written: {rows_written}")
    if missing > 0:
        print(f"Images not found and skipped: {missing}")

if __name__ == "__main__":
    main()
