import os
import argparse
from PIL import Image
import subprocess
import tempfile
import shutil

def main():
    parser = argparse.ArgumentParser(description="Selectively extract train/val/test splits directly using bsdtar.")
    parser.add_argument('--zip_path', type=str, required=True, help='Path to the massive dataset archive')
    parser.add_argument('--output_dir', type=str, required=True, help='Destination directory for the extracted images')
    parser.add_argument('--num_train', type=int, default=500, help='Number of train images per class (default: 500)')
    parser.add_argument('--num_val', type=int, default=100, help='Number of validation images per class (default: 100)')
    parser.add_argument('--num_test', type=int, default=100, help='Number of test images per class (default: 100)')
    
    args = parser.parse_args()

    zip_path = args.zip_path
    output_dir = args.output_dir
    total_to_extract = args.num_train + args.num_val + args.num_test

    # Create destination folders for train, val, and test
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, split, 'fake'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'real'), exist_ok=True)

    print(f"Opening archive: {zip_path}")
    print(f"We need {total_to_extract} total images per class (Train: {args.num_train}, Val: {args.num_val}, Test: {args.num_test})")
    
    try:
        print("Running bsdtar to scan file index (ignoring truncation errors)...")
        # Get the list of all files using bsdtar
        # We remove check=True because your file is truncated/corrupted at the end, 
        # so bsdtar will throw an exit code 1 when it hits the broken part.
        result = subprocess.run(['bsdtar', '-tf', zip_path], capture_output=True, text=True)
        all_files = result.stdout.splitlines()
        
        if result.returncode != 0:
            print("Note: bsdtar encountered errors reading the full archive (likely truncated). Proceeding with the files it did find!")
            
        fake_samples = []
        real_samples = []
        
        # Iterate through the index
        for f in all_files:
            if len(fake_samples) >= total_to_extract and len(real_samples) >= total_to_extract:
                break
                
            f_lower = f.lower()
            if not f_lower.endswith(('.jpg', '.jpeg', '.png')):
                continue
                
            if len(fake_samples) < total_to_extract and 'fake_10k/' in f:
                fake_samples.append(f)
            elif len(real_samples) < total_to_extract and ('real_10k/' in f or 'real_10/' in f):
                real_samples.append(f)
                
        print(f"Found {len(fake_samples)} fake images.")
        print(f"Found {len(real_samples)} real images.")
        
        # Split the samples so we never re-sample the same image
        train_fake = fake_samples[:args.num_train]
        val_fake = fake_samples[args.num_train : args.num_train + args.num_val]
        test_fake = fake_samples[args.num_train + args.num_val : total_to_extract]
        
        train_real = real_samples[:args.num_train]
        val_real = real_samples[args.num_train : args.num_train + args.num_val]
        test_real = real_samples[args.num_train + args.num_val : total_to_extract]

        all_selected_files = fake_samples + real_samples
        
        if not all_selected_files:
            print("No images found to extract!")
            return

        # Extract only the selected files using a temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            print(f"Extracting {len(all_selected_files)} files using bsdtar...")
            
            # Since command line limits can be exceeded if we pass 1000s of files, 
            # we write the files to extract into a text list and pass it to bsdtar
            list_file_path = os.path.join(tmpdir, "extract_list.txt")
            with open(list_file_path, 'w') as f:
                for file_path in all_selected_files:
                    f.write(file_path + '\n')
            
            # Remove check=True here as well, so it doesn't crash if it hits a broken file during extraction
            subprocess.run(['bsdtar', '-xf', zip_path, '-C', tmpdir, '-T', list_file_path])
            
            # Helper function to process extracted files
            def convert_and_move(file_list, dest_folder, label_name):
                print(f"Converting and moving {len(file_list)} {label_name} images...")
                for count, original_path in enumerate(file_list, 1):
                    extracted_file = os.path.join(tmpdir, original_path)
                    
                    if not os.path.exists(extracted_file):
                        print(f"Error: {original_path} was not extracted properly.")
                        continue
                        
                    try:
                        with Image.open(extracted_file) as img:
                            if img.mode != 'RGB':
                                img = img.convert('RGB')
                                
                            original_name = os.path.basename(original_path)
                            base_name = os.path.splitext(original_name)[0]
                            new_name = f"{base_name}.png"
                            
                            img.save(os.path.join(dest_folder, new_name), 'PNG')
                            
                        if count % 25 == 0:
                            print(f"  [{count}/{len(file_list)}] converted...")
                    except Exception as e:
                        print(f"Error processing {original_path}: {e}")

            print("\n--- Processing FAKE images ---")
            convert_and_move(train_fake, os.path.join(output_dir, 'train', 'fake'), "train")
            convert_and_move(val_fake, os.path.join(output_dir, 'val', 'fake'), "val")
            convert_and_move(test_fake, os.path.join(output_dir, 'test', 'fake'), "test")
            
            print("\n--- Processing REAL images ---")
            convert_and_move(train_real, os.path.join(output_dir, 'train', 'real'), "train")
            convert_and_move(val_real, os.path.join(output_dir, 'val', 'real'), "val")
            convert_and_move(test_real, os.path.join(output_dir, 'test', 'real'), "test")
            
        print(f"\nSuccess! Dataset is ready in: {output_dir}")
        
    except FileNotFoundError:
        print(f"Error: Could not find the archive at {zip_path} or 'bsdtar' is not installed.")
    except subprocess.CalledProcessError as e:
        print(f"bsdtar failed with error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
