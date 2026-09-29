import argparse
import zipfile
import random
import os
import io
try:
    from PIL import Image
except ImportError:
    print("Pillow library is required for image conversion. Please install it using: pip install Pillow")
    exit(1)

def extract_random_images(zip_path, output_folder, num_images):
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            # Get a list of all files in the zip
            all_files = z.namelist()
            
            # Filter out directories and non-image files
            image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')
            image_files = [f for f in all_files if f.lower().endswith(image_extensions) and not f.endswith('/')]
            
            if not image_files:
                print("No images found in the zip file.")
                return
                
            # Select N random images (no duplicates)
            num_to_extract = min(num_images, len(image_files))
            if num_to_extract < num_images:
                print(f"Warning: Only found {num_to_extract} images in the zip file. Extracting all of them.")
                
            selected_images = random.sample(image_files, num_to_extract)
            
            for img_path in selected_images:
                # To handle directories inside the zip, extract just the filename for the output
                file_name = os.path.basename(img_path)
                
                if not file_name:
                    continue # Skip if it's a directory (should be caught by filter, but just in case)

                # Read from zip
                with z.open(img_path) as file_in_zip:
                    file_content = file_in_zip.read()
                    
                name, ext = os.path.splitext(file_name)
                
                # Check if it's a JPG/JPEG and convert to PNG
                if ext.lower() in ('.jpg', '.jpeg'):
                    new_file_name = name + '.png'
                    output_path = os.path.join(output_folder, new_file_name)
                    try:
                        # Use PIL to properly convert the image format
                        img = Image.open(io.BytesIO(file_content))
                        img.save(output_path, 'PNG')
                        print(f"Extracted and converted: {file_name} -> {new_file_name}")
                    except Exception as e:
                        print(f"Failed to convert {file_name}: {e}")
                else:
                    # Write other formats as is
                    output_path = os.path.join(output_folder, file_name)
                    with open(output_path, 'wb') as f_out:
                        f_out.write(file_content)
                    print(f"Extracted: {file_name}")
                    
    except zipfile.BadZipFile:
        print(f"Error: The file '{zip_path}' is not a valid zip file.")
    except FileNotFoundError:
        print(f"Error: The file '{zip_path}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract random images from a zip file and convert JPGs to PNGs.")
    parser.add_argument("zip_path", help="Path to the input zip file")
    parser.add_argument("output_folder", help="Path to the output folder")
    parser.add_argument("num_images", type=int, help="Number of random images to extract (no duplicates)")
    
    args = parser.parse_args()
    
    if args.num_images <= 0:
        print("Error: Number of images must be greater than 0.")
    else:
        extract_random_images(args.zip_path, args.output_folder, args.num_images)
