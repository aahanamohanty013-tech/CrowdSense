import os
import argparse
import pandas as pd
from lwcc import LWCC
from glob import glob
from tqdm import tqdm

def process_folder(input_folder, output_csv, model_name="DM-Count", model_weights="SHA", resize_img=True):
    """
    Processes all images in a folder and saves crowd counts to a CSV file.
    """
    # Supported image extensions
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob(os.path.join(input_folder, ext)))
        image_paths.extend(glob(os.path.join(input_folder, ext.upper())))

    if not image_paths:
        print(f"No images found in {input_folder}")
        return

    print(f"Found {len(image_paths)} images. Loading model {model_name}...")
    
    # Load model once and reuse
    model = LWCC.load_model(model_name=model_name, model_weights=model_weights)
    
    results = []
    print("Processing images...")
    
    # LWCC can take a list of paths and return a dict
    # However, to show progress and handle potential errors per image, we'll process in chunks or individually
    # For now, let's use the list feature if it's efficient, otherwise individual
    
    try:
        # Batch processing (returns dict: {path: count})
        # Note: LWCC.get_count can take a list of paths
        counts_dict = LWCC.get_count(image_paths, model=model, resize_img=resize_img)
        
        for path, count in counts_dict.items():
            results.append({
                "image_path": path,
                "count": count
            })
    except Exception as e:
        print(f"Error during batch processing: {e}. Falling back to individual processing.")
        for path in tqdm(image_paths):
            try:
                count = LWCC.get_count(path, model=model, resize_img=resize_img)
                results.append({
                    "image_path": path,
                    "count": count
                })
            except Exception as e2:
                print(f"Error processing {path}: {e2}")
                results.append({
                    "image_path": path,
                    "count": "Error"
                })

    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Processing complete. Results saved to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LWCC Crowd Counter - Folder Processor")
    parser.add_argument("--input", type=str, required=True, help="Path to folder containing images")
    parser.add_argument("--output", type=str, default="counts.csv", help="Output CSV file path")
    parser.add_argument("--model", type=str, default="DM-Count", help="Model name (CSRNet, Bay, DM-Count, SFANet)")
    parser.add_argument("--weights", type=str, default="SHA", help="Model weights (SHA, SHB, QNRF)")
    parser.add_argument("--no-resize", action="store_false", dest="resize_img", help="Disable resizing for large dense images")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Input directory {args.input} does not exist.")
    else:
        process_folder(args.input, args.output, args.model, args.weights, args.resize_img)
