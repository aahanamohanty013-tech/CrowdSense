import os
import argparse
from lwcc import LWCC
import matplotlib.pyplot as plt

def count_single_image(image_path, model_name="DM-Count", model_weights="SHA"):
    """
    Counts people in a single image and prints the result.
    """
    if not os.path.exists(image_path):
        print(f"Error: Image {image_path} not found.")
        return

    print(f"Loading model {model_name}...")
    model = LWCC.load_model(model_name=model_name, model_weights=model_weights)
    
    print(f"Processing {image_path}...")
    count, density_map = LWCC.get_count(image_path, model=model, return_density=True)
    
    print("-" * 30)
    print(f"FINAL COUNT: {count:.2f}")
    print("-" * 30)
    
    # Save a quick visualization
    output_name = f"result_{os.path.basename(image_path)}"
    plt.imshow(density_map, cmap='jet')
    plt.title(f"Count: {count:.2f}")
    plt.axis('off')
    plt.savefig(output_name)
    print(f"Density map saved as: {output_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LWCC Single Image Counter")
    parser.add_argument("image", type=str, help="Path to the image file")
    parser.add_argument("--model", type=str, default="DM-Count", help="Model name")
    parser.add_argument("--weights", type=str, default="SHA", help="Weights (SHA/SHB/QNRF)")
    
    args = parser.parse_args()
    count_single_image(args.image, args.model, args.weights)
