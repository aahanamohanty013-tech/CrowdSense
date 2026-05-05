import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
from lwcc import LWCC
from PIL import Image

def visualize_density(image_path, model_name="DM-Count", model_weights="SHA", resize_img=True):
    """
    Visualizes the density map for a crowd image.
    """
    if not os.path.exists(image_path):
        print(f"Image {image_path} not found.")
        return

    print(f"Loading model {model_name}...")
    model = LWCC.load_model(model_name=model_name, model_weights=model_weights)
    
    print(f"Processing {image_path}...")
    # Get count and density map
    count, density_map = LWCC.get_count(image_path, model=model, return_density=True, resize_img=resize_img)
    
    print(f"Predicted Count: {count:.2f}")
    
    # Load original image for display
    img = Image.open(image_path).convert('RGB')
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    
    # Original Image
    axes[0].imshow(img)
    axes[0].set_title(f"Original Image\nPath: {os.path.basename(image_path)}")
    axes[0].axis('off')
    
    # Density Map
    # Density map is usually a 2D numpy array
    im_dm = axes[1].imshow(density_map, cmap='jet')
    axes[1].set_title(f"Predicted Density Map\nEstimated Count: {count:.2f}")
    axes[1].axis('off')
    
    # Add colorbar
    fig.colorbar(im_dm, ax=axes[1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    # Save the visualization
    output_path = f"density_{os.path.basename(image_path)}"
    plt.savefig(output_path)
    print(f"Visualization saved to {output_path}")
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LWCC Density Map Visualizer")
    parser.add_argument("--image", type=str, required=True, help="Path to the crowd image")
    parser.add_argument("--model", type=str, default="DM-Count", help="Model name")
    parser.add_argument("--weights", type=str, default="SHA", help="Model weights")
    parser.add_argument("--no-resize", action="store_false", dest="resize_img", help="Disable resizing")
    
    args = parser.parse_args()
    
    visualize_density(args.image, args.model, args.weights, args.resize_img)
