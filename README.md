# LWCC Crowd Counting Tool

This project provides scripts to process crowd images using the LWCC (LightWeight Crowd Counting) library.

## Features
- **Folder Processing**: Batch process a directory of images and export counts to a CSV file.
- **Density Visualization**: Generate heatmaps of crowd density for individual images.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Process a Folder of Images
To count people in all images within a folder and save to a CSV:
```bash
python crowd_counter.py --input path/to/images --output counts.csv
```

**Arguments:**
- `--input`: Path to folder containing images.
- `--output`: Path to save the CSV results (default: `counts.csv`).
- `--model`: Model name (default: `DM-Count`). Options: `CSRNet`, `Bay`, `DM-Count`, `SFANet`.
- `--weights`: Model weights (default: `SHA`). Options: `SHA` (dense), `SHB` (sparse), `QNRF` (diverse).
- `--no-resize`: Disable automatic image resizing (recommended for very high-res dense crowds).

### 2. Visualize Density Map
To generate a density map heatmap for a specific image:
```bash
python visualize.py --image path/to/image.jpg
```
The visualization will be saved as `density_<filename>.png`.

## Credits
Built using the [LWCC library](https://github.com/tersekmatija/lwcc).
