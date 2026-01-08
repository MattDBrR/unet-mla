# Cellpose Segmentation and Evaluation Script

## English

This script uses **Cellpose**, a deep learning model based on **U-Net**, with additional layers for instance segmentation.

Cellpose generates **instance segmentation masks** for microscopy images. To fairly compare Cellpose predictions with a standard U-Net or other segmentation methods, the masks are converted to **binary masks** (0 = background, 1 = cells). This ensures a fair comparison regardless of different instance labels or extra outputs.

### What the script does

1. Loads all images and the corresponding ST/SEG masks from a folder.
2. Runs Cellpose segmentation on each image.
3. Converts the Cellpose masks into binary masks.
4. Compares the binary masks with the ground truth masks for each image.
5. Computes two evaluation metrics:
   - **Dice coefficient**: Measures overlap between predicted and ground truth masks.
   - **IoU (Intersection over Union)**: Measures the ratio of intersection over union of masks.
6. Stores the metrics in lists and computes the **average score** across all images.

---

## Requirements

This script requires **Python 3.10 or 3.11** and the following Python packages:

- `numpy` – for array and numerical operations  
- `matplotlib` – for plotting images and masks  
- `tifffile` – for reading TIFF microscopy images  
- `scipy` – required by Cellpose  
- `cellpose` – the deep learning instance segmentation model

### GPU Recommendation

- A **GPU** is recommended for faster processing.  
- The script can run on CPU, but it will be significantly slower.

### Installation Instructions

It's recommended to use a virtual environment:

```bash
# Create virtual environment (optional)
python -m venv venv

# Activate the environment
source venv/bin/activate  # Linux / Mac
venv\Scripts\activate     # Windows

# Upgrade pip
pip install --upgrade pip

# Install required packages
pip install numpy matplotlib tifffile scipy
pip install git+https://www.github.com/mouseland/cellpose.git
```
Once installed, the script can be run directly on your images and masks folder by editing the paths in the script:

### Notes

The script converts Cellpose instance segmentation masks to binary masks for fair comparison with standard U-Net or other segmentation outputs.

Evaluation metrics used are Dice coefficient and IoU (Intersection over Union).
