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
