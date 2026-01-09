import os
import numpy as np
import matplotlib.pyplot as plt
from cellpose import models, core, io, plot

# Enable Cellpose progress printing
io.logger_setup()

# Check if GPU is available
if core.use_gpu() == False:
    raise ImportError("No GPU access, please run on GPU runtime")

# Load the Cellpose model with GPU enabled
model = models.CellposeModel(gpu=True)

# Paths
# CHANGE THIS to your folder path containing images and masks
image_folder = "choose your path/images"   # folder with images
mask_folder  = "choose your path/masks"    # folder with corresponding ST/SEG masks

# Get sorted list of images and masks
image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.tif')])
mask_files  = sorted([f for f in os.listdir(mask_folder)  if f.endswith('.tif')])

# Ensure same number of files
if len(image_files) != len(mask_files):
    print(f"Warning: {len(image_files)} images vs {len(mask_files)} masks")

# Helper functions
def dice_coefficient(mask1, mask2):
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)
    intersection = np.logical_and(mask1, mask2).sum()
    return 2 * intersection / (mask1.sum() + mask2.sum())

def iou_score(mask1, mask2):
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union

# Lists to store metrics
dice_list = []
iou_list  = []

# Process each image 
for img_file, mask_file in zip(image_files, mask_files):
    # Load image
    img_path = os.path.join(image_folder, img_file)
    img = io.imread(img_path)
    if img.ndim == 2:
        img_selected = img
    else:
        img_selected = img[:, :, 0] 

    # Cellpose segmentation
    masks, _, _ = model.eval(
        img_selected,
        batch_size=32,
        flow_threshold=0.4,
        cellprob_threshold=0.0,
        normalize={"tile_norm_blocksize":0}
    )

    # Convert to binary mask
    binary_mask = (masks > 0).astype(np.uint8)

    # Load corresponding ST/SEG mask
    mask_path = os.path.join(mask_folder, mask_file)
    gt_mask = io.imread(mask_path)
    gt_mask = (gt_mask > 0).astype(np.uint8)

    # Compute metrics
    dice = dice_coefficient(binary_mask, gt_mask)
    iou  = iou_score(binary_mask, gt_mask)

    dice_list.append(dice)
    iou_list.append(iou)

    print(f"{img_file}: Dice={dice:.4f}, IoU={iou:.4f}")

# Convert lists to numpy arrays
dice_array = np.array(dice_list)
iou_array  = np.array(iou_list)

# Compute averages
dice_mean = dice_array.mean()
iou_mean  = iou_array.mean()

print("\n--- Summary ---")
print(f"Average Dice coefficient: {dice_mean:.4f}")
print(f"Average IoU score: {iou_mean:.4f}")
