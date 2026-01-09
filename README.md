# U-Net : Convolutional Networks for Biomedical Image Segmentation

**Team:** Aya, Konstantinos, Joel & Matthieu

## 📋 Project Description

Implementation of U-Net architecture for biomedical image segmentation with support for multiple datasets including cell microscopy images. The project includes training, validation, and inference capabilities with comprehensive metrics tracking and visualization.

### Supported Datasets
- **DIC-HeLa**: Differential Interference Contrast microscopy of HeLa cells
- **PHC-U373**: Phase Contrast microscopy of U373 cells  
- **EM-SEG**: Electron Microscopy segmentation
- **MIXED**: Combined dataset from multiple sources

## 🚀 Features

- U-Net architecture with configurable skip connections
- Weighted loss functions with boundary emphasis
- Elastic deformation data augmentation
- Multiple segmentation metrics (IoU, Dice, Precision, Recall, F1)
- Learning rate scheduling with ReduceLROnPlateau
- Model checkpointing (best and last)
- Training progress visualization and plots
- Support for both binary and multi-class segmentation

## 📁 Project Structure

```
unet-mla/
├── data/                    # Dataset folder (see setup below)
├── weights/                 # Trained model weights (see setup below)
├── plots/                   # Training progress plots (generated)
├── interface.py             # Main training interface
├── u_net.py                 # U-Net model architecture
├── unet_dataset.py          # Dataset loader with augmentation
├── loss_functions.py        # Custom loss functions
├── metrics.py               # Segmentation metrics
├── test.ipynb              # Testing and inference notebook
└── README.md
```

## 🔧 Installation

### Requirements
- Python 3.8+
- PyTorch 1.12+
- CUDA (optional, for GPU acceleration)

### Install Dependencies

```bash
pip install torch torchvision torchaudio
pip install numpy scipy scikit-image
pip install tifffile opencv-python
pip install matplotlib tqdm
```

Or using a requirements file:

```bash
pip install -r requirements.txt
```

## 📦 Required Data and Weights

### 1. Data Folder
You must have a folder named **`data`** at the root of the project.

Download the data from the following Google Drive link:
🔗 https://drive.google.com/drive/folders/1zp8QdZq_mJDJMflOxZE_pubzlaoaFu_p?usp=sharing

**Expected structure:**
```
data/
├── dic-hela/
│   ├── train/
│   │   ├── 01/
│   │   ├── 02/
│   │   ├── 01_GT/
│   │   └── 02_GT/
│   └── test/
├── phc-u373/
├── em-seg/
└── mixed/
```

### 2. Weights Folder
You must also have a folder named **`weights`** containing the pretrained models.

Download the trained models from:
🔗 https://drive.google.com/drive/folders/1KUyPzjYjcPor9kpNOiWYp_6CqYbiDid6?usp=sharing

**Structure:**
```
weights/
├── dic-hela/
│   └── best0.XXXX.pt
├── phc-u373/
└── em-seg/
```

### ⚠️ Important Notes
- Make sure the folder names are exactly **`data`** and **`weights`**
- Do not rename files inside these folders unless you also update the code accordingly
- Both folders must be present before training or evaluation

## 🎯 Usage

### Training a Model

```python
from interface import Interface
from unet_dataset import DatasetName

# Initialize training interface
trainer = Interface(
    dataset_name=DatasetName.DIC_HELA,
    epochs=50,
    batch_size=8,
    learning_rate=1e-5,
    weight_decay=1e-4,
    val_split=0.2,
    augment=True,
    device="cuda",
    skip_connections=True,
    save_model_path="weights"
)

# Start training
train_losses, val_losses, train_ious, val_ious = trainer.train()
```

### Available Training Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `out_channels` | int | 2 | Number of output channels (1 for binary, 2 for binary classification) |
| `epochs` | int | 50 | Number of training epochs |
| `batch_size` | int | 8 | Batch size for training |
| `learning_rate` | float | 1e-5 | Initial learning rate |
| `weight_decay` | float | 1e-4 | Weight decay for optimizer |
| `val_split` | float | 0.2 | Validation split ratio |
| `augment` | bool | True | Enable data augmentation |
| `skip_connections` | bool | True | Enable U-Net skip connections |
| `metrics_thr` | float | 0.5 | Threshold for binary metrics |
| `plot` | bool | True | Generate training plots |

### Loading and Evaluating a Model

```python
# Load pretrained model
trainer = Interface(
    dataset_name=DatasetName.DIC_HELA,
    load_model_path="weights/dic-hela/best0.1234.pt",
    device="cuda"
)

# Run validation
val_loss, val_metrics = trainer.validate()
print(f"Validation IoU: {val_metrics['iou']:.4f}")
```

### Inference on Single Image

```python
import torch
from interface import Interface

# Load model
trainer = Interface(
    dataset_name=DatasetName.DIC_HELA,
    load_model_path="weights/dic-hela/best0.1234.pt"
)

# Load and predict
img, mask, _ = trainer.dataset_test[0]
img_cut, pred_mask = trainer.predict_proba(img, threshold=0.5)

# pred_mask contains the binary segmentation
```

### Using the Jupyter Notebook

Open `test.ipynb` for interactive examples including:
- Loading pretrained models
- Running predictions on test images
- Visualizing results
- Computing metrics

## 🏗️ Model Architecture

The U-Net architecture consists of:
- **Encoder (Contracting Path)**: 5 levels with max pooling
- **Bottleneck**: Dropout layer (p=0.5)
- **Decoder (Expanding Path)**: 4 levels with upsampling
- **Skip Connections**: Concatenation between encoder and decoder
- **Output Layer**: 1×1 convolution for final segmentation

### Key Features
- Batch normalization in each convolutional block
- ReLU activation functions
- Kaiming initialization for weights
- Configurable padding (original mode vs. same padding)

## 📊 Metrics

The following metrics are computed during training and validation:

- **IoU (Intersection over Union)**: Main metric for segmentation quality
- **Dice Coefficient**: Harmonic mean of precision and recall
- **Pixel Accuracy**: Overall pixel-wise accuracy
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall

## 🎨 Data Augmentation

The following augmentations are applied during training:

1. **Elastic Deformation**: Grid-based elastic transformations
   - Grid size: 3×3
   - Sigma: 10 pixels
   - Applied with 50% probability

2. **Random Horizontal Flip**: 50% probability
3. **Random Vertical Flip**: 50% probability
4. **Random Rotation**: ±15 degrees
5. **Random Translation**: ±28 pixels

## 📈 Training Output

After training, the following are generated:

1. **Model Checkpoints**:
   - `best{val_loss}.pt`: Best model based on validation loss
   - `last{val_loss}.pt`: Final model at end of training

2. **Training Plots** (in `plots/` folder):
   - Loss curves (train and validation)
   - IoU curves (train and validation)
   - Comparison with U-Net paper benchmarks

3. **Console Output**:
   - Epoch-wise loss and metrics
   - Learning rate updates
   - Best model notifications

## 🎓 Loss Functions

### For Binary Segmentation (`out_channels=1`)
**WeightedBCEWithLogitsLoss**: Binary cross-entropy with spatial weighting for boundary emphasis

### For Multi-class Segmentation (`out_channels=2`)
**UNetLoss**: Weighted cross-entropy loss with weight maps computed using:
- Background/foreground class balancing
- Boundary separation emphasis based on distance transforms

## 🔬 Benchmark Results

Target IoU scores from original U-Net paper:

| Dataset | Target IoU | Paper Reference |
|---------|-----------|-----------------|
| DIC-HeLa | 77.56% | U-Net Paper |
| PHC-U373 | 92.03% | U-Net Paper |

## 🐛 Troubleshooting

**CUDA Out of Memory:**
- Reduce `batch_size`
- Reduce image resolution
- Disable augmentation temporarily

**Poor Performance:**
- Increase `epochs`
- Adjust `learning_rate`
- Check data preprocessing
- Verify weight maps computation

**NaN Loss:**
- Check for invalid values in weight maps
- Reduce learning rate
- Verify data normalization

## 👥 Team

- Aya
- Konstantinos  
- Joel
- Matthieu

## 📄 License

This project is part of an academic assignment.

## 🙏 Acknowledgments

Based on the U-Net architecture from:
> Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI 2015.

## 📧 Contact

For questions or issues, please contact the team members or open an issue in the repository.
