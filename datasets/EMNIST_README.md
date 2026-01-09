# EMNIST Datasets Integration

## Overview
Three EMNIST dataset variants have been successfully integrated into the project:
- **EMNIST (ByClass)**: 62 classes (0-9, A-Z, a-z)
- **EMNIST Balanced**: 47 classes (balanced distribution)
- **EMNIST Letters**: 26 classes (A-Z only)

## Directory Structure
```
datasets/
├── emnist/
│   ├── data/
│   ├── download.py
│   ├── classes.txt
│   ├── train.csv
│   ├── val.csv
│   └── trainUval.csv
├── emnist_balanced/
│   ├── data/
│   ├── download.py
│   ├── classes.txt
│   ├── train.csv
│   ├── val.csv
│   └── trainUval.csv
└── emnist_letters/
    ├── data/
    ├── download.py
    ├── classes.txt
    ├── train.csv
    ├── val.csv
    └── trainUval.csv
```

## Downloading the Datasets

Run the download scripts from the project root:

```bash
# EMNIST ByClass (62 classes)
python datasets/emnist/download.py

# EMNIST Balanced (47 classes)
python datasets/emnist_balanced/download.py

# EMNIST Letters (26 classes)
python datasets/emnist_letters/download.py
```

## Generating Split CSV Files

After downloading, generate the train/val split CSV files:

```bash
# EMNIST ByClass
python training-inference/emnist_loader.py ./datasets/emnist/data byclass ./datasets/emnist

# EMNIST Balanced
python training-inference/emnist_loader.py ./datasets/emnist_balanced/data balanced ./datasets/emnist_balanced

# EMNIST Letters
python training-inference/emnist_loader.py ./datasets/emnist_letters/data letters ./datasets/emnist_letters
```

## Using EMNIST in Training

Update your training YAML configuration file to include EMNIST datasets. Example:

```yaml
datasets:
  cifar: ./datasets/cifar/data
  mnist: ./datasets/mnist/data
  emnist: ./datasets/emnist/data
  emnist_balanced: ./datasets/emnist_balanced/data
  emnist_letters: ./datasets/emnist_letters/data

do:
  - densenet_emnist
  - resnet_emnist_balanced
  - vgg_emnist_letters

config:
  checkpoint_dir: ./checkpoints
  experiments_dir: ./training-inference/train_runs
  device: cuda
```

Then run training:

```bash
python training-inference/run_train.py --config your_config.yaml
```

## Dataset Details

### EMNIST (ByClass)
- **Training samples**: 697,932
- **Test samples**: 116,323
- **Classes**: 62 (digits 0-9 + uppercase A-Z + lowercase a-z)
- **Image size**: 28x28 grayscale

### EMNIST Balanced
- **Training samples**: 112,800
- **Test samples**: 18,800
- **Classes**: 47 (balanced distribution of digits and letters)
- **Image size**: 28x28 grayscale

### EMNIST Letters
- **Training samples**: 124,800
- **Test samples**: 20,800
- **Classes**: 26 (uppercase letters A-Z only)
- **Image size**: 28x28 grayscale

## Implementation Details

### Loader Features
- Compatible with the existing MNIST loader structure
- Supports deterministic shuffling with fixed seed (1759143479)
- Handles EMNIST image rotation/transposition automatically
- Supports random label proportions for experiments
- Returns batches as dictionaries with 'image', 'label', and 'index' keys
- Images are normalized using MNIST statistics (mean=0.1307, std=0.3081)
- Images are padded to 32x32 to match project conventions

### Training Integration
- Automatically selects correct variant based on dataset name
- Uses same transforms as MNIST
- Returns num_classes attribute for model initialization
- Fully compatible with existing training infrastructure

## Notes
- EMNIST images are automatically transposed and flipped to match MNIST orientation
- The CSV files track original indices for reproducibility
- All three variants use the same loader with different variant parameters
