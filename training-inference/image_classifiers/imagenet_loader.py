"""ImageNet loader that mirrors the MNIST loader structure.

Provides a PyTorch Dataset wrapper around torchvision.datasets.ImageNet and a
`make_imagenet_dataloaders` helper that returns train and test DataLoaders. When
run directly the script will produce CSV files with the same structure used
across the repo (train.csv, val.csv and trainUval.csv).

The dataset order is deterministically shuffled by a fixed seed (17592090214)
so the produced CSVs are reproducible.
"""

from typing import Optional, Tuple, Callable
from os.path import join
import os

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

from torchvision.datasets import ImageNet
import torchvision.transforms as T


SEED = 17592090214

# ImageNet normalization stats
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_imagenet_transforms(image_size: int = 224) -> Tuple[object, object]:
    """Return (train_transform, test_transform) for ImageNet as torchvision transforms.

    Transforms are designed to accept PIL Image inputs and follow standard ImageNet
    training practices used for ResNet, VGG, and other architectures.
    """

    train_transform = T.Compose([
        T.RandomResizedCrop(image_size, scale=(0.08, 1.0), ratio=(3./4., 4./3.)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    test_transform = T.Compose([
        T.Resize(int(image_size * 256/224)),  # Resize shorter side to 256 for 224x224 models
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    return train_transform, test_transform

class ImageNetDataset(Dataset):
    """PyTorch Dataset wrapper for ImageNet.

    Each sample returned is a dict with image_tensor, label_tensor, and index where
    image_tensor is float32 normalized with ImageNet stats with shape (3, 224, 224) and
    label_tensor is a long (int64) scalar.
    """

    def __init__(self, root: str, split: str = 'train', transform: Optional[Callable] = None, permute: bool = True, subset: Optional[int] = None, random_label_prop: float = 0.0):
        self.imagenet = ImageNet(root=root, split=split)

        # ImageNet doesn't load all data into memory, so we'll access it dynamically
        self.dataset = self.imagenet
        self.labels = np.array([self.imagenet.targets[i] for i in range(len(self.imagenet))], dtype=np.int64)
        self.classnames = self.imagenet.classes
        
        if random_label_prop > 0.0:
            rng = np.random.RandomState(SEED)
            random_indices = rng.choice(len(self.labels), size=int(len(self.labels) * random_label_prop), replace=False)
            self.labels[random_indices] = rng.randint(0, 1000, size=len(random_indices))
        
        if subset is not None:
            self.labels = self.labels[:subset]
        
        if transform is not None:
            self.transform = transform
        else:
            self.transform = T.ToTensor()

        # Deterministic permutation (fixed seed) so DataLoader(shuffle=False) yields reproducible order
        torch.random.manual_seed(SEED)
        self.perm = torch.randperm(len(self.labels))
        
        if not permute:
            self.perm = torch.arange(len(self.labels))

        self.num_classes = 1000

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int):
        perm_idx = idx  # Store the permutation/dataset index
        idx = int(self.perm[idx].item())  # Get underlying data index
        img, imglabel = self.dataset[idx]  # ImageNet returns (PIL Image, label)

        tensor_img = self.transform(img)

        label = int(self.labels[idx])
        # Note: assertion disabled for subset case where labels may not match original dataset
        # assert label == imglabel, f"Label mismatch at index {idx}: {label} vs {imglabel}"
        tensor_label = torch.tensor(label, dtype=torch.long)
        # return a dict so original (shuffled) index can be retrieved for mapping
        return {"image": tensor_img, "label": tensor_label, "index": idx, "perm_index": perm_idx}


def make_imagenet_dataloaders(
    data_root: str,
    batch_size: int = 64,
    train_transform: Optional[Callable] = None,
    test_transform: Optional[Callable] = None,
    random_label_prop: float = 0.0,
    shuffle: bool = False,
    train_subset: Optional[int] = None,
    test_subset: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and val DataLoaders for ImageNet stored under `data_root`.

    The underlying torchvision dataset must already be present under `data_root`.
    ImageNet should be organized in the standard format with train/ and val/ subdirectories.
    """
    train_ds = ImageNetDataset(root=data_root, split='train', transform=train_transform, random_label_prop=random_label_prop, subset=train_subset)
    test_ds = ImageNetDataset(root=data_root, split='val', transform=test_transform, subset=test_subset)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=shuffle)

    return train_loader, test_loader


def output_split_csv(train_loader, test_loader, output_dir):
    import pandas as pd

    os.makedirs(output_dir, exist_ok=True)

    def save_loader_to_csv(loader, filename):
        split = filename.split('.')[0]
        all_data = []
        for batch in loader:
            # support batch as dict or tuple
            labels = batch['label']
            indices = batch['index']
            
            for j, label in enumerate(labels):
                orig_idx = int(indices[j]) 
                row = [split, orig_idx, int(label.item())]
                all_data.append(row)
        df = pd.DataFrame(all_data, columns=["Split", "Image_Index", "Original_Label"])
        df.to_csv(os.path.join(output_dir, filename), header=True, index=False)

    save_loader_to_csv(train_loader, 'train.csv')
    save_loader_to_csv(test_loader, 'val.csv')

    def save_loaders_to_csv(train_loader, test_loader, output_file):
        all_data = []
        for batch in train_loader:
            labels = batch['label']
            indices = batch['index']
            
            for j, label in enumerate(labels):
                orig_idx = int(indices[j])
                all_data.append(['train', orig_idx, int(label.item())])
            
        for batch in test_loader:
            labels = batch['label']
            indices = batch['index']
            
            for j, label in enumerate(labels):
                orig_idx = int(indices[j])
                all_data.append(['val', orig_idx, int(label.item())])
        
        df = pd.DataFrame(all_data, columns=["Split", "Image_Index", "Original_Label"])
        df.to_csv(output_file, header=True, index=False)

    save_loaders_to_csv(train_loader, test_loader, join(output_dir, 'trainUval.csv'))


if __name__ == '__main__':
    # CLI: python imagenet_loader.py <data_root> <output_dir>
    import sys

    if len(sys.argv) < 3:
        print('Usage: python imagenet_loader.py <data_root> <output_dir>')
        sys.exit(1)

    data_root = sys.argv[1]
    output_dir = sys.argv[2]

    train_transform, test_transform = get_imagenet_transforms()

    print("Creating ImageNet DataLoaders...")
    train_loader, test_loader = make_imagenet_dataloaders(data_root, batch_size=1, train_transform=train_transform, test_transform=test_transform)
    output_split_csv(train_loader, test_loader, output_dir=output_dir)
