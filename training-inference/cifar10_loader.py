"""CIFAR-10 loader that mirrors the MNIST loader structure.

Provides a PyTorch Dataset wrapper around torchvision.datasets.CIFAR10 and a
`make_cifar10_dataloaders` helper that returns train and test DataLoaders. When
run directly the script will produce CSV files with the same structure used
across the repo (train.csv, test.csv and trainUtest.csv).

The dataset order is deterministically shuffled by a fixed seed (1759209098)
so the produced CSVs are reproducible.
"""

from typing import Optional, Tuple, Callable
from os.path import join
import os

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

try:
    from torchvision.datasets import CIFAR10
    import torchvision.transforms as T
except Exception:
    CIFAR10 = None
    T = None


SEED = 1759209098


# CIFAR transforms (moved here from the model loader)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def get_cifar10_transforms(image_size: int = 32) -> Tuple[object, object]:
    """Return (train_transform, test_transform) for CIFAR-10 as torchvision transforms.

    Transforms are designed to accept torch.Tensor inputs (C,H,W). If torchvision
    is not available this will raise.
    """
    if T is None:
        raise RuntimeError('torchvision.transforms not available; cannot build transforms')

    train_transform = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    return train_transform, test_transform


class Cifar10Dataset(Dataset):
    """PyTorch Dataset wrapper for CIFAR-10.

    Each sample returned is a tuple (image_tensor, label_tensor) where
    image_tensor is float32 in range [0,1] with shape (3, 32, 32) and
    label_tensor is a long (int64) scalar.
    """

    def __init__(self, root: str, train: bool = True, transform: Optional[Callable] = None, permute: bool = True):
        if CIFAR10 is None:
            raise RuntimeError('torchvision is required for Cifar10Dataset')

        self.cifar = CIFAR10(root=root, train=train, download=False)

        # torchvision.CIFAR10 provides numpy arrays for data and a list for targets
        # data has shape (N, H, W, C)
        self.images = np.asarray(self.cifar.data)
        self.labels = np.asarray(self.cifar.targets, dtype=np.int64)
        self.transform = transform

        # Deterministic permutation (fixed seed) so DataLoader(shuffle=False) yields reproducible order
        torch.random.manual_seed(SEED)
        self.perm = torch.randperm(len(self.labels))
        
        if not permute:
            self.perm = torch.arange(len(self.labels))

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int):
        idx = int(self.perm[idx].item())
        img = self.images[idx].astype(np.float32) / 255.0  # (H, W, C) in [0,1]
        # convert to channel-first (C, H, W)
        img = np.transpose(img, (2, 0, 1)).copy()
        tensor_img = torch.from_numpy(img)
        if self.transform is not None:
            tensor_img = self.transform(tensor_img)
        label = int(self.labels[idx])
        tensor_label = torch.tensor(label, dtype=torch.long)
        # return a dict so original (shuffled) index can be retrieved for mapping
        return {"image": tensor_img, "label": tensor_label, "index": idx}


def make_cifar10_dataloaders(
    data_root: str,
    batch_size: int = 64,
    transform: Optional[Callable] = None,
    shuffle: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and test DataLoaders for CIFAR-10 stored under `data_root`.

    The underlying torchvision dataset must already be present under `data_root`.
    If not present, set download=True in the dataset constructor (not done here to
    avoid unexpected network activity).
    """
    train_ds = Cifar10Dataset(root=data_root, train=True, transform=transform)
    test_ds = Cifar10Dataset(root=data_root, train=False, transform=transform)

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
    save_loader_to_csv(test_loader, 'test.csv')

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
                all_data.append(['test', orig_idx, int(label.item())])
        
        df = pd.DataFrame(all_data, columns=["Split", "Image_Index", "Original_Label"])
        df.to_csv(output_file, header=True, index=False)

    save_loaders_to_csv(train_loader, test_loader, join(output_dir, 'trainUtest.csv'))


if __name__ == '__main__':
    # CLI: python cifar10_loader.py <data_root> <output_dir>
    import sys

    if len(sys.argv) < 3:
        print('Usage: python cifar10_loader.py <data_root> <output_dir>')
        sys.exit(1)

    data_root = sys.argv[1]
    output_dir = sys.argv[2]

    train_loader, test_loader = make_cifar10_dataloaders(data_root, batch_size=1)
    output_split_csv(train_loader, test_loader, output_dir=output_dir)
