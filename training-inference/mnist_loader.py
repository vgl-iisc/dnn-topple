"""MNIST loader converted to a PyTorch Dataset + DataLoader helper.

This preserves the original raw MNIST parsing but exposes a torch.utils.data.Dataset
(`MnistDataset`) and a convenience `make_mnist_dataloaders` function that returns
train and test DataLoaders.

Usage example:
    train_loader, test_loader = make_mnist_dataloaders(
        train_images_path, train_labels_path,
        test_images_path, test_labels_path,
        batch_size=64)
"""

import struct
from array import array
from os.path import join
from typing import Optional, Tuple, Callable

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def read_images_labels(images_filepath: str, labels_filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read MNIST images and labels from the original IDX files.

    Returns:
        images: numpy array of shape (N, 28, 28), dtype=uint8
        labels: numpy array of shape (N,), dtype=uint8
    """
    with open(labels_filepath, 'rb') as file:
        magic, size = struct.unpack(">II", file.read(8))
        if magic != 2049:
            raise ValueError(f'Magic number mismatch for labels: expected 2049, got {magic}')
        labels = array("B", file.read())

    with open(images_filepath, 'rb') as file:
        magic, size2, rows, cols = struct.unpack(">IIII", file.read(16))
        if magic != 2051:
            raise ValueError(f'Magic number mismatch for images: expected 2051, got {magic}')
        if size != size2:
            raise ValueError(f'Label count ({size}) does not match image count ({size2})')
        image_data = array("B", file.read())

    # Convert to numpy arrays for easier handling
    n = size
    images = np.frombuffer(image_data.tobytes(), dtype=np.uint8)
    images = images.reshape(n, rows, cols)
    labels = np.frombuffer(labels.tobytes(), dtype=np.uint8)

    return images, labels

SEED = 1759143479

class MnistDataset(Dataset):
    """PyTorch Dataset for raw MNIST IDX files.

    Each sample returned is a tuple (image_tensor, label_tensor) where
    image_tensor is float32 in range [0, 1] with shape (1, 28, 28) and
    label_tensor is a long (int64) scalar.
    """

    def __init__(self, images_filepath: str, labels_filepath: str, transform: Optional[Callable] = None):
        self.images, self.labels = read_images_labels(images_filepath, labels_filepath)
        self.transform = transform
        
        torch.random.manual_seed(SEED) 
        self.perm = torch.randperm(len(self.labels))

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int):
        idx = int(self.perm[idx].item())
        
        img = self.images[idx].astype(np.float32) / 255.0  # normalize to [0,1]
        # add channel dim
        img = np.expand_dims(img, 0)  # (1, 28, 28)
        tensor_img = torch.from_numpy(img)
        if self.transform is not None:
            tensor_img = self.transform(tensor_img)
        label = int(self.labels[idx])
        tensor_label = torch.tensor(label, dtype=torch.long)
        return tensor_img, tensor_label


def make_mnist_dataloaders(
    train_images_filepath: str,
    train_labels_filepath: str,
    test_images_filepath: str,
    test_labels_filepath: str,
    batch_size: int = 64,
    transform: Optional[Callable] = None,
    shuffle: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and test DataLoaders for MNIST IDX files.

    Args:
        train_images_filepath, train_labels_filepath, test_*: paths to IDX files
        batch_size: batch size for the DataLoaders
        shuffle: whether to shuffle the training DataLoader
        num_workers: number of worker processes for data loading
        pin_memory: pin memory for faster cuda transfer
        transform: optional callable applied to image tensors

    Returns:
        (train_loader, test_loader)
    """
    train_ds = MnistDataset(train_images_filepath, train_labels_filepath, transform=transform)
    test_ds = MnistDataset(test_images_filepath, test_labels_filepath, transform=transform)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=shuffle
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=shuffle
    )

    return train_loader, test_loader

def output_split_csv(train_loader, test_loader, output_dir):
    import pandas as pd
    import os

    os.makedirs(output_dir, exist_ok=True)

    def save_loader_to_csv(loader, filename):
        split = filename.split('.')[0]
        
        all_data = []
        i = 0
        for _, labels in loader:
            for label in labels:
                row = [split, i, label.item()]
                all_data.append(row)
                i += 1
        df = pd.DataFrame(all_data, columns=["Split", "Image_Index", "Original_Label"])
        df.to_csv(os.path.join(output_dir, filename), header=True, index=False)

    save_loader_to_csv(train_loader, 'train.csv')
    save_loader_to_csv(test_loader, 'test.csv')

    def save_loaders_to_csv(train_loader, test_loader, output_file):
        all_data = []
        i = 0
        for _, labels in train_loader:
            for label in labels:
                row = ['train', i, label.item()]
                i += 1   
                all_data.append(row)
        for _, labels in test_loader:
            for label in labels:
                row = ['test', i, label.item()]
                i += 1
                all_data.append(row)
        df = pd.DataFrame(all_data, columns=["Split", "Image_Index", "Original_Label"])
        df.to_csv(output_file, header=True, index=False)
    
    save_loaders_to_csv(train_loader, test_loader, join(output_dir, 'trainUtest.csv'))
    
if __name__ == "__main__":
    from sys import argv
    
    if len(argv) < 6:
        print("Usage: python mnist_loader.py <train_images> <train_labels> <test_images> <test_labels> <output_dir>")
        exit(1)
    
    train_loader, test_loader = make_mnist_dataloaders(
        argv[1],
        argv[2],
        argv[3],
        argv[4],
        batch_size=1
    )
    output_split_csv(train_loader, test_loader, output_dir=argv[5])