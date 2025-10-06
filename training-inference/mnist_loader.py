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
import torchvision.transforms as T


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


def get_mnist_transforms() -> Tuple[object, object]:
    """Return (train_transform, test_transform) for MNIST as torchvision transforms.

    These transforms assume input is a torch.Tensor with shape (1,H,W).
    """
    train_transform = T.Compose([
        T.ToTensor(),
        T.Pad(2),
        T.Normalize((0.1307,), (0.3081,)),
        T.RandomHorizontalFlip(),
    ])
    test_transform = T.Compose([
        T.ToTensor(),
        T.Pad(2),
        T.Normalize((0.1307,), (0.3081,)),
    ])
    return train_transform, test_transform

class MnistDataset(Dataset):
    """PyTorch Dataset for raw MNIST IDX files.

    Each sample returned is a tuple (image_tensor, label_tensor) where
    image_tensor is float32 in range [0, 1] with shape (1, 28, 28) and
    label_tensor is a long (int64) scalar.
    """

    def __init__(self, images_filepath: str, labels_filepath: str, transform: Optional[Callable] = None, permute: bool = True):
        self.images, self.labels = read_images_labels(images_filepath, labels_filepath)
        self.transform = transform

        if transform is None:
            self.transform = T.ToTensor()

        torch.random.manual_seed(SEED)
        self.perm = torch.randperm(len(self.labels))
        if not permute:
            self.perm = torch.arange(len(self.labels))

        self.num_classes = 10

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int):
        idx = int(self.perm[idx].item())
        img = np.repeat(self.images[idx][:, :, np.newaxis], repeats=3, axis=2)

        tensor_img = self.transform(img)

        label = int(self.labels[idx])
        tensor_label = torch.tensor(label, dtype=torch.long)
        # return dict so caller can map back to original (shuffled) index
        return {"image": tensor_img, "label": tensor_label, "index": idx}


def make_mnist_dataloaders(
    data_root: str,
    batch_size: int = 64,
    transform: Optional[Callable] = None,
    shuffle: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and test DataLoaders for MNIST IDX files.

    Args:
        data_root: path to MNIST base directory containing the 4 files:
            train-images-idx3-ubyte
            train-labels-idx1-ubyte
            t10k-images-idx3-ubyte
            t10k-labels-idx1-ubyte
        batch_size: batch size for the DataLoaders
        shuffle: whether to shuffle the training DataLoader
        num_workers: number of worker processes for data loading
        pin_memory: pin memory for faster cuda transfer
        transform: optional callable applied to image tensors

    Returns:
        (train_loader, test_loader)
    """

    train_images_filepath = join(data_root, "train-images.idx3-ubyte")
    train_labels_filepath = join(data_root, "train-labels.idx1-ubyte")
    test_images_filepath = join(data_root, "t10k-images.idx3-ubyte")
    test_labels_filepath = join(data_root, "t10k-labels.idx1-ubyte")

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
        for batch in loader:
            labels = batch['label']
            indices = batch['index']
            for j, label in enumerate(labels):
                orig_idx = int(indices[j])
                all_data.append([split, orig_idx, int(label.item())])
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
    
if __name__ == "__main__":
    from sys import argv
    
    if len(argv) < 6:
        print("Usage: python mnist_loader.py <data_root> <output_dir>")
        exit(1)
    
    train_loader, test_loader = make_mnist_dataloaders(
        data_root=argv[1],
        batch_size=1
    )
    output_split_csv(train_loader, test_loader, output_dir=argv[2])