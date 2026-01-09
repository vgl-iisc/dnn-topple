"""EMNIST loader converted to a PyTorch Dataset + DataLoader helper.

This follows the same structure as the MNIST loader but supports three EMNIST variants:
- emnist (ByClass): 62 classes (digits + upper + lower case letters)
- emnist_balanced (Balanced): 47 classes (balanced distribution)
- emnist_letters (Letters): 26 classes (only letters A-Z)

Usage example:
    train_loader, test_loader = make_emnist_dataloaders(
        data_root, variant='balanced', batch_size=64)
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
    """Read EMNIST images and labels from the original IDX files.

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

# Number of classes for each EMNIST variant
EMNIST_NUM_CLASSES = {
    'byclass': 62,
    'balanced': 47,
    'letters': 26
}


def get_emnist_transforms() -> Tuple[object, object]:
    """Return (train_transform, test_transform) for EMNIST as torchvision transforms.

    These transforms assume input is a torch.Tensor with shape (1,H,W).
    EMNIST images need to be transposed and flipped to match MNIST orientation.
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


class EmnistDataset(Dataset):
    """PyTorch Dataset for raw EMNIST IDX files.

    Each sample returned is a tuple (image_tensor, label_tensor) where
    image_tensor is float32 in range [0, 1] with shape (1, 28, 28) and
    label_tensor is a long (int64) scalar.
    """

    def __init__(self, images_filepath: str, labels_filepath: str, variant: str = 'balanced', 
                 transform: Optional[Callable] = None, permute: bool = True, 
                 subset: Optional[int] = None, random_label_prop: float = 0.0):
        self.images, self.labels = read_images_labels(images_filepath, labels_filepath)
        
        # EMNIST images are stored rotated, need to transpose and flip
        self.images = np.transpose(self.images, (0, 2, 1))
        self.images = np.flip(self.images, axis=1).copy()
        
        self.variant = variant
        self.num_classes = EMNIST_NUM_CLASSES.get(variant, 47)
                
        if transform is not None:
            self.transform = transform
        else:
            self.transform = T.ToTensor()
            
        if random_label_prop > 0.0:
            self.labels = np.array(self.labels, dtype=np.uint8)
            rng = np.random.RandomState(SEED)
            random_indices = rng.choice(len(self.labels), size=int(len(self.labels) * random_label_prop), replace=False)
            self.labels[random_indices] = rng.randint(0, self.num_classes, size=len(random_indices))
            
        if subset is not None:
            self.images = self.images[:subset]
            self.labels = self.labels[:subset]

        torch.random.manual_seed(SEED)
        self.perm = torch.randperm(len(self.labels))
        if not permute:
            self.perm = torch.arange(len(self.labels))

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


def make_emnist_dataloaders(
    data_root: str,
    variant: str = 'balanced',
    batch_size: int = 64,
    transform: Optional[Callable] = None,
    random_label_prop: float = 0.0,
    shuffle: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """Create train and test DataLoaders for EMNIST IDX files.

    Args:
        data_root: path to EMNIST base directory containing the 4 files:
            train-images.idx3-ubyte
            train-labels.idx1-ubyte
            t10k-images.idx3-ubyte
            t10k-labels.idx1-ubyte
        variant: EMNIST variant - 'byclass', 'balanced', or 'letters'
        batch_size: batch size for the DataLoaders
        shuffle: whether to shuffle the training DataLoader
        transform: optional callable applied to image tensors
        random_label_prop: proportion of labels to randomize (for experiments)

    Returns:
        (train_loader, test_loader)
    """

    train_images_filepath = join(data_root, "train-images.idx3-ubyte")
    train_labels_filepath = join(data_root, "train-labels.idx1-ubyte")
    test_images_filepath = join(data_root, "t10k-images.idx3-ubyte")
    test_labels_filepath = join(data_root, "t10k-labels.idx1-ubyte")

    train_ds = EmnistDataset(train_images_filepath, train_labels_filepath, variant=variant, 
                             transform=transform, random_label_prop=random_label_prop)
    test_ds = EmnistDataset(test_images_filepath, test_labels_filepath, variant=variant, 
                            transform=transform)

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
    

if __name__ == "__main__":
    from sys import argv
    
    if len(argv) < 4:
        print("Usage: python emnist_loader.py <data_root> <variant> <output_dir>")
        print("  variant: byclass, balanced, or letters")
        exit(1)
    
    train_loader, test_loader = make_emnist_dataloaders(
        data_root=argv[1],
        variant=argv[2],
        batch_size=1
    )
    output_split_csv(train_loader, test_loader, output_dir=argv[3])
