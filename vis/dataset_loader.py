import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../training-inference')))

from cifar10_loader import make_cifar10_dataloaders
from mnist_loader import make_mnist_dataloaders

from vis_utils import LossLandscapeExperiment

import numpy as np
import csv

from torch.utils.data import ConcatDataset

def load_dataset(exp: LossLandscapeExperiment):
	ds_name = exp.dataset.name
	
	csv_path = exp.dataset.get_split_path(exp.split)
 
	order = []
	current = None	
 
	with open(csv_path, 'r') as f:
		reader = csv.reader(f)
		lines = list(reader)[1:]  # Skip header

		for line in lines:
			split = line[0]
			if split != current:
				current = split
				order.append(current)

	root = os.path.join(exp.dataset.path, "data")

	if ds_name.lower() == "mnist":
		train_loader, test_loader = make_mnist_dataloaders(root, batch_size=1)
		
	elif ds_name.lower() == "cifar" or ds_name.lower() == "cifar10":
		train_loader, test_loader = make_cifar10_dataloaders(root, batch_size=1)
  

	ds = []
	print(order)
 
	for split in order:
		if split == "train":
			ds.append(train_loader.dataset)
		else:
			ds.append(test_loader.dataset)

	return ConcatDataset(ds)