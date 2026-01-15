import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../training-inference')))

from cifar10_loader import make_cifar10_dataloaders
from mnist_loader import make_mnist_dataloaders
from imagenet_loader import make_imagenet_dataloaders
from emnist_loader import make_emnist_dataloaders

from vis_utils import LossLandscapeExperiment

import numpy as np
import csv

from torch.utils.data import ConcatDataset
import streamlit as st

@st.cache_data(hash_funcs={LossLandscapeExperiment: LossLandscapeExperiment.__hash__})
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

	ds_name_l = ds_name.lower()

	if ds_name_l == "mnist":
		train_loader, test_loader = make_mnist_dataloaders(root, batch_size=1)
		
	elif ds_name_l == "cifar" or ds_name_l == "cifar10":
		train_loader, test_loader = make_cifar10_dataloaders(root, batch_size=1)
  
	elif ds_name_l == "imagenet" or ds_name_l == "imagenetsb":
		train_loader, test_loader = make_imagenet_dataloaders(root, batch_size=1)
  
	elif ds_name_l == "emnist_letters":
		train_loader, test_loader = make_emnist_dataloaders(root, batch_size=1, variant="letters")

	ds = []
	print(order)
 
	for split in order:
		if split == "train":
			ds.append(train_loader.dataset)
		else:
			ds.append(test_loader.dataset)

	return ConcatDataset(ds)