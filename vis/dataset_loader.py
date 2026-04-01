import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../training-inference/image_classifiers')))

from cifar10_loader import make_cifar10_dataloaders
from mnist_loader import make_mnist_dataloaders
from imagenet_loader import make_imagenet_dataloaders
from emnist_loader import make_emnist_dataloaders

from experiment import LossLandscapeExperiment, BertExperiment

import numpy as np
import csv

from torch.utils.data import ConcatDataset
import streamlit as st

@st.cache_data(hash_funcs={LossLandscapeExperiment: LossLandscapeExperiment.__hash__, BertExperiment: BertExperiment.__hash__})
def load_dataset(exp: LossLandscapeExperiment):
	if isinstance(exp, BertExperiment) or st.session_state.is_bert:
		return None  # BERT data explorer uses token table view, not image dataset
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
 
	for split in order:
		if split == "train":
			ds.append(train_loader.dataset)
		else:
			ds.append(test_loader.dataset)

	return ConcatDataset(ds)


@st.cache_data(hash_funcs={BertExperiment: BertExperiment.__hash__})
def load_bert_flat_losses(exp: BertExperiment):
	"""Load per-token flat losses (N_valid,) for a BERT experiment.

	Uses the cached flat_losses_e{epoch}.txt written by compute_contour_trees_bert.py.
	Falls back to computing from the raw loss tensor + token_coords if the cache is absent.
	"""
	import numpy as np
	import torch

	cache_path = os.path.join(
		st.session_state.complexes_dir,
		exp.split,
		f"flat_losses_e{exp.epoch}.txt",
	)

	if os.path.exists(cache_path):
		return np.loadtxt(cache_path)

	# Compute on-the-fly if the cache file is missing
	loss_path = os.path.join(
		st.session_state.landscapes_dir,
		"Losses", exp.split, f"losses_{exp.epoch_tag}.pt",
	)
	coords_path = os.path.join(
		st.session_state.complexes_dir,
		exp.split,
		f"token_coords_a{exp.tag}_{exp.epoch_tag}.pt",
	)
	losses = torch.load(loss_path, weights_only=True)
	coords = torch.load(coords_path, weights_only=True)
	return losses[coords[:, 0], coords[:, 1]].float().numpy()