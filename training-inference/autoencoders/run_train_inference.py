"""
This script expects a YAML configuration describing the dataset, model,
and training hyperparameters. 
Run:
	python run_train.py --config path/to/config.yaml -o path/to/output.log
"""

import argparse
import os
import yaml
import time

from tqdm import tqdm

import copy

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import cifar10_loader
import mnist_loader
import emnist_loader

from autoencoders import create_autoencoder

from logging import Logger, FileHandler, Formatter, StreamHandler

logger = Logger(__name__)

def set_seed(seed: int):
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	import random
	random.seed(seed)
	import numpy as np
	np.random.seed(seed)

def get_dataloaders(cfg):
	ds = cfg["dataset"]
	batch_size = cfg["batch_size"]
 
	random_prop = cfg.get("random_label_prop", 0.0)
	logger.info(f'Loader using random label proportion: {random_prop}')

	if ds == 'cifar10' or ds == 'cifar':
		train_tf, test_tf = cifar10_loader.get_cifar10_transforms()
		train_loader, test_loader = cifar10_loader.make_cifar10_dataloaders(
			cfg['data_root'], batch_size=batch_size, train_transform=train_tf, test_transform=test_tf, shuffle=True, random_label_prop=random_prop
		)
	elif ds == 'mnist':
		train_tf, _ = mnist_loader.get_mnist_transforms()
		train_loader, test_loader = mnist_loader.make_mnist_dataloaders(
			cfg['data_root'], batch_size=batch_size, transform=train_tf, shuffle=True, random_label_prop=random_prop
		)
	elif ds == 'emnist':
		train_tf, _ = emnist_loader.get_emnist_transforms()
		train_loader, test_loader = emnist_loader.make_emnist_dataloaders(
			cfg['data_root'], variant='byclass', batch_size=batch_size, transform=train_tf, shuffle=True, random_label_prop=random_prop
		)
	elif ds == 'emnist_balanced':
		train_tf, _ = emnist_loader.get_emnist_transforms()
		train_loader, test_loader = emnist_loader.make_emnist_dataloaders(
			cfg['data_root'], variant='balanced', batch_size=batch_size, transform=train_tf, shuffle=True, random_label_prop=random_prop
		)
	elif ds == 'emnist_letters':
		train_tf, _ = emnist_loader.get_emnist_transforms()
		train_loader, test_loader = emnist_loader.make_emnist_dataloaders(
			cfg['data_root'], variant='letters', batch_size=batch_size, transform=train_tf, shuffle=True, random_label_prop=random_prop
		)
	else:
		raise ValueError(f'Unsupported dataset: {ds}')
	return train_loader, test_loader

def build_model(cfg, device):
	arch = cfg['model']
	checkpoint = cfg.get('checkpoint', None)

	model = create_autoencoder(config_name=arch, device=device, pretrained_path=checkpoint)
	return model

def attach_collection_hooks(model, collection, collected_activations):
	"""Attach hooks to model layers to collect activations during forward pass."""
	def find_module(path):
		parts = path.split('.')
		indexed_parts = [tuple(part.replace("]", "").split("[")) if "[" in part else (part, None) for part in parts]
		
		mod = model
		for part, idx in indexed_parts:
			mod = getattr(mod, part)
			if idx is not None:
				mod = mod[int(idx)]
		return mod

	hooks = []
	for item in collection:
		path = item['path']
		tag = item['tag']
		
		collected_activations[tag] = []
		mod = find_module(path)
		logger.info(f"Attaching hook to {path} with tag {tag}: found {mod.__class__.__name__}")
		
		def hook_fn(m, i, o, tag=tag):
			collected_activations[tag].append(i[0].detach().cpu())
			
		hooks.append(mod.register_forward_hook(hook_fn))
	
	return hooks

def make_optimizer(model, cfg):
	lr = cfg["lr"]
	momentum = cfg["momentum"]
	decay = cfg.get("weight_decay", 0.0)

	return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=decay)

def make_lr_scheduler(optimizer, cfg):
	schedule_cfg = cfg.get('schedule', {})
	if schedule_cfg.get('type') == 'cosineWR':
		return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
			optimizer, 
			T_0=schedule_cfg.get('T0', 10), 
			T_mult=schedule_cfg.get('T_mult', 2)
		)
	else:
		return None

def train_epoch(model, loader, criterion, criterion_collect, optimizer, device, epoch, writer):
	model.train()
	running_loss = 0.0
	total = 0
	
	collected_losses = []
	collected_labels = []
	collected_recons = []
	collected_perm_indices = []
	
	for step, batch in tqdm(enumerate(loader), total=len(loader), desc=f'Epoch {epoch}'):
		if isinstance(batch, dict):
			images = batch['image']
			labels = batch['label']
			perm_idxs = batch['perm_index']
		else:
			raise ValueError("Expected batch to be a dict with 'image' and 'label' keys.")
		
		images = images.to(device)
		labels = labels.to(device)

		reconstructed = model(images)
		loss = criterion(reconstructed, images)
		
		# Compute per-sample losses for collection
		with torch.no_grad():
			loss_per_sample = criterion_collect(reconstructed, images)
			loss_per_sample = loss_per_sample.view(images.size(0), -1).mean(dim=1)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		running_loss += loss.item() * images.size(0)
		total += images.size(0)

		if writer is not None and step % 100 == 0:
			writer.add_scalar('train/batch_loss', loss.item(), epoch * len(loader) + step)
		
		# Always collect outputs
		collected_losses.append(loss_per_sample.detach())
		collected_labels.append(labels.detach())
		collected_recons.append(reconstructed.detach())
		collected_perm_indices.append(perm_idxs.to(dtype=torch.uint64))

	epoch_loss = running_loss / total
	
	# Always return collected outputs
	return epoch_loss, {
		'losses': collected_losses,
		'labels': collected_labels,
		'reconstructed': collected_recons,
		'perm_indices': collected_perm_indices
	}


def validate(model, loader, criterion, criterion_collect, device):
	model.eval()
	running_loss = 0.0
	total = 0
	
	collected_losses = []
	collected_labels = []
	collected_recons = []
	collected_perm_indices = []
	
	with torch.no_grad():
		for batch in tqdm(loader, total=len(loader), desc='Validation'):
			images = batch['image']
			labels = batch['label']
			perm_idxs = batch['perm_index']
			
			images = images.to(device)
			labels = labels.to(device)
			
			reconstructed = model(images)
			loss = criterion(reconstructed, images)
			
			loss_per_sample = criterion_collect(reconstructed, images)
			loss_per_sample = loss_per_sample.view(images.size(0), -1).mean(dim=1)

			running_loss += loss.item() * images.size(0)
			total += images.size(0)
			
			collected_losses.append(loss_per_sample.detach())
			collected_labels.append(labels.detach())
			collected_recons.append(reconstructed.detach())
			collected_perm_indices.append(perm_idxs.to(dtype=torch.uint64))
	
	# Always return collected outputs
	return running_loss / total, {
		'losses': collected_losses,
		'labels': collected_labels,
		'reconstructed': collected_recons,
		'perm_indices': collected_perm_indices
	}


def save_checkpoint(state, ckpt_dir, epoch):
	os.makedirs(ckpt_dir, exist_ok=True)
	path = os.path.join(ckpt_dir, f'checkpoint_e{epoch}.pt')
	torch.save(state, path)
	return path

def save_inference_outputs(epoch, train_outputs, val_outputs, train_activations, val_activations, output_root):
	"""Save inference outputs collected during training/validation."""
	logger.info(f'Saving inference outputs for epoch {epoch}...')
	
	epoch_output_dir = os.path.join(output_root)
	os.makedirs(epoch_output_dir, exist_ok=True)
	
	import numpy as np
	
	def write_output(split, output, collected_activations):
		loss_dir = os.path.join(epoch_output_dir, "Losses", split)
		recons_dir = os.path.join(epoch_output_dir, "Reconstructions", split)
		tens_dir = os.path.join(epoch_output_dir, "Tensors", split)
		
		os.makedirs(loss_dir, exist_ok=True)
		os.makedirs(recons_dir, exist_ok=True)
		os.makedirs(tens_dir, exist_ok=True)
		
		collected_losses = torch.cat(output['losses']).detach().cpu()
		collected_recons = torch.cat(output['reconstructed']).detach().cpu()
		collected_perm_indices = torch.cat(output['perm_indices']).cpu().to(dtype=torch.int64)
		
		# The perm_indices tell us the dataset/permutation index (ds[0], ds[1], ds[2], ...)
		# We need to reorder from shuffled collection order back to permutation order
		reordered_losses = torch.zeros_like(collected_losses)
		reordered_recons = torch.zeros_like(collected_recons)
		
		reordered_losses[collected_perm_indices] = collected_losses
		reordered_recons[collected_perm_indices] = collected_recons
		
		for tag, activations in collected_activations.items():
			stacked = torch.cat(activations).detach().cpu()
			collated = stacked.reshape(len(stacked), -1)
			
			reordered_activations = torch.zeros_like(collated)
			reordered_activations[collected_perm_indices] = collated
			
			logger.info(f'Saving activations for tag {tag}, shape: {reordered_activations.shape}')
			torch.save(reordered_activations, os.path.join(tens_dir, f"a{tag}_e{epoch}.pt"))
		
		collected_losses_np = reordered_losses.numpy().reshape(-1, 1).squeeze()
		torch.save(reordered_recons, os.path.join(recons_dir, f"reconstructions_e{epoch}.pt"))
		
		np.savetxt(os.path.join(loss_dir, f"losses_e{epoch}.txt"), collected_losses_np)
	
	write_output("train", train_outputs, train_activations)
	write_output("val", val_outputs, val_activations)
	
	# Combine train+val - need to adjust perm_indices for val to account for train dataset size
	combined_outputs = {}
	for key in ['losses', 'labels', 'reconstructed']:
		combined_outputs[key] = train_outputs[key] + val_outputs[key]
	train_size = int(torch.cat(train_outputs['losses']).shape[0])
	train_perm_indices = [idx.to(dtype=torch.int64) for idx in train_outputs['perm_indices']]
	val_perm_indices_offset = [idx.to(dtype=torch.int64) + train_size for idx in val_outputs['perm_indices']]
	combined_outputs['perm_indices'] = train_perm_indices + val_perm_indices_offset
	combined_activations = {tag: train_activations[tag] + val_activations[tag] for tag in train_activations}
	write_output("trainUval", combined_outputs, combined_activations)

	logger.info(f'Inference outputs saved for epoch {epoch}')

def do_run(cfg, device, tboard_base, checkpoints_base, inference_cfg, only_last_best=False):
	train_loader, test_loader = get_dataloaders(cfg)
	logger.info(f"Using config: {cfg}")

	model = build_model(cfg, device)

	criterion = nn.MSELoss()
	criterion_collect = nn.MSELoss(reduction='none')
	optimizer = make_optimizer(model, cfg)
	
	scheduler = make_lr_scheduler(optimizer, cfg)

	writer = SummaryWriter(tboard_base)

	set_seed(cfg["seed"])

	try:
		writer.add_text('config', yaml.dump(cfg))
		writer.add_text('inference_config', yaml.dump(inference_cfg))
	except Exception:
		pass

	epochs = cfg["epochs"]

	best_val_loss = float('inf')
	best_epoch = 0
	ckpt_dir = checkpoints_base

	for epoch in range(1, epochs + 1):
		t0 = time.time()
		logger.info(f'--- Epoch {epoch}/{epochs} ---')
		
		should_collect_inference = False
		epochs_config = inference_cfg.get('epochs', 'all')
		if epochs_config == 'all' or 'all' in (epochs_config if isinstance(epochs_config, list) else []):
			should_collect_inference = True
		elif "last" in (epochs_config if isinstance(epochs_config, list) else [epochs_config]) and epoch == epochs:
			should_collect_inference = True
		elif isinstance(epochs_config, list) and len(epochs_config) == 3 and all(isinstance(x, int) for x in epochs_config):
			# Format: [start, end, step] - check if (epoch-1) is in this range
			should_collect_inference = (epoch - 1) in list(range(epochs_config[0], epochs_config[1], epochs_config[2]))
		elif isinstance(epochs_config, list):
			# Format: list of specific epochs
			should_collect_inference = (epoch - 1) in epochs_config
		elif isinstance(epochs_config, int):
			# Format: interval - every N epochs
			should_collect_inference = ((epoch - 1) % epochs_config == 0)

		logger.info(f'Collecting inference outputs this epoch: {should_collect_inference}')
		
		train_hooks = []
		val_hooks = []
		train_activations = {}
		val_activations = {}
		
		if should_collect_inference:
			collection = inference_cfg.get('collect', [])
			train_hooks = attach_collection_hooks(model, collection, train_activations)
		
		train_loss, train_outputs = train_epoch(model, train_loader, criterion, criterion_collect, optimizer, device, epoch, writer)
		
		if should_collect_inference:
			for h in train_hooks:
				h.remove()
			val_hooks = attach_collection_hooks(model, collection, val_activations)
		
		if scheduler is not None:
			scheduler.step()
		logger.info('Evaluating on validation set...')
		
		val_loss, val_outputs = validate(model, test_loader, criterion, criterion_collect, device)
		
		# Remove val hooks
		if should_collect_inference:
			for h in val_hooks:
				h.remove()

		writer.add_scalar('train/loss', train_loss, epoch)
		writer.add_scalar('val/loss', val_loss, epoch)

		for i, param_group in enumerate(optimizer.param_groups):
			writer.add_scalar(f'optimizer/group_{i}_lr', param_group.get('lr', 0.0), epoch)

		epoch_time = time.time() - t0
		logger.info(f'Epoch {epoch}/{epochs} - train_loss: {train_loss:.6f}, val_loss: {val_loss:.6f} ({epoch_time:.1f}s)')

		if should_collect_inference:
			try:
				save_inference_outputs(epoch - 1, train_outputs, val_outputs, train_activations, val_activations, inference_cfg['output_root'])
			except Exception as e:
				logger.error(f'Error saving inference outputs for epoch {epoch}: {e}')

		state = {
			'epoch': epoch,
			'model_state_dict': model.state_dict(),
			'optimizer_state_dict': optimizer.state_dict(),
			'cfg': cfg,
		}
  
		if only_last_best:
			if epoch == epochs:
				save_checkpoint(state, ckpt_dir, epoch)
		else:
			save_checkpoint(state, ckpt_dir, epoch)

		if val_loss < best_val_loss:
			best_val_loss = val_loss
			if best_epoch > 0:
				try:
					os.remove(os.path.join(ckpt_dir, f'best_e{best_epoch}.pt'))
				except FileNotFoundError:
					pass
			best_epoch = epoch
			best_path = os.path.join(ckpt_dir, f'best_e{epoch}.pt')
			torch.save(state, best_path)
   
	logger.info(f'Best validation loss: {best_val_loss:.6f} @ {best_epoch}')

	writer.close()

def main(argv=None):
	parser = argparse.ArgumentParser()
	parser.add_argument('--config', '-c', required=True, help='YAML config file')
	parser.add_argument('--output_file', '-o', help='output file to log metrics to')
	parser.add_argument('--tensorboard_dir', default=None, help='override tensorboard dir from config')
	parser.add_argument('--checkpoint_dir', default=None, help='override checkpoint dir from config')
	args = parser.parse_args(argv)

	with open(args.config, 'r') as f:
		config = yaml.safe_load(f)
  
	train_cfg = config["train"]
	infer_cfg = config["infer"]

	# Setup logging
	log_formatter = Formatter('%(asctime)s - %(levelname)s - %(message)s')
	
	if args.output_file:
		file_handler = FileHandler(args.output_file)
		file_handler.setFormatter(log_formatter)
		logger.addHandler(file_handler)
		
	stream_handler = StreamHandler()
	stream_handler.setFormatter(log_formatter)
	logger.addHandler(stream_handler)
	logger.setLevel('INFO')

	# Get dataset root
	dataset_name = train_cfg['dataset']
	train_cfg['data_root'] = config['datasets'][dataset_name]
	
	device = torch.device(config["device"])
	logger.info(f'Using device: {device}')

	# Override tensorboard and checkpoint dirs if specified
	tensorboard_dir = args.tensorboard_dir or train_cfg.get('tensorboard_dir')
	checkpoint_dir = args.checkpoint_dir or train_cfg.get('checkpoints_dir')
	
	# Create run name and directories
	run_name = config["name"]
	datestr = time.strftime('%Y%m%d_%H%M%S')
	tb_dir = os.path.join(tensorboard_dir, run_name, datestr)
	ckpt_dir = os.path.join(checkpoint_dir, run_name, datestr)

	os.makedirs(tb_dir, exist_ok=True)
	os.makedirs(ckpt_dir, exist_ok=True)

	run_inference_cfg = copy.deepcopy(infer_cfg)
	run_inference_cfg['model'] = train_cfg['model']
	run_inference_cfg['output_root'] = os.path.join(infer_cfg['output_dir'], run_name, datestr)
	logger.info(f'Inference will be collected at: {run_inference_cfg["output_root"]}')
	logger.info(f'Inference epochs schedule: {run_inference_cfg.get("epochs", "all")}')
	logger.info(f'Inference collection layers: {run_inference_cfg.get("collect", [])}')

	only_last_best = train_cfg.get("only_last_best", False)

	logger.info(f'\n=== Starting run: {run_name} ===')
	do_run(train_cfg, device, tb_dir, ckpt_dir, inference_cfg=run_inference_cfg, only_last_best=only_last_best)
	logger.info(f'=== Finished run: {run_name} ===\n')


if __name__ == '__main__':
	main()