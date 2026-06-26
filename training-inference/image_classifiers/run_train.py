"""
This script expects a YAML configuration describing the dataset, model,
and training hyperparameters. 
Run:
	python run_train.py --config path/to/config.yaml -o path/to/output.log
"""

import argparse
import random
import copy
import gc
import os
import time
import yaml

import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import imagenet_loader
import cifar10_loader
import mnist_loader
import emnist_loader
import model_loader
import run_inference

from logging import Logger, FileHandler, Formatter, StreamHandler

logger = Logger(__name__)

BATCH_LOG_INTERVAL = 10

def set_seed(seed: int):
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	random.seed(seed)
	np.random.seed(seed)


# ---------------------------------------------------------------------------
# Memory utilities
# ---------------------------------------------------------------------------

def get_memory_usage(device=None):
	proc = psutil.Process(os.getpid())
	cpu_rss = proc.memory_info().rss
	cpu_vms = proc.memory_info().vms
	gpu_stats = {}
	if torch.cuda.is_available():
		if device is None:
			device = torch.device('cuda')
		gpu_stats = {
			'gpu_allocated':     torch.cuda.memory_allocated(device),
			'gpu_reserved':      torch.cuda.memory_reserved(device),
			'gpu_max_allocated': torch.cuda.max_memory_allocated(device),
			'gpu_max_reserved':  torch.cuda.max_memory_reserved(device),
		}
	return {'cpu_rss': cpu_rss, 'cpu_vms': cpu_vms, **gpu_stats}


def log_memory_stats(device, writer, epoch, step=None, phase='', prev_mem=None):
	mem = get_memory_usage(device)
	parts = [f'Memory [{phase}]:' if phase else 'Memory:']
	parts.append(f'CPU RSS={mem["cpu_rss"]/1e9:.2f}GB')
	parts.append(f'CPU VMS={mem["cpu_vms"]/1e9:.2f}GB')
	if 'gpu_allocated' in mem:
		parts.append(f'GPU alloc={mem["gpu_allocated"]/1e9:.2f}GB')
		parts.append(f'GPU reserved={mem["gpu_reserved"]/1e9:.2f}GB')
	if prev_mem is not None:
		parts.append(f'(CPU delta={(mem["cpu_rss"] - prev_mem["cpu_rss"])/1e6:.1f}MB)')
		if 'gpu_allocated' in mem and 'gpu_allocated' in prev_mem:
			parts.append(f'(GPU delta={(mem["gpu_allocated"] - prev_mem["gpu_allocated"])/1e6:.1f}MB)')
	logger.info(' '.join(parts))
	if writer is not None:
		gs = step if step is not None else epoch
		prefix = f'mem/{phase}/' if phase else 'mem/'
		writer.add_scalar(f'{prefix}cpu_rss', mem['cpu_rss'], gs)
		writer.add_scalar(f'{prefix}cpu_vms', mem['cpu_vms'], gs)
		if 'gpu_allocated' in mem:
			writer.add_scalar(f'{prefix}gpu_allocated',     mem['gpu_allocated'],     gs)
			writer.add_scalar(f'{prefix}gpu_reserved',      mem['gpu_reserved'],      gs)
			writer.add_scalar(f'{prefix}gpu_max_allocated', mem['gpu_max_allocated'], gs)
			writer.add_scalar(f'{prefix}gpu_max_reserved',  mem['gpu_max_reserved'],  gs)
	return mem


def log_collected_shapes(collected_act, tag=''):
	prefix = f'[{tag}] ' if tag else ''
	for k, v in collected_act.items():
		if isinstance(v, list) and len(v) > 0:
			logger.info(f'{prefix}Collected activations for tag {k!r}: {len(v)} batches, example shape: {v[0].shape}')
		else:
			logger.info(f'{prefix}No activations collected for tag {k!r} yet.')


# ---------------------------------------------------------------------------
# Reordering + subsampling utilities
# ---------------------------------------------------------------------------

def build_selection_lookup(total_size, selected_positions=None):
	if selected_positions is None:
		selected_positions = torch.arange(total_size, dtype=torch.int64)
	else:
		selected_positions = selected_positions.detach().cpu().to(dtype=torch.int64).reshape(-1)
	lookup = torch.full((total_size,), -1, dtype=torch.int64)
	lookup[selected_positions] = torch.arange(selected_positions.numel(), dtype=torch.int64)
	return selected_positions, lookup


def reorder_1d_batches(value_batches, perm_batches, selection_lookup, selected_size):
	"""Reorder per-sample 1-D outputs into canonical dataset order.
	Works for float32 (losses), int64 (labels, preds).
	"""
	first = value_batches[0].detach().cpu()
	reordered = torch.empty(selected_size, dtype=first.dtype)
	if first.dtype in (torch.int64, torch.long):
		reordered.fill_(-1)
	else:
		reordered.fill_(0.0)
	filled = 0
	for batch_vals, perm_idxs in zip(value_batches, perm_batches):
		vals  = batch_vals.detach().cpu()
		slots = selection_lookup[perm_idxs]
		mask  = slots >= 0
		if torch.any(mask):
			reordered[slots[mask]] = vals[mask]
			filled += int(mask.sum().item())
	if filled != selected_size:
		raise RuntimeError(f'Reordering mismatch: expected {selected_size} rows, filled {filled}.')
	return reordered


def build_fixed_balanced_indices(labels, ds_subsample, split_name):
	"""Build fixed random class-balanced subset indices."""
	if ds_subsample is None:
		return None
	logger.info(f"Building fixed balanced indices for split '{split_name}' with ds_subsample={ds_subsample}...")
	ds_subsample = int(ds_subsample)
	n = int(labels.shape[0])
	if ds_subsample <= 0 or ds_subsample >= n:
		if ds_subsample > n:
			logger.warning(f"Requested ds_subsample={ds_subsample} exceeds split size {n} for '{split_name}'. Using full split.")
		return None
	unique_classes, class_counts = torch.unique(labels, sorted=True, return_counts=True)
	n_classes = int(unique_classes.numel())
	if n_classes == 0:
		return None
	base      = ds_subsample // n_classes
	remainder = ds_subsample % n_classes
	if remainder != 0:
		logger.warning(
			f"Cannot sample exactly equally for split '{split_name}': ds_subsample={ds_subsample} "
			f"not divisible by {n_classes} classes. Using near-balanced allocation."
		)
	targets            = torch.full((n_classes,), base, dtype=torch.int64)
	targets[:remainder] += 1
	available          = class_counts.to(dtype=torch.int64)
	selected_per_class = torch.minimum(targets, available)
	deficit            = int(ds_subsample - selected_per_class.sum().item())
	if deficit > 0:
		capacity = available - selected_per_class
		for idx in torch.argsort(capacity, descending=True).tolist():
			if deficit <= 0:
				break
			take = min(int(capacity[idx].item()), deficit)
			if take > 0:
				selected_per_class[idx] += take
				deficit -= take
	if deficit > 0:
		logger.warning(
			f"Could not reach ds_subsample={ds_subsample} for split '{split_name}'. "
			f"Using {int(selected_per_class.sum().item())} samples."
		)
	selected_positions = []
	for cls_idx, cls in enumerate(unique_classes):
		k = int(selected_per_class[cls_idx].item())
		if k <= 0:
			continue
		cls_positions = torch.where(labels == cls)[0]
		if cls_positions.numel() > k:
			perm   = torch.randperm(int(cls_positions.numel()))
			chosen = cls_positions[perm[:k]]
		else:
			chosen = cls_positions
		selected_positions.append(chosen)
	if not selected_positions:
		return None
	selected_positions, _ = torch.sort(torch.cat(selected_positions))
	return selected_positions


def extract_batch(batch):
	"""Support both dict-batches (our datasets) and tuple batches."""
	if isinstance(batch, dict):
		images = batch['image']
		labels = batch['label']
	else:
		images, labels = batch
	return images, labels

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
	elif ds == "imagenet":
		train_tf, test_tf = imagenet_loader.get_imagenet_transforms()
		train_loader, test_loader = imagenet_loader.make_imagenet_dataloaders(
			cfg['data_root'], batch_size=batch_size, train_transform=train_tf, test_transform=test_tf, shuffle=True, random_label_prop=random_prop
		)
	else:
		raise ValueError(f'Unsupported dataset: {ds}')
	return train_loader, test_loader

def build_model(cfg, num_classes, device):
	arch = cfg['model']
	checkpoint = cfg.get('checkpoint', None)

	model = model_loader.create_model(arch, num_classes=num_classes, checkpoint=checkpoint, device=device)
	return model

def make_optimizer(model, cfg):
	opt_cfg = cfg.get('opt', {})
	typ     = opt_cfg.get('type', 'sgd').lower()
	lr      = float(cfg['lr'])
	decay   = float(cfg.get('weight_decay', 0.0))
	if typ == 'adamw':
		return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=decay)
	elif typ == 'adam':
		betas = tuple(cfg.get('betas', (0.9, 0.999)))
		return torch.optim.Adam(model.parameters(), lr=lr, betas=betas, weight_decay=decay)
	elif typ == 'sgd':
		momentum = float(cfg.get('momentum', 0.9))
		return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=decay)
	else:
		raise ValueError(f'Unsupported optimizer type: {typ!r}')


def make_lr_scheduler(optimizer, cfg, num_training_steps=None):
	"""Return a scheduler or None.

	For linear_warmup_decay the scheduler steps per batch; the caller must step
	it inside the training loop. For cosineWR / step it steps per epoch.
	"""
	schedule_cfg = cfg.get('schedule', {})
	stype = schedule_cfg.get('type', 'cosineWR')
	if stype == 'cosineWR':
		return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
			optimizer, T_0=schedule_cfg.get('T0', 10), T_mult=schedule_cfg.get('T_mult', 2)
		)
	elif stype == 'step':
		return torch.optim.lr_scheduler.StepLR(
			optimizer,
			step_size=schedule_cfg.get('step_size', 30),
			gamma=schedule_cfg.get('gamma', 0.1),
		)
	elif stype == 'linear_warmup_decay':
		if num_training_steps is None:
			raise ValueError('num_training_steps required for linear_warmup_decay scheduler.')
		warmup_steps = int(schedule_cfg.get('warmup_steps', 0))

		def lr_lambda(current_step):
			if current_step < warmup_steps:
				return float(current_step) / float(max(1, warmup_steps))
			remaining = num_training_steps - current_step
			total     = max(1, num_training_steps - warmup_steps)
			return max(0.0, float(remaining) / float(total))

		return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
	else:
		return None


def _is_batch_level_scheduler(cfg):
	return cfg.get('schedule', {}).get('type', '') == 'linear_warmup_decay'


def train_epoch(model, loader, criterion, optimizer, device, epoch, writer=None, collect_outputs=False, criterion_collect=None,
                scheduler=None, step_per_batch=False, acts=None, extra_metrics=frozenset()):
	model.train()
	running_loss = 0.0
	correct      = 0
	total        = 0
	acts = acts or {}

	assert not (collect_outputs and criterion_collect is None), \
		"criterion_collect must be provided if collect_outputs is True"

	need_input_grad = collect_outputs and 'gradnorm' in extra_metrics

	collected_losses       = [] if collect_outputs else None
	collected_labels       = [] if collect_outputs else None
	collected_preds        = [] if collect_outputs else None
	collected_perm_indices = [] if collect_outputs else None
	collected_extra        = {m: [] for m in extra_metrics} if collect_outputs else {}

	prev_mem = None

	for step, batch in tqdm(enumerate(loader), total=len(loader), desc=f'Epoch {epoch}'):
		if step % BATCH_LOG_INTERVAL == 0:
			global_step = epoch * len(loader) + step
			prev_mem = log_memory_stats(device, writer, epoch, global_step, f'train_batch_{step}', prev_mem)
			log_collected_shapes(acts, tag='train')

		if isinstance(batch, dict):
			images    = batch['image']
			labels    = batch['label']
			perm_idxs = batch.get('perm_index', None)
		else:
			images, labels = batch
			perm_idxs = None

		images = images.to(device)
		labels = labels.to(device)

		# Create a grad-tracked leaf tensor for input-space gradient norm.
		if need_input_grad:
			images_for_grad = images.detach().requires_grad_(True)
			outputs = model(images_for_grad)
		else:
			images_for_grad = None
			outputs = model(images)

		loss = criterion(outputs, labels)

		# Compute per-sample metrics BEFORE optimizer.zero_grad so retain_graph
		# keeps the graph alive for loss.backward() below.
		if collect_outputs:
			with torch.no_grad():
				loss_collect = criterion_collect(outputs, labels)
			if extra_metrics:
				batch_extra = compute_batch_extra_metrics(
					outputs, labels, extra_metrics,
					images_for_grad=images_for_grad,
					criterion_collect=criterion_collect,
				)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		if step_per_batch and scheduler is not None:
			scheduler.step()

		running_loss += loss.item() * images.size(0)
		_, preds = torch.max(outputs, 1)
		correct  += (preds == labels).sum().item()
		total    += labels.size(0)

		if writer is not None and step % BATCH_LOG_INTERVAL == 0:
			writer.add_scalar('train/batch_loss', loss.item(), epoch * len(loader) + step)

		if collect_outputs:
			collected_losses.append(loss_collect.detach().cpu())
			collected_labels.append(labels.detach().cpu())
			collected_preds.append(preds.detach().cpu())
			if perm_idxs is not None:
				collected_perm_indices.append(perm_idxs.to(dtype=torch.int64))
			if extra_metrics:
				for m, vals in batch_extra.items():
					collected_extra[m].append(vals)

	epoch_loss = running_loss / total
	epoch_acc  = correct / total

	if collect_outputs:
		output_dict = {
			'losses':       collected_losses,
			'labels':       collected_labels,
			'predicted':    collected_preds,
			'perm_indices': collected_perm_indices if collected_perm_indices else None,
		}
		output_dict.update(collected_extra)
		return epoch_loss, epoch_acc, output_dict
	return epoch_loss, epoch_acc


def validate(model, loader, criterion, device, collect_outputs=False, criterion_collect=None,
             epoch=0, writer=None, acts=None, extra_metrics=frozenset()):
	model.eval()
	running_loss = 0.0
	correct      = 0
	total        = 0
	acts = acts or {}

	assert not (collect_outputs and criterion_collect is None), \
		"criterion_collect must be provided if collect_outputs is True"

	need_input_grad = collect_outputs and 'gradnorm' in extra_metrics

	collected_losses       = [] if collect_outputs else None
	collected_labels       = [] if collect_outputs else None
	collected_preds        = [] if collect_outputs else None
	collected_perm_indices = [] if collect_outputs else None
	collected_extra        = {m: [] for m in extra_metrics} if collect_outputs else {}

	prev_mem = None

	with torch.no_grad():
		for step, batch in enumerate(tqdm(loader, total=len(loader), desc='Validation')):
			if step % BATCH_LOG_INTERVAL == 0:
				global_step = epoch * len(loader) + step
				prev_mem = log_memory_stats(device, writer, epoch, global_step, f'val_batch_{step}', prev_mem)
				log_collected_shapes(acts, tag='val')

			if isinstance(batch, dict):
				images    = batch['image']
				labels    = batch['label']
				perm_idxs = batch.get('perm_index', None)
			else:
				images, labels = batch
				perm_idxs = None

			images = images.to(device)
			labels = labels.to(device)

			if need_input_grad:
				# torch.enable_grad() overrides the outer no_grad() for this block.
				with torch.enable_grad():
					images_for_grad = images.detach().requires_grad_(True)
					outputs = model(images_for_grad)
					batch_extra = compute_batch_extra_metrics(
						outputs, labels, extra_metrics,
						images_for_grad=images_for_grad,
						criterion_collect=criterion_collect,
					)
				outputs = outputs.detach()
			else:
				outputs = model(images)
				if collect_outputs and extra_metrics:
					batch_extra = compute_batch_extra_metrics(outputs, labels, extra_metrics)
				else:
					batch_extra = {}

			loss = criterion(outputs, labels)

			if collect_outputs:
				loss_collect = criterion_collect(outputs, labels)

			running_loss += loss.item() * images.size(0)
			_, preds = torch.max(outputs, 1)
			correct  += (preds == labels).sum().item()
			total    += labels.size(0)

			if collect_outputs:
				collected_losses.append(loss_collect.detach().cpu())
				collected_labels.append(labels.detach().cpu())
				collected_preds.append(preds.detach().cpu())
				if perm_idxs is not None:
					collected_perm_indices.append(perm_idxs.to(dtype=torch.int64))
				if extra_metrics:
					for m, vals in batch_extra.items():
						collected_extra[m].append(vals)

	if collect_outputs:
		output_dict = {
			'losses':       collected_losses,
			'labels':       collected_labels,
			'predicted':    collected_preds,
			'perm_indices': collected_perm_indices if collected_perm_indices else None,
		}
		output_dict.update(collected_extra)
		return running_loss / total, correct / total, output_dict
	return running_loss / total, correct / total


def save_checkpoint(state, ckpt_dir, epoch):
	os.makedirs(ckpt_dir, exist_ok=True)
	path = os.path.join(ckpt_dir, f'checkpoint_e{epoch}.pt')
	torch.save(state, path)
	return path


# Recognised extra metric names and their output sub-directory names.
EXTRA_METRIC_NAMES = ('entropy', 'margin', 'gradnorm')
EXTRA_METRIC_DIRS  = {
	'entropy':  'Entropy',
	'margin':   'Margin',
	'gradnorm': 'GradNorm',
}


def compute_batch_extra_metrics(logits, labels, metrics, images_for_grad=None, criterion_collect=None):
	"""Compute requested per-sample extra metrics for one batch.

	Args:
		logits           : (B, C) — raw model output (may or may not have grad_fn).
		labels           : (B,) int64 on the same device as logits.
		metrics          : collection of metric names to compute.
		images_for_grad  : leaf tensor with requires_grad=True (needed for 'gradnorm').
		criterion_collect: CrossEntropyLoss(reduction='none') (needed for 'gradnorm').

	Returns:
		dict mapping metric name -> CPU float32 tensor of shape (B,).
	"""
	result = {}
	if not metrics:
		return result

	with torch.no_grad():
		if 'entropy' in metrics:
			probs   = torch.softmax(logits.detach().float(), dim=1)
			entropy = -(probs * probs.clamp(min=1e-12).log()).sum(dim=1)
			result['entropy'] = entropy.cpu()

		if 'margin' in metrics:
			k        = min(2, logits.shape[1])
			top2     = torch.topk(logits.detach().float(), k=k, dim=1).values
			margin   = top2[:, 0] - top2[:, 1] if k >= 2 else top2[:, 0]
			result['margin'] = margin.cpu()

	if 'gradnorm' in metrics:
		if images_for_grad is None or criterion_collect is None:
			raise ValueError('images_for_grad and criterion_collect are required for the gradnorm metric')
		per_sample_loss = criterion_collect(logits, labels)
		# retain_graph=True so the outer loss.backward() can still use the graph.
		grads = torch.autograd.grad(
			per_sample_loss.sum(), images_for_grad,
			create_graph=False, retain_graph=True,
		)[0]
		result['gradnorm'] = grads.detach().flatten(1).norm(dim=1).cpu()

	return result


def write_compiled_csv(epoch, train_data, val_data, output_root):
	"""Append per-epoch per-sample results to compiled_results.csv.

	train_data / val_data: (reordered_losses, reordered_preds, reordered_labels, selected_indices)
	"""
	csv_path = os.path.join(output_root, 'compiled_results.csv')
	rows = []
	for split, (reordered_losses, reordered_preds, reordered_labels, selected_indices) in (
		('train', train_data), ('val', val_data)
	):
		losses_np = reordered_losses.numpy()
		preds_np  = reordered_preds.numpy()
		labels_np = reordered_labels.numpy()
		idxs_np   = selected_indices.numpy()
		for i in range(len(labels_np)):
			rows.append({
				'Epoch_No':             epoch,
				'Split':                split,
				'Image_Index':          int(idxs_np[i]),
				'Image_Function_value': float(losses_np[i]),
				'Original_Label':       int(labels_np[i]),
				'Predicted_Label':      int(preds_np[i]),
				'Correct':              'CORRECT' if preds_np[i] == labels_np[i] else 'INCORRECT',
			})
	df = pd.DataFrame(rows)
	write_header = not os.path.exists(csv_path)
	df.to_csv(csv_path, mode='a', index=False, header=write_header)


def save_inference_outputs(
	epoch,
	train_outputs,
	val_outputs,
	train_activations,
	val_activations,
	output_root,
	ds_subsample=None,
	fixed_indices_by_split=None,
):
	"""Save per-sample inference outputs collected during training/validation.

	Saved tensors:
	  Losses/{split}/losses_e{epoch}.pt           (N,) float32
	  Predictions/{split}/predictions_e{epoch}.pt (N,) int64
	  Labels/{split}/labels_e{epoch}.pt           (N,) int64
	  Tensors/{split}/a{tag}_e{epoch}.pt          (N, flat_size) float32
	  Indices/{split}/selected_dataset_indices.txt 1-D sample perm indices
	"""
	logger.info(f'Saving inference outputs for epoch {epoch}...')
	os.makedirs(output_root, exist_ok=True)
	if fixed_indices_by_split is None:
		fixed_indices_by_split = {}

	csv_data = {}

	def write_output(split, output, collected_activations):
		loss_dir   = os.path.join(output_root, 'Losses',      split)
		preds_dir  = os.path.join(output_root, 'Predictions', split)
		labels_dir = os.path.join(output_root, 'Labels',      split)
		tens_dir   = os.path.join(output_root, 'Tensors',     split)
		idxs_dir   = os.path.join(output_root, 'Indices',     split)
		for d in (loss_dir, preds_dir, labels_dir, tens_dir, idxs_dir):
			os.makedirs(d, exist_ok=True)

		perm_batches   = [idx.detach().cpu().to(dtype=torch.int64).reshape(-1)
		                  for idx in output['perm_indices']]
		total_examples = sum(int(b.shape[0]) for b in perm_batches)

		# Full-dataset reorder of labels (needed to build balanced indices).
		_, full_lookup = build_selection_lookup(total_examples)
		reordered_labels_full = reorder_1d_batches(
			output['labels'], perm_batches, full_lookup, total_examples,
		)

		# Determine / reuse the fixed balanced subset.
		selected_positions = fixed_indices_by_split.get(split)
		if ds_subsample is not None and selected_positions is None:
			selected_positions = build_fixed_balanced_indices(
				reordered_labels_full.to(dtype=torch.int64), ds_subsample, split
			)
			fixed_indices_by_split[split] = selected_positions

		if selected_positions is not None:
			selected_positions = selected_positions.to(dtype=torch.int64)
			max_idx = int(selected_positions.max().item()) if selected_positions.numel() > 0 else -1
			if max_idx >= total_examples:
				logger.warning(f"Stored ds_subsample indices for {split!r} out of range; recomputing.")
				selected_positions = build_fixed_balanced_indices(
					reordered_labels_full.to(dtype=torch.int64), ds_subsample, split
				)
				fixed_indices_by_split[split] = selected_positions

		selected_perm_indices, selection_lookup = build_selection_lookup(total_examples, selected_positions)
		selected_size = selected_perm_indices.numel()

		reordered_losses = reorder_1d_batches(output['losses'],    perm_batches, selection_lookup, selected_size)
		reordered_preds  = reorder_1d_batches(output['predicted'], perm_batches, selection_lookup, selected_size)
		reordered_labels = reordered_labels_full[selected_perm_indices]

		torch.save(reordered_losses, os.path.join(loss_dir,   f'losses_e{epoch}.pt'))
		torch.save(reordered_preds,  os.path.join(preds_dir,  f'predictions_e{epoch}.pt'))
		torch.save(reordered_labels, os.path.join(labels_dir, f'labels_e{epoch}.pt'))
		np.savetxt(os.path.join(idxs_dir, 'selected_dataset_indices.txt'),
		           selected_perm_indices.numpy(), fmt='%d')

		# Activations: reorder from per-batch hook outputs into dataset order.
		for tag, activations in collected_activations.items():
			if not activations:
				continue
			logger.info(f'Saving activations for tag {tag!r} (split={split})...')
			first_batch = activations[0].detach().cpu()
			flat_size   = int(first_batch[0].reshape(-1).shape[0])
			reordered_acts = torch.zeros((selected_size, flat_size), dtype=torch.float32)
			for batch_acts, perm_idxs in zip(activations, perm_batches):
				acts_cpu = batch_acts.detach().cpu().reshape(batch_acts.shape[0], -1).float()
				slots    = selection_lookup[perm_idxs]
				mask     = slots >= 0
				if torch.any(mask):
					reordered_acts[slots[mask]] = acts_cpu[mask]
			logger.info(f'Activation shape for tag {tag!r}: {reordered_acts.shape}')
			torch.save(reordered_acts, os.path.join(tens_dir, f'a{tag}_e{epoch}.pt'))
			del reordered_acts
			gc.collect()

		if split in ('train', 'val'):
			csv_data[split] = (reordered_losses, reordered_preds, reordered_labels, selected_perm_indices)

		# Extra per-sample metrics (entropy, margin, gradnorm).
		for metric_name, dir_name in EXTRA_METRIC_DIRS.items():
			if metric_name not in output or not output[metric_name]:
				continue
			m_dir = os.path.join(output_root, dir_name, split)
			os.makedirs(m_dir, exist_ok=True)
			reordered_metric = reorder_1d_batches(
				output[metric_name], perm_batches, selection_lookup, selected_size
			)
			torch.save(reordered_metric, os.path.join(m_dir, f'{metric_name}_e{epoch}.pt'))
			del reordered_metric
			gc.collect()

		del reordered_losses, reordered_preds, reordered_labels
		del reordered_labels_full, full_lookup, selection_lookup, selected_perm_indices
		gc.collect()

	write_output('train', train_outputs, train_activations)
	write_output('val',   val_outputs,   val_activations)

	# Write compiled CSV for train + val splits.
	if 'train' in csv_data and 'val' in csv_data:
		write_compiled_csv(epoch, csv_data['train'], csv_data['val'], output_root)

	# trainUval: offset val perm_indices by train_size.
	train_total = sum(int(b.detach().cpu().shape[0]) for b in train_outputs['perm_indices'])
	combined_outputs = {}
	for key in ('losses', 'labels', 'predicted'):
		combined_outputs[key] = train_outputs[key] + val_outputs[key]
	combined_outputs['perm_indices'] = (
		[idx.to(dtype=torch.int64) for idx in train_outputs['perm_indices']] +
		[idx.to(dtype=torch.int64) + train_total for idx in val_outputs['perm_indices']]
	)
	# Include any collected extra metrics.
	for key in EXTRA_METRIC_NAMES:
		if key in train_outputs and train_outputs[key] and key in val_outputs and val_outputs[key]:
			combined_outputs[key] = train_outputs[key] + val_outputs[key]
	combined_acts = {
		tag: train_activations[tag] + val_activations[tag]
		for tag in train_activations
	}
	write_output('trainUval', combined_outputs, combined_acts)

	logger.info(f'Inference outputs saved for epoch {epoch}.')

def do_run(cfg, device, tboard_base, checkpoints_base, only_last_best=False, inference_cfg=None,
           start_checkpoint=None, start_epoch=1):
	train_loader, test_loader = get_dataloaders(cfg)
	num_classes = train_loader.dataset.num_classes

	logger.info(f'Using config: {cfg}')

	model     = build_model(cfg, num_classes, device)
	criterion = nn.CrossEntropyLoss()
	optimizer = make_optimizer(model, cfg)

	checkpoint_data = None
	if start_checkpoint is not None:
		logger.info(f'Loading start checkpoint: {start_checkpoint}')
		checkpoint_data = torch.load(start_checkpoint, map_location=device)
		model.load_state_dict(checkpoint_data['model_state_dict'])
		if 'optimizer_state_dict' in checkpoint_data:
			optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
		ckpt_epoch = checkpoint_data.get('epoch')
		if isinstance(ckpt_epoch, int) and start_epoch != (ckpt_epoch + 1):
			raise ValueError(
				f'start_epoch={start_epoch} but checkpoint epoch={ckpt_epoch}. '
				f'Set --start_epoch {ckpt_epoch + 1}.'
			)
		logger.info(f'Loaded checkpoint epoch={checkpoint_data.get("epoch", "unknown")}.')

	epochs             = cfg['epochs']
	num_training_steps = epochs * len(train_loader)
	scheduler          = make_lr_scheduler(optimizer, cfg, num_training_steps=num_training_steps)
	step_per_batch     = _is_batch_level_scheduler(cfg)

	if scheduler is not None:
		if checkpoint_data is not None and 'scheduler_state_dict' in checkpoint_data:
			scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
			logger.info('Restored scheduler state from checkpoint.')
		elif start_epoch > 1:
			logger.warning('No scheduler state in checkpoint; fast-forwarding scheduler.')
			for _ in range(1, start_epoch):
				scheduler.step()
			logger.info(f'Scheduler fast-forwarded by {start_epoch - 1} steps.')

	writer = SummaryWriter(tboard_base)
	set_seed(cfg['seed'])

	try:
		writer.add_text('config', yaml.dump(cfg))
		if inference_cfg is not None:
			writer.add_text('inference_config', yaml.dump(inference_cfg))
	except Exception:
		pass

	best_val_acc               = 0.0
	best_epoch                 = 0
	ckpt_dir                   = checkpoints_base
	prev_mem                   = get_memory_usage(device)
	ds_subsample               = None if inference_cfg is None else inference_cfg.get('ds_subsample', None)
	fixed_ds_subsample_indices = {}
	collection                 = [] if inference_cfg is None else inference_cfg.get('collect', [])
	criterion_collect          = nn.CrossEntropyLoss(reduction='none')

	# Extra per-sample metrics from the inference config.
	VALID_METRICS = set(EXTRA_METRIC_NAMES)
	raw_metrics   = [] if inference_cfg is None else inference_cfg.get('metrics', [])
	if isinstance(raw_metrics, str):
		raw_metrics = [m.strip() for m in raw_metrics.split(',') if m.strip()]
	extra_metrics = frozenset(raw_metrics)
	invalid = extra_metrics - VALID_METRICS
	if invalid:
		raise ValueError(f'Unknown metrics in inference config: {invalid}. Valid choices: {VALID_METRICS}')
	if extra_metrics:
		logger.info(f'Extra per-sample metrics enabled: {sorted(extra_metrics)}')

	for epoch in range(start_epoch, epochs + 1):
		t0 = time.time()
		logger.info(f'--- Epoch {epoch}/{epochs} ---')
		prev_mem = log_memory_stats(device, writer, epoch, None, f'epoch_{epoch}_start', prev_mem)

		# Epoch collection schedule.
		should_collect_inference = False
		if inference_cfg is not None:
			inference_schedule = inference_cfg.get('schedule', 'all')
			if inference_schedule == 'all' or (isinstance(inference_schedule, list) and 'all' in inference_schedule):
				should_collect_inference = True
			elif ('last' in (inference_schedule if isinstance(inference_schedule, list) else [inference_schedule])
			      and epoch == epochs):
				should_collect_inference = True
			elif (isinstance(inference_schedule, list) and len(inference_schedule) == 3
			      and all(isinstance(x, int) for x in inference_schedule)):
				should_collect_inference = (epoch - 1) in list(range(*inference_schedule))
			elif isinstance(inference_schedule, list):
				should_collect_inference = (epoch - 1) in inference_schedule
			elif isinstance(inference_schedule, int):
				should_collect_inference = ((epoch - 1) % inference_schedule == 0)

		logger.info(f'Collecting inference outputs this epoch: {should_collect_inference}')

		train_activations = {}
		val_activations   = {}
		train_hooks       = []
		val_hooks         = []

		if should_collect_inference:
			train_hooks = run_inference.attach_collection_hooks(model, collection, train_activations)

		train_result = train_epoch(
			model, train_loader, criterion, optimizer, device, epoch, writer,
			collect_outputs=should_collect_inference,
			criterion_collect=criterion_collect if should_collect_inference else None,
			scheduler=scheduler,
			step_per_batch=step_per_batch,
			acts=train_activations,
			extra_metrics=extra_metrics if should_collect_inference else frozenset(),
		)
		if should_collect_inference:
			train_loss, train_acc, train_outputs = train_result
		else:
			train_loss, train_acc = train_result

		if should_collect_inference:
			for h in train_hooks:
				h.remove()
			val_hooks = run_inference.attach_collection_hooks(model, collection, val_activations)

		if scheduler is not None and not step_per_batch:
			scheduler.step()

		logger.info('Evaluating on validation set...')
		val_result = validate(
			model, test_loader, criterion, device,
			collect_outputs=should_collect_inference,
			criterion_collect=criterion_collect if should_collect_inference else None,
			epoch=epoch,
			writer=writer,
			acts=val_activations,
			extra_metrics=extra_metrics if should_collect_inference else frozenset(),
		)
		if should_collect_inference:
			val_loss, val_acc, val_outputs = val_result
		else:
			val_loss, val_acc = val_result

		if should_collect_inference:
			for h in val_hooks:
				h.remove()

		gc.collect()
		prev_mem = log_memory_stats(device, writer, epoch, None, f'epoch_{epoch}_end', prev_mem)

		writer.add_scalar('train/loss',     train_loss, epoch)
		writer.add_scalar('train/accuracy', train_acc,  epoch)
		writer.add_scalar('val/loss',       val_loss,   epoch)
		writer.add_scalar('val/accuracy',   val_acc,    epoch)
		for i, pg in enumerate(optimizer.param_groups):
			writer.add_scalar(f'optimizer/group_{i}_lr', pg.get('lr', 0.0), epoch)

		epoch_time = time.time() - t0
		logger.info(
			f'Epoch {epoch}/{epochs} - train_loss={train_loss:.4f} train_acc={train_acc:.4f} '
			f'val_loss={val_loss:.4f} val_acc={val_acc:.4f} ({epoch_time:.1f}s)'
		)

		if should_collect_inference:
			try:
				save_inference_outputs(
					epoch - 1,
					train_outputs, val_outputs,
					train_activations, val_activations,
					inference_cfg['output_root'],
					ds_subsample=ds_subsample,
					fixed_indices_by_split=fixed_ds_subsample_indices,
				)
			except Exception as e:
				logger.error(f'Error saving inference outputs for epoch {epoch}: {e}')
			finally:
				train_outputs.clear()
				val_outputs.clear()
				train_activations.clear()
				val_activations.clear()
				gc.collect()
				post_save_mem = get_memory_usage(device)
				logger.info(
					f'Memory after inference save epoch {epoch}: '
					f'CPU RSS={post_save_mem["cpu_rss"]/1e9:.2f}GB '
					f'GPU alloc={post_save_mem.get("gpu_allocated", 0)/1e9:.2f}GB'
				)

		state = {
			'epoch':                epoch,
			'model_state_dict':     model.state_dict(),
			'optimizer_state_dict': optimizer.state_dict(),
			'cfg':                  cfg,
		}
		if scheduler is not None:
			state['scheduler_state_dict'] = scheduler.state_dict()

		if only_last_best:
			if epoch == epochs:
				save_checkpoint(state, ckpt_dir, epoch)
		else:
			save_checkpoint(state, ckpt_dir, epoch)

		if val_acc > best_val_acc:
			best_val_acc = val_acc
			if best_epoch > 0:
				try:
					os.remove(os.path.join(ckpt_dir, f'best_e{best_epoch}.pt'))
				except FileNotFoundError:
					pass
			best_epoch = epoch
			torch.save(state, os.path.join(ckpt_dir, f'best_e{epoch}.pt'))

	logger.info(f'Best validation accuracy: {best_val_acc:.4f} @ epoch {best_epoch}')
	writer.close()

def main(argv=None):
	"""Single-run entrypoint.

	Expected config structure
	-------------------------
	datasets:
	  <dataset_name>: <path>

	name: <run_name>
	device: cuda:0   # or cpu

	train:
	  seed: ...
	  model: resnet18
	  dataset: <dataset_name>
	  batch_size: 128
	  epochs: 150
	  lr: 0.0001
	  weight_decay: 0.0005
	  opt:
	    type: sgd          # adamw | adam | sgd
	  schedule:
	    type: cosineWR     # cosineWR | step | linear_warmup_decay
	  checkpoints_dir: /path/to/checkpoints/
	  tensorboard_dir: /path/to/tensorboard/
	  only_last_best: false   # optional

	infer:                    # omit section to skip inference collection
	  output_dir: /path/to/infer/
	  collect:
	    - path: layer4
	      tag: 2
	  epochs: all             # all | last | N | [start, end, step] | [e1, e2, ...]
	  ds_subsample: null      # optional
	  metrics: []             # optional: entropy, margin, gradnorm
	"""
	parser = argparse.ArgumentParser(description='Image classifier training + inference pipeline')
	parser.add_argument('--config',           '-c', required=True,        help='YAML config file')
	parser.add_argument('--output_file',      '-o', default=None,         help='log file base path (optional; timestamped on write)')
	parser.add_argument('--tensorboard_dir',        default=None,         help='override train.tensorboard_dir from config')
	parser.add_argument('--checkpoint_dir',         default=None,         help='override train.checkpoints_dir from config')
	parser.add_argument('--start_checkpoint',        default=None,        help='checkpoint path to resume training from')
	parser.add_argument('--start_epoch',             type=int, default=1, help='epoch number to resume at (1-based)')
	args = parser.parse_args(argv)

	if args.start_epoch < 1:
		raise ValueError(f'--start_epoch must be >= 1, got {args.start_epoch}')

	with open(args.config, 'r') as f:
		cfg = yaml.safe_load(f)

	log_formatter = Formatter('%(asctime)s - %(levelname)s - %(message)s')
	if args.output_file:
		fname        = os.path.splitext(args.output_file)[0]
		log_filename = f'{fname}_{time.strftime("%Y%m%d_%H%M%S")}.log'
		fh = FileHandler(log_filename, mode='w')
		fh.setFormatter(log_formatter)
		logger.addHandler(fh)
	sh = StreamHandler()
	sh.setFormatter(log_formatter)
	logger.addHandler(sh)
	logger.setLevel('INFO')

	train_cfg = copy.deepcopy(cfg['train'])
	infer_cfg = cfg.get('infer', None)
	run_name  = cfg['name']
	device    = torch.device(cfg.get('device', 'cpu'))

	# Resolve dataset data_root
	dataset_name            = train_cfg['dataset']
	train_cfg['data_root']  = cfg['datasets'][dataset_name]

	# CLI dir overrides
	if args.tensorboard_dir:
		train_cfg['tensorboard_dir'] = args.tensorboard_dir
	if args.checkpoint_dir:
		train_cfg['checkpoints_dir'] = args.checkpoint_dir

	only_last_best = train_cfg.get('only_last_best', False)

	datestr  = time.strftime('%Y%m%d_%H%M%S')
	tb_dir   = os.path.join(train_cfg['tensorboard_dir'], run_name, datestr)
	ckpt_dir = os.path.join(train_cfg['checkpoints_dir'], run_name, datestr)
	os.makedirs(tb_dir,   exist_ok=True)
	os.makedirs(ckpt_dir, exist_ok=True)

	# Build the inference config that do_run expects
	run_inference_cfg = None
	if infer_cfg is not None:
		run_inference_cfg = copy.deepcopy(infer_cfg)
		run_inference_cfg['schedule']    = infer_cfg.get('epochs', 'all')
		run_inference_cfg['output_root'] = os.path.join(infer_cfg['output_dir'], run_name, datestr)
		logger.info(f'Inference output root:      {run_inference_cfg["output_root"]}')
		logger.info(f'Inference epochs schedule:  {run_inference_cfg["schedule"]}')
		logger.info(f'Inference collection layers:{run_inference_cfg.get("collect", [])}')

	logger.info(f'Using device: {device}')
	logger.info(f'\n=== Starting run: {run_name} ===')
	do_run(
		train_cfg, device, tb_dir, ckpt_dir,
		only_last_best=only_last_best,
		inference_cfg=run_inference_cfg,
		start_checkpoint=args.start_checkpoint,
		start_epoch=args.start_epoch,
	)
	logger.info(f'=== Finished run: {run_name} ===\n')


if __name__ == '__main__':
	main()

