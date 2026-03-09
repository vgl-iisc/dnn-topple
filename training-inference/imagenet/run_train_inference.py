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

import psutil
import gc

import imagenet_loader
import cifar10_loader
import mnist_loader
import emnist_loader
import model_loader

from logging import Logger, FileHandler, Formatter, StreamHandler

logger = Logger(__name__)

# Memory logging interval (in batches)
BATCH_LOG_INTERVAL = 10

def set_seed(seed: int):
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	import random
	random.seed(seed)
	import numpy as np
	np.random.seed(seed)


def get_memory_usage(device=None):
	"""Return a dict of current CPU and GPU memory usage."""
	proc = psutil.Process(os.getpid())
	cpu_rss = proc.memory_info().rss
	cpu_vms = proc.memory_info().vms

	gpu_stats = {}
	if torch.cuda.is_available():
		if device is None:
			device = torch.device('cuda')
		gpu_stats = {
			"gpu_allocated": torch.cuda.memory_allocated(device),
			"gpu_reserved": torch.cuda.memory_reserved(device),
			"gpu_max_allocated": torch.cuda.max_memory_allocated(device),
			"gpu_max_reserved": torch.cuda.max_memory_reserved(device),
		}

	return {
		"cpu_rss": cpu_rss,
		"cpu_vms": cpu_vms,
		**gpu_stats,
	}


def log_memory_stats(device, writer, epoch, step=None, phase="", prev_mem=None):
	"""Log memory stats to both logger and tensorboard."""
	mem = get_memory_usage(device)
	
	# Construct log message
	log_parts = []
	if phase:
		log_parts.append(f"Memory [{phase}]:")
	else:
		log_parts.append("Memory:")
	
	log_parts.append(f"CPU RSS={mem['cpu_rss']/1e9:.2f}GB")
	log_parts.append(f"CPU VMS={mem['cpu_vms']/1e9:.2f}GB")
	
	if 'gpu_allocated' in mem:
		log_parts.append(f"GPU alloc={mem['gpu_allocated']/1e9:.2f}GB")
		log_parts.append(f"GPU reserved={mem['gpu_reserved']/1e9:.2f}GB")
	
	if prev_mem is not None:
		delta_cpu_rss = mem['cpu_rss'] - prev_mem['cpu_rss']
		log_parts.append(f"(CPU delta={delta_cpu_rss/1e6:.1f}MB)")
		if 'gpu_allocated' in mem and 'gpu_allocated' in prev_mem:
			delta_gpu = mem['gpu_allocated'] - prev_mem['gpu_allocated']
			log_parts.append(f"(GPU delta={delta_gpu/1e6:.1f}MB)")
	
	logger.info(" ".join(log_parts))
	
	# Log to tensorboard
	if writer is not None:
		# Determine the global step for tensorboard
		if step is not None:
			global_step = step
		else:
			global_step = epoch
		
		prefix = f"mem/{phase}/" if phase else "mem/"
		
		writer.add_scalar(f'{prefix}cpu_rss', mem['cpu_rss'], global_step)
		writer.add_scalar(f'{prefix}cpu_vms', mem['cpu_vms'], global_step)
		if 'gpu_allocated' in mem:
			writer.add_scalar(f'{prefix}gpu_allocated', mem['gpu_allocated'], global_step)
			writer.add_scalar(f'{prefix}gpu_reserved', mem['gpu_reserved'], global_step)
			writer.add_scalar(f'{prefix}gpu_max_allocated', mem['gpu_max_allocated'], global_step)
			writer.add_scalar(f'{prefix}gpu_max_reserved', mem['gpu_max_reserved'], global_step)
	
	return mem

def log_collected_shapes(collected_preds):
	logger.info(f"Collected predictions count: {len(collected_preds)}, example shape: {collected_preds[0].shape if len(collected_preds) > 0 else 'N/A'}")

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

def get_dataloaders(cfg):
	ds = cfg["dataset"]
	batch_size = cfg["batch_size"]
 
	random_prop = cfg.get("random_prop", 0.0)
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
	lr = float(cfg["lr"])
	betas = tuple(cfg.get("betas", (0.9, 0.999)))
	decay = float(cfg.get("weight_decay", 0.0))

	return torch.optim.Adam(model.parameters(), lr=lr, betas=betas, weight_decay=decay)

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
	correct = 0
	total = 0
	
	collected_losses = []
	collected_labels = []
	collected_preds = []
	collected_perm_indices = []
	
	prev_mem = None
	
	for step, batch in tqdm(enumerate(loader), total=len(loader), desc=f'Epoch {epoch}'):
		# Log memory every MEMORY_LOG_INTERVAL batches
		if step % BATCH_LOG_INTERVAL == 0:
			global_step = epoch * len(loader) + step
			prev_mem = log_memory_stats(device, writer, epoch, global_step, f"train_batch_{step}", prev_mem)
			log_collected_shapes(collected_preds)
		
		images = batch['image']
		labels = batch['label']
		perm_idxs = batch['perm_index']
		
		images = images.to(device)
		labels = labels.to(device)

		outputs = model(images)
		loss = criterion(outputs, labels)
		
		# Compute per-sample losses for collection
		with torch.no_grad():
			loss_collect = criterion_collect(outputs, labels)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		running_loss += loss.item() * images.size(0)
		_, preds = torch.max(outputs, 1)
		correct += (preds == labels).sum().item()
		total += labels.size(0)

		if writer is not None and step % BATCH_LOG_INTERVAL == 0:
			writer.add_scalar('train/batch_loss', loss.item(), epoch * len(loader) + step)
		
		# Always collect outputs
		collected_losses.append(loss_collect.detach())
		collected_labels.append(labels.detach())
		collected_preds.append(preds.detach())
		collected_perm_indices.append(perm_idxs.to(dtype=torch.int64))

	epoch_loss = running_loss / total
	epoch_acc = correct / total
	
	return epoch_loss, epoch_acc, {
		'losses': collected_losses,
		'labels': collected_labels,
		'predicted': collected_preds,
		'perm_indices': collected_perm_indices
	}


def validate(model, loader, criterion, criterion_collect, device, epoch=0, writer=None):
	model.eval()
	running_loss = 0.0
	correct = 0
	total = 0
	
	collected_losses = []
	collected_labels = []
	collected_preds = []
	collected_perm_indices = []
	
	prev_mem = None
	
	with torch.no_grad():
		for step, batch in enumerate(tqdm(loader, total=len(loader), desc='Validation')):
			# Log memory every MEMORY_LOG_INTERVAL batches
			if step % BATCH_LOG_INTERVAL == 0:
				global_step = epoch * len(loader) + step
				prev_mem = log_memory_stats(device, writer, epoch, global_step, f"val_batch_{step}", prev_mem)
				log_collected_shapes(collected_preds)
			
			images = batch['image']
			labels = batch['label']
			perm_idxs = batch['perm_index']
			
			images = images.to(device)
			labels = labels.to(device)
			outputs = model(images)
			loss = criterion(outputs, labels)
			
			loss_collect = criterion_collect(outputs, labels)

			running_loss += loss.item() * images.size(0)
			_, preds = torch.max(outputs, 1)
			correct += (preds == labels).sum().item()
			total += labels.size(0)
			
			collected_losses.append(loss_collect.detach())
			collected_labels.append(labels.detach())
			collected_preds.append(preds.detach())
			collected_perm_indices.append(perm_idxs.to(dtype=torch.int64))
	
	return running_loss / total, correct / total, {
		'losses': collected_losses,
		'labels': collected_labels,
		'predicted': collected_preds,
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
	
	# Create output directory for this epoch
	epoch_output_dir = os.path.join(output_root)
	os.makedirs(epoch_output_dir, exist_ok=True)
	
	import numpy as np
	
	def write_output(split, output, collected_activations):
		loss_dir = os.path.join(epoch_output_dir, "Losses", split)
		preds_dir = os.path.join(epoch_output_dir, "Predictions", split)
		tens_dir = os.path.join(epoch_output_dir, "Tensors", split)
		
		os.makedirs(loss_dir, exist_ok=True)
		os.makedirs(preds_dir, exist_ok=True)
		os.makedirs(tens_dir, exist_ok=True)
		
		collected_losses = torch.cat(output['losses']).detach().cpu()
		collected_preds = torch.cat(output['predicted']).detach().cpu()
		collected_perm_indices = torch.cat(output['perm_indices']).cpu().to(dtype=torch.int64)
		
		# The perm_indices tell us the dataset/permutation index (ds[0], ds[1], ds[2], ...)
		# We need to reorder from shuffled collection order back to permutation order
		reordered_losses = torch.zeros_like(collected_losses)
		reordered_preds = torch.zeros_like(collected_preds)
		
		reordered_losses[collected_perm_indices] = collected_losses
		reordered_preds[collected_perm_indices] = collected_preds
		
		for tag, activations in collected_activations.items():
			stacked = torch.cat(activations).detach().cpu()
			collated = stacked.reshape(len(stacked), -1)
			
			reordered_activations = torch.zeros_like(collated)
			reordered_activations[collected_perm_indices] = collated
			
			logger.info(f'Saving activations for tag {tag}, shape: {reordered_activations.shape}')
			torch.save(reordered_activations, os.path.join(tens_dir, f"a{tag}_e{epoch}.pt"))
		
		# Convert to numpy for saving
		collected_losses_np = reordered_losses.numpy().reshape(-1, 1).squeeze()
		collected_preds_np = reordered_preds.numpy().reshape(-1, 1).squeeze()
		
		np.savetxt(os.path.join(loss_dir, f"losses_e{epoch}.txt"), collected_losses_np)
		np.savetxt(os.path.join(preds_dir, f"predictions_e{epoch}.txt"), collected_preds_np, fmt='%d')
	
	write_output("train", train_outputs, train_activations)
	write_output("val", val_outputs, val_activations)
	
	# Combine train+val - need to adjust perm_indices for val to account for train dataset size
	combined_outputs = {}
	for key in ['losses', 'labels', 'predicted']:
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
	num_classes = train_loader.dataset.num_classes
 
	logger.info(f"Using config: {cfg}")

	model = build_model(cfg, num_classes, device)

	criterion = nn.CrossEntropyLoss()
	criterion_collect = nn.CrossEntropyLoss(reduction='none')
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

	best_val_acc = 0.0
	best_epoch = 0
	ckpt_dir = checkpoints_base

	# Track memory usage over epochs to help detect leaks
	prev_mem = get_memory_usage(device)
	logger.info(f"Initial memory - CPU RSS: {prev_mem['cpu_rss']} bytes, GPU allocated: {prev_mem.get('gpu_allocated', 'N/A')} bytes")

	for epoch in range(1, epochs + 1):
		t0 = time.time()
		logger.info(f'--- Epoch {epoch}/{epochs} ---')
		
		# Log memory before start of epoch
		prev_mem = log_memory_stats(device, writer, epoch, None, f"epoch_{epoch}_start", prev_mem)
		
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
		
		train_loss, train_acc, train_outputs = train_epoch(model, train_loader, criterion, criterion_collect, optimizer, device, epoch, writer)
		
		if should_collect_inference:
			for h in train_hooks:
				h.remove()
			val_hooks = attach_collection_hooks(model, collection, val_activations)
		
		if scheduler is not None:
			scheduler.step()
		logger.info('Evaluating on validation set...')
		
		val_loss, val_acc, val_outputs = validate(model, test_loader, criterion, criterion_collect, device, epoch, writer)
		
		# Remove val hooks
		if should_collect_inference:
			for h in val_hooks:
				h.remove()

		# Memory tracking: run a GC and log current usage and delta from previous epoch
		gc.collect()
		current_mem = get_memory_usage(device)
		delta_cpu_rss = current_mem['cpu_rss'] - prev_mem['cpu_rss']
		delta_cpu_vms = current_mem['cpu_vms'] - prev_mem['cpu_vms']
		logger.info(
			f"Memory after epoch {epoch}: CPU RSS={current_mem['cpu_rss']} (+{delta_cpu_rss}), "
			f"CPU VMS={current_mem['cpu_vms']} (+{delta_cpu_vms}), "
			f"GPU allocated={current_mem.get('gpu_allocated', 'N/A')} (+{current_mem.get('gpu_allocated', 0) - prev_mem.get('gpu_allocated', 0)})"
		)
		
		# Add memory scalars to tensorboard
		writer.add_scalar('mem/cpu_rss', current_mem['cpu_rss'], epoch)
		writer.add_scalar('mem/cpu_vms', current_mem['cpu_vms'], epoch)
		if 'gpu_allocated' in current_mem:
			writer.add_scalar('mem/gpu_allocated', current_mem['gpu_allocated'], epoch)
			writer.add_scalar('mem/gpu_reserved', current_mem['gpu_reserved'], epoch)
			writer.add_scalar('mem/gpu_max_allocated', current_mem['gpu_max_allocated'], epoch)
			writer.add_scalar('mem/gpu_max_reserved', current_mem['gpu_max_reserved'], epoch)
		
		prev_mem = current_mem

		writer.add_scalar('train/loss', train_loss, epoch)
		writer.add_scalar('train/accuracy', train_acc, epoch)
		writer.add_scalar('val/loss', val_loss, epoch)
		writer.add_scalar('val/accuracy', val_acc, epoch)

		# log lr
		for i, param_group in enumerate(optimizer.param_groups):
			writer.add_scalar(f'optimizer/group_{i}_lr', param_group.get('lr', 0.0), epoch)

		epoch_time = time.time() - t0
		logger.info(f'Epoch {epoch}/{epochs} - train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f} ({epoch_time:.1f}s)')

		# Save inference outputs if collected
		if should_collect_inference:
			try:
				save_inference_outputs(epoch - 1, train_outputs, val_outputs, train_activations, val_activations, inference_cfg['output_root'])
			except Exception as e:
				logger.error(f'Error saving inference outputs for epoch {epoch}: {e}')

		# checkpoint
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


		if val_acc > best_val_acc:
			best_val_acc = val_acc
			if best_epoch > 0:
				try:
					os.remove(os.path.join(ckpt_dir, f'best_e{best_epoch}.pt'))
				except FileNotFoundError:
					pass
			best_epoch = epoch
			best_path = os.path.join(ckpt_dir, f'best_e{epoch}.pt')
			torch.save(state, best_path)
   
	logger.info(f'Best validation accuracy: {best_val_acc:.4f} @ {best_epoch}')

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
	
	infer_output_dir = infer_cfg['output_dir']

	run_infer_cfg = copy.deepcopy(infer_cfg)
	run_infer_cfg['model'] = train_cfg['model']
	run_infer_cfg['output_root'] = os.path.join(infer_output_dir, run_name, datestr)

	runs = [(run_name, train_cfg, device, tb_dir, ckpt_dir, run_infer_cfg)]
	
	random_values = config.get("random", [])
	if len(random_values) > 0:
		runs = []
		for r in random_values:
			tcfg = copy.deepcopy(train_cfg)
			tcfg["random_prop"] = r
			name = f"{run_name}-r{r}"
			
			tb_dir = os.path.join(tensorboard_dir, name, datestr)
			ckpt_dir = os.path.join(checkpoint_dir, name, datestr)
			
			icfg = copy.deepcopy(run_infer_cfg)
			icfg['output_root'] = os.path.join(infer_output_dir, name, datestr)

			runs.append((name, tcfg, device, tb_dir, ckpt_dir, icfg))


	for (name, tcfg, device, tb_dir, ckpt_dir, icfg) in runs:
		os.makedirs(tb_dir, exist_ok=True)
		os.makedirs(ckpt_dir, exist_ok=True)

		logger.info(f'Inference will be collected at: {icfg["output_root"]}')
		logger.info(f'Inference epochs schedule: {icfg.get("epochs", "all")}')
		logger.info(f'Inference collection layers: {icfg.get("collect", [])}')

		only_last_best = tcfg.get("only_last_best", False)

		logger.info(f'\n=== Starting run: {name} ===')
		do_run(tcfg, device, tb_dir, ckpt_dir, icfg, only_last_best=only_last_best)
		logger.info(f'=== Finished run: {name} ===\n')


if __name__ == '__main__':
	main()