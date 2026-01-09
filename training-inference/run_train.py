"""
This script expects a YAML configuration describing the dataset, model,
and training hyperparameters. 
Run:
	python run_train.py --config path/to/config.yaml -o path/to/output.log
"""

import argparse
import os
from sched import scheduler
import yaml
import time

from tqdm import tqdm

import copy

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import imagenet_loader
import cifar10_loader
import mnist_loader
import emnist_loader
import model_loader

import run_inference

from logging import Logger, FileHandler, Formatter, StreamHandler

logger = Logger(__name__)

def set_seed(seed: int):
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	import random
	random.seed(seed)
	import numpy as np
	np.random.seed(seed)

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
	lr = cfg["lr"]
	momentum = cfg["momentum"]
	decay = cfg.get("weight_decay", 0.0)

	return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=decay)

def train_epoch(model, loader, criterion, optimizer, device, epoch, writer=None, collect_outputs=False):
	model.train()
	running_loss = 0.0
	correct = 0
	total = 0
	
	# Optional collection for inference
	collected_losses = [] if collect_outputs else None
	collected_labels = [] if collect_outputs else None
	collected_preds = [] if collect_outputs else None
	collected_indices = [] if collect_outputs else None
	
	for step, batch in tqdm(enumerate(loader), total=len(loader), desc=f'Epoch {epoch}'):
		if isinstance(batch, dict):
			images = batch['image']
			labels = batch['label']
			idxs = batch.get('index', None)
		else:
			images, labels = batch
			idxs = None
		
		images = images.to(device)
		labels = labels.to(device)

		outputs = model(images)
		loss = criterion(outputs, labels)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		running_loss += loss.item() * images.size(0)
		_, preds = torch.max(outputs, 1)
		correct += (preds == labels).sum().item()
		total += labels.size(0)

		if writer is not None and step % 100 == 0:
			writer.add_scalar('train/batch_loss', loss.item(), epoch * len(loader) + step)
		
		if collect_outputs:
			collected_losses.append(loss.detach())
			collected_labels.append(labels.detach())
			collected_preds.append(preds.detach())
			if idxs is not None:
				collected_indices.append(idxs.to(dtype=torch.uint64))

	epoch_loss = running_loss / total
	epoch_acc = correct / total
	
	if collect_outputs:
		return epoch_loss, epoch_acc, {
			'losses': collected_losses,
			'labels': collected_labels,
			'predicted': collected_preds,
			'indices': collected_indices if collected_indices else None
		}
	return epoch_loss, epoch_acc


def validate(model, loader, criterion, device, collect_outputs=False):
	model.eval()
	running_loss = 0.0
	correct = 0
	total = 0
	
	# Optional collection for inference
	collected_losses = [] if collect_outputs else None
	collected_labels = [] if collect_outputs else None
	collected_preds = [] if collect_outputs else None
	collected_indices = [] if collect_outputs else None
	
	with torch.no_grad():
		for batch in tqdm(loader, total=len(loader), desc='Validation'):
			if isinstance(batch, dict):
				images = batch['image']
				labels = batch['label']
				idxs = batch.get('index', None)
			else:
				images, labels = batch
				idxs = None
			
			images = images.to(device)
			labels = labels.to(device)
			outputs = model(images)
			loss = criterion(outputs, labels)
			running_loss += loss.item() * images.size(0)
			_, preds = torch.max(outputs, 1)
			correct += (preds == labels).sum().item()
			total += labels.size(0)
			
			if collect_outputs:
				collected_losses.append(loss.detach())
				collected_labels.append(labels.detach())
				collected_preds.append(preds.detach())
				if idxs is not None:
					collected_indices.append(idxs.to(dtype=torch.uint64))
	
	if collect_outputs:
		return running_loss / total, correct / total, {
			'losses': collected_losses,
			'labels': collected_labels,
			'predicted': collected_preds,
			'indices': collected_indices if collected_indices else None
		}
	return running_loss / total, correct / total


def save_checkpoint(state, ckpt_dir, epoch):
	os.makedirs(ckpt_dir, exist_ok=True)
	path = os.path.join(ckpt_dir, f'checkpoint_e{epoch}.pt')
	torch.save(state, path)
	return path

def save_inference_outputs(epoch, train_outputs, val_outputs, train_activations, val_activations, output_root):
	"""Save inference outputs collected during training/validation."""
	logger.info(f'Saving inference outputs for epoch {epoch}...')
	
	# Create output directory for this epoch
	epoch_output_dir = os.path.join(output_root, f'epoch_{epoch}')
	os.makedirs(epoch_output_dir, exist_ok=True)
	
	import numpy as np
	
	# Write outputs
	def write_output(split, output, collected_activations):
		loss_dir = os.path.join(epoch_output_dir, "Losses", split)
		preds_dir = os.path.join(epoch_output_dir, "Predictions", split)
		tens_dir = os.path.join(epoch_output_dir, "Tensors", split)
		
		os.makedirs(loss_dir, exist_ok=True)
		os.makedirs(preds_dir, exist_ok=True)
		os.makedirs(tens_dir, exist_ok=True)
		
		collected_losses = torch.cat(output['losses']).detach().cpu().numpy().reshape(-1, 1).squeeze()
		collected_preds = torch.cat(output['predicted']).detach().cpu().numpy().reshape(-1, 1).squeeze()
		
		for tag, activations in collected_activations.items():
			stacked = torch.cat(activations).detach().cpu()
			collated = stacked.reshape(len(stacked), -1)
			torch.save(collated, os.path.join(tens_dir, f"a{tag}_e{epoch}.pt"))
		
		np.savetxt(os.path.join(loss_dir, f"losses_e{epoch}.txt"), collected_losses)
		np.savetxt(os.path.join(preds_dir, f"predictions_e{epoch}.txt"), collected_preds, fmt='%d')
	
	write_output("train", train_outputs, train_activations)
	write_output("val", val_outputs, val_activations)
	
	# Combine train+val
	combined_outputs = {key: train_outputs[key] + val_outputs[key] for key in train_outputs}
	combined_activations = {tag: train_activations[tag] + val_activations[tag] for tag in train_activations}
	write_output("trainUval", combined_outputs, combined_activations)
	
	logger.info(f'Inference outputs saved for epoch {epoch}')

def do_run(cfg, device, tboard_base, checkpoints_base, only_last_best=False, inference_cfg=None):
	train_loader, test_loader = get_dataloaders(cfg)
	num_classes = train_loader.dataset.num_classes
 
	logger.info(f"Using config: {cfg}")

	model = build_model(cfg, num_classes, device)

	# Use reduction='none' if collecting inference outputs, otherwise use default 'mean'
	criterion = nn.CrossEntropyLoss(reduction='none' if inference_cfg else 'mean')
	optimizer = make_optimizer(model, cfg)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

	writer = SummaryWriter(tboard_base)

	set_seed(cfg["seed"])

	# log hyperparameters as text and with add_hparams (requires a metric)
	try:
		writer.add_text('config', yaml.dump(cfg))
	except Exception:
		pass

	epochs = cfg["epochs"]

	best_val_acc = 0.0
	best_epoch = 0
	ckpt_dir = checkpoints_base

	for epoch in range(1, epochs + 1):
		t0 = time.time()
		logger.info(f'--- Epoch {epoch}/{epochs} ---')
		
		# Determine if we should collect inference outputs this epoch
		should_collect_inference = False
		if inference_cfg is not None:
			inference_schedule = inference_cfg.get('schedule', 'all')
			if inference_schedule == 'all':
				should_collect_inference = True
			elif inference_schedule == 'last' and epoch == epochs:
				should_collect_inference = True
			elif isinstance(inference_schedule, list) and len(inference_schedule) == 3:
				# Format: [start, end, step] - check if (epoch-1) is in this range
				should_collect_inference = (epoch - 1) in list(range(inference_schedule[0], inference_schedule[1], inference_schedule[2]))
			elif isinstance(inference_schedule, list):
				# Format: list of specific epochs
				should_collect_inference = (epoch - 1) in inference_schedule
			elif isinstance(inference_schedule, int):
				# Format: interval - every N epochs
				should_collect_inference = ((epoch - 1) % inference_schedule == 0)
		
		# Attach hooks if collecting inference outputs
		train_hooks = []
		val_hooks = []
		train_activations = {}
		val_activations = {}
		
		if should_collect_inference:
			model_arch = inference_cfg['model']
			collection = inference_cfg.get('collect', {}).get(model_arch, [])
			train_hooks = run_inference.attach_collection_hooks(model, collection, train_activations)
		
		# Training pass (with optional collection)
		train_result = train_epoch(model, train_loader, criterion, optimizer, device, epoch, writer, collect_outputs=should_collect_inference)
		if should_collect_inference:
			train_loss, train_acc, train_outputs = train_result
		else:
			train_loss, train_acc = train_result
			
		# Remove train hooks and attach val hooks
		if should_collect_inference:
			for h in train_hooks:
				h.remove()
			val_hooks = run_inference.attach_collection_hooks(model, collection, val_activations)
		
		scheduler.step()
		logger.info('Evaluating on validation set...')
		
		# Validation pass (with optional collection)
		val_result = validate(model, test_loader, criterion, device, collect_outputs=should_collect_inference)
		if should_collect_inference:
			val_loss, val_acc, val_outputs = val_result
		else:
			val_loss, val_acc = val_result
			
		# Remove val hooks
		if should_collect_inference:
			for h in val_hooks:
				h.remove()

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
				os.remove(os.path.join(ckpt_dir, f'best_e{best_epoch}.pt'))
			best_epoch = epoch
			best_path = os.path.join(ckpt_dir, f'best_e{epoch}.pt')
			torch.save(state, best_path)
   
	logger.info(f'Best validation accuracy: {best_val_acc:.4f} @ {best_epoch}')

	writer.close()

def main(argv=None):
	parser = argparse.ArgumentParser()
	parser.add_argument('--config', '-c', required=True, help='YAML config file')
	parser.add_argument('--output_file', '-o', required=True, help='output file to log metrics to')
	parser.add_argument('--tensorboard_dir', default=None, help='override tensorboard dir from config')
	parser.add_argument('--checkpoint_dir', default=None, help='override checkpoint dir from config')
	parser.add_argument('--inference_config', default=None, help='YAML config for running inference during training')
	args = parser.parse_args(argv)

	with open(args.config, 'r') as f:
		global_cfg = yaml.safe_load(f)

	log_formatter = Formatter('%(asctime)s - %(levelname)s - %(message)s')
	file_handler = FileHandler(args.output_file)
	file_handler.setFormatter(log_formatter)
	stream_handler = StreamHandler()
	stream_handler.setFormatter(log_formatter)
	logger.addHandler(file_handler)
	logger.addHandler(stream_handler)
	logger.setLevel('INFO')

	runs = {}

	random_intervals = global_cfg["config"].get("random", None)
 
	logger.info(f'Found random intervals: {random_intervals}')
 
	train_runs_dir = global_cfg["config"]["experiments_dir"]
	for run_file in os.listdir(train_runs_dir):
		if not run_file.endswith('.yaml'):
			continue

		with open(os.path.join(train_runs_dir, run_file), 'r') as f:
			run_cfg = yaml.safe_load(f)
			assert 'runs' in run_cfg, f'Invalid run file {run_file}, missing "runs" key'

			for k, v in run_cfg['runs'].items():
				cfg = copy.deepcopy(run_cfg)
				cfg.pop('runs', None)
				cfg.update(v)
				run_object = cfg
				run_object["data_root"] = global_cfg["datasets"][run_object["dataset"]]

				if random_intervals is not None:
					for prop in random_intervals:
						run_copy = copy.deepcopy(run_object)
						run_copy["name"] = f"{k}-r{prop}"
						run_copy["random_label_prop"] = prop
						runs[f"{k}-r{prop}"] = run_copy
				else:
					run_object["random"] = 0.0
					runs[k] = run_object

	todo = global_cfg["do"]
	global_cfg = global_cfg["config"]

	only_last_best = global_cfg.get("only_last_best", False)

	# allow CLI overrides
	if args.tensorboard_dir:
		global_cfg['tensorboard_dir'] = args.tensorboard_dir
	if args.checkpoint_dir:
		global_cfg['checkpoint_dir'] = args.checkpoint_dir

	# Load inference config if provided
	inference_cfg = None
	if args.inference_config:
		logger.info(f'Loading inference config from {args.inference_config}')
		with open(args.inference_config, 'r') as f:
			inference_cfg = yaml.safe_load(f)

	device = torch.device(global_cfg.get('device', 'cpu'))

	if random_intervals is not None:
		todo_new = []
		for name in todo:
			for prop in random_intervals:
				todo_new.append(f"{name}-r{prop}")
    
		todo = todo_new

	for name in todo:
		cfg = runs[name]

		datestr = time.strftime('%Y%m%d_%H%M%S')
		tb_dir = os.path.join(global_cfg['tensorboard_dir'], name, datestr)
		ckpt_dir = os.path.join(global_cfg['checkpoint_dir'], name, datestr)

		os.makedirs(tb_dir, exist_ok=False)
		os.makedirs(ckpt_dir, exist_ok=False)

		# Prepare inference config for this run
		run_inference_cfg = None
		if inference_cfg:
			run_inference_cfg = copy.deepcopy(inference_cfg)
			run_inference_cfg['model'] = cfg['model']
			run_inference_cfg['output_root'] = os.path.join(inference_cfg["save_dir"], name, datestr)
   
			if name not in inference_cfg["do"]:
				run_inference_cfg = None  # skip inference for this run
				logger.info(f'Skipping inference collection for run {name}')
			else:
				logger.info(f'Inference will be collected for run {name} at: {run_inference_cfg["output_root"]}')
				run_inference_cfg['collect'] = inference_cfg["collect"][cfg["model"]]
				run_inference_cfg["schedule"] = inference_cfg["epochs"][cfg["dataset"]]
				logger.info(f'Inference collection schedule: {run_inference_cfg["schedule"]}')
				logger.info(f'Inference collection layers: {run_inference_cfg["collect"]}')


		logger.info(f'\n=== Starting run: {name} ===')
		do_run(cfg, device, tb_dir, ckpt_dir, only_last_best=only_last_best, inference_cfg=run_inference_cfg)
		logger.info(f'=== Finished run: {name} ===\n')


if __name__ == '__main__':
	main()

