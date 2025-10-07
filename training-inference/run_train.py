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

import cifar10_loader
import mnist_loader
import model_loader

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

	if ds == 'cifar10':
		train_tf, _ = cifar10_loader.get_cifar10_transforms()
		train_loader, test_loader = cifar10_loader.make_cifar10_dataloaders(
			cfg['data_root'], batch_size=batch_size, transform=train_tf, shuffle=True
		)
	elif ds == 'mnist':
		train_tf, _ = mnist_loader.get_mnist_transforms()
		train_loader, test_loader = mnist_loader.make_mnist_dataloaders(
			cfg['data_root'], batch_size=batch_size, transform=train_tf, shuffle=True
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

def train_epoch(model, loader, criterion, optimizer, device, epoch, writer=None):
	model.train()
	running_loss = 0.0
	correct = 0
	total = 0
	for step, batch in tqdm(enumerate(loader), total=len(loader), desc=f'Epoch {epoch}'):
		images, labels = extract_batch(batch)
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

	epoch_loss = running_loss / total
	epoch_acc = correct / total
	return epoch_loss, epoch_acc


def validate(model, loader, criterion, device):
	model.eval()
	running_loss = 0.0
	correct = 0
	total = 0
	with torch.no_grad():
		for batch in tqdm(loader, total=len(loader), desc='Validation'):
			images, labels = extract_batch(batch)
			images = images.to(device)
			labels = labels.to(device)
			outputs = model(images)
			loss = criterion(outputs, labels)
			running_loss += loss.item() * images.size(0)
			_, preds = torch.max(outputs, 1)
			correct += (preds == labels).sum().item()
			total += labels.size(0)
	return running_loss / total, correct / total


def save_checkpoint(state, ckpt_dir, epoch):
	os.makedirs(ckpt_dir, exist_ok=True)
	path = os.path.join(ckpt_dir, f'checkpoint_epoch_{epoch}.pt')
	torch.save(state, path)
	return path

def do_run(cfg, device, tboard_base, checkpoints_base):
	train_loader, test_loader = get_dataloaders(cfg)
	num_classes = train_loader.dataset.num_classes
 
	logger.info(f"Using config: {cfg}")

	model = build_model(cfg, num_classes, device)

	criterion = nn.CrossEntropyLoss()
	optimizer = make_optimizer(model, cfg)

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
		train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch, writer)
		logger.info('Evaluating on validation set...')
		val_loss, val_acc = validate(model, test_loader, criterion, device)

		writer.add_scalar('train/loss', train_loss, epoch)
		writer.add_scalar('train/accuracy', train_acc, epoch)
		writer.add_scalar('val/loss', val_loss, epoch)
		writer.add_scalar('val/accuracy', val_acc, epoch)

		# log lr
		for i, param_group in enumerate(optimizer.param_groups):
			writer.add_scalar(f'optimizer/group_{i}_lr', param_group.get('lr', 0.0), epoch)

		epoch_time = time.time() - t0
		logger.info(f'Epoch {epoch}/{epochs} - train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f} ({epoch_time:.1f}s)')

		# checkpoint
		state = {
			'epoch': epoch,
			'model_state_dict': model.state_dict(),
			'optimizer_state_dict': optimizer.state_dict(),
			'cfg': cfg,
		}
		save_checkpoint(state, ckpt_dir, epoch)
		if val_acc > best_val_acc:
			best_val_acc = val_acc
			best_epoch = epoch
			best_path = os.path.join(ckpt_dir, 'best.pt')
			torch.save(state, best_path)
   
	logger.info(f'Best validation accuracy: {best_val_acc:.4f} @ {best_epoch}')

	writer.close()

def main(argv=None):
	parser = argparse.ArgumentParser()
	parser.add_argument('--config', '-c', required=True, help='YAML config file')
	parser.add_argument('--output_file', '-o', required=True, help='output file to log metrics to')
	parser.add_argument('--tensorboard_dir', default=None, help='override tensorboard dir from config')
	parser.add_argument('--checkpoint_dir', default=None, help='override checkpoint dir from config')
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

	train_runs_dir = os.path.join(os.path.dirname(__file__), 'train_runs')
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
				runs[k] = cfg
				runs[k]["data_root"] = global_cfg["datasets"][runs[k]["dataset"]]

	todo = global_cfg["do"]
	global_cfg = global_cfg["config"]

	# allow CLI overrides
	if args.tensorboard_dir:
		global_cfg['tensorboard_dir'] = args.tensorboard_dir
	if args.checkpoint_dir:
		global_cfg['checkpoint_dir'] = args.checkpoint_dir

	device = torch.device(global_cfg.get('device', 'cpu'))

	for name in todo:
		cfg = runs[name]

		datestr = time.strftime('%Y%m%d_%H%M%S')
		tb_dir = os.path.join(global_cfg['tensorboard_dir'], name, datestr)
		ckpt_dir = os.path.join(global_cfg['checkpoint_dir'], name, datestr)

		os.makedirs(tb_dir, exist_ok=False)
		os.makedirs(ckpt_dir, exist_ok=False)

		logger.info(f'\n=== Starting run: {name} ===')
		do_run(cfg, device, tb_dir, ckpt_dir)
		logger.info(f'=== Finished run: {name} ===\n')


if __name__ == '__main__':
	main()

