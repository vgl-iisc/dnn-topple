"""
This script expects a YAML configuration describing the inference runs.
Run:
	python run_inference.py --config path/to/config.yaml -o path/to/output.log
"""

import argparse
import os
import yaml
import time

from tqdm import tqdm

from glob import glob

import torch
import torch.nn as nn
import torchvision as tv

import imagenet_loader
import cifar10_loader
import mnist_loader
import emnist_loader
import model_loader

import shutil

import pandas as pd
import numpy as np

from logging import Logger, FileHandler, Formatter, StreamHandler

logger = Logger(__name__)

def extract_batch(batch):
	"""Support both dict-batches (our datasets) and tuple batches."""
	if isinstance(batch, dict):
		images = batch['image']
		labels = batch['label']
		idxs = batch['index'].to(dtype=torch.uint64)
	else:
		images, labels = batch
		idxs = torch.tensor([0 for i in range(len(batch))], dtype=torch.uint64)
	return images, labels, idxs

BATCH_SIZE = 512

def get_dataloaders(dataset, data_root, random_label_prop=0.0):
	
	if dataset == 'cifar10' or dataset == 'cifar':
		_, test_tf = cifar10_loader.get_cifar10_transforms()
		train_loader, test_loader = cifar10_loader.make_cifar10_dataloaders(
			data_root, batch_size=BATCH_SIZE, train_transform=test_tf, test_transform=test_tf, shuffle=False, random_label_prop=random_label_prop
		)
	elif dataset == 'mnist':
		_, test_tf = mnist_loader.get_mnist_transforms()
		train_loader, test_loader = mnist_loader.make_mnist_dataloaders(
			data_root, batch_size=BATCH_SIZE, transform=test_tf, shuffle=False, random_label_prop=random_label_prop
		)
	elif dataset == 'emnist':
		_, test_tf = emnist_loader.get_emnist_transforms()
		train_loader, test_loader = emnist_loader.make_emnist_dataloaders(
			data_root, variant='byclass', batch_size=BATCH_SIZE, transform=test_tf, shuffle=False, random_label_prop=random_label_prop
		)
	elif dataset == 'emnist_balanced':
		_, test_tf = emnist_loader.get_emnist_transforms()
		train_loader, test_loader = emnist_loader.make_emnist_dataloaders(
			data_root, variant='balanced', batch_size=BATCH_SIZE, transform=test_tf, shuffle=False, random_label_prop=random_label_prop
		)
	elif dataset == 'emnist_letters':
		_, test_tf = emnist_loader.get_emnist_transforms()
		train_loader, test_loader = emnist_loader.make_emnist_dataloaders(
			data_root, variant='letters', batch_size=BATCH_SIZE, transform=test_tf, shuffle=False, random_label_prop=random_label_prop
		)
	elif dataset == "imagenetpt":
		test_tf = tv.models.ResNet50_Weights.IMAGENET1K_V1.transforms()
		train_loader, test_loader = imagenet_loader.make_imagenet_dataloaders(data_root, batch_size=int(BATCH_SIZE / 1.5), train_transform=test_tf, test_transform=test_tf, shuffle=False)
	else:
		raise ValueError(f'Unsupported dataset: {dataset}')
	return train_loader, test_loader

def build_model(arch, num_classes, device, checkpoint=None):
	model = model_loader.create_model(arch, num_classes=num_classes, checkpoint=checkpoint, device=device)
	model.eval()
	return model

def infer(model, epoch, loader, criterion, device, collected_activations=None, save_every_batches=None, act_flush_dir=None): 
	losses = []
	labels = []
	predicted = []
	correct= []
	idxs = []

	streaming = (collected_activations is not None and save_every_batches is not None and act_flush_dir is not None)
	n_act_chunks = 0

	if streaming:
		os.makedirs(act_flush_dir, exist_ok=True)

	def flush_activations(epoch, batch_i=None):
		nonlocal n_act_chunks
		loc = f" at batch {batch_i}" if batch_i is not None else " (final)"
		logger.info(f"Flushing activation chunk {n_act_chunks}{loc} to {act_flush_dir}")
		for tag, acts in collected_activations.items():
			if acts:
				torch.save(torch.cat(acts).cpu(), os.path.join(act_flush_dir, f'a{tag}_e{epoch}_chunk{n_act_chunks}.pt'))
				acts.clear()
		n_act_chunks += 1
 
	with torch.no_grad():
		for batch_i, batch in enumerate(tqdm(loader, total=len(loader), desc='Inferring')):
			image, label, idx = extract_batch(batch)
			image = image.to(device)
			label = label.to(device)
			output = model(image)

			loss = criterion(output, label)

			pred = torch.argmax(output, dim=1)

			losses.append(loss)
			labels.append(label)
			predicted.append(pred)
			correct.append(pred == label)
			idxs.append(idx)

			if streaming and (batch_i + 1) % save_every_batches == 0:
				flush_activations(epoch=epoch, batch_i=batch_i)

	if streaming and any(len(acts) > 0 for acts in collected_activations.values()):
		flush_activations(epoch=epoch, batch_i=None)
   
	total = len(loader.dataset)
	avg_loss = (torch.cat(losses).sum()).item() / total
	accuracy = torch.cat(correct)
	accuracy = len(accuracy[accuracy == True]) / total

	result = {
		"loss": avg_loss,
		"accuracy": accuracy,
		"losses": losses,
		"labels": labels,
		"predicted": predicted,
		"indices": idxs
	}
	if streaming:
		result['act_flush_dir'] = act_flush_dir
		result['n_act_chunks'] = n_act_chunks
	return result

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

	base = ds_subsample // n_classes
	remainder = ds_subsample % n_classes

	if remainder != 0:
		logger.warning(
			f"Cannot sample exactly equally for split '{split_name}': ds_subsample={ds_subsample} not divisible by {n_classes} classes. "
			f"Using near-balanced allocation."
		)

	targets = torch.full((n_classes,), base, dtype=torch.int64)
	if remainder > 0:
		targets[:remainder] += 1

	available = class_counts.to(dtype=torch.int64)
	selected_per_class = torch.minimum(targets, available)
	deficit = int(ds_subsample - selected_per_class.sum().item())

	if torch.any(selected_per_class < targets):
		logger.warning(
			f"Equal per-class sampling not possible for split '{split_name}' due to class counts. "
			f"Requested targets={targets.tolist()}, available={available.tolist()}."
		)

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
			perm = torch.randperm(int(cls_positions.numel()))
			chosen = cls_positions[perm[:k]]
		else:
			chosen = cls_positions
		selected_positions.append(chosen)

	if len(selected_positions) == 0:
		return None

	selected_positions = torch.cat(selected_positions)
	selected_positions, _ = torch.sort(selected_positions)
	return selected_positions


def attach_collection_hooks(model, collection, collected_input_activations):
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
		
		collected_input_activations[tag] = []

		mod = find_module(path)
		logger.info(f"Attaching hook to {path} with tag {tag}: found {mod.__class__.__name__}")
  
		def hook_fn(m, i, o, tag=tag):
			collected_input_activations[tag].append(i[0].detach().cpu())
     
		hooks.append(mod.register_forward_hook(hook_fn))
  
	return hooks

def do_run(dataset, data_root, arch, checkpoints_dir, collection, epochs, output_root, device, random_prop=0.0, ds_subsample=None, save_every_batches=None):
	
	def write_output(split, output, collected_activations, epoch, extra_act_tmp_dir=None, extra_n_act_chunks=0):
		loss_dir = os.path.join(output_root, "Losses", split)
		preds_dir = os.path.join(output_root, "Predictions", split)
		tens_dir = os.path.join(output_root, "Tensors", split)

		os.makedirs(loss_dir, exist_ok=True)
		os.makedirs(preds_dir, exist_ok=True)
		os.makedirs(tens_dir, exist_ok=True)

		avg_loss = output["loss"]
		accuracy = output["accuracy"]
		
		with open(os.path.join(output_root, f"metrics_e{epoch}"), "w") as f:
			f.write(f"avg loss: {avg_loss}\naccuracy: {accuracy}\n")

		all_losses = torch.cat(output["losses"]).detach().cpu()
		all_preds = torch.cat(output["predicted"]).detach().cpu()

		collected_losses = all_losses.numpy().reshape(-1, 1).squeeze()
		collected_preds = all_preds.numpy().reshape(-1, 1).squeeze()

		assert collected_losses.shape == collected_preds.shape
		np.savetxt(os.path.join(loss_dir, f"losses_e{epoch}.txt"), collected_losses)
		np.savetxt(os.path.join(preds_dir, f"predictions_e{epoch}.txt"), collected_preds, fmt='%d')

		act_flush_dir = output.get('act_flush_dir')

		if act_flush_dir is not None:
			# Streaming mode: chunks already written to disk by infer(), nothing to do here
			pass
		else:
			for tag, activations in collected_activations.items():
				stacked = torch.cat(activations).detach().cpu()
				logger.info(f"Activations {tag} shape {stacked.shape}")
				collated = stacked.reshape(len(stacked), -1)
				logger.info(f"Writing activations for tag {tag} with shape {collated.shape}")
				torch.save(collated, os.path.join(tens_dir, f"a{tag}_e{epoch}.pt"))
  	
	def write_compiled_csv(epoch_results, output_root):
		
		data = {
			"Epoch_No": [],
			"Split": [], 
			"Image_Index": [],
			"Image_Function_value": [],
			"Original_Label": [],
			"Predicted_Label": [],
			"Correct": []
		}
	
		for epoch, results in epoch_results.items():
			for split in results:
				res = results[split]

				labels = torch.cat(res['labels']).detach().cpu().numpy()
				idxs = torch.cat(res['indices']).detach().cpu().numpy()
				losses = torch.cat(res['losses']).detach().cpu().numpy()
				preds = torch.cat(res['predicted']).detach().cpu().numpy()

				num_samples = len(labels)
				data["Epoch_No"].extend([epoch] * num_samples)
				data["Split"].extend([split] * num_samples)

				data["Image_Index"].extend(idxs)
				data["Image_Function_value"].extend(losses)
				data["Original_Label"].extend(labels)
				data["Predicted_Label"].extend(preds)
				data["Correct"].extend(["CORRECT" if c == l else "INCORRECT" for c, l in zip(preds, labels)])
	
		df = pd.DataFrame(data)
		df.to_csv(os.path.join(output_root, "compiled_results.csv"), index=False)

	train_loader, test_loader = get_dataloaders(dataset, data_root, random_label_prop=random_prop)
	num_classes = train_loader.dataset.num_classes

	if ds_subsample is not None:
		def make_subset_loader(loader, split_name):
			raw_labels = torch.from_numpy(loader.dataset.labels.astype(np.int64))
			indices = build_fixed_balanced_indices(raw_labels, ds_subsample, split_name)
			if indices is None:
				return loader
			idxs_dir = os.path.join(output_root, "Indices", split_name)
			os.makedirs(idxs_dir, exist_ok=True)
			np.savetxt(os.path.join(idxs_dir, "selected_dataset_indices.txt"), indices.numpy(), fmt='%d')
			logger.info(f"Subsampled '{split_name}' to {len(indices)} samples (from {len(raw_labels)})")
			subset = torch.utils.data.Subset(loader.dataset, indices.numpy())
			return torch.utils.data.DataLoader(
				subset,
				batch_size=loader.batch_size,
				shuffle=False,
				num_workers=loader.num_workers,
				pin_memory=loader.pin_memory,
			)
		train_loader = make_subset_loader(train_loader, 'train')
		test_loader = make_subset_loader(test_loader, 'val')

	if checkpoints_dir is not None:
		logger.info(f"Looking for checkpoints in {checkpoints_dir}")
		best_epoch = int(glob("best_e*.pt", root_dir=checkpoints_dir)[0].split("_e")[-1].split(".pt")[0])
		run_epochs = set()

		all_epochs = glob("checkpoint_e*.pt", root_dir=checkpoints_dir)
		all_epochs = [int(f.split("_e")[-1].split(".pt")[0]) for f in all_epochs]
		last_epoch = max(all_epochs)
	
		logger.info(f"Best epoch: {best_epoch}, last epoch: {last_epoch}")
	
		for epoch_range in epochs:
			if epoch_range == "best":
				run_epochs.add(best_epoch)
			elif epoch_range == "last":
				run_epochs.add(last_epoch)
			elif epoch_range == "all":
				run_epochs.update(all_epochs)
				run_epochs.add(0)
			elif isinstance(epoch_range, int):
				run_epochs.add(epoch_range)
			elif isinstance(epoch_range, list):
				run_epochs.update(range(*map(int, epoch_range)))
	else:
		run_epochs = {0}
		best_epoch = 0
		last_epoch = 0
	
	logger.info(f"Will run inference for epochs: {sorted(run_epochs)}")
	epoch_results = {}
 
	for epoch in run_epochs:
		logger.info(f"Processing epoch {epoch}...")

		if epoch == 0:
			ckpt = None
		elif epoch == best_epoch:
			ckpt = os.path.join(checkpoints_dir, f"best_e{epoch}.pt")
		else:
			ckpt = os.path.join(checkpoints_dir, f"checkpoint_e{epoch}.pt")
   
		model = build_model(arch, num_classes, device, checkpoint=ckpt)
		
		collected_input_activations_train = {}
		hooks = attach_collection_hooks(model, collection, collected_input_activations_train)
		criterion = nn.CrossEntropyLoss(reduction="none")

		if save_every_batches is not None:
			train_act_tmp = os.path.join(output_root, "Tensors", "train")
			train_results = infer(model, epoch, train_loader, criterion, device,
								  collected_activations=collected_input_activations_train,
								  save_every_batches=save_every_batches,
								  act_flush_dir=train_act_tmp)
		else:
			train_results = infer(model, epoch, train_loader, criterion, device)

		for h in hooks:
			h.remove()

		if save_every_batches is not None:
			logger.info(f"Writing train outputs for epoch {epoch}...")
			write_output("train", train_results, collected_input_activations_train, epoch)
			epoch_results[epoch] = {
				"Train": train_results
			}
			logger.info(f"Skipping val/trainUval for epoch {epoch} (streaming mode)")
			continue

		collected_input_activations_val = {}
		hooks = attach_collection_hooks(model, collection, collected_input_activations_val)

		if save_every_batches is not None:
			val_act_tmp = os.path.join(output_root, "_tmp", f"val_e{epoch}")
			val_results = infer(model, epoch, test_loader, criterion, device,
								collected_activations=collected_input_activations_val,
								save_every_batches=save_every_batches,
								act_flush_dir=val_act_tmp)
		else:
			val_results = infer(model, epoch, test_loader, criterion, device)
  
		for h in hooks:
			h.remove()

		logger.info(f"Writing train outputs for epoch {epoch}...")
		write_output("train", train_results, collected_input_activations_train, epoch)
		logger.info(f"Writing val outputs for epoch {epoch}...")
		write_output("val", val_results, collected_input_activations_val, epoch)

		logger.info(f"Combining train+val outputs for epoch {epoch}...")
		if save_every_batches is not None:
			# Chunks still exist on disk; build combined result pointing to both dirs.
			# Activation lists are empty (all flushed); pass them through for the non-streaming
			# code path inside write_output which will be skipped anyway.
			combined_results = {
				key: train_results[key] + val_results[key]
				for key in ('loss', 'accuracy', 'losses', 'labels', 'predicted', 'indices')
			}
			combined_results['act_tmp_dir'] = train_results['act_tmp_dir']
			combined_results['n_act_chunks'] = train_results['n_act_chunks']
			combined_collected_activations = {}
		else:
			combined_results = {key: train_results[key] + val_results[key] for key in train_results}
			combined_collected_activations = {tag: collected_input_activations_train[tag] + collected_input_activations_val[tag] for tag in collected_input_activations_train}

		logger.info(f"Writing combined train+val outputs for epoch {epoch}...")
		if save_every_batches is not None:
			write_output("trainUval", combined_results, combined_collected_activations, epoch,
						 extra_act_tmp_dir=val_results['act_tmp_dir'],
						 extra_n_act_chunks=val_results['n_act_chunks'])
			# All three writes done — now safe to delete intermediate chunk dirs
			for d in [train_results['act_tmp_dir'], val_results['act_tmp_dir']]:
				if os.path.exists(d):
					shutil.rmtree(d)
					logger.info(f"Deleted intermediate activation chunks at {d}")
		else:
			write_output("trainUval", combined_results, combined_collected_activations, epoch)

		epoch_results[epoch] = {
			"Train": train_results,
			"Val": val_results
		}

	logger.info(f"Writing compiled CSV...")
	write_compiled_csv(epoch_results, output_root)


def main(argv=None):
	parser = argparse.ArgumentParser()
	parser.add_argument('--config', '-c', required=True, help='YAML config file')
	parser.add_argument('--output_file', '-o', required=True, help='output file to log metrics to')
	args = parser.parse_args(argv)

	with open(args.config, 'r') as f:
		cfg = yaml.safe_load(f)

	log_formatter = Formatter('%(asctime)s - %(levelname)s - %(message)s')
	file_handler = FileHandler(args.output_file)
	file_handler.setFormatter(log_formatter)
	stream_handler = StreamHandler()
	stream_handler.setFormatter(log_formatter)
	logger.addHandler(file_handler)
	logger.addHandler(stream_handler)
	logger.setLevel('INFO')

	base_checkpoints_dir = cfg['checkpoints_dir']
	base_output_root = cfg['save_dir']
	device = torch.device(cfg.get('device', 'cpu'))
 
	logger.info(f"Using device: {device}")
 
	todo = cfg["do"]
 
	random = cfg.get("random", None)
 
	if random is not None:
		todo_new = []
		for name in todo:
			for prop in random:
				todo_new.append(f"{name}-r{prop}")
	
		todo = todo_new
 
	for task in todo:
		logger.info(f"Starting task: {task}")

		if random is not None:
			task_cfg = cfg["runs"][task.split("-r")[0]]
		else:
			task_cfg = cfg["runs"][task]
  
		start_time = time.time()

		random_prop = 0.0
  
		if random is not None:
			random_prop = float(task.split("-r")[-1])
			logger.info(f"Using random label proportion: {random_prop}")
  
		model = task_cfg["model"]
		dataset = task_cfg["dataset"]
  
		collect = cfg["collect"][model]
		epochs = cfg["epochs"][dataset]
  
		if not base_checkpoints_dir is None:
			checkpoints_dir = os.path.join(base_checkpoints_dir, task)
			output_root = os.path.join(base_output_root, task)
	
			biggest_name = None
			biggest = 0
			if len(glob("*.pt", root_dir=checkpoints_dir)) == 0:
				sorted_dirs = sorted(list(os.listdir(checkpoints_dir)))
				
				for dir in sorted_dirs:
					dir_path = os.path.join(checkpoints_dir, dir)
					if not os.path.isdir(dir_path):
						continue

					weights = glob("*.pt", root_dir=dir_path)
					if len(weights) >= biggest:
						biggest = len(weights)
						biggest_name = dir

				checkpoints_dir = os.path.join(checkpoints_dir, biggest_name)
				logger.info(f"found biggest weights dir for {task}: {checkpoints_dir} ({biggest})")
		else:
			logger.info(f"No checkpoints_dir specified, using random/init weights for {task}")
			checkpoints_dir = None
			output_root = os.path.join(base_output_root, task)

		do_run(dataset, cfg["datasets"][dataset], model, checkpoints_dir, collect, epochs, output_root, device, random_prop=random_prop, ds_subsample=cfg.get('ds_subsample', None), save_every_batches=cfg.get('save_every_batches', None))

		elapsed = time.time() - start_time
		logger.info(f"Finished task: {task} in {elapsed:.2f} seconds")

if __name__ == '__main__':
	main()