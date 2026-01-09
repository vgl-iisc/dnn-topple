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
import model_loader

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

def infer(model, loader, criterion, device): 
	losses = []
	labels = []
	predicted = []
	correct= []
	idxs = []
 
	with torch.no_grad():
		for batch in tqdm(loader, total=len(loader), desc='Inferring'):
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
   
	total = len(loader.dataset)
	avg_loss = (torch.cat(losses).sum()).item() / total
	accuracy = torch.cat(correct)
	accuracy = len(accuracy[accuracy == True]) / total

	return {
		"loss": avg_loss,
		"accuracy": accuracy,
		"losses": losses,
		"labels": labels,
		"predicted": predicted,
		"indices": idxs
	}

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

def do_run(dataset, data_root, arch, checkpoints_dir, collection, epochs, output_root, device, random_prop=0.0):
	
	def write_output(split, output, collected_activations, epoch):
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
  
		collected_losses = torch.cat(output["losses"]).detach().cpu().numpy().reshape(-1, 1).squeeze()
		collected_preds = torch.cat(output["predicted"]).detach().cpu().numpy().reshape(-1, 1).squeeze()

   
		for tag, activations in collected_activations.items():
			stacked = torch.cat(activations).detach().cpu()
			logger.info(f"Activations {tag} shape {stacked.shape}")
			collated = stacked.reshape(len(stacked), -1)
			logger.info(f"Writing activations for tag {tag} with shape {collated.shape}")
			torch.save(collated, os.path.join(tens_dir, f"a{tag}_e{epoch}.pt"))
		
		assert collected_losses.shape == collected_preds.shape
		np.savetxt(os.path.join(loss_dir, f"losses_e{epoch}.txt"), collected_losses)
		np.savetxt(os.path.join(preds_dir, f"predictions_e{epoch}.txt"), collected_preds, fmt='%d')
  	
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
		else:
			ckpt = os.path.join(checkpoints_dir, f"checkpoint_e{epoch}.pt")
   
		model = build_model(arch, num_classes, device, checkpoint=ckpt)
		
		collected_input_activations_train = {}
		hooks = attach_collection_hooks(model, collection, collected_input_activations_train)
		criterion = nn.CrossEntropyLoss(reduction="none")
		train_results = infer(model, train_loader, criterion, device)

		for h in hooks:
			h.remove()

		collected_input_activations_val = {}
		hooks = attach_collection_hooks(model, collection, collected_input_activations_val)
		val_results = infer(model, test_loader, criterion, device)
  
		for h in hooks:
			h.remove()

		logger.info(f"Writing train outputs for epoch {epoch}...")
		write_output("train", train_results, collected_input_activations_train, epoch)
		logger.info(f"Writing val outputs for epoch {epoch}...")
		write_output("val", val_results, collected_input_activations_val, epoch)

		logger.info(f"Combining train+val outputs for epoch {epoch}...")
		combined_results = {key: train_results[key] + val_results[key] for key in train_results}
		combined_collected_activations = {tag: collected_input_activations_train[tag] + collected_input_activations_val[tag] for tag in collected_input_activations_train}

		logger.info(f"Writing combined train+val outputs for epoch {epoch}...")
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

		do_run(dataset, cfg["datasets"][dataset], model, checkpoints_dir, collect, epochs, output_root, device, random_prop=random_prop)

		elapsed = time.time() - start_time
		logger.info(f"Finished task: {task} in {elapsed:.2f} seconds")

if __name__ == '__main__':
	main()