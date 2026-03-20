import os
import re
import shutil
import logging
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count, log_to_stderr

import glob
from collections import Counter

import networkx as nx
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import pairwise_distances

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from experiments.balance_metrics.experiment import find_all_datasets


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(processName)s - %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CPUS = 24

@dataclass
class TensorExperiment:
	model: str
	dataset: str
	dataset_label: str
	randomness: float
	split: str
	layer: str
	epoch: int
	tensor_path: str


@dataclass
class AdjExperiment:
	model: str
	dataset: str
	dataset_label: str
	randomness: float
	split: str
	epoch: int
	anchor: str
	adj_path: str


def parse_dataset_randomness(dataset_label: str) -> tuple[str, float]:
	"""Parse dataset label like 'mnist-r0.4' into ('mnist', 0.4). Defaults to randomness=0.0."""
	match = re.match(r"^(.*)-r([0-9]*\.?[0-9]+)$", dataset_label)
	if not match:
		return dataset_label, 0.0
	return match.group(1), float(match.group(2))


def parse_tensor_metadata(data_dir: str, tensor_path: str) -> TensorExperiment:
	rel_path = os.path.relpath(tensor_path, data_dir)
	parts = rel_path.split(os.sep)

	if "Tensors" not in parts:
		raise ValueError(f"Tensor path does not include 'Tensors': {tensor_path}")

	tensors_idx = parts.index("Tensors")
	if tensors_idx < 1 or tensors_idx + 1 >= len(parts):
		raise ValueError(f"Unexpected tensor directory layout: {tensor_path}")

	model_data = parts[tensors_idx - 1]
	split = parts[tensors_idx + 1]
	filename = os.path.basename(tensor_path)

	match = re.match(r"^(.+)_e(\d+)\.(txt|pt|npy|npz)$", filename)
	if not match:
		raise ValueError(f"Unexpected tensor filename format: {filename}")

	layer = match.group(1)
	epoch = int(match.group(2))

	md_parts = model_data.split("_")
	if len(md_parts) < 2:
		raise ValueError(f"Unable to parse model/dataset from '{model_data}'")

	model = md_parts[0]
	dataset_label = "_".join(md_parts[1:])
	dataset, randomness = parse_dataset_randomness(dataset_label)

	return TensorExperiment(
		model=model,
		dataset=dataset,
		dataset_label=dataset_label,
		randomness=randomness,
		split=split,
		layer=layer,
		epoch=epoch,
		tensor_path=tensor_path,
	)


def find_all_tensor_files(data_dir: str) -> list[str]:
	tensor_files = []
	valid_exts = {".txt", ".pt", ".npy", ".npz"}

	for root, _, files in os.walk(data_dir):
		if "Tensors" not in root:
			continue

		for file_name in files:
			base, ext = os.path.splitext(file_name)
			if ext.lower() not in valid_exts:
				continue
			tensor_files.append(os.path.join(root, file_name))

	return sorted(tensor_files)


def load_tensor_data(tensor_path: str) -> np.ndarray:
	if tensor_path.endswith(".pt"):
		tensor = torch.load(tensor_path)
		if isinstance(tensor, torch.Tensor):
			arr = tensor.detach().cpu().numpy()
		else:
			arr = np.asarray(tensor)
	elif tensor_path.endswith(".txt"):
		arr = np.loadtxt(tensor_path)
	elif tensor_path.endswith(".npy"):
		arr = np.load(tensor_path)
	elif tensor_path.endswith(".npz"):
		npz = np.load(tensor_path)
		if "arr_0" not in npz:
			raise ValueError(f"NPZ tensor does not have default array 'arr_0': {tensor_path}")
		arr = npz["arr_0"]
	else:
		raise ValueError(f"Unsupported tensor extension: {tensor_path}")

	arr = np.asarray(arr)
	if arr.ndim == 1:
		arr = arr.reshape(-1, 1)
	if arr.ndim > 2:
		arr = arr.reshape(arr.shape[0], -1)

	return arr.astype(np.float64, copy=False)


def _sample_intra_class_distances(X: np.ndarray, y: np.ndarray, max_pairs_per_class: int, rng: np.random.Generator) -> np.ndarray:
	dists = []
	for cls in np.unique(y):
		indices = np.where(y == cls)[0]
		n = len(indices)
		if n < 2:
			continue

		if n * (n - 1) // 2 <= max_pairs_per_class:
			block = X[indices]
			pair_d = pairwise_distances(block)
			i_upper = np.triu_indices(n, k=1)
			dists.append(pair_d[i_upper])
			continue

		n_samples = max_pairs_per_class
		i = rng.integers(0, n, size=n_samples)
		j = rng.integers(0, n, size=n_samples)
		valid = i != j
		i = i[valid]
		j = j[valid]
		if i.size == 0:
			continue

		xi = X[indices[i]]
		xj = X[indices[j]]
		dists.append(np.linalg.norm(xi - xj, axis=1))

	if len(dists) == 0:
		return np.array([], dtype=np.float64)
	return np.concatenate(dists)


def _sample_inter_class_distances(X: np.ndarray, y: np.ndarray, max_pairs: int, rng: np.random.Generator) -> np.ndarray:
	n = len(y)
	if n < 2:
		return np.array([], dtype=np.float64)

	collected = []
	remaining = max_pairs
	attempts = 0
	max_attempts = max_pairs * 10

	while remaining > 0 and attempts < max_attempts:
		batch = min(remaining * 2, 10000)
		i = rng.integers(0, n, size=batch)
		j = rng.integers(0, n, size=batch)
		valid = (i != j) & (y[i] != y[j])
		i = i[valid]
		j = j[valid]
		if i.size > 0:
			d = np.linalg.norm(X[i] - X[j], axis=1)
			if d.size > remaining:
				d = d[:remaining]
			collected.append(d)
			remaining -= d.size
		attempts += 1

	if len(collected) == 0:
		return np.array([], dtype=np.float64)
	return np.concatenate(collected)


def compute_separability_metrics(
	X: np.ndarray,
	y: np.ndarray,
	k_neighbors: int = 5,
	max_intra_pairs_per_class: int = 10000,
	max_inter_pairs: int = 100000,
	random_seed: int = 42,
) -> dict:
	if X.shape[0] != y.shape[0]:
		raise ValueError(f"Number of points ({X.shape[0]}) does not match number of labels ({y.shape[0]})")

	if X.shape[0] < 2:
		return {
			"num_points": int(X.shape[0]),
			"num_classes": int(len(np.unique(y))),
			"intra_class_mean_distance": np.nan,
			"inter_class_mean_distance": np.nan,
			"intra_inter_distance_ratio": np.nan,
			"centroid_mean_separation": np.nan,
			"centroid_min_separation": np.nan,
			"neighborhood_purity": np.nan,
			"knn_accuracy": np.nan,
		}

	# rng = np.random.default_rng(random_seed)
	classes = np.unique(y)

	# intra = _sample_intra_class_distances(X, y, max_pairs_per_class=max_intra_pairs_per_class, rng=rng)
	# inter = _sample_inter_class_distances(X, y, max_pairs=max_inter_pairs, rng=rng)

	# intra_mean = float(np.mean(intra)) if intra.size > 0 else np.nan
	# inter_mean = float(np.mean(inter)) if inter.size > 0 else np.nan

	# if np.isnan(intra_mean) or np.isnan(inter_mean) or inter_mean == 0.0:
	# 	ratio = np.nan
	# else:
	# 	ratio = intra_mean / inter_mean

	# centroids = []
	# for cls in classes:
	# 	centroids.append(X[y == cls].mean(axis=0))
	# centroids = np.asarray(centroids)

	# if centroids.shape[0] >= 2:
	# 	centroid_d = pairwise_distances(centroids)
	# 	i_upper = np.triu_indices(centroid_d.shape[0], k=1)
	# 	centroid_vals = centroid_d[i_upper]
	# 	centroid_mean = float(np.mean(centroid_vals))
	# 	centroid_min = float(np.min(centroid_vals))
	# else:
	# 	centroid_mean = np.nan
	# 	centroid_min = np.nan

	purity = np.nan
	knn_acc = np.nan

	return {
		"num_points": int(X.shape[0]),
		"num_classes": int(len(classes)),
		"intra_class_mean_distance": np.nan,
		"inter_class_mean_distance": np.nan,
		"intra_inter_distance_ratio": np.nan,
		"centroid_mean_separation": np.nan,
		"centroid_min_separation": np.nan,
		"neighborhood_purity": purity,
		"knn_accuracy": knn_acc,
	}


def find_knn_dirs(knn_graphs_dir: str) -> dict[int, str]:
	"""Scan knn_graphs_dir for knns_cross_epoch_* subdirs and return {k: path}."""
 
	if "knn" in knn_graphs_dir:
		return {20: knn_graphs_dir}
 
	k_dirs: dict[int, str] = {}
	for entry in os.scandir(knn_graphs_dir):
		if entry.is_dir():
			m = re.match(r"knns_cross_epoch_(\d+)$", entry.name)
			if m:
				k_dirs[int(m.group(1))] = entry.path
	return k_dirs


def find_adj_file(knn_dir: str, model: str, dataset_label: str, split: str, epoch: int, k: int, layer: str):
	"""Return the adjacency list file path for model/dataset/split/epoch/k, or None if not found."""
	folder = os.path.join(knn_dir, f"{model}_{dataset_label}", split)
	search_path = os.path.join(folder, f"adj_{layer}_e{epoch}_{k}_connected.txt")
	if not os.path.isdir(folder):
		return (None, search_path)
	matches = glob.glob(os.path.join(folder, f"adj_{layer}_e{epoch}_{k}_connected.txt"))
	return (matches[0], search_path) if matches else (None, search_path)


def discover_adj_tasks(k_dirs: dict[int, str]) -> list[AdjExperiment]:
	"""
	Discover unique (model, dataset, split, epoch, anchor) tasks by scanning one reference k directory.
	Each discovered task can then be evaluated across all available k values.
	"""
	if not k_dirs:
		return []

	ref_k = min(k_dirs.keys())
	ref_dir = k_dirs[ref_k]
	pattern = os.path.join(ref_dir, "*", "*", f"adj_a*_e*_{ref_k}_connected.txt")

	unique_tasks: dict[tuple[str, str, str, int, str], AdjExperiment] = {}
	for adj_path in glob.glob(pattern):
		rel = os.path.relpath(adj_path, ref_dir)
		parts = rel.split(os.sep)
		if len(parts) != 3:
			continue

		model_dataset, split, filename = parts
		name_match = re.match(rf"^adj_(.+)_e(\d+)_{ref_k}_connected\.txt$", filename)
		if not name_match:
			continue

		anchor = name_match.group(1)
		epoch = int(name_match.group(2))

		md_parts = model_dataset.split("_")
		if len(md_parts) < 2:
			continue

		model = md_parts[0]
		dataset_label = "_".join(md_parts[1:])
		dataset, randomness = parse_dataset_randomness(dataset_label)
		key = (model, dataset_label, split, epoch, anchor)
		if key not in unique_tasks:
			unique_tasks[key] = AdjExperiment(
				model=model,
				dataset=dataset,
				dataset_label=dataset_label,
				randomness=randomness,
				split=split,
				epoch=epoch,
				anchor=anchor,
				adj_path=adj_path,
			)

	return [unique_tasks[k] for k in sorted(unique_tasks.keys())]


def compute_knn_metrics_from_graph(adj_file: str, labels: np.ndarray) -> tuple[float, float]:
	"""
	Compute neighbourhood purity and knn accuracy from a pre-built KNN adjacency list.
	Nodes are expected to be integer string indices matching positions in `labels`.
	Returns (neighbourhood_purity, knn_accuracy).
	"""
	G = nx.read_adjlist(adj_file)
	n = labels.shape[0]
	purity_scores: list[float] = []
	correct = 0
	total = 0
	logger.info(f"Computing KNN metrics from graph {adj_file} with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
	for node in G.nodes():
		idx = int(node)
		if idx >= n:
			continue
		neighbors = [int(nb) for nb in G.neighbors(node) if int(nb) < n]
		if not neighbors:
			continue
		neighbor_labels = labels[neighbors]
		true_label = int(labels[idx])
		purity_scores.append(float(np.mean(neighbor_labels == true_label)))
		pred = Counter(neighbor_labels.tolist()).most_common(1)[0][0]
		correct += int(pred == true_label)
		total += 1
	purity = float(np.mean(purity_scores)) if purity_scores else np.nan
	knn_acc = float(correct / total) if total > 0 else np.nan
	return purity, knn_acc


def process_tensor_file(
	worker_id: int,
	tensor_path: str,
	data_dir: str,
	datasets_dir: str,
	k_neighbors: int,
	max_intra_pairs_per_class: int,
	max_inter_pairs: int,
	random_seed: int,
	k_dirs: dict[int, str] | None = None,
	per_file_log_dir: str | None = None,
) -> dict:
	log = logging.getLogger(__name__)
	log.info(f"{worker_id}: Processing tensor file {tensor_path}")

	meta = parse_tensor_metadata(data_dir, tensor_path)
	datasets = find_all_datasets(datasets_dir)

	dataset_key = meta.dataset if meta.dataset in datasets else meta.dataset_label
	if dataset_key not in datasets:
		raise ValueError(f"Dataset '{meta.dataset}' not found in {datasets_dir}")

	dataset = datasets[dataset_key]
	if meta.split not in dataset.labels_by_split:
		raise ValueError(f"Split '{meta.split}' not found for dataset '{dataset_key}'")

	labels = np.asarray(dataset.labels_by_split[meta.split], dtype=int)
	X = load_tensor_data(tensor_path)

	if X.shape[0] != labels.shape[0]:
		raise ValueError(
			f"Size mismatch for {tensor_path}: latent rows={X.shape[0]}, labels={labels.shape[0]} "
			f"(dataset={meta.dataset}, split={meta.split})"
		)

	metrics = compute_separability_metrics(
		X,
		labels,
		k_neighbors=k_neighbors,
		max_intra_pairs_per_class=max_intra_pairs_per_class,
		max_inter_pairs=max_inter_pairs,
		random_seed=random_seed,
	)

	result = {
		"dataset": meta.dataset,
		"randomness": meta.randomness,
		"split": meta.split,
		"model": meta.model,
		"epoch": meta.epoch,
		"layer": meta.layer,
		"tensor_path": tensor_path,
		"k_neighbors": k_neighbors,
		**metrics,
	}

	if k_dirs is not None:
		for k, knn_dir in sorted(k_dirs.items()):
			adj_file, search_path = find_adj_file(knn_dir, meta.model, meta.dataset_label, meta.split, meta.epoch, k)

			if adj_file is None:
				log.warning(f"{worker_id}: No adj file for {meta.model}_{meta.dataset}/{meta.split} epoch={meta.epoch} k={k} (searched {search_path})")
				result[f"neighbourhood_purity_{k}"] = np.nan
				result[f"knn_accuracy_{k}"] = np.nan
			else:
				purity, acc = compute_knn_metrics_from_graph(adj_file, labels)
				result[f"neighbourhood_purity_{k}"] = purity
				result[f"knn_accuracy_{k}"] = acc

	if per_file_log_dir is not None:
		safe_name = os.path.relpath(tensor_path, data_dir).replace(os.sep, "__").replace("/", "__")
		log_csv = os.path.join(per_file_log_dir, f"{safe_name}.csv")
		pd.DataFrame([result]).to_csv(log_csv, index=False)
		log.info(f"{worker_id}: Logged result to {log_csv}")

	log.info(f"{worker_id}: Completed tensor file {tensor_path}")
	return result


def process_adj_file(
	worker_id: int,
	adj_task: AdjExperiment,
	datasets_dir: str,
	k_dirs: dict[int, str] | None = None,
	per_file_log_dir: str | None = None,
) -> dict:
	log = logging.getLogger(__name__)
	log.info(f"{worker_id}: Processing adjacency task {adj_task.model}_{adj_task.dataset}/{adj_task.split} e{adj_task.epoch}")

	datasets = find_all_datasets(datasets_dir)
	dataset_key = adj_task.dataset if adj_task.dataset in datasets else adj_task.dataset_label
	if dataset_key not in datasets:
		raise ValueError(f"Dataset '{adj_task.dataset}' not found in {datasets_dir}")

	dataset = datasets[dataset_key]
	if adj_task.split not in dataset.labels_by_split:
		raise ValueError(f"Split '{adj_task.split}' not found for dataset '{dataset_key}'")

	labels = np.asarray(dataset.labels_by_split[adj_task.split], dtype=int)

	result = {
		"dataset": adj_task.dataset,
		"randomness": adj_task.randomness,
		"split": adj_task.split,
		"model": adj_task.model,
		"epoch": adj_task.epoch,
		"layer": adj_task.anchor,
		"tensor_path": np.nan,
		"k_neighbors": np.nan,
		"num_points": int(labels.shape[0]),
		"num_classes": int(len(np.unique(labels))),
		"intra_class_mean_distance": np.nan,
		"inter_class_mean_distance": np.nan,
		"intra_inter_distance_ratio": np.nan,
		"centroid_mean_separation": np.nan,
		"centroid_min_separation": np.nan,
		"neighborhood_purity": np.nan,
		"knn_accuracy": np.nan,
	}

	if k_dirs is not None:
		for k, knn_dir in sorted(k_dirs.items()):
			adj_file, search_path = find_adj_file(knn_dir, adj_task.model, adj_task.dataset_label, adj_task.split, adj_task.epoch, k, adj_task.anchor)
			if adj_file is None:
				log.warning(
					f"{worker_id}: No adj file for {adj_task.model}_{adj_task.dataset}/{adj_task.split} "
					f"epoch={adj_task.epoch} k={k} (searched {search_path})"
				)
				result[f"neighbourhood_purity_{k}"] = np.nan
				result[f"knn_accuracy_{k}"] = np.nan
			else:
				purity, acc = compute_knn_metrics_from_graph(adj_file, labels)
				result[f"neighbourhood_purity_{k}"] = purity
				result[f"knn_accuracy_{k}"] = acc

	if per_file_log_dir is not None:
		safe_name = f"{adj_task.model}_{adj_task.dataset_label}__{adj_task.split}__e{adj_task.epoch}.csv"
		log_csv = os.path.join(per_file_log_dir, safe_name)
		pd.DataFrame([result]).to_csv(log_csv, index=False)
		log.info(f"{worker_id}: Logged result to {log_csv}")

	log.info(f"{worker_id}: Completed adjacency task {adj_task.model}_{adj_task.dataset}/{adj_task.split} e{adj_task.epoch}")
	return result


def main(
	datasets_dir: str,
	data_dir: str,
	output_path: str,
	k_neighbors: int = 5,
	max_intra_pairs_per_class: int = 10000,
	max_inter_pairs: int = 100000,
	random_seed: int = 42,
	knn_graphs_dir: str | None = None,
) -> None:
	log_to_stderr(logging.INFO)
	logger.info("Starting geometric separability computation")

	tensor_files = find_all_tensor_files(data_dir)
	logger.info(f"Found {len(tensor_files)} tensor files to process")

	N_workers = max(1, CPUS)

	k_dirs = find_knn_dirs(knn_graphs_dir) if knn_graphs_dir is not None else None
	if k_dirs is not None:
		logger.info(f"Found KNN graph dirs for k values: {sorted(k_dirs.keys())}")

	output_dir = os.path.dirname(os.path.abspath(output_path))
	per_file_log_dir = os.path.join(output_dir, "per_file_logs")
	os.makedirs(per_file_log_dir, exist_ok=True)
	logger.info(f"Per-file logs will be written to {per_file_log_dir}")

	task_args = []
	if len(tensor_files) > 0:
		for i in range(N_workers):
			worker_files = tensor_files[i::N_workers]
			for tensor_path in worker_files:
				task_args.append(
					(
						i,
						tensor_path,
						data_dir,
						datasets_dir,
						k_neighbors,
						max_intra_pairs_per_class,
						max_inter_pairs,
						random_seed,
						k_dirs,
						per_file_log_dir,
					)
				)

		logger.info(f"Starting tensor processing: {len(task_args)} files across {N_workers} workers")
		with Pool(N_workers) as processes:
			results = processes.starmap(process_tensor_file, task_args)
	else:
		if not k_dirs:
			logger.warning("No tensor files found and no KNN graph directories provided/found")
			return

		adj_tasks = discover_adj_tasks(k_dirs)
		if len(adj_tasks) == 0:
			logger.warning("No tensor files found and no adjacency tasks discovered from KNN graph directories")
			return

		for i in range(N_workers):
			worker_tasks = adj_tasks[i::N_workers]
			for adj_task in worker_tasks:
				task_args.append((i, adj_task, datasets_dir, k_dirs, per_file_log_dir))

		logger.info(f"No tensor files found; starting adjacency-only processing: {len(task_args)} tasks across {N_workers} workers")
		with Pool(N_workers) as processes:
			results = processes.starmap(process_adj_file, task_args)

	results = [r for r in results if r is not None]
	if len(results) == 0:
		logger.error("No valid results were generated")
		return

	results_df = pd.DataFrame(results)
	os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None
	results_df.to_csv(output_path, index=False)
	logger.info(f"Saved separability metrics to {output_path}")

	shutil.rmtree(per_file_log_dir, ignore_errors=True)
	logger.info(f"Removed per-file log directory {per_file_log_dir}")


if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser(
		description="Compute latent-space geometric separability metrics for every tensor file in a data directory."
	)
	parser.add_argument("datasets_dir", type=str, help="Path to datasets directory.")
	parser.add_argument("data_dir", type=str, help="Path to landscape data directory containing Tensors folders.")
	parser.add_argument("output_path", type=str, help="Path to output CSV file.")
	parser.add_argument(
		"--max_intra_pairs_per_class",
		type=int,
		default=10000,
		help="Maximum sampled intra-class pairs per class for distance estimation.",
	)
	parser.add_argument(
		"--max_inter_pairs",
		type=int,
		default=100000,
		help="Maximum sampled inter-class pairs for distance estimation.",
	)
	parser.add_argument("--seed", type=int, default=124234, help="Random seed for pair sampling.")
	parser.add_argument(
		"--knn_graphs_dir",
		type=str,
		default=None,
		help="Parent directory containing knns_cross_epoch_* subdirs with pre-built KNN adjacency lists.",
	)

	args = parser.parse_args()

	main(
		datasets_dir=args.datasets_dir,
		data_dir=args.data_dir,
		output_path=args.output_path,
		max_intra_pairs_per_class=args.max_intra_pairs_per_class,
		max_inter_pairs=args.max_inter_pairs,
		random_seed=args.seed,
		knn_graphs_dir=args.knn_graphs_dir,
	)
