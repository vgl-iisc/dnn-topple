import argparse
import os
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "scripts")))

from scripts.get_accuracies import process_file

SEPARABILITY_METRICS = [
	"intra_class_mean_distance",
	"inter_class_mean_distance",
	"intra_inter_distance_ratio",
	"centroid_mean_separation",
	"centroid_min_separation",
	"centroid_mean_correlation",
	"neighborhood_purity",
	"knn_accuracy",
]


def _extract_split_acc(accuracy_df: pd.DataFrame, split_name: str) -> float:
	split_cols = [c for c in accuracy_df.columns if c.lower() in ("split", "set", "phase")]
	if len(split_cols) == 0 or "accuracy" not in accuracy_df.columns:
		return np.nan

	split_col = split_cols[0]
	matches = accuracy_df[accuracy_df[split_col].astype(str).str.lower() == split_name.lower()]
	if len(matches) == 0:
		return np.nan

	return float(matches["accuracy"].iloc[0])


def _compiled_results_candidates(data_dir: str, model: str, dataset: str, randomness: float) -> list[str]:
	candidates = []
	if randomness > 0.0:
		candidates.append(os.path.join(data_dir, f"{model}_{dataset}-r{randomness:g}", "compiled_results.csv"))
	candidates.append(os.path.join(data_dir, f"{model}_{dataset}", "compiled_results.csv"))
	return candidates


def attach_classification_accuracies(metrics_df: pd.DataFrame, data_dir: str) -> pd.DataFrame:
	df = metrics_df.copy()
	train_accs = []
	val_accs = []
	cache = {}

	for _, row in df.iterrows():
		model = str(row.get("model", ""))
		dataset = str(row.get("dataset", ""))
		epoch = int(float(row.get("epoch", 0)))
		randomness = float(row.get("randomness", 0.0))
		cache_key = (model, dataset, randomness, epoch)
		if cache_key not in cache:
			compiled_res = None
			for candidate in _compiled_results_candidates(data_dir, model, dataset, randomness):
				if os.path.exists(candidate):
					compiled_res = candidate
					break
			if compiled_res is None:
				cache[cache_key] = (np.nan, np.nan)
			else:
				try:
					acc_df = process_file(compiled_res, epoch)
					cache[cache_key] = (
						_extract_split_acc(acc_df, "Train"),
						_extract_split_acc(acc_df, "Val"),
					)
				except Exception:
					cache[cache_key] = (np.nan, np.nan)

		train_acc, val_acc = cache[cache_key]
		train_accs.append(train_acc)
		val_accs.append(val_acc)

	df["train_acc"] = train_accs
	df["val_acc"] = val_accs
	return df


def _get_all_metrics(data: pd.DataFrame) -> list[str]:
	"""Return static separability metrics present in the dataframe plus any knn-graph derived columns.
	Excludes columns that are entirely NaN (e.g. disabled computations).
	"""
	metrics = [m for m in SEPARABILITY_METRICS if m in data.columns and data[m].notna().any()]
	for col in sorted(data.columns):
		if col.startswith("neighbourhood_purity_") or col.startswith("knn_accuracy_"):
			if data[col].notna().any():
				metrics.append(col)
	return metrics


def _corr_entry(group: pd.DataFrame, metric: str) -> dict:
	return {
		"n": len(group),
		"train_pearson": group["train_acc"].corr(group[metric]),
		"val_pearson": group["val_acc"].corr(group[metric]),
		"train_spearman": group["train_acc"].corr(group[metric], method="spearman"),
		"val_spearman": group["val_acc"].corr(group[metric], method="spearman"),
	}


def compute_correlations(data: pd.DataFrame, out_csv: str | None = None) -> pd.DataFrame:
	correlations = {}
	metrics = _get_all_metrics(data)
	has_randomness = "randomness" in data.columns

	for layer, layer_group in data.groupby("layer"):
		for metric in metrics:
			correlations[f"full__{layer}__{metric}"] = _corr_entry(layer_group, metric)

		for dataset_name, group in layer_group.groupby("dataset"):
			for metric in metrics:
				correlations[f"{dataset_name}__{layer}__{metric}"] = _corr_entry(group, metric)

		for model_name, group in layer_group.groupby("model"):
			for metric in metrics:
				correlations[f"{model_name}__{layer}__{metric}"] = _corr_entry(group, metric)

		for (dataset_name, model_name), group in layer_group.groupby(["dataset", "model"]):
			for metric in metrics:
				correlations[f"{dataset_name}__{model_name}__{layer}__{metric}"] = _corr_entry(group, metric)

		if has_randomness:
			for randomness, group in layer_group.groupby("randomness"):
				for metric in metrics:
					correlations[f"randomness_{randomness:g}__{layer}__{metric}"] = _corr_entry(group, metric)

			for (dataset_name, randomness), group in layer_group.groupby(["dataset", "randomness"]):
				for metric in metrics:
					correlations[f"{dataset_name}__randomness_{randomness:g}__{layer}__{metric}"] = _corr_entry(group, metric)

			for (model_name, randomness), group in layer_group.groupby(["model", "randomness"]):
				for metric in metrics:
					correlations[f"{model_name}__randomness_{randomness:g}__{layer}__{metric}"] = _corr_entry(group, metric)

			for (dataset_name, model_name, randomness), group in layer_group.groupby(["dataset", "model", "randomness"]):
				for metric in metrics:
					correlations[f"{dataset_name}__{model_name}__randomness_{randomness:g}__{layer}__{metric}"] = _corr_entry(group, metric)

	corr_df = pd.DataFrame.from_dict(correlations, orient="index")

	if out_csv is not None:
		corr_df.to_csv(out_csv)
		print(f"Correlation results saved to {out_csv}")

	return corr_df


def run_separability_metrics(
	python_executable: str,
	datasets_dir: str,
	data_dir: str,
	output_csv: str,
	max_intra_pairs_per_class: int,
	max_inter_pairs: int,
	seed: int,
	knn_graphs_dir: str | None = None,
) -> None:
	cmd = [
		python_executable,
		"experiments/separability_metrics/geometric_separability_metrics.py",
		datasets_dir,
		data_dir,
		output_csv,
		"--max_intra_pairs_per_class",
		str(max_intra_pairs_per_class),
		"--max_inter_pairs",
		str(max_inter_pairs),
		"--seed",
		str(seed),
	]
	if knn_graphs_dir is not None:
		cmd += ["--knn_graphs_dir", knn_graphs_dir]
	print("Running separability metrics...")
	subprocess.run(cmd, check=True)



def main() -> None:
	parser = argparse.ArgumentParser(
		description="Run geometric separability metrics and save metric-accuracy correlations."
	)
	parser.add_argument("datasets_dir", type=str, help="Path to datasets directory.")
	parser.add_argument("data_dir", type=str, help="Path to landscape data directory.")
	parser.add_argument("output_dir", type=str, help="Directory where CSV outputs are saved.")
	parser.add_argument("--output_prefix", type=str, default="geometric_separability", help="Prefix for output files.")
	parser.add_argument("--python_executable", type=str, default=sys.executable, help="Python executable used to launch metric script.")
	parser.add_argument(
		"--knn_graphs_dir",
		type=str,
		default=None,
		help="Parent directory containing knns_cross_epoch_* subdirs with pre-built KNN adjacency lists.",
	)
	parser.add_argument("--max_intra_pairs_per_class", type=int, default=10000)
	parser.add_argument("--max_inter_pairs", type=int, default=100000)
	parser.add_argument("--seed", type=int, default=42)

	args = parser.parse_args()

	os.makedirs(args.output_dir, exist_ok=True)

	metrics_csv = os.path.join(args.output_dir, f"{args.output_prefix}.csv")
	run_separability_metrics(
		python_executable=args.python_executable,
		datasets_dir=args.datasets_dir,
		data_dir=args.data_dir,
		output_csv=metrics_csv,
		max_intra_pairs_per_class=args.max_intra_pairs_per_class,
		max_inter_pairs=args.max_inter_pairs,
		seed=args.seed,
		knn_graphs_dir=args.knn_graphs_dir,
	)

	df = pd.read_csv(metrics_csv)
	df = attach_classification_accuracies(df, args.data_dir)
	df.to_csv(metrics_csv, index=False)

	corrs_csv = metrics_csv.replace(".csv", "_corrs.csv")
	compute_correlations(df, out_csv=corrs_csv)

	print(f"Saved metrics to {metrics_csv}")


if __name__ == "__main__":
	main()
