import numpy as np
import pandas as pd

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'vis')))

from scripts.get_accuracies import process_file
from scripts.chart_simplification_valleys import get_valley_vs_thresh
import scripts.tree_metrics as tm
import vis.experiment as exp
import vis.vis_utils as vu

from sklearn.linear_model import LinearRegression
from lmfit.models import ExponentialModel
import scipy.stats as stats

import matplotlib.pyplot as plt

METRICS = ["average_branching_factor", "colless_index", "sackin_index", "total_cophenetic_index", "average_colless_index", "average_sackin_index", "average_total_cophenetic_index"]
# THRESH_SELECTION = "valley=classes"
THRESH_SELECTION = 1e-2

def get_data(csv_path: str) -> pd.DataFrame:
	data = pd.read_csv(csv_path)
	return data

def ignore_outliers(df: pd.DataFrame, ignore_percentile: float = 0.0, ignore_std: float = 0.0) -> pd.DataFrame:
	if ignore_percentile > 0.0:
		df = df.copy()

		for metric in METRICS:
			if metric == "average_branching_factor":
				continue

			threshold = np.percentile(df[metric].fillna(0.0), ignore_percentile)
			if np.isnan(threshold):
				print(f"Percentile computation for metric {metric} resulted in NaN; skipping.")
				continue
			print(f"Ignoring above {ignore_percentile} percentile value {threshold} for metric {metric}")
			df = df[df[metric] < threshold]
  
	if ignore_std > 0.0:
		df = df.copy()

		for metric in METRICS:
			mean = df[metric].mean()
			std = df[metric].std()
			upper_bound = mean + ignore_std * std
			lower_bound = mean - ignore_std * std
			print(f"Ignoring values outside {ignore_std} std dev ({lower_bound}, {upper_bound}) for metric {metric}")
			df = df[(df[metric] >= lower_bound) & (df[metric] <= upper_bound)]
  
	return df

def plot_epoch_metrics_vs_accuracies(csv_path: str, out_dir: str, ignore_percentile: float = 0.0, ignore_std: float = 0.0):
	df = pd.read_csv(csv_path)

	epoch_vs_acc_path = os.path.join(out_dir, "epoch_vs_accuracy")
	merged_path = os.path.join(out_dir, "merged_plots")
 
	os.makedirs(epoch_vs_acc_path, exist_ok=True)
	os.makedirs(merged_path, exist_ok=True)

	base_filename = lambda ds, model, k, layer, split, thresh_mode: f"{model}_{ds}_{split}_k{k}_{layer}_{thresh_mode}"
 
	for name, group in df.groupby(['dataset', 'model', 'k', 'layer', 'split', 'thresh_mode']):
		ds, model, k, layer, split, thresh_selection = name
		group.sort_values(by='epoch', inplace=True)

		group = ignore_outliers(group, ignore_percentile=ignore_percentile, ignore_std=ignore_std)
    
		fig = plt.figure()
		plt.plot(group['epoch'], group['train_acc'], label='Train Accuracy')
		plt.plot(group['epoch'], group['val_acc'], label='Val Accuracy')
		plt.xlabel('Epoch')
		plt.ylabel('Accuracy')	
		plt.title('Epoch vs Accuracy')
		plt.legend()
		fig.tight_layout()
		fig.savefig(os.path.join(epoch_vs_acc_path, f"{base_filename(ds, model, k, layer, split, thresh_selection)}_accuracy.png"))
		plt.close(fig)
     
		for metric in METRICS:
			path_acc_vs_metric = os.path.join(out_dir, f"accuracy_vs_{metric}")
			os.makedirs(path_acc_vs_metric, exist_ok=True)
      
			fig = plt.figure()
			plt.scatter(group['train_acc'], group[metric], label='Train Accuracy')
			plt.scatter(group['val_acc'], group[metric], label='Val Accuracy')
			plt.xlabel('Accuracy')
			plt.ylabel(metric)
			plt.title(f'Accuracy vs {metric}')
			plt.legend()
			fig.tight_layout()	
			fig.savefig(os.path.join(path_acc_vs_metric, f"{base_filename(ds, model, k, layer, split, thresh_selection)}.png"))
			plt.close(fig)
   
			path_epoch_vs_metric = os.path.join(out_dir, f"epoch_vs_{metric}")
			os.makedirs(path_epoch_vs_metric, exist_ok=True)
			fig = plt.figure()
			plt.plot(group['epoch'], group[metric], label=metric)
			plt.xlabel('Epoch')
			plt.title(f'Epoch vs {metric}')
			plt.legend()
			fig.tight_layout()
			fig.savefig(os.path.join(path_epoch_vs_metric, f"{base_filename(ds, model, k, layer, split, thresh_selection)}.png"))
			plt.close(fig)
   
			path_epoch_vs_metric_acc = os.path.join(merged_path, f"epoch_vs_{metric}_and_accuracy")
			os.makedirs(path_epoch_vs_metric_acc, exist_ok=True)
			fig, ax1 = plt.subplots()
			color = 'tab:blue'
			ax1.set_xlabel('Epoch')
			ax1.set_ylabel(metric, color=color)
			ax1.plot(group['epoch'], group[metric], color=color, label=metric)
			ax1.tick_params(axis='y', labelcolor=color)
			ax2 = ax1.twinx()
			color = 'tab:orange'
			ax2.set_ylabel('Accuracy', color=color)
			ax2.plot(group['epoch'], group['train_acc'], color='green', label='Train Accuracy')
			ax2.plot(group['epoch'], group['val_acc'], color='orange', label='Val Accuracy')
			ax2.tick_params(axis='y', labelcolor=color)
			fig.tight_layout()
			fig.savefig(os.path.join(path_epoch_vs_metric_acc, f"{base_filename(ds, model, k, layer, split, thresh_selection)}.png"))
			plt.close(fig)

def compute_correlations(data: pd.DataFrame, out_csv: str | None = None, ignore_percentile: float = 0.0, ignore_std: float = 0.0) -> pd.DataFrame:
	correlations = {}

	data_full = ignore_outliers(data, ignore_percentile=ignore_percentile, ignore_std=ignore_std)

	for metric in METRICS:
		train_pearson = data_full['train_acc'].corr(data_full[metric])
		val_pearson = data_full['val_acc'].corr(data_full[metric])
		
		train_spearman = data_full['train_acc'].corr(data_full[metric], method='spearman')
		val_spearman = data_full['val_acc'].corr(data_full[metric], method='spearman')
  
		correlations[f"full_{metric}"] = {
			'n': len(data_full),
			'train_pearson': train_pearson,
			'val_pearson': val_pearson,
			'train_spearman': train_spearman,
			'val_spearman': val_spearman
		}
  
	by_dataset = data.groupby('dataset')
	for ds_name, group in by_dataset:
		group = ignore_outliers(group, ignore_percentile=ignore_percentile, ignore_std=ignore_std)
     
		for metric in METRICS:
			train_pearson = group['train_acc'].corr(group[metric])
			val_pearson = group['val_acc'].corr(group[metric])
	  
			train_spearman = group['train_acc'].corr(group[metric], method='spearman')
			val_spearman = group['val_acc'].corr(group[metric], method='spearman')
	  
			correlations[f"{ds_name}_{metric}"] = {
				'n': len(group),
				'train_pearson': train_pearson,
				'val_pearson': val_pearson,
				'train_spearman': train_spearman,
				'val_spearman': val_spearman
			}
   
	by_model = data.groupby('model')
	for model_name, group in by_model:
		group = ignore_outliers(group, ignore_percentile=ignore_percentile, ignore_std=ignore_std)
     
		for metric in METRICS:
			train_pearson = group['train_acc'].corr(group[metric])
			val_pearson = group['val_acc'].corr(group[metric])
	  
			train_spearman = group['train_acc'].corr(group[metric], method='spearman')
			val_spearman = group['val_acc'].corr(group[metric], method='spearman')
	  
			correlations[f"{model_name}_{metric}"] = {
				'n': len(group),
				'train_pearson': train_pearson,
				'val_pearson': val_pearson,
				'train_spearman': train_spearman,
				'val_spearman': val_spearman
			}
   
	by_both = data.groupby(['dataset', 'model'])
	for (ds_name, model_name), group in by_both:
		group = ignore_outliers(group, ignore_percentile=ignore_percentile, ignore_std=ignore_std)
     
		for metric in METRICS:
			train_pearson = group['train_acc'].corr(group[metric])
			val_pearson = group['val_acc'].corr(group[metric])
	  
			train_spearman = group['train_acc'].corr(group[metric], method='spearman')
			val_spearman = group['val_acc'].corr(group[metric], method='spearman')
	  
			correlations[f"{ds_name}_{model_name}_{metric}"] = {
				'n': len(group),
    			'train_pearson': train_pearson,
				'val_pearson': val_pearson,
				'train_spearman': train_spearman,
				'val_spearman': val_spearman
			}

	corr_df = pd.DataFrame.from_dict(correlations, orient='index')
 
	if out_csv is not None:
		corr_df.to_csv(out_csv)
		print(f"Correlation results saved to {out_csv}")
 
	return corr_df

def plot_linear_regression(X, y, xlabel: str, ylabel: str, out_path: str) -> None:
	X = X.values.reshape(-1, 1)
	y = y.values.reshape(-1, 1)

	model = LinearRegression()
	model.fit(X, y)
	r2 = model.score(X, y)

	y_pred = model.predict(X)

	plt.figure()
	plt.scatter(X, y, color='blue', label=f'Data points (n = {len(y)})')
	plt.plot(X, y_pred, color='red', label=f'Linear regression (r² = {r2:.3f})')
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.title(f'Linear Regression: {ylabel} vs {xlabel}')
	plt.legend()
	plt.tight_layout()
	plt.savefig(out_path)
	plt.close()

def plot_linear_regressions(data: pd.DataFrame, out_dir: str, ignore_percentile: float = 0.0, ignore_std: float = 0.0) -> None:
	data_full = ignore_outliers(data, ignore_percentile=ignore_percentile, ignore_std=ignore_std)
 
	metric_dirs = {metric: os.path.join(out_dir, metric) for metric in METRICS}
	for metric in METRICS:
		os.makedirs(metric_dirs[metric], exist_ok=True)
 
	for metric in METRICS:
		df_copy = data_full[[metric, 'train_acc', 'val_acc']].dropna()
     
		plot_linear_regression(
			df_copy[metric],
			df_copy['train_acc'],
			xlabel=metric,
			ylabel='Train Accuracy',
			out_path=os.path.join(metric_dirs[metric], f'linear_regression_train_acc_vs_{metric}.png')
		)
  
		plot_linear_regression(
			df_copy[metric],
			df_copy['val_acc'],
			xlabel=metric,
			ylabel='Val Accuracy',
			out_path=os.path.join(metric_dirs[metric], f'linear_regression_val_acc_vs_{metric}.png')
		)
  
	by_dataset = data.groupby('dataset')
	for ds_name, group in by_dataset:
		group = ignore_outliers(group, ignore_percentile=ignore_percentile, ignore_std=ignore_std)
	 
		for metric in METRICS:
			df_copy = group[[metric, 'train_acc', 'val_acc']].dropna()
      
			plot_linear_regression(
				df_copy[metric],
				df_copy['train_acc'],
				xlabel=metric,
				ylabel='Train Accuracy',
				out_path=os.path.join(metric_dirs[metric], f'linear_regression_{ds_name}_train_acc_vs_{metric}.png')
			)
	  
			plot_linear_regression(
				df_copy[metric],
				df_copy['val_acc'],
				xlabel=metric,
				ylabel='Val Accuracy',
				out_path=os.path.join(metric_dirs[metric], f'linear_regression_{ds_name}_val_acc_vs_{metric}.png')
			)
   
	by_model = data.groupby('model')
	for model_name, group in by_model:
		group = ignore_outliers(group, ignore_percentile=ignore_percentile, ignore_std=ignore_std)
	 
		for metric in METRICS:
			df_copy = group[[metric, 'train_acc', 'val_acc']].dropna()
      
			plot_linear_regression(
				df_copy[metric],
				df_copy['train_acc'],
				xlabel=metric,
				ylabel='Train Accuracy',
				out_path=os.path.join(metric_dirs[metric], f'linear_regression_{model_name}_train_acc_vs_{metric}.png')
			)
	  
			plot_linear_regression(
				df_copy[metric],
				df_copy['val_acc'],
				xlabel=metric,
				ylabel='Val Accuracy',
				out_path=os.path.join(metric_dirs[metric], f'linear_regression_{model_name}_val_acc_vs_{metric}.png')
			)
   
	by_both = data.groupby(['dataset', 'model'])
	for (ds_name, model_name), group in by_both:
		group = ignore_outliers(group, ignore_percentile=ignore_percentile, ignore_std=ignore_std)
	 
		for metric in METRICS:
			df_copy = group[[metric, 'train_acc', 'val_acc']].dropna()
      
			plot_linear_regression(
				df_copy[metric],
				df_copy['train_acc'],
				xlabel=metric,
				ylabel='Train Accuracy',
				out_path=os.path.join(metric_dirs[metric], f'linear_regression_{ds_name}_{model_name}_train_acc_vs_{metric}.png')
			)
	  
			plot_linear_regression(
				df_copy[metric],
				df_copy['val_acc'],
				xlabel=metric,
				ylabel='Val Accuracy',
				out_path=os.path.join(metric_dirs[metric], f'linear_regression_{ds_name}_{model_name}_val_acc_vs_{metric}.png')
			)

def exp_hypothesis(x, a, b, c):
	return a * np.exp(b * x) + c

def plot_exp_regression(X, y, xlabel: str, ylabel: str, out_path: str) -> None:
	X = X.values
	y = y.values

	regressor = ExponentialModel()
	res = regressor.fit(y, x=X)
	 
	x_query = np.linspace(min(X), max(X), 100)
	y_query = res.eval(x=x_query)
	r2 = res.rsquared

	plt.figure()
	plt.scatter(X, y, color='blue', label=f'Data points (n = {len(y)})')
	plt.plot(x_query, y_query, color='red', label=f'Exponential fit (r² = {r2:.3f})')
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.title(f'Exponential Fit: {ylabel} vs {xlabel}')
	plt.legend()
	plt.tight_layout()
	plt.savefig(out_path)
	plt.close()

def plot_exp_regressions(data: pd.DataFrame, out_dir: str, ignore_percentile: float = 0.0, ignore_std: float = 0.0) -> None:
	data_full = ignore_outliers(data, ignore_percentile=ignore_percentile, ignore_std=ignore_std)
 
	metric_dirs = {metric: os.path.join(out_dir, metric) for metric in METRICS}
	for metric in METRICS:
		os.makedirs(metric_dirs[metric], exist_ok=True)
 
	for metric in METRICS:
		df_copy = data_full[[metric, 'train_acc', 'val_acc']].dropna()
	 
		plot_exp_regression(
			df_copy[metric],
			df_copy['train_acc'],
			xlabel=metric,
			ylabel='Train Accuracy',
			out_path=os.path.join(metric_dirs[metric], f'exp_regression_train_acc_vs_{metric}.png')
		)
  
		plot_exp_regression(
			df_copy[metric],
			df_copy['val_acc'],
			xlabel=metric,
			ylabel='Val Accuracy',
			out_path=os.path.join(metric_dirs[metric], f'exp_regression_val_acc_vs_{metric}.png')
		)
  
	by_dataset = data.groupby('dataset')
	for ds_name, group in by_dataset:
		group = ignore_outliers(group, ignore_percentile=ignore_percentile, ignore_std=ignore_std)
	 
		for metric in METRICS:
			df_copy = group[[metric, 'train_acc', 'val_acc']].dropna()
	  
			plot_exp_regression(
				df_copy[metric],
				df_copy['train_acc'],
				xlabel=metric,
				ylabel='Train Accuracy',
				out_path=os.path.join(metric_dirs[metric], f'exp_regression_{ds_name}_train_acc_vs_{metric}.png')
			)
	  
			plot_exp_regression(
				df_copy[metric],
				df_copy['val_acc'],
				xlabel=metric,
				ylabel='Val Accuracy',
				out_path=os.path.join(metric_dirs[metric], f'exp_regression_{ds_name}_val_acc_vs_{metric}.png')
			)
    
	by_model = data.groupby('model')
	for model_name, group in by_model:
		group = ignore_outliers(group, ignore_percentile=ignore_percentile, ignore_std=ignore_std)
	 
		for metric in METRICS:
			df_copy = group[[metric, 'train_acc', 'val_acc']].dropna()
	  
			plot_exp_regression(
				df_copy[metric],
				df_copy['train_acc'],
				xlabel=metric,
				ylabel='Train Accuracy',
				out_path=os.path.join(metric_dirs[metric], f'exp_regression_{model_name}_train_acc_vs_{metric}.png')
			)
	  
			plot_exp_regression(
				df_copy[metric],
				df_copy['val_acc'],
				xlabel=metric,
				ylabel='Val Accuracy',
				out_path=os.path.join(metric_dirs[metric], f'exp_regression_{model_name}_val_acc_vs_{metric}.png')
			)
   
	by_both = data.groupby(['dataset', 'model'])
	for (ds_name, model_name), group in by_both:
		group = ignore_outliers(group, ignore_percentile=ignore_percentile, ignore_std=ignore_std)
	 
		for metric in METRICS:
			df_copy = group[[metric, 'train_acc', 'val_acc']].dropna()
	  
			plot_exp_regression(
				df_copy[metric],
				df_copy['train_acc'],
				xlabel=metric,
				ylabel='Train Accuracy',
				out_path=os.path.join(metric_dirs[metric], f'exp_regression_{ds_name}_{model_name}_train_acc_vs_{metric}.png')
			)
	  
			plot_exp_regression(
				df_copy[metric],
				df_copy['val_acc'],
				xlabel=metric,
				ylabel='Val Accuracy',
				out_path=os.path.join(metric_dirs[metric], f'exp_regression_{ds_name}_{model_name}_val_acc_vs_{metric}.png')
			)

def main(datasets_dir: str, data_dir: str, ct_dir: str, output_path: str) -> None:
	datasets = exp.find_all_datasets(datasets_dir)
	experiments = exp.find_all_experiments(datasets, data_dir, ct_dir)

	results = []

	for experiment in experiments:
		print(f"Processing experiment: {experiment}")

		paths = experiment.get_paths(data_dir, ct_dir)
		tree_path = paths["ctree"]

		fns, num_min = get_valley_vs_thresh(tree_path)
  
		if THRESH_SELECTION == "valley=classes":
			wanted_num_valleys = len(experiment.dataset.classes)
		elif THRESH_SELECTION == "valley=2classes":
			wanted_num_valleys = 2 * len(experiment.dataset.classes)
		elif type(THRESH_SELECTION) is float:
			wanted_num_valleys = None
			thresh = THRESH_SELECTION
		else:
			raise ValueError(f"Unknown THRESH_SELECTION: {THRESH_SELECTION}")

		if wanted_num_valleys is not None:
			if not wanted_num_valleys in num_min:
				wanted_num_valleys = max(num_min)
				print(f"Desired number of valleys {wanted_num_valleys} not found. Using maximum available: {wanted_num_valleys}")
			thresh = fns[num_min.index(wanted_num_valleys)]
		tree = vu.rooted_tree_from_exp(experiment, thresh, data_dir, ct_dir)

		metrics = tm.compute_tree_imbalance_metrics(tree)

		accuracy = process_file(paths["compiled_res"], experiment.epoch)

		result = {
			"dataset": experiment.dataset.name,
			"split": experiment.split,
			"model": experiment.model,
			"k": experiment.k,
			"epoch": experiment.epoch,
			"layer": experiment.layer,
			"thresh": thresh,
			"thresh_mode": str(THRESH_SELECTION),
			"node_count": tree.number_of_nodes(),
			**metrics
		}

		result["train_acc"] = accuracy.loc[accuracy['Split'] == 'Train', 'accuracy'].values[0] if 'Train' in accuracy['Split'].values else np.nan
		result["val_acc"] = accuracy.loc[accuracy['Split'] == 'Val', 'accuracy'].values[0] if 'Val' in accuracy['Split'].values else np.nan

		results.append(result)

	results_df = pd.DataFrame(results)
	results_df.to_csv(output_path, index=False)
	print(f"Results saved to {output_path}")

if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser(description="Compute tree imbalance metrics and correlate with accuracy.")
	parser.add_argument("datasets_dir", type=str, help="Path to datasets directory.")
	parser.add_argument("data_dir", type=str, help="Path to data directory.")
	parser.add_argument("ct_dir", type=str, help="Path to contour trees directory.")
	parser.add_argument("output_path", type=str, help="Path to output CSV file.")

	args = parser.parse_args()

	print(f"Using threshold selection mode: {THRESH_SELECTION}, writing to {args.output_path}, continue?")
	input()

	main(args.datasets_dir, args.data_dir, args.ct_dir, args.output_path)