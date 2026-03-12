from subprocess import run
from utils import *

LLVIS_PYTHON = "C:/Users/Santripta/miniconda3/envs/llvis/python.exe"
DISKANNPY_PYTHON = "C:/Users/Santripta/miniconda3/envs/diskannpy/python.exe"

data_dir = f"D:/CrossEpochLatents/"
knn_path = lambda k: f"D:/knns_cross_epoch_{k}"
stree_path = lambda k: f"D:/strees_cross_epoch_{k}"
balance_path = lambda k: f"experiment_data/balance_metrics/cross_epoch_k{k}_1e-6.csv"

Ks = [15, 20, 25, 30, 35, 40, 45, 60, 80]

for k in Ks:
	print(f"Computing k-NN complexes for k={k}...")
	run([DISKANNPY_PYTHON, "scripts/compute_knn_complexes.py", data_dir, knn_path(k), str(k), "k", "train,val"], check=True)
	
	print(f"Computing split trees for k={k}...")
	run([LLVIS_PYTHON, "scripts/compute_contour_trees.py", data_dir, knn_path(k), stree_path(k), "s"], check=True)
 
	print(f"Computing balance metrics for k={k}...")
	run([LLVIS_PYTHON, "experiments/balance_metrics/balance_metrics.py", "datasets/", data_dir, stree_path(k), balance_path(k)], check=True)
 
	data = get_data(balance_path(k))
	compute_correlations(data, balance_path(k).replace(".csv", "_corr.csv"))