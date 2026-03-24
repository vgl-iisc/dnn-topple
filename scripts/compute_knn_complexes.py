"""
Goes through all landscapes in the data directory and computes
(minimally connected) k-NN graphs for each of them, saving them appropriately.
"""

import torch
from knn_graph import compute_knn_graph
# from rn_graph import compute_rn_graph

import os

from sys import argv
import numpy as np

import networkx as nx

from multiprocessing import Pool, cpu_count, log_to_stderr, get_logger
from timeit import default_timer as timer

import pickle
import logging

CPUS = cpu_count() - 6

def save_name_txt(file, k, connected):
    return f"adj_{file[len('vectors_'):-4]}_{k}" + ("_connected" if connected else "")

def save_name_pt(file, k, connected):
    return f"adj_{os.path.splitext(file)[0]}_{k}" + ("_connected" if connected else "")

def process_files(id, data_dir, complexes_dir, root, files, max_k, exact, method):
        times = {}
        log = get_logger()

        for tensor_file in files:
            tensor_path = os.path.join(root, tensor_file)
            
            save_name_fn = save_name_txt
            if tensor_path.endswith('.pt'):
                data = torch.load(tensor_path).numpy()
                save_name_fn = save_name_pt
            else:
                data = np.loadtxt(tensor_path)

            log.info(f"{id}: Loaded data from {tensor_path} with shape {data.shape}")
            
            if data.shape not in times:
                times[data.shape] = []

            if not complexes_dir.endswith("/") and data_dir.endswith("/"):
                complexes_dir += '/'

            save_basepath = root.replace(data_dir, complexes_dir).replace(f"Tensors{os.sep}", f"")
            os.makedirs(save_basepath, exist_ok=True)
            
            possible_save_names = [save_name_fn(tensor_file, max_k, b) for b in [True, False]]
            save_paths = [os.path.join(save_basepath, f"{name}.txt") for name in possible_save_names]
            if any(os.path.exists(path) for path in save_paths):
                log.info(f"{id}: Found existing files for {tensor_path} with k={max_k}, skipping")
                continue

            # if method == 'r':
            if method == 'r':
                start = timer()
                raise NotImplementedError()
                # try:
                #     Gmax = compute_rn_graph(data, complexity=75, graph_degree=60, num_threads=1, prefix=str(id))
                # except Exception as e:
                #     log.error(f"{id}: Error computing RN graph for {tensor_path}: {e}")
                #     continue
            else:
                start = timer()
                Gmax = compute_knn_graph(data, n_neighbors=max_k)
            
            end = timer()
            times[data.shape].append(end - start)
            
            if not nx.is_connected(Gmax):
                log.info(f"{id}: Warning: max_k={max_k} does not yield a connected graph for {tensor_path}, skipping")
                continue

            if exact:
                name = save_name_fn(tensor_file, max_k, True)
                nx.write_adjlist(Gmax, os.path.join(save_basepath, f"{name}.txt"))
                log.info(f"{id}: Done with {tensor_path}, exact k={max_k} -> {name}")
                continue

            l = 1
            r = max_k

            while l < r:
                mid = l + (r - l) // 2
                G = compute_knn_graph(data, n_neighbors=mid)

                name = save_name_fn(tensor_file, mid, False)

                if nx.is_connected(G):
                    r = mid
                    name += "_connected"
                else:
                    l = mid + 1

                nx.write_adjlist(G, os.path.join(save_basepath, f"{name}.txt"))

            k = l
            log.info(f"{id}: Done with {tensor_path}, min connected k={k}")
        
        return times

def main():
    if len(argv) < 5:
        print("Usage: python compute_knn_complexes.py <data_dir> <complexes_dir> <max_k> <r|k> (rng vs knn) [ignore_splits]")
        return
    
    ignore_splits = []
    if len(argv) >= 6:
        ignore_splits = argv[5].split(",")
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(processName)s - %(levelname)s: %(message)s')

    log_to_stderr(logging.INFO)

    logging.info(f"Starting k-NN complex computation with ignore_splits={ignore_splits}")

    data_dir = argv[1]
    complexes_dir = argv[2]
    max_k = int(argv[3])
    method = argv[4]
    
    exact = True
    times_dict = {}

    logging.info(f"Starting k-NN complex computation in {data_dir}, saving to {complexes_dir}, max_k={max_k}, method={method}")

    for root, dirs, files in os.walk(data_dir):
        logging.info(f"Processing directory: {root}")
        if not "Tensors" in root:
            continue
        
        skip = False
        for split in ignore_splits:
            root_nice = root.replace("\\", "/")
            if split in root_nice.split("/"):
                skip = True
                logging.info(f"Skipping {root} due to ignore_splits")
                break
        
        if skip:
            continue

        tensor_files = [f for f in files if f.startswith("vectors_") and f.endswith(".txt")]
        tensor_files_2 = [f for f in files if f.endswith(".pt")]

        if len(tensor_files_2) > 0:
            tensor_files = tensor_files_2

        logging.info(f"Found {len(tensor_files)} tensor files in {root}")

        if len(tensor_files) == 0:
            continue

        groups = []

        N_groups = max(1, CPUS)
        for i in range(N_groups):
            groups.append((i, data_dir, complexes_dir, root, tensor_files[i::N_groups], max_k, True, method))

        logging.info(f"Starting {root}: {len(tensor_files)} files, {len(groups)} groups: {list(map(lambda x: len(x[4]), groups))}")

        processes = Pool(N_groups)
        all_times = processes.starmap(process_files, groups)
        processes.close()
        processes.join()

        for times in all_times:
            for k, v in times.items():
                if k not in times_dict:
                    times_dict[k] = []
                times_dict[k].extend(v)
        
        logging.info(f"Done with {root}")
    
    name = "rn_times.pkl" if method == 'r' else "knn_times.pkl"
    with open(name, "wb") as f:
        pickle.dump(times_dict, f)
                
if __name__ == "__main__":
    main()