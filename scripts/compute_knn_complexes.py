"""
Goes through all landscapes in the data directory and computes
(minimally connected) k-NN graphs for each of them, saving them appropriately.
"""

import torch
from knn_graph import compute_knn_graph
# from rng_graph import compute_rng_graph

import os

from sys import argv
import numpy as np

import networkx as nx

from multiprocessing import Pool, cpu_count, log_to_stderr, get_logger

import logging

def save_name_txt(file, k, connected):
    return f"adj_{file[len("vectors_"):-4]}_{k}" + ("_connected" if connected else "")

def save_name_pt(file, k, connected):
    return f"adj_{os.path.splitext(file)[0]}_{k}" + ("_connected" if connected else "")

def process_files(id, data_dir, complexes_dir, root, files, max_k, exact, method):
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

            if not complexes_dir.endswith("/") and data_dir.endswith("/"):
                complexes_dir += '/"'

            save_basepath = root.replace(data_dir, complexes_dir).replace(f"Tensors{os.sep}", f"")
            os.makedirs(save_basepath, exist_ok=True)

            # if method == 'r':
            if False:
                pass
            else:
                Gmax = compute_knn_graph(data, n_neighbors=max_k)
            
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

def main():
    if len(argv) != 5:
        print("Usage: python compute_knn_complexes.py <data_dir> <complexes_dir> <max_k> <r|k> (rng vs knn)")
        return
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(processName)s - %(levelname)s: %(message)s')

    log_to_stderr(logging.INFO)

    data_dir = argv[1]
    complexes_dir = argv[2]
    max_k = int(argv[3])
    method = argv[4]
    
    exact = True

    for root, dirs, files in os.walk(data_dir):
        if not "Tensors" in root:
            continue

        tensor_files = [f for f in files if f.startswith("vectors_") and f.endswith(".txt")]
        tensor_files_2 = [f for f in files if f.endswith(".pt")]

        if len(tensor_files_2) > 0:
            tensor_files = tensor_files_2

        if len(tensor_files) == 0:
            continue

        groups = []

        N_groups = max(1, (cpu_count() // 2) - 1)
        for i in range(N_groups):
            groups.append((i, data_dir, complexes_dir, root, tensor_files[i::N_groups], max_k, True, method))

        logging.info(f"Starting {root}: {len(tensor_files)} files, {len(groups)} groups: {list(map(lambda x: len(x[4]), groups))}")

        processes = Pool(N_groups)
        processes.starmap(process_files, groups)
        processes.close()
        processes.join()

        logging.info(f"Done with {root}")
                
if __name__ == "__main__":
    main()