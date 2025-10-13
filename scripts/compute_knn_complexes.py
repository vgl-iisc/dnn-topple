"""
Goes through all landscapes in the data directory and computes
(minimally connected) k-NN graphs for each of them, saving them appropriately.
"""

from knn_graph import compute_knn_graph

import os

from sys import argv
import numpy as np

import networkx as nx

def main():
    if len(argv) != 4 and len(argv) != 5:
        print("Usage: python compute_knn_complexes.py <data_dir> <complexes_dir> <max_k> [exact_k]")
        return
    
    data_dir = argv[1]
    complexes_dir = argv[2]
    max_k = int(argv[3])
    
    exact = len(argv) == 5

    for root, dirs, files in os.walk(data_dir):
        if not "Tensors" in root:
            continue

        tensor_files = [f for f in files if f.startswith("vectors_") and f.endswith(".txt")]

        for tensor_file in tensor_files:
            tensor_path = os.path.join(root, tensor_file)
            data = np.loadtxt(tensor_path)
            print(f"Loaded data from {tensor_path} with shape {data.shape}")

            # do this as a binary search for the smallest k that gives a connected graph

            l = 1
            r = max_k

            save_basepath = root.replace(data_dir, complexes_dir).replace(f"Tensors{os.sep}", f"")
            os.makedirs(save_basepath, exist_ok=True)

            Gmax = compute_knn_graph(data, n_neighbors=max_k)
            
            if not nx.is_connected(Gmax):
                print(f"Warning: max_k={max_k} does not yield a connected graph for {tensor_path}, skipping")
                continue

            if exact:
                name = f"adj_{tensor_file[len('vectors_'):-4]}_{max_k}_connected"
                nx.write_adjlist(Gmax, os.path.join(save_basepath, f"{name}.txt"))
                print(f"Done with {tensor_path}, exact k={max_k}")
                continue

            while l < r:
                mid = l + (r - l) // 2
                G = compute_knn_graph(data, n_neighbors=mid)

                name = f"adj_{tensor_file[len("vectors_"):-4]}_{mid}"

                if nx.is_connected(G):
                    r = mid
                    name += "_connected"
                else:
                    l = mid + 1

                nx.write_adjlist(G, os.path.join(save_basepath, f"{name}.txt"))

            k = l
            print(f"Done with {tensor_path}, min connected k={k}")
                
if __name__ == "__main__":
    main()