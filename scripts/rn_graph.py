"""
Script to compute the k-nearest neighbors graph from a set of high-dim data points.

Usage:
python rn_graph.py <tensors_file.pt> <output_file>

Saves the graph in networkx adjacency list format to the specified output file.

You probably don't want to run this directly, but rather use the wrapper script compute_knn_complexes.py
"""

import diskannpy as dap
import networkx as nx
import numpy as np
import torch

import shutil

from scipy.sparse import csr_matrix

from sys import argv
import os

def load_packed_vamana(file_path):
    with open(file_path, 'rb') as f:
        index_size = np.fromfile(f, dtype=np.uint64, count=1)[0]
        max_deg = np.fromfile(f, dtype=np.uint32, count=1)[0]
        entry_point = np.fromfile(f, dtype=np.uint32, count=1)[0]
        num_frozen = np.fromfile(f, dtype=np.uint64, count=1)[0]
        
        data = np.fromfile(f, dtype=np.uint32)
    
    indptr = [0]
    indices = []
    curr = 0
    while curr < len(data):
        gk = data[curr]
        indptr.append(indptr[-1] + gk)
        indices.append(data[curr + 1 : curr + 1 + gk])
        curr += 1 + gk
        
    final_indices = np.concatenate(indices)
    final_indptr = np.array(indptr, dtype=np.int32)
    
    n_nodes = len(final_indptr) - 1
    
    adj_matrix = csr_matrix(
        (np.ones(len(final_indices), dtype=np.int8), final_indices, final_indptr),
        shape=(n_nodes, n_nodes)
    )

    return adj_matrix

def compute_rn_graph(data: np.ndarray, complexity: int = 75, graph_degree: int = 60, num_threads: int = 1) -> nx.Graph:
    """
    Computes the approx rngraph (vamana index) for the given data.

    Parameters:
    - data: np.ndarray, shape (n_samples, n_features)
        The input data points.
    - complexity: int
        The complexity parameter for the Vamana index. Higher values lead to better recall but slower query times.
    - graph_degree: int
        The graph degree parameter for the Vamana index. Higher values lead to better recall but slower query times.
    - num_threads: int
        The number of threads to use for building the index.
    """

    shutil.rmtree("tmp", ignore_errors=True)
    os.makedirs("tmp", exist_ok=True)

    dap.build_memory_index(data, index_prefix="tmp", distance_metric="l2", index_directory="tmp", complexity=complexity, graph_degree=graph_degree, num_threads=num_threads) # type: ignore

    adj = load_packed_vamana("tmp/tmp")
    
    return nx.from_scipy_sparse_array(adj)


def main():
    if len(argv) != 3:
        print("Usage: python rn_graph.py <tensors_file.pt> <output_file>")
        return

    data_file = argv[1]
    output_graph_file = argv[2]

    if data_file.endswith('.pt'):
        data = torch.load(data_file).numpy()
    else:
        data = np.loadtxt(data_file)
    print(f"Loaded data from {data_file} with shape {data.shape}")

    G = compute_rn_graph(data)

    os.makedirs(os.path.dirname(output_graph_file), exist_ok=True)

    print("Graph is connected:", nx.is_connected(G))

    nx.write_adjlist(G, output_graph_file)
    print(f"Graph saved to {output_graph_file}")

if __name__ == "__main__":
    main()