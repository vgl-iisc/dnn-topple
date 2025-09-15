"""
Script to compute the k-nearest neighbors graph from a set of high-dim data points.

Usage:
python knn_graph.py <tensors_file.txt> <n_neighbours> <output_file>

Saves the graph in networkx adjacency list format to the specified output file.

You probably don't want to run this directly, but rather use the wrapper script compute_knn_complexes.py
"""

from sklearn.neighbors import kneighbors_graph
import networkx as nx
import numpy as np

from sys import argv
import os

def compute_knn_graph(data: np.ndarray, n_neighbors=5) -> nx.Graph:
    """
    Computes the k-nearest neighbors graph for the given data.

    Parameters:
    - data: np.ndarray, shape (n_samples, n_features)
        The input data points.
    - n_neighbors: int
        The number of neighbors to use for each sample.
    - mode: str
        The mode to use for the graph. Options are 'connectivity' or 'distance'.
    """

    adj = kneighbors_graph(data, n_neighbors=n_neighbors, mode="connectivity", include_self=False)

    G = nx.from_scipy_sparse_array(adj)

    return G


def main():
    if len(argv) != 4:
        print("Usage: python knn_graph.py <tensors_file.txt> <n_neighbours> <output_file>")
        return

    data_file = argv[1]
    n_neighbors = int(argv[2])
    output_graph_file = argv[3]

    data = np.loadtxt(data_file)
    print(f"Loaded data from {data_file} with shape {data.shape}")

    G = compute_knn_graph(data, n_neighbors=n_neighbors)

    os.makedirs(os.path.dirname(output_graph_file), exist_ok=True)

    print("Graph is connected:", nx.is_connected(G))

    nx.write_adjlist(G, output_graph_file)
    print(f"Graph saved to {output_graph_file}")

if __name__ == "__main__":
    main()