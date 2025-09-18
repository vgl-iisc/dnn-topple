"""
Contains utility functions for various tasks.
"""

import networkx as nx
import numpy as np

def load_graph_from_adjlist_file(file_path: str):
    """
    Loads a graph from a NetworkX adjacency list file.

    Parameters:
    - file_path: str
        Path to the adjacency list file.

    Returns:
    - G: nx.Graph
        The loaded graph.
    """

    G = nx.read_adjlist(file_path)
    return G

def get_adjlist_from_graph(G: nx.Graph):
    """
    Converts a NetworkX graph to an adjacency list format suitable for GraphScalarFunction.

    Parameters:
    - G: nx.Graph
        The input graph.

    Returns:
    - adjlist: list of lists
        The adjacency list representation of the graph.
    """

    adjlist = [[] for _ in range(G.number_of_nodes())]

    for node, nbr_dict in G.adjacency():
        adj = list(map(int, nbr_dict.keys()))
        adjlist[node] = adj

    return adjlist

def load_simplification_thresholds_from_tree(tree_files: str) -> list[float]:
    with open(f"{tree_files}.order.dat", "r") as f:
        no_simpl = int(f.readline().strip())

    thresh = [float(f) for f in np.fromfile(f"{tree_files}.order.bin", dtype=np.float32, offset=4*no_simpl, count=no_simpl)]

    # add no simplification (0.0) and deduplicate
    thresh_set = set([0.0] + thresh)
    thresh = sorted(list(thresh_set))

    return thresh