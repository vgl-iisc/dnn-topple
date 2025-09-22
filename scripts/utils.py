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

def load_order_and_wts(tree_files: str) -> tuple[list[int], list[float]]:
    with open(f"{tree_files}.order.dat", "r") as f:
        no_simpl = int(f.readline().strip())
        
    with open(f"{tree_files}.order.bin", "rb") as file:
        order = [int(f) for f in np.fromfile(file, dtype=np.uint32, count=no_simpl)]
        wts = [float(f) for f in np.fromfile(file, dtype=np.float32, count=no_simpl)]

    return order, wts