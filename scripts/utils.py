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

    adjlist = []

    for _, nbr_dict in G.adjacency():
        adj = []
        
        adj.append(list(nbr_dict.keys()))
        adjlist.append(adj)

    return adjlist