"""
Computes a contour tree given a graph and scalar function on its nodes.

Usage:
python compute_ctree.py <adjlist_file.txt> <scalar_function_file.txt> <output_directory>
"""

import networkx as nx
import numpy as np

import pyct as ct

from sys import argv
import os

import pickle

from utils import get_adjlist_from_graph

def compute_contour_tree(G: nx.Graph, scalar_function: np.ndarray):
    """
    Computes the contour tree for the given graph and scalar function.

    Parameters:
    - G: nx.Graph
        The input graph.
    - scalar_function: np.ndarray, shape (n_nodes,)
        The scalar function defined on the nodes of the graph.
    """

    # Create a GraphScalarFunction object
    gsf = ct.GraphScalarFunction()
    gsf.initialize(G.number_of_nodes())

    gsf.loadGraphFromAdjList(get_adjlist_from_graph(G))
    gsf.updateFnValues(list(scalar_function))

    # Compute the contour tree
    tree = ct.MergeTree()
    tree_type = ct.TreeType.TypeContourTree

    tree.computeTree(gsf, tree_type)

    return tree

def main():
    if len(argv) != 4:
        print("Usage: python compute_ctree.py <adjlist_file.txt> <scalar_function_file.txt> <output_directory>")
        return
    
    

if __name__ == "__main__":
    main()