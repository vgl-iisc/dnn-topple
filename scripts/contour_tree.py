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
    
    gsf.loadGraphFromAdjList(get_adjlist_from_graph(G))
    gsf.updateFnValues(list(scalar_function))

    # Compute the contour tree
    tree = ct.MergeTree()
    tree_type = ct.TreeType.TypeContourTree

    tree.computeTree(gsf, tree_type)

    return tree

def compute_and_save_contour_tree(adjlist_file: str, scalar_fn_file: str, output_directory: str):

    os.makedirs(output_directory, exist_ok=True)

    # Load the graph
    G = nx.read_adjlist(adjlist_file, nodetype=int)
    print(f"Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    filename = os.path.basename(adjlist_file)
    name, _ = os.path.splitext(filename)
    name = name.replace("adj_", "ctree_").replace("_connected", "")

    # Load the scalar function
    scalar_function = np.loadtxt(scalar_fn_file)
    print(f"Loaded scalar function with {len(scalar_function)} values.")

    if len(scalar_function) != G.number_of_nodes():
        raise ValueError("The length of the scalar function must match the number of nodes in the graph.")

    # Compute the contour tree
    contour_tree = compute_contour_tree(G, scalar_function)
    print("Computed contour tree.")

    outfile = os.path.join(output_directory, f"{name}")
    contour_tree.output(outfile, ct.TreeType.TypeContourTree)
    
    print(f"Saved contour tree to {output_directory}.")
    print(f"Computing hierarchical simplification.")

    ctdata = ct.ContourTreeData()
    
    try:
        ctdata.loadBinFile(outfile)

        sim = ct.SimplifyCT()
        sim.setInput(ctdata)

        sim_fn = ct.Persistence(ctdata)

        sim.simplify(sim_fn)
        sim.outputOrder(outfile, False)
    except Exception as e:
        print(f"Error during simplification: {e}")

    print(f"Saved simplification order to {output_directory}.")

def main():
    if len(argv) != 4:
        print("Usage: python compute_ctree.py <adjlist_file.txt> <scalar_function_file.txt> <output_directory>")
        return
    
    adjlist_file = argv[1]
    scalar_fn_file = argv[2]
    output_directory = argv[3]
    
    compute_and_save_contour_tree(adjlist_file, scalar_fn_file, output_directory)    
    

if __name__ == "__main__":
    input()
    main()