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

from multiprocessing import get_logger

import torch

from utils import get_adjlist_from_graph

def compute_contour_tree(G: nx.Graph, scalar_function: np.ndarray, tree_type: ct.TreeType = ct.TreeType.TypeContourTree) -> ct.MergeTree:
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

    tree.computeTree(gsf, tree_type)

    return tree

tree_type_name = {
    ct.TreeType.TypeContourTree: "contour_tree",
    ct.TreeType.TypeSplitTree: "split_tree",
    ct.TreeType.TypeJoinTree: "join_tree"
}

def compute_and_save_contour_tree(adjlist_file: str, scalar_fn_file: str, output_directory: str, tree_type: ct.TreeType = ct.TreeType.TypeContourTree, sim_type="pers", worker_id: int = 0):
    """
    Compute and save contour tree for a single graph-scalar pair.
    
    This function writes directly to disk since ct.ContourTree objects are not serializable.
    """
    log = get_logger()

    os.makedirs(output_directory, exist_ok=True)

    # Load the graph
    G = nx.read_adjlist(adjlist_file, nodetype=int)
    log.info(f"{worker_id}: Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges from {adjlist_file}")

    filename = os.path.basename(adjlist_file)
    name, _ = os.path.splitext(filename)
    name = name.replace("adj_", "ctree_").replace("_connected", "")

    # Load the scalar function
    scalar_function = np.loadtxt(scalar_fn_file) if scalar_fn_file.endswith(".txt") else torch.load(scalar_fn_file)
    log.info(f"{worker_id}: Loaded scalar function with {len(scalar_function)} values from {scalar_fn_file}")

    if len(scalar_function) != G.number_of_nodes():
        log.error(f"{worker_id}: Scalar function length {len(scalar_function)} does not match number of nodes {G.number_of_nodes()}. Skipping.")
        return

    # Compute the contour tree
    contour_tree = compute_contour_tree(G, scalar_function, tree_type)
    log.info(f"{worker_id}: Computed {tree_type_name[tree_type]}")

    outfile = os.path.join(output_directory, f"{name}")
    contour_tree.output(outfile, tree_type)
    
    log.info(f"{worker_id}: Saved {tree_type_name[tree_type]} to {output_directory}")
    log.info(f"{worker_id}: Computing hierarchical simplification")

    ctdata = ct.ContourTreeData()
    
    try:
        ctdata.loadBinFile(outfile)

        sim = ct.SimplifyCT()
        sim.setInput(ctdata)

        if sim_type == "pers":
            sim_fn = ct.Persistence(ctdata)
        elif sim_type == "vol":
            part_file = os.path.join(output_directory, f"{name}.part.raw")
            sim_fn = ct.Volume(ctdata, part_file)
            print(f"{worker_id}: Using volume simplification with partition file {part_file}")
        else:
            raise ValueError(f"Unknown simplification type: {sim_type}")

        sim.simplify(sim_fn)
        sim.outputOrder(outfile, False)
        log.info(f"{worker_id}: Saved simplification order to {output_directory}")
    except Exception as e:
        log.error(f"{worker_id}: Error during simplification: {e}")

def process_contour_trees(worker_id: int, tasks: list, tree_type: ct.TreeType, sim_type: str = "pers"):
    """
    Worker function to process a batch of contour tree computations.
    
    Parameters:
    - worker_id: int
        Worker identifier for logging
    - tasks: list of tuples (adjlist_file, scalar_fn_file, output_directory)
        List of tasks to process
    - tree_type: ct.TreeType
        The type of tree to compute
    """
    log = get_logger()
    log.info(f"{worker_id}: Starting processing of {len(tasks)} tasks")
    
    for adjlist_file, scalar_fn_file, output_directory in tasks:
        try:
            compute_and_save_contour_tree(adjlist_file, scalar_fn_file, output_directory, tree_type, sim_type, worker_id)
        except Exception as e:
            log.error(f"{worker_id}: Error processing {adjlist_file} with {scalar_fn_file}: {e}")
    
    log.info(f"{worker_id}: Completed all tasks")


def main():
    if len(argv) != 5:
        print("Usage: python compute_ctree.py <adjlist_file.txt> <scalar_function_file.txt> <output_directory> <type: c | s | j (contour, split, or join)>")
        return
    
    adjlist_file = argv[1]
    scalar_fn_file = argv[2]
    output_directory = argv[3]
    ct_type = argv[4].strip().lower()
    
    if ct_type not in ["c", "s", "j"]:
        print("Error: ct_type must be one of 'c', 's', or 'j'")
        return
    
    tree_type = {
        "c": ct.TreeType.TypeContourTree,
        "s": ct.TreeType.TypeSplitTree,
        "j": ct.TreeType.TypeJoinTree
    }
    tree_type = tree_type[ct_type]

    compute_and_save_contour_tree(adjlist_file, scalar_fn_file, output_directory, tree_type, worker_id=0)  


if __name__ == "__main__":
    main()