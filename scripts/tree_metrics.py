import networkx as nx
import numpy as np

def average_branching_factor(tree: nx.DiGraph) -> float:
	"""Compute the average branching factor of a (reversed) tree.

	The average branching factor is defined as the average number of children
	per internal node (nodes with at least one child).

	Args:
		tree (nx.DiGraph): A directed graph representing the tree.

	Returns:
		float: The average branching factor of the tree. Returns 0.0 if there
			   are no internal nodes.
	"""
	internal_nodes = [n for n in tree.nodes if tree.in_degree(n) > 0]
	if len(internal_nodes) == 0:
		return 0.0

	total_children = sum(tree.in_degree(n) for n in internal_nodes)
	avg_branching_factor = total_children / len(internal_nodes)
	return avg_branching_factor

def colless_index(tree: nx.DiGraph) -> tuple[float, float, int, int]:
	"""Compute the Colless index of a rooted (reversed) binary tree. Augmented for contour trees.

	The Colless index is a measure of tree imbalance. It is defined as the sum
	of the absolute differences in sizes of the left and right subtrees for all
	internal nodes.

	Args:
		tree (nx.DiGraph): A directed graph representing the rooted binary tree.

	Returns:
		float: The Colless index of the tree.
	"""
	colless_sum = 0.0
	colless_nodes = 0
	missing = 0

	rev = tree.reverse()

	for n in rev.nodes:
		if rev.out_degree(n) == 2:  # Internal node with two children
			left_child, right_child = list(rev.successors(n))[:2]
   
			left_tree = nx.algorithms.traversal.dfs_tree(rev, left_child)
			right_tree = nx.algorithms.traversal.dfs_tree(rev, right_child)
   
			left_size = 0
			for edge in left_tree.edges:
				left_size += rev.get_edge_data(*edge)["volume"]
    
			right_size = 0
			for edge in right_tree.edges:
				right_size += rev.get_edge_data(*edge)["volume"]

			colless_sum += abs(left_size - right_size)
			colless_nodes += 1
		elif rev.out_degree(n) > 0:
			missing += 1

	average_colless_sum = colless_sum / colless_nodes if colless_nodes > 0 else np.nan

	return colless_sum, average_colless_sum, colless_nodes, missing

def sackin_index(tree: nx.DiGraph) -> tuple[float, float, int]:
	"""Compute Sackin's index of a rooted (reversed) tree. Augmented for contour trees.

	Sackin's index is defined as the sum of the depths of all leaves in the tree.

	Args:
		tree (nx.DiGraph): A directed graph representing the rooted tree.

	Returns:
		float: Sackin's index of the tree.
		float: Average Sackin's index of the tree.
		int: Number of leaves in the tree.
	"""
	
	rev = tree.reverse()
	root = list(nx.topological_sort(rev))[0]
 
	leaves = [n for n in rev.nodes if rev.out_degree(n) == 0]
	sackin_sum = 0.0
 
	paths = nx.single_source_shortest_path(rev, root)

	for leaf in leaves:
		path = paths[leaf]

		for i in range(len(path) - 1):
			edge_data = rev.get_edge_data(path[i], path[i + 1])
			sackin_sum += edge_data["volume"]

	return sackin_sum, sackin_sum / len(leaves) if len(leaves) > 0 else np.nan, len(leaves)

def compute_tree_imbalance_metrics(tree: nx.DiGraph) -> dict:
	"""Compute various tree imbalance metrics for a given tree.

	Args:
		tree (nx.DiGraph): A directed graph representing the tree.
	Returns:
		dict: A dictionary containing the computed metrics.
	"""
 
	sackin_idx, avg_sackin_idx, leaf_count = sackin_index(tree)
 
	colless_idx, avg_colless_idx, total_colless, missing_colless = colless_index(tree)
	missing_colless_frac = np.nan
 
	if tree.number_of_nodes() - leaf_count > 0:
		missing_colless_frac = missing_colless / (tree.number_of_nodes() - leaf_count)
  
	total_volume = 0
	for edge in tree.edges:
		total_volume += tree.get_edge_data(*edge)["volume"]
  
	metrics = {
		"node_count": tree.number_of_nodes(),
		"average_branching_factor": average_branching_factor(tree),
		"colless_index": colless_idx,
		"average_colless_index": avg_colless_idx,
		"total_colless": total_colless,
		"missing_colless": missing_colless,
		"missing_colless_frac": missing_colless_frac,
		"sackin_index": sackin_idx,
		"average_sackin_index": avg_sackin_idx,
		"leaf_count": leaf_count,
		"total_volume": total_volume
	}
  
	return metrics