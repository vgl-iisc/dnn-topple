from itertools import combinations
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

def colless_index(tree: nx.DiGraph) -> tuple[float, int]:
	"""Compute the Colless index of a rooted (reversed) binary tree.

	The Colless index is a measure of tree imbalance. It is defined as the sum
	of the absolute differences in sizes of the left and right subtrees for all
	internal nodes.

	Args:
		tree (nx.DiGraph): A directed graph representing the rooted binary tree.

	Returns:
		float: The Colless index of the tree.
	"""
	colless_sum = 0.0
	missing = 0

	rev = tree.reverse()

	for n in rev.nodes:
		if rev.out_degree(n) == 2:  # Internal node with two children
			left_child, right_child = list(rev.successors(n))[:2]

			left_size = nx.algorithms.traversal.dfs_tree(rev, left_child).number_of_nodes()
			right_size = nx.algorithms.traversal.dfs_tree(rev, right_child).number_of_nodes()
			colless_sum += abs(left_size - right_size)
		elif rev.out_degree(n) > 0:
			print(f"Node {n} is not binary; skipping in Colless index calculation.")
			missing += 1

	return colless_sum, missing

def total_cophenetic_index(tree: nx.DiGraph) -> float:
	"""Compute the total cophenetic index of a rooted (reversed) tree.

	The total cophenetic index is defined as the sum of the depths of the
	least common ancestors (LCAs) for all pairs of leaves in the tree.

	Args:
		tree (nx.DiGraph): A directed graph representing the rooted tree.

	Returns:
		float: The total cophenetic index of the tree.
	"""
	tree = tree.reverse()
 
	root = list(nx.topological_sort(tree))[0]
	leaves = [n for n in tree.nodes if tree.out_degree(n) == 0]
	cophenetic_sum = 0.0
 
	pairs = combinations(leaves, 2)

	lcas = map(lambda p: p[1], nx.all_pairs_lowest_common_ancestor(tree, pairs))
	lengths = nx.single_source_shortest_path_length(tree, root)
 
	for lca in lcas:
		if lca not in lengths:
			print(f"LCA {lca} not found in lengths; skipping.")
			return np.nan
		cophenetic_sum += lengths[lca]

	return cophenetic_sum

def sackin_index(tree: nx.DiGraph) -> float:
	"""Compute Sackin's index of a rooted (reversed) tree.

	Sackin's index is defined as the sum of the depths of all leaves in the tree.

	Args:
		tree (nx.DiGraph): A directed graph representing the rooted tree.

	Returns:
		float: Sackin's index of the tree.
	"""
	
	rev = tree.reverse()
	root = list(nx.topological_sort(rev))[0]
 
	leaves = [n for n in rev.nodes if rev.out_degree(n) == 0]
	sackin_sum = 0.0
 
	lengths = nx.single_source_shortest_path_length(rev, root)

	for leaf in leaves:
		if leaf not in lengths:
			print(f"Leaf {leaf} not found in lengths; skipping.")
			return np.nan
		sackin_sum += lengths[leaf]

	return sackin_sum

def compute_tree_imbalance_metrics(tree: nx.DiGraph) -> dict:
	"""Compute various tree imbalance metrics for a given tree.

	Args:
		tree (nx.DiGraph): A directed graph representing the tree.
	Returns:
		dict: A dictionary containing the computed metrics.
	"""
 
	colless_idx, missing_colless = colless_index(tree)
 
	if missing_colless > 3:
		print(f"too many non-binary nodes ({missing_colless}); skipping colless.")
     
		return {
			"average_branching_factor": average_branching_factor(tree),
			"colless_index": np.nan,
			"missing_colless": missing_colless,
			"total_cophenetic_index": total_cophenetic_index(tree),
			"sackin_index": sackin_index(tree)
		}
 
	metrics = {
		"average_branching_factor": average_branching_factor(tree),
		"colless_index": colless_idx,
		"missing_colless": missing_colless,
		"total_cophenetic_index": total_cophenetic_index(tree),
		"sackin_index": sackin_index(tree)
	}
	return metrics