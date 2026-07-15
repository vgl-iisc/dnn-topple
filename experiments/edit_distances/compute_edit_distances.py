"""
Compute pairwise merge-tree edit distances for all trees in a directory.

Usage
-----
    python compute_edit_distances.py <tree_dir> <output_path>
        [--simpl FLOAT]
        [--method {branch,path,constrained}]
        [--pattern REGEX]
        [--workers INT]

Output
------
A ``.npz`` file containing:
  distance_matrix  – (N, N) float64 symmetric distance matrix
  names            – (N,)   array of tree-stem strings (row/column labels)
  method           – scalar string recording which distance was used

Example
-------
    python compute_edit_distances.py \\
        "/media/santripta/Santripta 1/strees_cross_epoch_60/resnet_cifar/trainUval" \\
        resnet_cifar_trainUval_branch.npz \\
        --method branch --workers 8
"""

import sys
import os
import argparse
import numpy as np
import networkx as nx
from multiprocessing import Pool

# ---------------------------------------------------------------------------
# Path setup: find repo root so we can import from both experiments/ and deps/
# ---------------------------------------------------------------------------
_HERE      = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
_EDIST_ROOT = os.path.join(_REPO_ROOT, 'deps', 'edit-dists')

sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _EDIST_ROOT)

from load_mergetrees import load_mergetree_dir, find_root, parse_stem
from mergetree.mergetree import computeBranchDecomposition
from mted.baseMetrics import cost_wasserstein_branch_squared
from mted.branch_mapping_dist import branchMappingDistance
from mted.path_mapping_dist import pathMappingDistance
from mted.constrained_edist_mt import editDistance_constrained


# ---------------------------------------------------------------------------
# Tree preparation
# ---------------------------------------------------------------------------

def prepare_tree(g: nx.Graph) -> tuple[nx.DiGraph, int]:
    """Apply branch decomposition and return a rooted DFS tree.

    Parameters
    ----------
    g : nx.Graph
        Undirected merge-tree graph as returned by
        :func:`load_mergetrees.pyct_to_mergetree_nx`.

    Returns
    -------
    (rooted_tree, root_id) : (nx.DiGraph, int)
        ``rooted_tree`` is the DFS-directed version with node attributes
        copied from *g*.  ``root_id`` is the root node index.
    """
    root_id = find_root(g)
    computeBranchDecomposition(g, root_id)
    rt = nx.dfs_tree(g, root_id)
    # copy node attributes (scalar, type, birth, death, …) into directed tree
    rt.add_nodes_from((n, g.nodes[n]) for n in rt.nodes)
    return rt, root_id


# ---------------------------------------------------------------------------
# Pair-wise distance worker (top-level so it is picklable for multiprocessing)
# ---------------------------------------------------------------------------

def _compute_pair(args: tuple) -> tuple[int, int, float]:
    i, j, rt1, root1, rt2, root2, method = args
    cost = cost_wasserstein_branch_squared
    if method == 'branch':
        d = branchMappingDistance(rt1, root1, rt2, root2, cost, sqrt=True)
    elif method == 'path':
        d = pathMappingDistance(rt1, root1, rt2, root2)
    elif method == 'constrained':
        d = editDistance_constrained(rt1, root1, rt2, root2, cost, sqrt=True)
    else:
        raise ValueError(f'Unknown method: {method!r}')
    return i, j, float(d)


# ---------------------------------------------------------------------------
# Distance matrix
# ---------------------------------------------------------------------------

def compute_distance_matrix(
    trees: dict[str, nx.Graph],
    method: str = 'branch',
    n_workers: int = 1,
) -> tuple[np.ndarray, list[str]]:
    """Compute a symmetric pairwise distance matrix.

    Parameters
    ----------
    trees : dict[str, nx.Graph]
        Mapping from tree name to graph (as returned by
        :func:`load_mergetrees.load_mergetree_dir`).
    method : str
        One of ``'branch'``, ``'path'``, ``'constrained'``.
    n_workers : int
        Number of parallel worker processes.  Use ``1`` for serial execution.

    Returns
    -------
    (D, names) : (np.ndarray, list[str])
        ``D`` is an ``(N, N)`` symmetric distance matrix.
        ``names`` is the ordered list of tree stems corresponding to rows/cols.
    """
    names = sorted(trees.keys())
    N = len(names)

    print(f'Preparing {N} trees (branch decomposition)…')
    prepared = {name: prepare_tree(trees[name]) for name in names}

    pairs = [
        (i, j, *prepared[names[i]], *prepared[names[j]], method)
        for i in range(N)
        for j in range(i + 1, N)
    ]

    D = np.zeros((N, N), dtype=np.float64)

    if n_workers > 1:
        with Pool(n_workers) as pool:
            for k, (i, j, d) in enumerate(pool.imap_unordered(_compute_pair, pairs)):
                D[i, j] = D[j, i] = d
                if (k + 1) % 50 == 0 or k + 1 == len(pairs):
                    print(f'  {k + 1}/{len(pairs)} pairs done')
    else:
        for k, args in enumerate(pairs):
            i, j, d = _compute_pair(args)
            D[i, j] = D[j, i] = d
            if (k + 1) % 50 == 0 or k + 1 == len(pairs):
                print(f'  {k + 1}/{len(pairs)} pairs done')

    return D, names


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Compute pairwise merge-tree edit distances for all trees in a directory.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        'tree_dir',
        help='Directory containing merge tree files (*.order.dat)',
    )
    parser.add_argument(
        'output_path',
        help='Output .npz file path',
    )
    parser.add_argument(
        '--simpl', type=float, default=0.0,
        help='Persistence simplification threshold',
    )
    parser.add_argument(
        '--method', choices=['branch', 'path', 'constrained'], default='branch',
        help='Edit-distance method',
    )
    parser.add_argument(
        '--pattern', default=None,
        help='Regex filter applied to tree stem names (optional)',
    )
    parser.add_argument(
        '--workers', type=int, default=1,
        help='Number of parallel worker processes',
    )
    args = parser.parse_args()

    print(f'Loading merge trees from: {args.tree_dir}  (simpl={args.simpl})')
    trees = load_mergetree_dir(args.tree_dir, simpl=args.simpl, pattern=args.pattern)
    if not trees:
        print('No merge trees found — check the directory path.', file=sys.stderr)
        sys.exit(1)
    print(f'Loaded {len(trees)} trees.')

    n_pairs = len(trees) * (len(trees) - 1) // 2
    print(f'Computing {args.method} distances ({n_pairs} pairs, {args.workers} workers)…')
    D, names = compute_distance_matrix(trees, method=args.method, n_workers=args.workers)

    # Parse structured metadata from stem names where possible
    metas  = [parse_stem(n) for n in names]
    layers = np.array([m.layer if m else '' for m in metas])
    epochs = np.array([m.epoch if m else -1 for m in metas], dtype=np.int64)
    ks     = np.array([m.k     if m else -1 for m in metas], dtype=np.int64)

    np.savez(
        args.output_path,
        distance_matrix=D,
        names=np.array(names),
        layers=layers,
        epochs=epochs,
        ks=ks,
        method=np.str_(args.method),
    )
    print(f'Saved → {args.output_path}')


if __name__ == '__main__':
    main()
