"""
Load pyct merge trees from a directory and convert them to NetworkX graphs
using the node/edge field names expected by deps/edit-dists/mergetree/mergetree.py.

Node attributes
---------------
  scalar   : float  – function value at the critical point
  type     : int    – critical-point type (pyct constant cast to int:
                       MINIMUM=0, SADDLE=1, MAXIMUM=3 in TTK convention)
  birth    : float  – initialised to 0.0; populated by computeBranchDecomposition
  death    : float  – initialised to 0.0; populated by computeBranchDecomposition
  position : tuple  – (0.0, 0.0, 0.0) placeholder (no spatial data in pyct)

Edge attributes
---------------
  persistence : float – |scalar(to) - scalar(frm)|
  regionSize  : int   – arc membership count (0 when no partition is loaded)

Nodes are re-indexed to 0..N-1 to satisfy the contiguous-index assumption of
the edit-distance algorithms in deps/edit-dists/mted/.

Typical usage
-------------
    from load_mergetrees import load_mergetree_dir, find_root
    from mergetree.mergetree import computeBranchDecomposition

    trees = load_mergetree_dir('/path/to/strees_dir', simpl=0.0)
    for name, g in trees.items():
        root = find_root(g)
        computeBranchDecomposition(g, root)
        # g is now ready for edit-distance computation
"""

import os
import re
from typing import NamedTuple

import networkx as nx
import pyct as ct


_ORDER_DAT_SUFFIX = '.order.dat'

# Matches stems like: ctree_a1_e0_60  or  ctree_alayer4_e12_15
_STEM_RE = re.compile(r'^ctree_a(.+)_e(\d+)_(\d+)$')


class TreeMeta(NamedTuple):
    layer: str   # tag after 'a', e.g. '1', 'layer4'
    epoch: int
    k: int


def parse_stem(stem: str) -> 'TreeMeta | None':
    """Parse a merge-tree stem name into its (layer, epoch, k) components.

    Parameters
    ----------
    stem : str
        Stem name as returned by :func:`load_mergetree_dir`, e.g.
        ``'ctree_a1_e0_60'``.

    Returns
    -------
    TreeMeta or None
        A named tuple ``(layer, epoch, k)`` on success, or ``None`` if the
        stem does not match the expected pattern.

    Examples
    --------
    >>> parse_stem('ctree_a1_e0_60')
    TreeMeta(layer='1', epoch=0, k=60)
    >>> parse_stem('ctree_a2_e149_60')
    TreeMeta(layer='2', epoch=149, k=60)
    """
    m = _STEM_RE.match(stem)
    if m is None:
        return None
    return TreeMeta(layer=m.group(1), epoch=int(m.group(2)), k=int(m.group(3)))


def pyct_to_mergetree_nx(tree_path: str, simpl: float = 0.0) -> nx.Graph:
    """Convert a single pyct merge tree to a NetworkX graph.

    Parameters
    ----------
    tree_path : str
        Base path of the merge tree **without extension**, e.g.
        ``'/data/strees/ctree_a1_e0_60'``.
        The files ``<tree_path>.order.dat``, ``.order.bin``, ``.rg.bin``,
        and ``.rg.dat`` must exist alongside it.
    simpl : float
        Persistence simplification threshold forwarded to
        ``topo.getArcFeatures``.  Default ``0.0`` applies no simplification.

    Returns
    -------
    nx.Graph
        Undirected graph with nodes labelled ``0 .. N-1``.

    Notes
    -----
    ``birth`` and ``death`` are initialised to ``0.0``.  Call
    ``mergetree.computeBranchDecomposition(g, root_id)`` before passing the
    graph to any edit-distance function that uses those fields.
    """
    topo = ct.TopologicalFeatures()
    topo.loadData(tree_path)
    features = topo.getArcFeatures(-1, simpl)

    # Collect unique nodes: original pyct ID -> (scalar, type)
    node_data: dict[int, tuple[float, int]] = {}
    for f in features:
        node_data[f.frm] = (float(f.fn_frm), int(f.type_frm))
        node_data[f.to]  = (float(f.fn_to),  int(f.type_to))

    # Remap to contiguous 0-indexed integers required by the edit-distance code
    orig_ids = sorted(node_data.keys())
    id_map   = {orig: new for new, orig in enumerate(orig_ids)}

    g = nx.Graph()
    for orig_id, (scalar, ntype) in node_data.items():
        g.add_node(
            id_map[orig_id],
            scalar=scalar,
            type=ntype,
            birth=0.0,
            death=0.0,
            position=(0.0, 0.0, 0.0),
        )

    for f in features:
        u = id_map[f.frm]
        v = id_map[f.to]
        g.add_edge(
            u, v,
            persistence=float(f.pers),
            regionSize=int(getattr(f, 'size', 0)),
        )

    return g


def find_root(g: nx.Graph) -> int:
    """Return the node id of the join-tree root (global maximum by scalar).

    For a split tree loaded via pyct (where arcs always go from lower to higher
    scalar), the root is the node with the highest scalar value.
    """
    return max(g.nodes, key=lambda n: g.nodes[n]['scalar'])


def load_mergetree_dir(
    directory: str,
    simpl: float = 0.0,
    pattern: str | None = None,
) -> dict[str, nx.Graph]:
    """Load all merge trees in *directory* and return them as a dict.

    Parameters
    ----------
    directory : str
        Directory to scan.  Every file ending in ``.order.dat`` is treated as
        a merge tree stem.
    simpl : float
        Persistence simplification threshold forwarded to
        :func:`pyct_to_mergetree_nx`.
    pattern : str or None
        Optional regex applied to each stem name; only matching stems are
        loaded.  Pass ``None`` (default) to load everything.

    Returns
    -------
    dict[str, nx.Graph]
        Keys are stem names (e.g. ``'ctree_a1_e0_60'``), values are the
        corresponding NetworkX graphs.

    Example
    -------
    >>> trees = load_mergetree_dir(
    ...     '/media/santripta/Santripta 1/strees_cross_epoch_60/resnet_cifar/trainUval',
    ...     simpl=0.0,
    ... )
    >>> print(list(trees)[:3])
    ['ctree_a1_e0_60', 'ctree_a1_e100_60', 'ctree_a1_e101_60']
    """
    compiled = re.compile(pattern) if pattern else None
    trees: dict[str, nx.Graph] = {}

    for fname in sorted(os.listdir(directory)):
        if not fname.endswith(_ORDER_DAT_SUFFIX):
            continue
        stem = fname[: -len(_ORDER_DAT_SUFFIX)]
        if compiled and not compiled.search(stem):
            continue
        tree_path = os.path.join(directory, stem)
        trees[stem] = pyct_to_mergetree_nx(tree_path, simpl=simpl)

    return trees
