"""
Script to compute the k-nearest neighbors graph from a set of high-dim data points.

Usage:
python rn_graph.py <tensors_file.pt> <output_file>

Saves the graph in networkx adjacency list format to the specified output file.

You probably don't want to run this directly, but rather use the wrapper script compute_knn_complexes.py
"""

try:
    import diskannpy as dap
except ImportError:
    dap = None

import networkx as nx
import numpy as np
import torch

import shutil

from scipy.sparse import csr_matrix

from sys import argv
import os

def load_packed_vamana(file_path):
    with open(file_path, 'rb') as f:
        index_size = np.fromfile(f, dtype=np.uint64, count=1)[0]
        max_deg = np.fromfile(f, dtype=np.uint32, count=1)[0]
        entry_point = np.fromfile(f, dtype=np.uint32, count=1)[0]
        num_frozen = np.fromfile(f, dtype=np.uint64, count=1)[0]
        
        data = np.fromfile(f, dtype=np.uint32)
    
    indptr = [0]
    indices = []
    curr = 0
    while curr < len(data):
        gk = data[curr]
        indptr.append(indptr[-1] + gk)
        indices.append(data[curr + 1 : curr + 1 + gk])
        curr += 1 + gk
        
    final_indices = np.concatenate(indices)
    final_indptr = np.array(indptr, dtype=np.int32)
    
    n_nodes = len(final_indptr) - 1
    
    adj_matrix = csr_matrix(
        (np.ones(len(final_indices), dtype=np.int8), final_indices, final_indptr),
        shape=(n_nodes, n_nodes)
    )

    return adj_matrix

RNG_BASEDIR = "tmp_rng/"


def repair_low_degree(data: np.ndarray, work_dir: str, prefix: str, graph: nx.Graph, min_degree: int, num_threads: int = 1) -> None:
    """
    For every node whose graph degree is below min_degree, query the Vamana
    index for additional nearest neighbors and splice the missing edges in-place.
    """

    low_degree = [n for n, d in graph.degree() if d < min_degree]
    if not low_degree:
        return

    print(f"repair_low_degree: {len(low_degree)} nodes below min_degree={min_degree} out of {len(graph)} total nodes")

    idx = dap.StaticMemoryIndex(
        index_directory=work_dir,
        num_threads=num_threads,
        initial_search_complexity=64,
        index_prefix=prefix,
    )

    for i in low_degree:
        neighbors, _ = idx.search(data[i].astype(np.float32), k_neighbors=min_degree + 1, complexity=64)
        # neighbors[0] is the query point itself; skip it
        for j in neighbors:
            j = int(j)
            if j != i and not graph.has_edge(i, j):
                graph.add_edge(i, j)


def compute_rn_graph(data: np.ndarray, min_neighbours=None, metric = 'e', complexity: int = 75, graph_degree: int = 60, num_threads: int = 1, prefix: str = "tmp") -> nx.Graph:
    """
    Computes the approx rngraph (vamana index) for the given data.

    Parameters:
    - data: np.ndarray, shape (n_samples, n_features)
        The input data points.
    - min_neighbours: int | None
        The minimum number of neighbors to use for each sample in the Vamana index. This is a lower bound on the number of neighbors each node will have in the resulting graph.
    - metric: str
        The distance metric to use. Options are 'e' for euclidean or 'c' for cosine.
    - complexity: int
        The complexity parameter for the Vamana index. Higher values lead to better recall but slower query times.
    - graph_degree: int
        The graph degree parameter for the Vamana index. Higher values lead to better recall but slower query times.
    - num_threads: int
        The number of threads to use for building the index.
    - prefix: str
        A prefix for temporary files created during index building. This is useful to avoid conflicts when running multiple instances in parallel.
    """

    if dap is None:
        raise ImportError(
            "diskannpy is required for RNG computation but is not installed. "
            "Install it with: pip install diskannpy"
        )

    work_dir = os.path.join(RNG_BASEDIR, prefix)

    shutil.rmtree(work_dir, ignore_errors=True)
    os.makedirs(work_dir, exist_ok=True)

    metric = 'l2' if metric == 'e' else 'cosine'

    dap.build_memory_index(
        data,
        index_prefix=prefix,
        distance_metric=metric,
        index_directory=work_dir,
        complexity=complexity,
        graph_degree=graph_degree,
        num_threads=num_threads,
    )  # type: ignore

    packed_graph_path = os.path.join(work_dir, prefix)
    adj = load_packed_vamana(packed_graph_path)
    graph = nx.from_scipy_sparse_array(adj)

    if min_neighbours is not None:
        repair_low_degree(data, work_dir, prefix, graph, min_neighbours, num_threads)
    
    return graph


def compute_ann_graph(data: np.ndarray, n_neighbors: int = 5, metric: str = 'e', complexity: int = 75, graph_degree: int = 60, num_threads: int = 1, prefix: str = "tmp") -> nx.Graph:
    """
    Computes an approximate k-NN graph using the DiskANN/Vamana index backend.
    Builds the index once, then batch-queries every point for its k nearest
    neighbors.  Edges are added undirected (union of both directions).

    Parameters mirror compute_rn_graph; n_neighbors replaces min_neighbours.
    """
    if dap is None:
        raise ImportError(
            "diskannpy is required for ANN computation but is not installed. "
            "Install it with: pip install diskannpy"
        )

    work_dir = os.path.join(RNG_BASEDIR, prefix)
    shutil.rmtree(work_dir, ignore_errors=True)
    os.makedirs(work_dir, exist_ok=True)

    dap_metric = 'l2' if metric == 'e' else 'cosine'

    dap.build_memory_index(
        data,
        index_prefix=prefix,
        distance_metric=dap_metric,
        index_directory=work_dir,
        complexity=complexity,
        graph_degree=graph_degree,
        num_threads=num_threads,
    )  # type: ignore

    idx = dap.StaticMemoryIndex(
        index_directory=work_dir,
        num_threads=num_threads,
        initial_search_complexity=complexity,
        index_prefix=prefix,
    )

    # k+1 because the query point itself is included in the results
    ids, _ = idx.batch_search(
        data.astype(np.float32),
        k_neighbors=n_neighbors + 1,
        complexity=complexity,
        num_threads=num_threads,
    )

    n = len(data)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i, neighbors in enumerate(ids):
        for j in neighbors:
            j = int(j)
            if j != i:
                G.add_edge(i, j)

    return G


def compute_ann_disk_graph(data: np.ndarray, n_neighbors: int = 5, metric: str = 'e', complexity: int = 75, graph_degree: int = 60, num_threads: int = 1, prefix: str = "tmp", max_mem: float = 4.0) -> nx.Graph:
    """
    Computes an approximate k-NN graph using the DiskANN disk index backend.
    Suitable for very large datasets that do not fit in memory.
    Both build and search memory budgets are set to max_mem GB.

    Note: DiskANN disk indices do not support cosine metric; only 'e' (l2) is supported.

    Parameters mirror compute_ann_graph, with the addition of:
    - max_mem: float
        Maximum memory budget in GB for both building and searching the disk index.
    """
    if dap is None:
        raise ImportError(
            "diskannpy is required for ANN disk computation but is not installed. "
            "Install it with: pip install diskannpy"
        )
    if metric != 'e':
        raise ValueError(
            "DiskANN disk indices do not support cosine metric. Only 'e' (l2) is supported."
        )

    work_dir = os.path.join(RNG_BASEDIR, prefix)
    shutil.rmtree(work_dir, ignore_errors=True)
    os.makedirs(work_dir, exist_ok=True)

    dap.build_disk_index(
        data.astype(np.float32),
        distance_metric='l2',
        index_directory=work_dir,
        complexity=complexity,
        graph_degree=graph_degree,
        search_memory_maximum=max_mem,
        build_memory_maximum=max_mem,
        num_threads=num_threads,
        index_prefix=prefix,
    )  # type: ignore

    idx = dap.StaticDiskIndex(
        index_directory=work_dir,
        num_threads=num_threads,
        num_nodes_to_cache=0,
        index_prefix=prefix,
    )

    # k+1 because the query point itself is included in the results
    ids, _ = idx.batch_search(
        data.astype(np.float32),
        k_neighbors=n_neighbors + 1,
        complexity=complexity,
        num_threads=num_threads,
    )

    n = len(data)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i, neighbors in enumerate(ids):
        for j in neighbors:
            j = int(j)
            if j != i:
                G.add_edge(i, j)

    return G


def _find_index_prefix(index_dir: str):
    """
    Auto-detect the DiskANN index prefix and type in index_dir.

    Returns (prefix, is_disk) where is_disk is True for disk indices, False for
    memory indices.  Raises FileNotFoundError if no recognisable index is found.

    Detection rules:
      - Any file ending in ``_disk.index``  → disk index; prefix = stem before that suffix.
      - Any file ending in ``_metadata.bin`` → memory index; prefix = stem before that suffix.
    Disk check is attempted first because disk builds also produce a _metadata.bin file.
    """
    for f in os.listdir(index_dir):
        if f.endswith('_disk.index'):
            return f[:-len('_disk.index')], True
    for f in os.listdir(index_dir):
        if f.endswith('_metadata.bin'):
            return f[:-len('_metadata.bin')], False
    raise FileNotFoundError(f"No DiskANN index found in {index_dir!r}")


def query_existing_index(index_dir: str, data: np.ndarray, n_neighbors: int,
                          complexity: int = 75, num_threads: int = 1) -> nx.Graph:
    """
    Query a precomputed DiskANN index (memory or disk) stored in index_dir.

    The index prefix is auto-detected via _find_index_prefix.  Both memory
    (StaticMemoryIndex) and disk (StaticDiskIndex) indices are supported.
    Returns an undirected k-NN graph (edges added as union of both search directions).

    Parameters
    ----------
    index_dir : str
        Directory containing the precomputed DiskANN index files.
    data : np.ndarray, shape (n_samples, n_features)
        Query vectors (the same vectors used to build the index).
    n_neighbors : int
        Number of nearest neighbours to retrieve per point.
    complexity : int
        Search-list size passed to batch_search.
    num_threads : int
        Number of threads for batch_search.
    """
    if dap is None:
        raise ImportError(
            "diskannpy is required for querying an existing index but is not installed. "
            "Install it with: pip install diskannpy"
        )

    prefix, is_disk = _find_index_prefix(index_dir)

    if is_disk:
        idx = dap.StaticDiskIndex(
            index_directory=index_dir,
            num_threads=num_threads,
            num_nodes_to_cache=0,
            index_prefix=prefix,
        )
    else:
        idx = dap.StaticMemoryIndex(
            index_directory=index_dir,
            num_threads=num_threads,
            initial_search_complexity=complexity,
            index_prefix=prefix,
        )

    # k+1 because the query point itself is almost always returned
    ids, _ = idx.batch_search(
        data.astype(np.float32),
        k_neighbors=n_neighbors + 1,
        complexity=complexity,
        num_threads=num_threads,
    )

    n = len(data)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i, neighbors in enumerate(ids):
        for j in neighbors:
            j = int(j)
            if j != i:
                G.add_edge(i, j)

    return G


def main():
    if len(argv) != 3:
        print("Usage: python rn_graph.py <tensors_file.pt> <output_file>")
        return

    data_file = argv[1]
    output_graph_file = argv[2]

    if data_file.endswith('.pt'):
        data = torch.load(data_file).numpy()
    else:
        data = np.loadtxt(data_file)
    print(f"Loaded data from {data_file} with shape {data.shape}")

    G = compute_rn_graph(data)

    os.makedirs(os.path.dirname(output_graph_file), exist_ok=True)

    print("Graph is connected:", nx.is_connected(G))

    nx.write_adjlist(G, output_graph_file)
    print(f"Graph saved to {output_graph_file}")

if __name__ == "__main__":
    main()