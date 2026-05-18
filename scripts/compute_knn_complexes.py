"""
Goes through all landscapes in the data directory and computes
(minimally connected) k-NN graphs for each of them, saving them appropriately.
"""

import torch
from knn_graph import compute_knn_graph
from rn_graph import compute_rn_graph

import os
import re

import argparse
import numpy as np

import networkx as nx

from multiprocessing import Pool, cpu_count, log_to_stderr, get_logger
from timeit import default_timer as timer

import pickle
import logging

# Matches chunk files produced by run_inference.py: a{tag}_e{epoch}_chunk{n}.pt
CHUNK_RE = re.compile(r'^(a.+_e\d+)_chunk(\d+)\.pt$')

def save_name_txt(file, k, connected):
    return f"adj_{file[len('vectors_'):-4]}_{k}" + ("_connected" if connected else "")

def save_name_pt(file, k, connected):
    return f"adj_{os.path.splitext(file)[0]}_{k}" + ("_connected" if connected else "")

def _available_memory_bytes():
    try:
        import psutil
        return psutil.virtual_memory().available
    except ImportError:
        return None

def _group_chunks(files):
    """Split files into chunk groups (base -> sorted filenames) and non-chunk files."""
    chunk_groups = {}
    non_chunk = []
    for f in files:
        m = CHUNK_RE.match(f)
        if m:
            chunk_groups.setdefault(m.group(1), []).append(f)
        else:
            non_chunk.append(f)
    for base in chunk_groups:
        chunk_groups[base].sort(key=lambda f: int(CHUNK_RE.match(f).group(2)))

    logging.info(f"Grouped {len(files)} files into {len(chunk_groups)} chunk groups and {len(non_chunk)} non-chunk files")
    return chunk_groups, non_chunk

def process_files(id, data_dir, complexes_dir, root, files, max_k, exact, method, metric="e", cpus_per_worker=1):
        times = {}
        log = get_logger()

        chunk_groups, non_chunk_files = _group_chunks(files)

        # Build a unified work list: (tensor_file, save_name_fn, data_loader)
        # data_loader is a zero-arg callable that returns a numpy array, or None if skipped.
        work = []

        for tensor_file in non_chunk_files:
            tensor_path = os.path.join(root, tensor_file)
            if tensor_path.endswith('.pt'):
                work.append((tensor_file, save_name_pt, lambda p=tensor_path: torch.load(p, weights_only=True).numpy()))
            else:
                work.append((tensor_file, save_name_txt, lambda p=tensor_path: np.loadtxt(p)))

        for base, chunk_files in chunk_groups.items():
            tensor_file = f"{base}.pt"
            chunk_paths = [os.path.join(root, f) for f in chunk_files]
            total_bytes = sum(os.path.getsize(p) for p in chunk_paths)
            avail = _available_memory_bytes()
            if avail is not None and total_bytes > avail * 0.8:
                log.info(
                    f"{id}: Skipping {tensor_file} (chunked): "
                    f"total size {total_bytes/1e9:.2f} GB exceeds 80% of available memory {avail/1e9:.2f} GB"
                )
                continue
            def _load_chunks(paths=chunk_paths, t_file=tensor_file, bytes_=total_bytes):
                log.info(f"{id}: Stitching {len(paths)} chunks for {t_file} (~{bytes_/1e9:.2f} GB)")
                chunks = [torch.load(p, weights_only=True) for p in paths]
                data = torch.cat(chunks).numpy()
                del chunks
                return data
            work.append((tensor_file, save_name_pt, _load_chunks))

        if not complexes_dir.endswith("/") and data_dir.endswith("/"):
            complexes_dir += '/'

        for tensor_file, save_name_fn, load_data in work:
            tensor_path = os.path.join(root, tensor_file)

            save_basepath = root.replace(data_dir, complexes_dir).replace(f"Tensors{os.sep}", f"")
            os.makedirs(save_basepath, exist_ok=True)

            possible_save_names = [save_name_fn(tensor_file, max_k, b) for b in [True, False]]
            save_paths = [os.path.join(save_basepath, f"{name}.txt") for name in possible_save_names]
            if any(os.path.exists(path) for path in save_paths):
                log.info(f"{id}: Found existing files for {tensor_path} with k={max_k}, skipping")
                continue

            data = load_data()
            log.info(f"{id}: Loaded data from {tensor_path} with shape {data.shape}")

            if data.shape not in times:
                times[data.shape] = []

            if method == 'r':
                start = timer()
                Gmax = compute_rn_graph(data, min_neighbours=max_k, metric=metric, complexity=75, graph_degree=60, num_threads=cpus_per_worker, prefix=f"worker_{id}")
            else:
                start = timer()
                Gmax = compute_knn_graph(data, n_neighbors=max_k, metric=metric, cpus=cpus_per_worker)
            
            end = timer()
            times[data.shape].append(end - start)
            
            if not nx.is_connected(Gmax):
                log.info(f"{id}: Warning: max_k={max_k} does not yield a connected graph for {tensor_path}, skipping")
                continue

            if exact:
                name = save_name_fn(tensor_file, max_k, True)
                nx.write_adjlist(Gmax, os.path.join(save_basepath, f"{name}.txt"))
                log.info(f"{id}: Done with {tensor_path}, exact k={max_k} -> {name}")
                continue

            l = 1
            r = max_k

            while l < r:
                mid = l + (r - l) // 2
                G = compute_knn_graph(data, n_neighbors=mid, metric=metric, cpus=cpus_per_worker)

                name = save_name_fn(tensor_file, mid, False)

                if nx.is_connected(G):
                    r = mid
                    name += "_connected"
                else:
                    l = mid + 1

                nx.write_adjlist(G, os.path.join(save_basepath, f"{name}.txt"))

            k = l
            log.info(f"{id}: Done with {tensor_path}, min connected k={k}")
        
        return times

def main():
    default_cpus = int(cpu_count() * 3/4)

    parser = argparse.ArgumentParser(description="Compute (minimally connected) k-NN graphs for all landscapes in a data directory.")
    parser.add_argument("data_dir", help="Root directory containing landscape tensors")
    parser.add_argument("complexes_dir", help="Output directory for adjacency lists")
    parser.add_argument("max_k", type=int, help="Maximum k for k-NN graph construction")
    parser.add_argument("method", choices=["r", "k"], help="Graph method: 'r' for RNG, 'k' for k-NN")
    parser.add_argument("--ignore-splits", dest="ignore_splits", default="", metavar="SPLITS",
                        help="Comma-separated list of split names to skip (use '.' to mean none)")
    parser.add_argument("--ignore-tags", dest="ignore_tags", default="", metavar="TAGS",
                        help="Comma-separated list of filename tags to exclude (use '.' to mean none)")
    parser.add_argument("--cpus", type=int, default=default_cpus,
                        help=f"Total CPU threads to use (default: {default_cpus})")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: same as --cpus)")

    args = parser.parse_args()

    ignore_splits = [s for s in args.ignore_splits.split(",") if s and s != "."]
    ignore_tags = [t for t in args.ignore_tags.split(",") if t and t != "."]

    cpus = args.cpus
    workers = args.workers if args.workers is not None else cpus
    cpus_per_worker = max(1, cpus // workers)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(processName)s - %(levelname)s: %(message)s')

    log_to_stderr(logging.INFO)

    logging.info(f"Starting k-NN complex computation with ignore_splits={ignore_splits} and ignore_tags={ignore_tags}")

    data_dir = args.data_dir
    complexes_dir = args.complexes_dir
    max_k = args.max_k
    method = args.method
    
    exact = True
    metric = "e"
    times_dict = {}

    logging.info(f"Starting k-NN complex computation in {data_dir}, saving to {complexes_dir}, max_k={max_k}, method={method}, CPUS={cpus}, cpus_per_worker={cpus_per_worker}")

    for root, dirs, files in os.walk(data_dir):
        logging.info(f"Processing directory: {root}")
        if not "Tensors" in root:
            continue
        
        skip = False
        for split in ignore_splits:
            root_nice = root.replace("\\", "/")
            if split in root_nice.split("/"):
                skip = True
                logging.info(f"Skipping {root} due to ignore_splits")
                break
        
        if skip:
            continue

        tensor_files = [f for f in files if f.startswith("vectors_") and f.endswith(".txt")]
        tensor_files_2 = [f for f in files if f.endswith(".pt") and all(tag not in f for tag in ignore_tags)]

        if len(tensor_files_2) > 0:
            tensor_files = tensor_files_2

        logging.info(f"Found {len(tensor_files)} tensor files in {root}")

        if len(tensor_files) == 0:
            continue

        groups = []

        N_groups = max(1, workers)
        for i in range(N_groups):
            groups.append((i, data_dir, complexes_dir, root, tensor_files[i::N_groups], max_k, exact, method, metric, cpus_per_worker))

        logging.info(f"Starting {root}: {len(tensor_files)} files, {len(groups)} groups: {list(map(lambda x: len(x[4]), groups))}")

        processes = Pool(N_groups)
        all_times = processes.starmap(process_files, groups)
        processes.close()
        processes.join()

        for times in all_times:
            for k, v in times.items():
                if k not in times_dict:
                    times_dict[k] = []
                times_dict[k].extend(v)
        
        logging.info(f"Done with {root}")
    
    name = "rn_times.pkl" if method == 'r' else "knn_times.pkl"
    with open(name, "wb") as f:
        pickle.dump(times_dict, f)
                
if __name__ == "__main__":
    main()