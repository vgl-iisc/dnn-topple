"""
Goes through all landscapes in the data directory and computes
(minimally connected) k-NN graphs for each of them, saving them appropriately.

With --transformer, handles 3D BERT/transformer activation tensors
(N_sent, max_words, hidden): each valid word position (where a paired mask
file's value is True) is treated as an independent vector for the k-NN
computation, and a token-provenance file is written alongside each adjacency
list so downstream analysis can map k-NN node indices back to
(sentence_index, word_index) coordinates.

Directory conventions (standard mode):
  Tensors/{split}/vectors_*.txt   or   *.pt   - flat activation tensors

Directory conventions (--transformer mode):
  Tensors/{split}/a{tag}_e{epoch}.pt   - activation tensor (N_sent, max_words, hidden)
  Tensors/{split}/mask_e{epoch}.pt     - word mask (N_sent, max_words) bool

Output (--transformer):
  {split}/adj_a{tag}_e{epoch}_{k}_connected.txt
  {split}/token_coords_a{tag}_e{epoch}.pt  — LongTensor (N_valid, 2): sentence/word indices
"""

import torch
from knn_graph import compute_knn_graph
from rn_graph import compute_rn_graph, compute_ann_graph, compute_ann_disk_graph, query_existing_index

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

# ---------------------------------------------------------------------------
# Transformer path – helpers
# ---------------------------------------------------------------------------

# Matches activation files: alayer11_e0.pt, aclassifier_e3.pt, etc.
_ACT_RE = re.compile(r'^a(.+)_e(\d+)\.pt$')


def _mask_filename(tag: str, epoch: str) -> str:
    return f'mask_e{epoch}.pt'


def _transformer_save_name(act_file: str, k: int, connected: bool) -> str:
    base = os.path.splitext(act_file)[0]
    suffix = '_connected' if connected else ''
    return f'adj_{base}_{k}{suffix}'


def _provenance_name(act_file: str) -> str:
    base = os.path.splitext(act_file)[0]
    return f'token_coords_{base}.pt'


# ---------------------------------------------------------------------------
# Worker: standard path
# ---------------------------------------------------------------------------

def process_files(id, data_dir, complexes_dir, root, files, max_k, exact, method, metric="e", cpus_per_worker=1, fill_rng=False, disk=False, max_mem=None, existing_index=None, index_store_dir=None):
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

            task_index_dir = None
            if existing_index is not None:
                task_name = os.path.splitext(tensor_file)[0]
                task_index_dir = os.path.join(existing_index, task_name)
                if not os.path.isdir(task_index_dir):
                    log.info(f"{id}: No index directory at {task_index_dir}, skipping {tensor_file}")
                    continue

            data = load_data()
            log.info(f"{id}: Loaded data from {tensor_path} with shape {data.shape}")

            if data.shape not in times:
                times[data.shape] = []

            task_name = os.path.splitext(tensor_file)[0]

            if method == 'r':
                min_k_rng = max_k if fill_rng else None
                start = timer()
                Gmax = compute_rn_graph(data, min_neighbours=min_k_rng, metric=metric, complexity=75, graph_degree=60, num_threads=cpus_per_worker, prefix=f"worker_{id}", index_store_dir=index_store_dir, task_name=task_name)
            elif method == 'ann':
                start = timer()
                if task_index_dir is not None:
                    Gmax = query_existing_index(task_index_dir, data, max_k, complexity=75, num_threads=cpus_per_worker)
                elif disk:
                    assert max_mem is not None
                    Gmax = compute_ann_disk_graph(data, n_neighbors=max_k, metric=metric, complexity=75, graph_degree=60, num_threads=cpus_per_worker, prefix=f"worker_{id}", max_mem=max_mem, index_store_dir=index_store_dir, task_name=task_name)
                else:
                    Gmax = compute_ann_graph(data, n_neighbors=max_k, metric=metric, complexity=75, graph_degree=60, num_threads=cpus_per_worker, prefix=f"worker_{id}", index_store_dir=index_store_dir, task_name=task_name)
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


# ---------------------------------------------------------------------------
# Worker: transformer path
# ---------------------------------------------------------------------------

def process_files_transformer(worker_id, data_dir, complexes_dir, root, act_files, max_k, method, metric, cpus_per_worker=1, fill_rng=False, disk=False, max_mem=None, existing_index=None, index_store_dir=None):
    """Worker: load activations, flatten with mask, run k-NN, save graph + provenance."""
    times = {}
    log = get_logger()

    for act_file in act_files:
        m = _ACT_RE.match(act_file)
        if m is None:
            log.warning(f'{worker_id}: Skipping unexpected filename {act_file!r}')
            continue
        tag, epoch_str = m.group(1), m.group(2)

        act_path  = os.path.join(root, act_file)
        mask_path = os.path.join(root, _mask_filename(tag, epoch_str))

        if not os.path.exists(mask_path):
            log.warning(f'{worker_id}: No mask file for {act_file!r} (expected {mask_path}), skipping')
            continue

        save_basepath = root.replace(data_dir, complexes_dir).replace(f'Tensors{os.sep}', '')
        os.makedirs(save_basepath, exist_ok=True)

        adj_names = [_transformer_save_name(act_file, max_k, b) for b in (True, False)]
        adj_paths = [os.path.join(save_basepath, f'{n}.txt') for n in adj_names]
        prov_path = os.path.join(save_basepath, _provenance_name(act_file))

        if any(os.path.exists(p) for p in adj_paths):
            log.info(f'{worker_id}: Output exists for {act_file}, k={max_k} - skipping')
            continue

        task_index_dir = None
        if existing_index is not None:
            task_name = os.path.splitext(act_file)[0]
            task_index_dir = os.path.join(existing_index, task_name)
            if not os.path.isdir(task_index_dir):
                log.info(f'{worker_id}: No index directory at {task_index_dir}, skipping {act_file}')
                continue

        acts = torch.load(act_path, weights_only=True)   # (N_sent, max_words, hidden)
        mask = torch.load(mask_path, weights_only=True)  # (N_sent, max_words) bool

        if acts.ndim != 3 or mask.ndim != 2:
            log.warning(f'{worker_id}: Unexpected tensor shapes for {act_file}: '
                        f'acts={tuple(acts.shape)}, mask={tuple(mask.shape)} - skipping')
            continue

        N_sent, max_words, hidden = acts.shape
        assert mask.shape == (N_sent, max_words), (
            f'Shape mismatch: acts {acts.shape}, mask {mask.shape}')

        acts_flat = acts[mask].float().numpy()          # (N_valid, hidden)
        N_valid   = acts_flat.shape[0]

        sent_idx, word_idx = torch.where(mask)
        coords = torch.stack([sent_idx, word_idx], dim=1)  # (N_valid, 2) int64

        log.info(f'{worker_id}: {act_file} - {N_sent} sents, {N_valid} valid tokens, hidden={hidden}')

        if acts_flat.shape not in times:
            times[acts_flat.shape] = []

        torch.save(coords, prov_path)

        task_name = os.path.splitext(act_file)[0]

        if method == 'r':
            start = timer()
            min_k_rng = max_k if fill_rng else None
            G = compute_rn_graph(acts_flat, min_neighbours=min_k_rng, metric=metric,
                                 complexity=75, graph_degree=60,
                                 num_threads=cpus_per_worker, prefix=f'worker_{worker_id}',
                                 index_store_dir=index_store_dir, task_name=task_name)
            elapsed = timer() - start
            effective_k = max_k
        elif method == 'ann':
            start = timer()
            if task_index_dir is not None:
                G = query_existing_index(task_index_dir, acts_flat, max_k, complexity=75, num_threads=cpus_per_worker)
            elif disk:
                assert max_mem is not None
                G = compute_ann_disk_graph(acts_flat, n_neighbors=max_k, metric=metric,
                                          complexity=75, graph_degree=60,
                                          num_threads=cpus_per_worker, prefix=f'worker_{worker_id}',
                                          max_mem=max_mem, index_store_dir=index_store_dir, task_name=task_name)
            else:
                G = compute_ann_graph(acts_flat, n_neighbors=max_k, metric=metric,
                                      complexity=75, graph_degree=60,
                                      num_threads=cpus_per_worker, prefix=f'worker_{worker_id}',
                                      index_store_dir=index_store_dir, task_name=task_name)
            elapsed = timer() - start
            effective_k = max_k
        else:
            effective_k = max_k
            if N_valid <= max_k:
                log.warning(f'{worker_id}: N_valid={N_valid} <= max_k={max_k} for {act_file}, '
                            f'clamping k to {N_valid - 1}')
                effective_k = N_valid - 1

            start = timer()
            G = compute_knn_graph(acts_flat, n_neighbors=effective_k, metric=metric, cpus=cpus_per_worker)
            elapsed = timer() - start

        times[acts_flat.shape].append(elapsed)

        connected = nx.is_connected(G)
        if not connected:
            log.info(f'{worker_id}: k={effective_k} graph NOT connected for {act_file}')

        name = _transformer_save_name(act_file, effective_k, connected)
        nx.write_adjlist(G, os.path.join(save_basepath, f'{name}.txt'))
        log.info(f'{worker_id}: Saved {act_file} \u2192 {name}.txt  ({elapsed:.1f}s, connected={connected})')

    return times


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    default_cpus = int(cpu_count() * 3/4)

    parser = argparse.ArgumentParser(
        description=(
            "Compute (minimally connected) k-NN graphs for all landscapes in a data directory. "
            "With --transformer, handles 3D BERT/transformer activation tensors and writes "
            "token-provenance files alongside each adjacency list."
        )
    )
    parser.add_argument("data_dir", help="Root directory containing landscape tensors")
    parser.add_argument("complexes_dir", help="Output directory for adjacency lists")
    parser.add_argument("max_k", type=int, help="Maximum k for k-NN graph construction")
    parser.add_argument("method", choices=["r", "k", "ann"], help="Graph method: 'r' for RNG, 'k' for k-NN, 'ann' for approximate k-NN via DiskANN")
    parser.add_argument("--transformer", action="store_true",
                        help="Use the transformer/BERT path: expects 3D activation tensors with paired mask files")
    parser.add_argument("--metric", default="e", choices=["e", "c"],
                        help="Distance metric: 'e' euclidean, 'c' cosine (default: e)")
    parser.add_argument("--ignore-splits", dest="ignore_splits", default="", metavar="SPLITS",
                        help="Comma-separated list of split names to skip (use '.' to mean none)")
    parser.add_argument("--ignore-tags", dest="ignore_tags", default="", metavar="TAGS",
                        help="Comma-separated list of filename tags to exclude (use '.' to mean none)")
    parser.add_argument("--cpus", type=int, default=default_cpus,
                        help=f"Total CPU threads to use (default: {default_cpus})")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of parallel workers (default: same as --cpus)")
    parser.add_argument("--fill_rng", action="store_true", default=False, 
                        help="Fill in the RN graph to ensure minimum degree of max_k")
    parser.add_argument("--disk", action="store_true", default=False,
                        help="Use DiskANN disk indices instead of memory indices (only valid with method 'ann')")
    parser.add_argument("--max_mem", type=float, default=None, metavar="GB",
                        help="Memory budget in GB for DiskANN disk index build and search (required with --disk)")
    parser.add_argument("--existing_index", default=None, metavar="DIR",
                        help="Directory of precomputed DiskANN indices (only valid with method 'ann'). "
                             "For each task, looks for a subdirectory named after the task stem, "
                             "e.g. <DIR>/ablock10_attn_e0/. Files with no matching subdirectory are skipped.")
    parser.add_argument("--index_store_dir", default=None, metavar="DIR",
                        help="If provided, built DiskANN indices are stored persistently under this "
                             "directory instead of the default temporary location (tmp_rng/). "
                             "Each index is saved in a subdirectory named <task_stem>_<worker_prefix>, "
                             "e.g. <DIR>/ablock10_attn_e0_worker_0/. Existing subdirectories are "
                             "reused rather than overwritten. Only valid with methods 'r', 'ann'.")

    args = parser.parse_args()

    if args.disk:
        if args.method != 'ann':
            parser.error("--disk can only be used with method 'ann'")
        if args.max_mem is None:
            parser.error("--disk requires --max_mem <GB>")
    if args.existing_index is not None and args.method != 'ann':
        parser.error("--existing_index can only be used with method 'ann'")
    if args.index_store_dir is not None and args.method == 'k':
        parser.error("--index_store_dir is only valid with methods 'r' and 'ann' (not 'k')")

    ignore_splits = [s for s in args.ignore_splits.split(",") if s and s != "."]
    ignore_tags   = [t for t in args.ignore_tags.split(",")   if t and t != "."]

    cpus = args.cpus
    workers = args.workers if args.workers is not None else cpus
    cpus_per_worker = max(1, cpus // workers)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(processName)s - %(levelname)s: %(message)s')
    log_to_stderr(logging.INFO)

    data_dir      = args.data_dir
    complexes_dir = args.complexes_dir
    max_k         = args.max_k
    method        = args.method
    metric        = args.metric
    fill_rng      = args.fill_rng
    disk          = args.disk
    max_mem       = args.max_mem
    existing_index = args.existing_index
    index_store_dir = args.index_store_dir
    mode = "transformer" if args.transformer else "standard"
    logging.info(
        f"Starting k-NN complex computation: mode={mode} data_dir={data_dir} "
        f"complexes_dir={complexes_dir} max_k={max_k} method={method} metric={metric} "
        f"ignore_splits={ignore_splits} ignore_tags={ignore_tags} "
        f"cpus={cpus} workers={workers} disk={disk} max_mem={max_mem} existing_index={existing_index}"
    )

    exact = True
    times_dict = {}

    for root, dirs, files in os.walk(data_dir):
        logging.info(f"Processing directory: {root}")
        if "Tensors" not in root:
            continue

        root_parts = root.replace("\\", "/").split("/")
        if any(s in root_parts for s in ignore_splits):
            logging.info(f"Skipping {root} (ignore_splits)")
            continue

        N_groups = max(1, workers)

        if args.transformer:
            act_files = [
                f for f in files
                if _ACT_RE.match(f) and all(tag not in f for tag in ignore_tags)
            ]
            logging.info(f"Found {len(act_files)} activation files in {root}")
            if not act_files:
                continue

            groups = [
                (i, data_dir, complexes_dir, root,
                 act_files[i::N_groups], max_k, method, metric, cpus_per_worker, fill_rng, disk, max_mem, existing_index, index_store_dir)
                for i in range(N_groups)
                if act_files[i::N_groups]
            ]
            worker_fn = process_files_transformer

        else:
            tensor_files = [f for f in files if f.startswith("vectors_") and f.endswith(".txt")]
            tensor_files_a = [f for f in files if f.endswith(".pt")]
            tensor_files_filtered = [f for f in tensor_files_a if all(tag not in f for tag in ignore_tags) or "143" in f]
            if tensor_files_filtered:
                tensor_files = tensor_files_filtered

            logging.info(f"Found {len(tensor_files)} tensor files in {root} (total_tensor_files={len(tensor_files_a)})")
            if not tensor_files:
                continue

            groups = [
                (i, data_dir, complexes_dir, root,
                 tensor_files[i::N_groups], max_k, exact, method, metric, cpus_per_worker, fill_rng, disk, max_mem, existing_index, index_store_dir)
                for i in range(N_groups)
            ]
            worker_fn = process_files

        logging.info(
            f"Starting {root}: {len(groups)} groups: {[len(g[4]) for g in groups]}"
        )

        pool = Pool(len(groups))
        all_times = pool.starmap(worker_fn, groups)
        pool.close()
        pool.join()

        for times in all_times:
            for k, v in times.items():
                times_dict.setdefault(k, []).extend(v)

        logging.info(f"Done with {root}")

    if args.transformer:
        prefix = "rn" if method == 'r' else ("ann" if method == 'ann' else "knn")
        name = f"{prefix}_times_transformer_{max_k}_{metric}.pkl"
    elif method == 'r':
        name = "rn_times.pkl"
    elif method == 'ann':
        name = "ann_times.pkl"
    else:
        name = "knn_times.pkl"

    with open(name, "wb") as f:
        pickle.dump(times_dict, f)

    logging.info(f"Finished. Timing saved to {name}")
                
if __name__ == "__main__":
    main()