"""
Alternate compute_knn_complexes for the BERT NER training-inference pipeline.

Activation tensors from that pipeline are 3D: (N_sent, max_words, 768).
Each valid word position (where mask[i, j] is True) is treated as an
independent 768-dimensional vector for the k-NN computation.

Alongside every adjacency list the script also writes a token-provenance
file so downstream analysis can map k-NN node indices back to
(sentence_index, word_index) coordinates.

Directory conventions expected under data_dir:
  Tensors/{split}/a{tag}_e{epoch}.pt   - activation tensor (N_sent, max_words, 768)
  Tensors/{split}/mask_e{epoch}.pt     - word mask        (N_sent, max_words)  bool
  (Losses / Predictions / Labels dirs are ignored here)

Output written under complexes_dir (mirrors data_dir sub-structure, Tensors/ stripped):
  {split}/adj_a{tag}_e{epoch}_{k}_connected.txt   - NetworkX adjacency list
  {split}/token_coords_a{tag}_e{epoch}.pt          - LongTensor (N_valid, 2):
                                                      [:, 0] = sentence index
                                                      [:, 1] = word index

Usage:
  python compute_knn_complexes_bert.py <data_dir> <complexes_dir> <max_k> <k|r> [ignore_splits]

  data_dir        root of inference output, e.g.
                  /media/santripta/data2/BERT/landscape_data/bert_ner_conll/<run>/
  complexes_dir   where to write adjacency lists and provenance files
  max_k           maximum number of neighbours
  k|r             always use 'k' (r-graph not implemented)
  ignore_splits   optional comma-separated split names to skip (e.g. train,val)
"""

import logging
import os
import pickle
import re
import sys
from multiprocessing import Pool, cpu_count, log_to_stderr, get_logger
from timeit import default_timer as timer

import networkx as nx
import numpy as np
import torch

from knn_graph import compute_knn_graph

CPUS = 15

# Regex that matches activation files saved by run_train_inference.py:
#   alayer11_e0.pt, aclassifier_e3.pt, etc.
_ACT_RE = re.compile(r'^a(.+)_e(\d+)\.pt$')


def _mask_filename(tag: str, epoch: str) -> str:
    return f'mask_e{epoch}.pt'


def _save_name(act_file: str, k: int, connected: bool) -> str:
    base = os.path.splitext(act_file)[0]          # e.g. 'alayer11_e0'
    suffix = '_connected' if connected else ''
    return f'adj_{base}_{k}{suffix}'


def _provenance_name(act_file: str) -> str:
    base = os.path.splitext(act_file)[0]          # e.g. 'alayer11_e0'
    return f'token_coords_{base}.pt'


def process_files(worker_id, data_dir, complexes_dir, root, act_files, max_k, method, metric):
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

        # ------------------------------------------------------------------ #
        # Build output paths
        # ------------------------------------------------------------------ #
        save_basepath = root.replace(data_dir, complexes_dir).replace(f'Tensors{os.sep}', '')
        os.makedirs(save_basepath, exist_ok=True)

        adj_names  = [_save_name(act_file, max_k, b) for b in (True, False)]
        adj_paths  = [os.path.join(save_basepath, f'{n}.txt') for n in adj_names]
        prov_path  = os.path.join(save_basepath, _provenance_name(act_file))

        if any(os.path.exists(p) for p in adj_paths):
            log.info(f'{worker_id}: Output exists for {act_file}, k={max_k} - skipping')
            continue

        # ------------------------------------------------------------------ #
        # Load and flatten
        # ------------------------------------------------------------------ #
        acts = torch.load(act_path)   # (N_sent, max_words, 768)
        mask = torch.load(mask_path)  # (N_sent, max_words)  bool

        if acts.ndim != 3 or mask.ndim != 2:
            log.warning(f'{worker_id}: Unexpected tensor shapes for {act_file}: '
                        f'acts={tuple(acts.shape)}, mask={tuple(mask.shape)} - skipping')
            continue

        N_sent, max_words, hidden = acts.shape
        assert mask.shape == (N_sent, max_words), (
            f'Shape mismatch: acts {acts.shape}, mask {mask.shape}')

        # One row per valid word token
        acts_flat = acts[mask].float().numpy()          # (N_valid, 768)
        N_valid   = acts_flat.shape[0]

        # Provenance: (sent_idx, word_idx) for each valid token
        sent_idx, word_idx = torch.where(mask)          # each (N_valid,)
        coords = torch.stack([sent_idx, word_idx], dim=1)  # (N_valid, 2) int64

        log.info(f'{worker_id}: {act_file} - {N_sent} sents, {N_valid} valid tokens, '
                 f'hidden={hidden}')

        if acts_flat.shape not in times:
            times[acts_flat.shape] = []

        # ------------------------------------------------------------------ #
        # Save provenance (once; independent of k)
        # ------------------------------------------------------------------ #
        torch.save(coords, prov_path)

        # ------------------------------------------------------------------ #
        # k-NN graph
        # ------------------------------------------------------------------ #
        if N_valid <= max_k:
            log.warning(f'{worker_id}: N_valid={N_valid} <= max_k={max_k} for {act_file}, '
                        f'clamping k to {N_valid - 1}')
            effective_k = N_valid - 1
        else:
            effective_k = max_k

        start = timer()
        G = compute_knn_graph(acts_flat, n_neighbors=effective_k, metric=metric)
        elapsed = timer() - start
        times[acts_flat.shape].append(elapsed)

        connected = nx.is_connected(G)
        if not connected:
            log.info(f'{worker_id}: k={effective_k} graph NOT connected for {act_file}')

        name = _save_name(act_file, effective_k, connected)
        out_path = os.path.join(save_basepath, f'{name}.txt')
        nx.write_adjlist(G, out_path)
        log.info(f'{worker_id}: Saved {act_file} → {name}.txt  ({elapsed:.1f}s, '
                 f'connected={connected})')

    return times


def main():
    if len(sys.argv) < 6:
        print('Usage: python compute_knn_complexes_bert.py '
              '<data_dir> <complexes_dir> <max_k> <k|r> <e|c> [ignore_splits]')
        return

    ignore_splits = []
    if len(sys.argv) >= 7:
        ignore_splits = sys.argv[6].split(',')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(processName)s - %(levelname)s: %(message)s',
    )
    log_to_stderr(logging.INFO)

    data_dir      = sys.argv[1]
    complexes_dir = sys.argv[2]
    max_k         = int(sys.argv[3])
    method        = sys.argv[4]
    distance      = sys.argv[5]  # 'e' for euclidean, 'c' for cosine

    if method != 'k':
        raise NotImplementedError('Only k-NN method is supported (pass k as 4th arg)')

    logging.info(f'BERT k-NN: data_dir={data_dir}  complexes_dir={complexes_dir}  '
                 f'max_k={max_k} metric={distance} ignore_splits={ignore_splits}')

    times_dict = {}

    for root, dirs, files in os.walk(data_dir):

        if 'Tensors' not in root:
            continue

        # Apply split filter
        root_parts = root.replace('\\', '/').split('/')
        if any(s in root_parts for s in ignore_splits):
            logging.info(f'Skipping {root} (ignore_splits)')
            continue

        # Only activation files (a*.pt), not mask files
        act_files = [f for f in files if _ACT_RE.match(f)]

        logging.info(f'Processing {root}: {len(act_files)} activation files')
        if not act_files:
            continue


        groups = [
            (i, data_dir, complexes_dir, root,
             act_files[i::CPUS], max_k, method, distance)
            for i in range(CPUS)
            if act_files[i::CPUS]
        ]

        pool = Pool(len(groups))
        all_times = pool.starmap(process_files, groups)
        pool.close()
        pool.join()

        for times in all_times:
            for shape, vals in times.items():
                times_dict.setdefault(shape, []).extend(vals)

        logging.info(f'Done with {root}')

    name = f'knn_times_bert_{max_k}_{distance}.pkl'
    with open(name, 'wb') as f:
        pickle.dump(times_dict, f)

    logging.info(f'Finished. Timing saved to {name}')


if __name__ == '__main__':
    main()
