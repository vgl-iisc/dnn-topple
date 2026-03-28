"""
Compute contour / split / join trees for BERT NER landscapes.

Data layout expected under data_dir (e.g. D:/BERT/landscape_data/bert_ner_conll):
  Losses/{split}/losses_e{epoch}.pt          (N_sent, max_words) float32

KNN adjacency lists under complexes_dir (e.g. D:/BERT/knn_complexes_115):
  {split}/adj_a{tag}_e{epoch}_{k}_connected.txt
  {split}/token_coords_a{tag}_e{epoch}.pt    (N_valid, 2) int64
                                              col 0 = sentence index
                                              col 1 = word index

Per-node scalar values are derived by indexing into the loss tensor via
token_coords: flat_losses[i] = losses[coords[i,0], coords[i,1]].
A cached flat-loss text file is written alongside the adjacency lists so it
is only computed once per (split, epoch).

Output contour trees are written to ctrees_dir, preserving the {split}/
sub-directory structure.

Usage:
  python compute_contour_trees_bert.py \\
      <data_dir> <complexes_dir> <ctrees_dir> <type: c|s|j>

  type  c  contour tree
        s  split tree
        j  join tree
"""

import logging
import os
import re
import sys
from multiprocessing import Pool, log_to_stderr

import numpy as np
import torch

# contour_tree.py lives in the same directory as this script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from contour_tree import process_contour_trees  # noqa: E402
import pyct as ct  # noqa: E402

CPUS = 24

# Matches only the "connected" adjacency files produced by compute_knn_complexes_bert.py
_ADJ_RE = re.compile(r'^adj_a(.+)_e(\d+)_\d+_connected\.txt$')


def get_or_create_flat_losses(data_dir, split, epoch, split_complex_dir, tag):
    """Flatten and cache per-token losses for a given (split, epoch).

    Uses token_coords for the given tag/epoch to index into the 2-D loss
    tensor.  Because the mask (and therefore token_coords) is identical for
    every tag at the same epoch, any tag can be used as the source of coords;
    the result is cached as flat_losses_e{epoch}.txt so it is computed once.

    Returns the path to the cached flat-losses text file.
    """
    cache_path = os.path.join(split_complex_dir, f'flat_losses_e{epoch}.txt')
    if os.path.exists(cache_path):
        return cache_path

    loss_path   = os.path.join(data_dir, 'Losses', split, f'losses_e{epoch}.pt')
    coords_path = os.path.join(split_complex_dir, f'token_coords_a{tag}_e{epoch}.pt')

    if not os.path.exists(loss_path):
        raise FileNotFoundError(f'Loss file not found: {loss_path}')
    if not os.path.exists(coords_path):
        raise FileNotFoundError(f'Token-coords file not found: {coords_path}')

    losses = torch.load(loss_path,   weights_only=True)  # (N_sent, max_words)
    coords = torch.load(coords_path, weights_only=True)  # (N_valid, 2)

    flat_losses = losses[coords[:, 0], coords[:, 1]].float().numpy()  # (N_valid,)

    np.savetxt(cache_path, flat_losses)
    logging.info(f'Wrote flat losses ({len(flat_losses):,} tokens) → {cache_path}')
    return cache_path


def main():
    if len(sys.argv) != 5:
        print(
            'Usage: python compute_contour_trees_bert.py '
            '<data_dir> <complexes_dir> <ctrees_dir> <type: c|s|j>'
        )
        return

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(processName)s - %(levelname)s: %(message)s',
    )
    log_to_stderr(logging.INFO)

    data_dir      = sys.argv[1].rstrip('/\\')
    complexes_dir = sys.argv[2].rstrip('/\\')
    ctrees_dir    = sys.argv[3].rstrip('/\\')
    ct_type       = sys.argv[4].strip().lower()

    if ct_type not in ('c', 's', 'j'):
        print("Error: type must be one of 'c', 's', or 'j'")
        return

    tree_type = {
        'c': ct.TreeType.TypeContourTree,
        's': ct.TreeType.TypeSplitTree,
        'j': ct.TreeType.TypeJoinTree,
    }[ct_type]

    all_tasks = []

    for split_name in sorted(os.listdir(complexes_dir)):
        split_complex_dir = os.path.join(complexes_dir, split_name)
        if not os.path.isdir(split_complex_dir):
            continue

        adj_files = sorted(f for f in os.listdir(split_complex_dir) if _ADJ_RE.match(f))
        if not adj_files:
            logging.info(f'No connected adjacency files in {split_complex_dir}, skipping')
            continue

        output_split_dir = os.path.join(ctrees_dir, split_name)
        os.makedirs(output_split_dir, exist_ok=True)

        logging.info(f'Split {split_name!r}: {len(adj_files)} connected adj files')

        for adj_file in adj_files:
            m = _ADJ_RE.match(adj_file)
            tag, epoch = m.group(1), m.group(2)

            try:
                scalar_path = get_or_create_flat_losses(
                    data_dir, split_name, epoch, split_complex_dir, tag
                )
            except FileNotFoundError as exc:
                logging.warning(f'Skipping {adj_file}: {exc}')
                continue

            adj_path = os.path.join(split_complex_dir, adj_file)
            all_tasks.append((adj_path, scalar_path, output_split_dir))

    if not all_tasks:
        logging.info('No tasks found — nothing to compute.')
        return

    logging.info(f'Total tasks: {len(all_tasks)}')

    N_workers   = min(CPUS, len(all_tasks))
    task_groups = [
        (i, all_tasks[i::N_workers], tree_type)
        for i in range(N_workers)
        if all_tasks[i::N_workers]
    ]

    logging.info(
        f'Workers: {len(task_groups)}, tasks per worker: '
        f'{[len(g[1]) for g in task_groups]}'
    )

    pool = Pool(len(task_groups))
    pool.starmap(process_contour_trees, task_groups)
    pool.close()
    pool.join()

    logging.info('All contour tree computations completed.')
    print('Done')


if __name__ == '__main__':
    main()
