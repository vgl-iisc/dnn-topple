"""
Computes contour trees for all landscapes available

Usage:
python compute_contour_trees.py <data_dir> <complexes_dir> <ct_dir> s|c|j
"""

import networkx as nx
import numpy as np

import pyct as ct

from glob import glob
from sys import argv
import os

from multiprocessing import Pool, cpu_count, log_to_stderr
import logging

from contour_tree import process_contour_trees

CPUS = 24

def main():
    if len(argv) < 5:
        print("Usage: python compute_contour_trees.py <data_dir> <complexes_dir> <ct_dir> <type: c | s | j (contour, split, or join)> [scalar-field: Losses | GradNorm | Entropy | Margin]")
        return
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(processName)s - %(levelname)s: %(message)s')
    log_to_stderr(logging.INFO)
    
    data_dir = argv[1].rstrip("\\/")
    complexes_dir = argv[2].rstrip("\\/")
    ctrees_dir = argv[3].rstrip("\\/")
    ct_type = argv[4].strip().lower()
    scalar_field = argv[5].strip() if len(argv) > 5 else "Losses"

    if ct_type not in ["c", "s", "j"]:
        print("Error: ct_type must be one of 'c', 's', or 'j'")
        return

    tree_type = {
        "c": ct.TreeType.TypeContourTree,
        "s": ct.TreeType.TypeSplitTree,
        "j": ct.TreeType.TypeJoinTree
    }
    
    tree_type = tree_type[ct_type]

    # Collect all tasks first
    all_tasks = []
    
    for (root, dirs, files) in os.walk(data_dir):
        if not scalar_field in root:
            continue
    
        scalar_files = [f for f in files if f.startswith(scalar_field.lower())]

        if len(scalar_files) == 0:
            continue

        # not in the argand plane
        complex_root = root.replace(f"{scalar_field}{os.sep}", "").replace(data_dir, complexes_dir)

        output_root = complex_root.replace(complexes_dir, ctrees_dir)
        os.makedirs(output_root, exist_ok=True)

        for f in scalar_files:
            epoch = os.path.splitext(f)[0][f.find("_e"):].split("_")[1]

            complexes = glob(f"adj_*_{epoch}_*_connected.txt", root_dir=complex_root)

            for g in complexes:
                scalar_file = os.path.join(root, f)
                complex_file = os.path.join(complex_root, g)
                all_tasks.append((complex_file, scalar_file, output_root))
    
    if len(all_tasks) == 0:
        logging.info("No tasks found to process")
        return
    
    # Distribute tasks across workers
    N_workers = max(1, CPUS)
    task_groups = []
    
    for i in range(N_workers):
        worker_tasks = all_tasks[i::N_workers]
        if len(worker_tasks) > 0:
            task_groups.append((i, worker_tasks, tree_type))
    
    logging.info(f"Starting processing: {len(all_tasks)} total tasks across {len(task_groups)} workers")
    logging.info(f"Tasks per worker: {[len(group[1]) for group in task_groups]}")
    
    # Create pool and process tasks
    processes = Pool(len(task_groups))
    processes.starmap(process_contour_trees, task_groups)
    processes.close()
    processes.join()
    
    logging.info("All contour tree computations completed")
    print("Done")



if __name__ == "__main__":
    main()
