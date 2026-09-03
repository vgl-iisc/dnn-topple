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
import re

from multiprocessing import Pool, cpu_count, log_to_stderr
import logging

from contour_tree import process_contour_trees

CPUS = 24

SCALAR_FIELD_SPECS = {
    "losses": {
        "root_dir": "Losses",
        "prefixes": ("losses_",),
        "match_layer": False,
    },
    "entropy": {
        "root_dir": "Entropy",
        "prefixes": ("entropy_",),
        "match_layer": False,
    },
    "margin": {
        "root_dir": "Margin",
        "prefixes": ("margin_",),
        "match_layer": False,
    },
    "gradnorm": {
        "root_dir": "GradNorm",
        "prefixes": ("gradnorm_",),
        "exclude_prefixes": ("gradnorm_importance_", "gradnorm_imp_"),
        "match_layer": False,
    },
    "gradimp": {
        "root_dir": "GradNorm",
        "prefixes": ("gradnorm_importance_", "gradnorm_imp_"),
        "match_layer": False,
    },
    "gradimportance": {
        "root_dir": "GradNorm",
        "prefixes": ("gradnorm_importance_", "gradnorm_imp_"),
        "match_layer": False,
    },
    "l2": {
        "root_dir": "Tensors",
        "prefixes": ("l2_",),
        "match_layer": True,
    },
}

EPOCH_TAG_RE = re.compile(r"_(e[^_\.]+)")
L2_FILE_RE = re.compile(r"^l2_(.+)_(e[^_\.]+)\.pt$")


def normalize_scalar_field(name):
    return name.strip().lower().replace("_", "")


def get_scalar_field_spec(name):
    return SCALAR_FIELD_SPECS.get(normalize_scalar_field(name))


def matches_scalar_file(filename, spec):
    if not any(filename.startswith(prefix) for prefix in spec["prefixes"]):
        return False

    for excluded_prefix in spec.get("exclude_prefixes", ()): 
        if filename.startswith(excluded_prefix):
            return False

    return True


def extract_epoch_tag(filename):
    match = EPOCH_TAG_RE.search(os.path.splitext(filename)[0])
    return match.group(1) if match is not None else None


def extract_l2_layer_and_epoch(filename):
    match = L2_FILE_RE.match(filename)
    if match is None:
        return None, None
    return match.group(1), match.group(2)


def l2_layer_candidates(layer_tag):
    layer_tag = layer_tag.strip()
    if not layer_tag:
        return []
    if layer_tag.startswith("a"):
        return [layer_tag]
    return [layer_tag, f"a{layer_tag}"]

def main():
    if len(argv) < 5:
        print("Usage: python compute_contour_trees.py <data_dir> <complexes_dir> <ct_dir> <type: c | s | j (contour, split, or join)> [scalar-field: Losses | GradNorm | GradImp | Entropy | Margin | L2] [simplification-type: pers | vol]")
        return
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(processName)s - %(levelname)s: %(message)s')
    log_to_stderr(logging.INFO)
    
    data_dir = argv[1].rstrip("\\/")
    complexes_dir = argv[2].rstrip("\\/")
    ctrees_dir = argv[3].rstrip("\\/")
    ct_type = argv[4].strip().lower()
    scalar_field = argv[5].strip() if len(argv) > 5 else "Losses"
    sim_type = argv[6].strip().lower() if len(argv) > 6 else "pers"
    scalar_spec = get_scalar_field_spec(scalar_field)

    if ct_type not in ["c", "s", "j"]:
        print("Error: ct_type must be one of 'c', 's', or 'j'")
        return
    if scalar_spec is None:
        print("Error: scalar_field must be one of 'Losses', 'GradNorm', 'GradImp', 'GradImportance', 'Entropy', 'Margin', or 'L2'")
        return
    if sim_type not in ["pers", "vol"]:
        print("Error: simplification-type must be one of 'pers' or 'vol'")
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
        root_parts = root.replace("\\", "/").split("/")
        if scalar_spec["root_dir"] not in root_parts:
            continue

        scalar_files = [f for f in files if matches_scalar_file(f, scalar_spec)]

        if len(scalar_files) == 0:
            continue

        # not in the argand plane
        complex_root = root.replace(f"{scalar_spec['root_dir']}{os.sep}", "").replace(data_dir, complexes_dir)

        output_root = complex_root.replace(complexes_dir, ctrees_dir)
        os.makedirs(output_root, exist_ok=True)

        for f in scalar_files:
            if scalar_spec["match_layer"]:
                layer_tag, epoch_tag = extract_l2_layer_and_epoch(f)
                if layer_tag is None or epoch_tag is None:
                    logging.warning(f"Skipping unrecognized L2 scalar filename: {os.path.join(root, f)}")
                    continue
                complexes = []
                for candidate_layer in l2_layer_candidates(layer_tag):
                    complexes.extend(glob(f"adj_{candidate_layer}_{epoch_tag}_*_connected.txt", root_dir=complex_root))
            else:
                epoch_tag = extract_epoch_tag(f)
                if epoch_tag is None:
                    logging.warning(f"Skipping scalar file without epoch tag: {os.path.join(root, f)}")
                    continue
                complexes = glob(f"adj_*_{epoch_tag}_*_connected.txt", root_dir=complex_root)

            for g in complexes:
                scalar_file = os.path.join(root, f)
                complex_file = os.path.join(complex_root, g)

                out_file_name = os.path.basename(complex_file).replace("adj_", "ctree_").replace("_connected", "")
                out_file_name, _ = os.path.splitext(out_file_name)

                out_file_name += ".rg.dat"
                outfile = os.path.join(output_root, out_file_name)

                if os.path.exists(outfile):
                    logging.info(f"Skipping existing output file: {outfile}")
                    continue

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
            task_groups.append((i, worker_tasks, tree_type, sim_type))
    
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
