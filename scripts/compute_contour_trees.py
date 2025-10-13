"""
Computes contour trees for all landscapes available

Usage:
python compute_contour_trees.py <data_dir> <complexes_dir> <ct_dir>
"""

import networkx as nx
import numpy as np

import pyct as ct

from glob import glob
from sys import argv
import os

from contour_tree import compute_and_save_contour_tree

def main():
    if len(argv) != 4:
        print("Usage: python compute_contour_trees.py <data_dir> <complexes_dir> <ct_dir>")
        return
    
    data_dir = argv[1].strip("\\/")
    complexes_dir = argv[2].strip("\\/")
    ctrees_dir = argv[3].strip("\\/")

    for (root, dirs, files) in os.walk(data_dir):
        if not "Losses" in root:
            continue
    
        scalar_files = [f for f in files if f.startswith("loss")]

        if len(scalar_files) == 0:
            continue

        # not in the argand plane
        complex_root = root.replace(f"Losses{os.sep}", "").replace(data_dir, complexes_dir)

        output_root = complex_root.replace(complexes_dir, ctrees_dir)
        os.makedirs(output_root, exist_ok=True)

        for f in scalar_files:
            epoch = os.path.splitext(f)[0][f.find("_e"):].split("_")[1]

            complexes = glob(f"adj_*_{epoch}_*_connected.txt", root_dir=complex_root)

            for g in complexes:
                scalar_file = os.path.join(root, f)
                complex_file = os.path.join(complex_root, g)

                print(f"Processing {scalar_file} with {complex_file}")
                compute_and_save_contour_tree(complex_file, scalar_file, output_root)
                print()

    print("Done")



if __name__ == "__main__":
    main()
