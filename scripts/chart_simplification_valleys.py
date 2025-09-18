"""
Computes charts relating simplification level and number of valleys (minima-saddle arcs in the contour tree)

Usage:
python chart_simplification_valleys.py <ct_dir> <charts_output_dir>
"""

from glob import glob
from sys import argv, stderr
import os

import pyct as ct
import numpy as np
import matplotlib.pyplot as plt

from utils import load_simplification_thresholds_from_tree

def load_features(tree_name: str, threshold: float):
    topo = ct.TopologicalFeatures()
    topo.loadData(tree_name)

    features = topo.getArcFeatures(-1, threshold)

    return features
    

def process_chart(root: str, tree_name: str, ct_dir: str, charts_dir: str):
    data = ct.ContourTreeData()
    data.loadBinFile(os.path.join(root, tree_name))
    
    thresh = load_simplification_thresholds_from_tree(os.path.join(root, tree_name)) 

    features = [load_features(os.path.join(root, tree_name), t)[0] for t in thresh]

    # map-filter to only minima-saddle arcs
    for i, _ in enumerate(thresh):
        features[i] = [f for f in features[i] if data.type[data.nodeMap[f.frm]] == ct.MINIMUM]

    num_features = [len(ft) for ft in features]
    print(num_features, file=stderr)

    os.makedirs(os.path.join(root.replace(ct_dir, charts_dir)), exist_ok=True)

    fig, ax = plt.subplots()
    
    ax.set_ylim(0.0, 20)
    ax.plot(thresh, num_features)

    ax.set_xlabel("Simplification Threshold")
    ax.set_ylabel("#Min-Saddle Arcs")
    ax.set_title(f"{tree_name}")
    fig.tight_layout()
    plt.savefig(os.path.join(root.replace(ct_dir, charts_dir), f"{tree_name}_features_simpl.png"))
    plt.close(fig)

def main():
    if len(argv) != 3:
        print("Usage: python chart_simplification_valleys.py <ct_dir> <charts_output_dir>")
        return
    
    plt.style.use("fivethirtyeight")
    
    ct_dir = argv[1].strip("\\/")
    charts_dir = argv[2].strip("\\/")

    for (root, dirs, files) in os.walk(ct_dir):
        trees = glob("*.order.bin", root_dir=root)

        if len(trees) == 0:
            continue

        tree_names = [t.split(".order.bin")[0] for t in trees]
        
        for tree in tree_names:
            process_chart(root, tree, ct_dir, charts_dir)


if __name__ == "__main__":
    main()