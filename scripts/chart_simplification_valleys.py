"""
Computes charts relating simplification level and number of valleys (minima-saddle arcs in the contour tree)

Usage:
python chart_simplification_valleys.py <ct_dir> <charts_output_dir>
"""

from glob import glob
from sys import argv, stderr
import os

from tqdm import tqdm

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
    
    outpath = os.path.join(root.replace(ct_dir, charts_dir), f"{tree_name}_features_simpl.png")

    print(f"Processing chart for {tree_name}...")
    if os.path.exists(outpath):
        print(f"Chart already exists, skipping: {outpath}")
        return

    os.makedirs(os.path.dirname(outpath), exist_ok=True)

    thresh = load_simplification_thresholds_from_tree(os.path.join(root, tree_name))

    thresh = np.linspace(0, np.max(thresh), num=500)

    features = [load_features(os.path.join(root, tree_name), t)[0] for t in thresh]
    print(f"Using {len(thresh)} simplification thresholds")
    print(f"Loaded features for all thresholds for {outpath}: {sum([len(f) for f in features])} features")

    # map-filter to only minima-saddle arcs
    for i, _ in tqdm(enumerate(thresh), total=len(thresh), desc="Filtering to minima-saddle arcs"):
        features[i] = [f for f in features[i] if data.type[data.nodeMap[f.frm]] == ct.MINIMUM]

    num_features = [len(ft) for ft in features]

    print(f"Making chart for {outpath}")

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [1, 1]})
    fig.subplots_adjust(hspace=0.025)
    
    ax_bot.plot(thresh, num_features)
    ax_top.plot(thresh, num_features)

    med = np.mean(num_features)
    max = np.max(num_features)

    ax_bot.set_ylim(0, med * 2.5)
    ax_top.set_ylim(max * 0.75, max * 1.25)

    ax_top.spines.bottom.set_visible(False)
    ax_bot.spines.top.set_visible(False)
    ax_top.xaxis.tick_top()
    ax_top.tick_params(labeltop=False)
    ax_bot.xaxis.tick_bottom()

    ax_bot.set_xlabel("Simplification Threshold")
    ax_bot.set_ylabel("#Min-Saddle Arcs")
    ax_top.set_title(f"{tree_name}")
    fig.tight_layout()
    plt.savefig(outpath)
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