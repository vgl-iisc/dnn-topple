import streamlit as st
from experiment import LossLandscapeExperiment
import numpy as np
import pyct as ct

def get_preds(exp: LossLandscapeExperiment) -> list[int]:
    pred_path = exp.get_paths()["predictions"]
    
    with open(pred_path, "rb") as f:
        preds = np.loadtxt(f, dtype=np.int32).reshape(-1)
    
    return preds.tolist()

def get_labels(exp: LossLandscapeExperiment) -> list[int]:
    return exp.dataset.labels_by_split[exp.split]

def get_tree(exp: LossLandscapeExperiment):
    ctree_name = exp.get_paths()["ctree"]
    topo = ct.TopologicalFeatures() # type: ignore
    topo.loadData(ctree_name)
            
    return topo.ctdata, topo

def get_order_and_weights(exp: LossLandscapeExperiment) -> tuple[list[int], list[float]]:
    tree_files = exp.get_paths()["ctree"]
    with open(f"{tree_files}.order.dat", "r") as f:
        no_simpl = int(f.readline().strip())
        
    with open(f"{tree_files}.order.bin", "rb") as file:
        order = [int(f) for f in np.fromfile(file, dtype=np.uint32, count=no_simpl)]
        wts = [float(f) for f in np.fromfile(file, dtype=np.float32, count=no_simpl)]

    return order, wts

def get_partition(exp: LossLandscapeExperiment) -> list[int]:
    count = len(exp.dataset.labels_by_split[exp.split])
    
    with open(f"{exp.get_paths()['ctree']}.part.raw", "rb") as f:
        parts = np.fromfile(f, dtype=np.uint32, count=count)
        
    return parts.tolist()

def get_valley_vs_thresh(exp: LossLandscapeExperiment) -> tuple[list[float], list[int]]:
    data, _ = get_tree(exp)
    simpl = ct.SimplifyCT() # type: ignore
    simpl.setInput(data)

    order, wts = get_order_and_weights(exp)
    fns, num_min, _ = simpl.getSimplificationPlot(order, wts)

    return fns, num_min