import streamlit as st
from experiment import LossLandscapeExperiment

import numpy as np
import pyct as ct

@st.cache_resource(hash_funcs={LossLandscapeExperiment: LossLandscapeExperiment.__hash__})
def get_preds(exp: LossLandscapeExperiment) -> list[int]:
    if st.session_state.no_preds:
        return get_labels(exp)
    
    pred_path = exp.get_paths(data_dir=st.session_state.landscapes_dir, ct_dir=st.session_state.ct_dir)["predictions"]
    
    with open(pred_path, "rb") as f:
        preds = np.loadtxt(f, dtype=np.int32).reshape(-1)
    
    return preds.tolist()

@st.cache_resource(hash_funcs={LossLandscapeExperiment: LossLandscapeExperiment.__hash__})
def get_labels(exp: LossLandscapeExperiment) -> list[int]:
    return exp.dataset.labels_by_split[exp.split]

@st.cache_resource(hash_funcs={LossLandscapeExperiment: LossLandscapeExperiment.__hash__})
def get_tree(exp: LossLandscapeExperiment):
    ctree_name = exp.get_paths(data_dir=st.session_state.landscapes_dir, ct_dir=st.session_state.ct_dir)["ctree"]
    topo = ct.TopologicalFeatures() # type: ignore
    topo.loadData(ctree_name)
            
    return topo.ctdata, topo

@st.cache_resource(hash_funcs={LossLandscapeExperiment: LossLandscapeExperiment.__hash__})
def get_order_and_weights(exp: LossLandscapeExperiment) -> tuple[list[int], list[float]]:
    tree_files = exp.get_paths(data_dir=st.session_state.landscapes_dir, ct_dir=st.session_state.ct_dir)["ctree"]
    with open(f"{tree_files}.order.dat", "r") as f:
        no_simpl = int(f.readline().strip())
        
    with open(f"{tree_files}.order.bin", "rb") as file:
        order = [int(f) for f in np.fromfile(file, dtype=np.uint32, count=no_simpl)]
        wts = [float(f) for f in np.fromfile(file, dtype=np.float32, count=no_simpl)]

    return order, wts

@st.cache_resource(hash_funcs={LossLandscapeExperiment: LossLandscapeExperiment.__hash__})
def get_partition(exp: LossLandscapeExperiment) -> list[int]:
    count = len(exp.dataset.labels_by_split[exp.split])
    
    with open(f"{exp.get_paths(data_dir=st.session_state.landscapes_dir, ct_dir=st.session_state.ct_dir)['ctree']}.part.raw", "rb") as f:
        parts = np.fromfile(f, dtype=np.uint32, count=count)
        
    return parts.tolist()

@st.cache_resource(hash_funcs={LossLandscapeExperiment: LossLandscapeExperiment.__hash__})
def get_valley_vs_thresh(exp: LossLandscapeExperiment) -> tuple[list[float], list[int]]:
    data, _ = get_tree(exp)
    simpl = ct.SimplifyCT() # type: ignore
    simpl.setInput(data)

    order, wts = get_order_and_weights(exp)
    fns, num_min, _ = simpl.getSimplificationPlot(order, wts)

    return fns, num_min

def get_filtered_cps_vs_thresh(exp: LossLandscapeExperiment, fnstart_range, fnend_range, types):
    data, _ = get_tree(exp)
    simpl = ct.SimplifyCT() # type: ignore
    simpl.setInput(data)

    order, wts = get_order_and_weights(exp)
    fns, counts = simpl.getFilteredSimplificationPlot(order, wts, fnstart_range[0], fnstart_range[1], fnend_range[0], fnend_range[1], types)

    return fns, counts

@st.cache_resource(hash_funcs={LossLandscapeExperiment: LossLandscapeExperiment.__hash__})
def compute_arc_features(exp: LossLandscapeExperiment, simpl: float):
    """
    Computes rich features using the C++ implementation.
    Returns a list of RichFeature objects with all metadata populated.
    """
    _, topo = get_tree(exp)
    partition = get_partition(exp)
    labels = get_labels(exp)
    preds = get_preds(exp)
    
    # Get class sizes for computing class coverage
    class_sizes = [exp.dataset.class_size_by_split[exp.split][cls] for cls in exp.dataset.classes]
    
    # Use C++ implementation for fast computation
    features = ct.computeRichFeatures(topo, -1, simpl, partition, labels, preds, class_sizes) # type: ignore
    
    return features