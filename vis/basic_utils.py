import streamlit as st
from experiment import LossLandscapeExperiment
from feature import RichFeature
from coverage_utils import get_class_coverage

import numpy as np
import pyct as ct

@st.cache_resource(hash_funcs={LossLandscapeExperiment: LossLandscapeExperiment.__hash__})
def get_preds(exp: LossLandscapeExperiment) -> list[int]:
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
def compute_arc_features(exp: LossLandscapeExperiment, simpl: float) -> list[RichFeature]:
    data, topo = get_tree(exp)
    features = [RichFeature(id, f, data) for id, f in enumerate(topo.getArcFeatures(-1, simpl)[0])]
    return features

def make_arc_map(features: list[RichFeature]):
    arc_map = {}
    for feat in features:
        for arc_id in feat.arcs:
            assert arc_id not in arc_map, "Arc belongs to multiple features!"
            arc_map[arc_id] = feat
    
    return arc_map

@st.cache_data(hash_funcs={LossLandscapeExperiment: LossLandscapeExperiment.__hash__, list: lambda x: hash(tuple(f.id for f in x))})
def compute_feature_map(exp: LossLandscapeExperiment, features: list[RichFeature], data_dir: str, ct_dir: str) -> list[int]:
    """
    Computes a mapping from contour tree node (i.e. an embedded vector for a data point) to the contour tree feature it belongs to and vice versa.
    Also populates rich data in the feature objects.
    """

    ctree_path = exp.get_paths(data_dir, ct_dir)["ctree"]

    labels = exp.dataset.labels_by_split[exp.split]
    count = len(labels)
    
    with open(f"{ctree_path}.part.raw", "rb") as f:
        parts = np.fromfile(f, dtype=np.uint32, count=count)
        
    arc_map = make_arc_map(features)

    point2feat = [-1] * count
    print("FMAP COUNT:", count, len(parts))
        
    for i, arc_id in enumerate(parts):
        assert arc_id in arc_map, "Arc not found in feature map!"
        
        # resolved_idx = data.nodeMap[i]
        resolved_idx = i
        
        feat = arc_map[arc_id]
        point2feat[resolved_idx] = arc_map[arc_id]
        
        feat.members.add(resolved_idx)
        feat.size += 1
        
        label = labels[resolved_idx]
        
        if label not in feat.class_counts:
            feat.class_counts[label] = 0
        feat.class_counts[label] += 1
        

    for feat in features:
        if len(feat.class_counts) == 0:
            feat.majority_class = labels[feat.frm]
            feat.major_class_size = 0
            continue
        
        feat.majority_class = max(list(feat.class_counts.keys()), key=lambda k: feat.class_counts[k])
        feat.major_class_size = feat.class_counts[feat.majority_class]
    
    # TODO: restore assertions
    # print("Feature map computed. Matched points:", sum([f.size for f in features]), "Expected:", count)
    assert len(point2feat) == count, "Point to feature map size mismatch!"
    assert set.union(*[feat.members for feat in features]) == set(range(count)), "Some nodes are not mapped to any feature!"
        
    for feat in features:
        for i in range(len(exp.dataset.classes)):
            feat.class_coverage[i] = get_class_coverage(exp, feat.class_counts.get(i, 0), i)

    return point2feat