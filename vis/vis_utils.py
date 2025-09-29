import streamlit as st
from experiment import LossLandscapeExperiment

import pandas as pd
import numpy as np
import pyct as ct

import os

def find_steady_simplification_states(thresh: list[float], num_min: list[int], tol: float = 0.05) -> tuple[list[tuple[float, float]], list[int]]:
    """Find thresholds where the number of minima remains steady for at least `tol` steps.

    Args:
        thresh (list[float]): List of thresholds.
        num_min (list[int]): Corresponding list of number of minima.
        tol (float, optional):  Minimum proportion of the range of thresholds for which the number of minima should remain constant to be considered steady. 
                                Defaults to 0.05.

    Returns:
        tuple[list[float, float], list[int]]: List of steady threshold ranges and their corresponding number of minima.
    """
    steady_thresh = []
    steady_minima = []
    
    range = max(thresh) - min(thresh)
    stable_span = tol * range
    
    n = len(num_min)
    
    i = 0
    while i < n:
        thr_i = thresh[i]
        min_i = num_min[i]
        
        j = i + 1
        while j < n and num_min[j] == min_i:
            j += 1
        
        corrected_j = min(j, n - 1)
        
        if thresh[corrected_j] - thr_i >= stable_span:
            steady_thresh.append((thr_i, thresh[corrected_j]))
            steady_minima.append(min_i)
            i = j  # Skip past the end of this stable segment
        else:
            i += 1 

    
    return steady_thresh, steady_minima

TYPE_STRING = {
    ct.REGULAR: "regular",
    ct.MINIMUM: "minima",
    ct.SADDLE: "saddle",
    ct.MAXIMUM: "maxima"
}

def get_class_coverage(exp: LossLandscapeExperiment, count: int, class_idx: int) -> float:
    ds = exp.dataset
    total = ds.class_size_by_split[exp.split][ds.classes[class_idx]]
    
    return round(count / total if total > 0 else 0.0, 4)

class RichFeature:
    def __init__(self, id: int, feature: ct.Feature, data: ct.ContourTreeData):
        self.id = id
                
        # confirmed correct, not going through nodeMap throws IndexError
        self.frm = data.nodeMap[feature.frm]
        self.to = data.nodeMap[feature.to]
        
        self.fn_frm = data.fnVals[self.frm]
        self.fn_to = data.fnVals[self.to]
        
        self.type_frm = data.type[self.frm]
        self.type_to = data.type[self.to]
                
        self.type_string = TYPE_STRING[self.type_frm] + "-" + TYPE_STRING[self.type_to]
        
        assert self.fn_to >= self.fn_frm
        
        self.pers = self.fn_to - self.fn_frm
        
        self.arcs = set(feature.arcs)
        
        # initialized later in compute_feature_map
        self.members = set()
        self.size = 0
        self.class_counts: dict[int, int] = {}
        self.class_coverage: dict[int, float] = {}
        self.majority_class = -1
        self.major_class_size = 0
        
        # initialized later in compute_feature_coverage_data
        self.pred_class_counts: dict[int, int] = {}
        self.confusion: dict[tuple[int, int], int] = {}
        self.pred_correct: int = 0
        self.pred_incorrect: int = 0 # INV = self.size - self.pred_correct
        self.pred_accuracy: float = 0.0
        

def compute_arc_features(exp: LossLandscapeExperiment, simpl: float):
    ctree_name = exp.get_paths(st.session_state.landscapes_dir, st.session_state.ct_dir)["ctree"]
    
    topo = ct.TopologicalFeatures()
    topo.loadData(ctree_name)

    data = topo.ctdata
    features = [RichFeature(id, f, data) for id, f in enumerate(topo.getArcFeatures(-1, simpl)[0])]

    return features

def make_arc_map(features: list[RichFeature]):
    arc_map = {}
    for feat in features:
        for arc_id in feat.arcs:
            assert arc_id not in arc_map, "Arc belongs to multiple features!"
            arc_map[arc_id] = feat
    
    return arc_map

def compute_feature_map(exp: LossLandscapeExperiment, features: list[RichFeature]) -> list[int]:
    """
    Computes a mapping from contour tree node (i.e. an embedded vector for a data point) to the contour tree feature it belongs to and vice versa.
    Also populates rich data in the feature objects.
    """
    
    ctree_path = exp.get_paths(st.session_state.landscapes_dir, st.session_state.ct_dir)["ctree"]
    
    labels = exp.dataset.labels_by_split[exp.split]
    count = len(labels)
    
    with open(f"{ctree_path}.part.raw", "rb") as f:
        parts = np.fromfile(f, dtype=np.uint32, count=count)
        
    arc_map = make_arc_map(features)

    point2feat = [-1] * count
        
    for i, arc_id in enumerate(parts):
        assert arc_id in arc_map, "Arc not found in feature map!"
        
        feat = arc_map[arc_id]
        point2feat[i] = arc_map[arc_id]
        
        feat.members.add(i)
        feat.size += 1
        
        label = labels[i]
        
        if label not in feat.class_counts:
            feat.class_counts[label] = 0
        feat.class_counts[label] += 1
        

    for feat in features:
        if len(feat.class_counts) == 0:
            continue
        
        feat.majority_class = max(list(feat.class_counts.keys()), key=lambda k: feat.class_counts[k])
        feat.major_class_size = feat.class_counts[feat.majority_class]
        
    assert set.union(*[feat.members for feat in features]) == set(range(count)), "Some nodes are not mapped to any feature!"
        
    for feat in features:
        for i in range(len(exp.dataset.classes)):
            feat.class_coverage[i] = get_class_coverage(exp, feat.class_counts.get(i, 0), i)

    return point2feat

def load_preds(exp: LossLandscapeExperiment) -> list[int]:
    pred_path = exp.get_paths(st.session_state.landscapes_dir, st.session_state.ct_dir)["predictions"]
    
    with open(pred_path, "rb") as f:
        preds = np.loadtxt(f, dtype=np.int32).reshape(-1)
    
    return preds.tolist()

def compute_feature_coverage_data(exp: LossLandscapeExperiment, features: list[RichFeature], preds: list[int]):
    """
    Computes per-feature coverage data, incorporating model predictions.
    """

    labels = exp.dataset.labels_by_split[exp.split]
    
    for feat in features:
        for dp in feat.members:
            true_label = labels[dp]
            pred_label = preds[dp]
            
            feat.pred_class_counts[pred_label] = feat.pred_class_counts.get(pred_label, 0) + 1
            feat.confusion[(true_label, pred_label)] = feat.confusion.get((true_label, pred_label), 0) + 1
            
            if true_label == pred_label:
                feat.pred_correct += 1
            else:
                feat.pred_incorrect += 1
                
        feat.pred_accuracy = feat.pred_correct / feat.size if feat.size > 0 else 0.0


def class2color(idx: int) -> str:
    colors = [
        "#1f77b4",  # blue
        "#ff7f0e",  # orange
        "#2ca02c",  # green
        "#d62728",  # red
        "#9467bd",  # purple
        "#8c564b",  # brown
        "#e377c2",  # pink
        "#7f7f7f",  # gray
        "#bcbd22",  # yellow-green
        "#17becf",  # cyan
        "#393b79",  # dark blue
        "#637939",  # dark green
        "#8c6d31",  # dark brown
        "#843c39",  # dark red
        "#7b4173",  # dark purple
        "#5254a3",  # medium blue
    ]
    
    return colors[idx % len(colors)]