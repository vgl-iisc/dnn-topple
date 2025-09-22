import streamlit as st
from experiment import LossLandscapeExperiment

import pyct as ct

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

class RichFeature:
    def __init__(self, feature: ct.Feature, data: ct.ContourTreeData):
        self.frm = data.nodeMap[feature.frm]
        self.to = data.nodeMap[feature.to]
        
        self.fn_frm = data.fnVals[self.frm]
        self.fn_to = data.fnVals[self.to]
        
        self.type_frm = data.type[self.frm]
        self.type_to = data.type[self.to]
        
        self.type_string = TYPE_STRING[self.type_frm] + "-" + TYPE_STRING[self.type_to]
        
        assert self.fn_to >= self.fn_frm
        
        self.pers = self.fn_to - self.fn_frm
        
        self.members = set()
        
        for arc_id in feature.arcs:
            arc = data.arcs[arc_id]
            assert arc.id == arc_id
            
            self.members.add(arc.frm)
            self.members.add(arc.to)
        
        self.size = len(self.members)
        

def compute_arc_features(exp: LossLandscapeExperiment, simpl: float):
    exp_paths = exp.get_paths(st.session_state.landscapes_dir, st.session_state.ct_dir)

    ctree_name = exp_paths["ctree"]
    
    topo = ct.TopologicalFeatures()
    topo.loadData(ctree_name)

    data = topo.ctdata
    features = [RichFeature(f, data) for f in topo.getArcFeatures(-1, simpl)[0]]

    return features