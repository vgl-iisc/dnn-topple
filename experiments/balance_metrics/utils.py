import numpy as np
import pyct as ct
import torch

import os
from glob import glob
import pandas as pd

from experiment import Dataset, LossLandscapeExperiment, NER_LABELS, BertExperiment

import networkx as nx

import pyct as ct

TYPE_STRING = {
	ct.REGULAR: "REGULAR",
	ct.MINIMUM: "MINIMA",
	ct.SADDLE: "SADDLE",
	ct.MAXIMUM: "MAXIMA"
}

CP_COLORING = {
    ct.MINIMUM: "#1f77b4", # type: ignore
    ct.SADDLE: "#ff7f0e", # type: ignore
    ct.MAXIMUM: "#ca2e29", # type: ignore
    ct.REGULAR: "#d822df", # type: ignore
}

stree_path = "C:\\home\\sumatra_llvis_data\\strees_pt\\resnet50pt_imagenet\\trainUval"
data_path = "C:\\home\\sumatra_llvis_data\\landscape_data_pt"
datasets_path = ".\\datasets"

class RichFeature:
	def __init__(self, id: int, feature: ct.Feature, data: ct.ContourTreeData):
		self.id = id
				
		self.frm = feature.frm
		self.to = feature.to
		
		frm_corrected = data.nodeMap[self.frm]
		to_corrected = data.nodeMap[self.to]
		
		self.fn_frm = data.fnVals[frm_corrected]
		self.fn_to = data.fnVals[to_corrected]
		
		self.type_frm = data.type[frm_corrected]
		self.type_to = data.type[to_corrected]
				
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
    """
    Compute rich features using the same pyct pipeline as the vis code path.
    This populates feature size/coverage/majority metadata used as edge volume.
    """
    topo = ct.TopologicalFeatures()
    topo.loadData(exp.get_paths()["ctree"])

    data = topo.ctdata
    partition = get_partition(exp)
    # labels = get_labels(exp)
    # preds = get_preds(exp)
    labels = [0] * len(partition)
    preds = [0] * len(partition)
    
    class_sizes = [100000 for _ in exp.dataset.classes]

    features = ct.computeRichFeatures(topo, -1, simpl, partition, labels, preds, class_sizes)  # type: ignore

    return features, data

def make_arc_map(features: list[RichFeature]):
    arc_map = {}
    for feat in features:
        for arc_id in feat.arcs:
            assert arc_id not in arc_map, "Arc belongs to multiple features!"
            arc_map[arc_id] = feat
    
    return arc_map

def _bert_token_coords(exp: BertExperiment):
    """Load (N_valid, 2) token-coords tensor for BERT experiments."""
    coords_path = os.path.join(
        exp.complexes_dir, exp.split,
        f"token_coords_a{exp.tag}_{exp.epoch_tag}.pt",
    )
    return torch.load(coords_path, weights_only=True)


def get_preds(exp) -> list[int]:
    if exp.is_bert:
        preds_2d = torch.load(exp.get_paths()["predictions"], weights_only=True)
        coords = _bert_token_coords(exp)
        flat = preds_2d[coords[:, 0], coords[:, 1]].numpy()
        flat = np.where(flat < 0, 0, flat).astype(int)
        return flat.tolist()
    pred_path = exp.get_paths()["predictions"]
    with open(pred_path, "rb") as f:
        preds = np.loadtxt(f, dtype=np.int32).reshape(-1)
    return preds.tolist()


def get_labels(exp) -> list[int]:
    if exp.is_bert:
        labels_2d = torch.load(exp.get_paths()["labels"], weights_only=True)
        coords = _bert_token_coords(exp)
        flat = labels_2d[coords[:, 0], coords[:, 1]].numpy()
        flat = np.where(flat < 0, 0, flat).astype(int)
        flat_list = flat.tolist()
        # Populate BertDataset so compute_tree_graph can use labels_by_split
        split = exp.split
        ds = exp.dataset
        counts: dict[str, int] = {cls: 0 for cls in NER_LABELS}
        for lbl in flat_list:
            counts[NER_LABELS[lbl]] += 1
        ds.labels_by_split[split] = flat_list
        ds.size_by_split[split] = len(flat_list)
        ds.class_size_by_split[split] = counts
        ds.largest_class_size_by_split[split] = max(counts.values()) if counts else 0
        return flat_list
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

def get_partition(exp) -> list[int]:
    if exp.is_bert:
        # Use token_coords length to determine N_valid (avoids dependency on labels_by_split
        # which may not be populated yet when get_partition is called first).
        coords = _bert_token_coords(exp)
        count = len(coords)
    else:
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

def rooted_tree_from_exp(exp: LossLandscapeExperiment, simpl: float, data_dir: str, ct_dir: str) -> nx.DiGraph:
    features = compute_arc_features(exp, simpl)[0]
    nxg = compute_tree_graph(exp, features, "None", "None", 0)
    
    return nxg

def compute_tree_graph(exp: LossLandscapeExperiment, features: list[ct.RichFeature], steiner_mode: str, ego_origin: str, ego_radius: int, simplify_saddles: bool = False):
    useful_nodes = set()
    minima_nodes = set()
    maxima_nodes = set()
    for i, f in enumerate(features):
        useful_nodes.add((f.frm, f.type_frm, f.fn_frm, CP_COLORING[f.type_frm]))
        useful_nodes.add((f.to, f.type_to, f.fn_to, CP_COLORING[f.type_to]))
        
        if f.type_frm == ct.MINIMUM: # type: ignore
            minima_nodes.add(f.frm)
        
        if f.type_to == ct.MAXIMUM: # type: ignore
            maxima_nodes.add(f.to)

    nxg = nx.DiGraph()
    for node_id, node_type, node_fn, node_color in useful_nodes:
        idx = node_id
        nxg.add_node(idx)

    for i, f in enumerate(features):
        nxg.add_edge(
        f.frm,
        f.to,
        color="#888888",
        width=2,
        feature_id=f.id,
        persistence=float(f.pers),
        volume=int(f.size),
        )

    return nxg

def load_order_and_wts(tree_files: str) -> tuple[list[int], list[float]]:
    with open(f"{tree_files}.order.dat", "r") as f:
        no_simpl = int(f.readline().strip())
        
    with open(f"{tree_files}.order.bin", "rb") as file:
        order = [int(f) for f in np.fromfile(file, dtype=np.uint32, count=no_simpl)]
        wts = [float(f) for f in np.fromfile(file, dtype=np.float32, count=no_simpl)]

    return order, wts