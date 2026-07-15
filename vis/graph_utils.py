from basic_utils import compute_arc_features
from experiment import LossLandscapeExperiment, BertExperiment

import pyct as ct
import streamlit as st
import networkx as nx

CP_COLORING = {
    ct.MINIMUM: "#1f77b4", # type: ignore
    ct.SADDLE: "#ff7f0e", # type: ignore
    ct.MAXIMUM: "#ca2e29", # type: ignore
    ct.REGULAR: "#d822df", # type: ignore
}

def rooted_tree_from_exp(exp: LossLandscapeExperiment, simpl: float, data_dir: str, ct_dir: str) -> nx.DiGraph:
    features = compute_arc_features(exp, simpl)
    nxg = compute_tree_graph(exp, features, "None", "None", 0)
    
    return nxg

@st.cache_data(hash_funcs={LossLandscapeExperiment: LossLandscapeExperiment.__hash__, BertExperiment: BertExperiment.__hash__, list: lambda x: hash(tuple(f.id for f in x))})
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
        if idx == len(exp.dataset.labels_by_split[exp.split]):
            idx -= 1
        cls = exp.dataset.classes[exp.dataset.labels_by_split[exp.split][idx]]
        nxg.add_node(idx, label=cls, color=node_color, cp_type=node_type, fn_val=node_fn, title=f"ID: {idx}\nLoss: {node_fn}\nClass: {cls}")

    for i, f in enumerate(features):
        class_label = exp.dataset.classes[f.majority_class]
        majority_share = f.major_class_size / f.size if f.size > 0 else 0.0
        nxg.add_edge(
        f.frm,
        f.to,
        label=f"{f.id}: {class_label} ({f.size})",
        title=f"ID: {f.id}\nPersistence: {f.pers}\nVolume: {f.size}\nMajority: {class_label}\nMajority Share: {majority_share}",
        color="#888888",
        width=2,
        feature_id=f.id,
        persistence=float(f.pers),
        volume=int(f.size),
        majority=class_label,
        )

    if steiner_mode != "None":
        steiner_v = list(minima_nodes) if steiner_mode == "Minima" else list(maxima_nodes)
        nxg_undir = nx.algorithms.approximation.steiner_tree(nxg.to_undirected(), steiner_v, weight="edge_count")
        nxg = nxg.subgraph(nxg_undir.nodes).to_directed()
    
    if ego_origin != "None":
        if type(ego_origin) is tuple:
            ego_verts = [v for v in ego_origin if v in nxg.nodes]
        else:
            ego_verts = list(minima_nodes) if ego_origin == "Minima" else list(maxima_nodes)
        
        full_graph = nx.DiGraph()
        
        for v in ego_verts:
            ego_g = nx.ego_graph(nxg, v, radius=ego_radius, undirected=True)
            full_graph = nx.compose(full_graph, ego_g)
            
        nxg = full_graph

    return nxg
