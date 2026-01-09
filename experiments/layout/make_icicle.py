import pyct as ct
from utils import RichFeature, make_arc_map
import plotly.graph_objects as go
import numpy as np

def resolve_sizes(ctree_path: str, feats: list[RichFeature]) -> list[RichFeature]:
	with open(f"{ctree_path}.part.raw", "rb") as f:
		parts = np.fromfile(f, dtype=np.uint32)
		
	print(f"Loaded {len(parts)} parts from {ctree_path}.part.raw")
	
	arc_map = make_arc_map(feats)
 
	for i, aid in enumerate(parts):
		feat = arc_map.get(aid, None)
		assert feat is not None, f"Arc id {aid} from parts file not found in features!"
		feat.size += 1
  
	return feats

def compute_prevs(feats: list[RichFeature]):
	prevs = {}
	for f in feats:
		if f.frm not in prevs:
			prevs[f.frm] = []
		prevs[f.frm].append(f.id)
	
	prevs_edges = []
	for f in feats:
		if f.to in prevs:
			assert len(prevs[f.to]) == 1
			prevs_edges.append(prevs[f.to][0])
		else:
			prevs_edges.append(-1)
   
	return prevs_edges

EPS = 1e-8

def make_icicle_inner(feats: list[RichFeature], use_colors: str = "vol", style="remainder"):
	labels = [f"{f.id}" for f in feats]
	parents = [str(prev) if prev != -1 else "" for prev in compute_prevs(feats)]
	values = [f.size for f in feats]
 
	if use_colors == "vol":
		colors = [np.log(v + EPS) for v in values]
		average_col = np.mean(colors)
		marker = dict(colors=colors, colorscale='RdBu_r', cmid=average_col)
	elif use_colors == "loss":
		colors = [np.log((f.fn_frm + f.fn_to) / 2 + EPS) for f in feats]
		average_col = np.mean(colors)
		marker = dict(colors=colors, colorscale='RdBu_r', cmid=average_col)
	else:
		marker = None
 
	fig = go.Figure(go.Icicle(
		labels=labels,
		parents=parents,
		values=values,
		root_color="lightgrey",
  		tiling = dict(orientation='v'),
		marker=marker,
		branchvalues=style
	))
 
	fig.update_layout(margin = dict(t=50, l=25, r=25, b=25))
	return fig

def make_icicle(ctree_path: str, topk: int = -1, thresh: float = 1e-6, style="remainder"):
	feats = ct.TopologicalFeatures()
	feats.loadData(ctree_path)
	
	arc_feats, top_k = feats.getArcFeatures(topk, thresh)
	arc_feats = [RichFeature(id, f, feats.ctdata) for id, f in enumerate(arc_feats)]
 
	arc_feats = resolve_sizes(ctree_path, arc_feats)
  
	return make_icicle_inner(arc_feats, style=style)