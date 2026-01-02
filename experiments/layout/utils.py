import pyct as ct
import networkx as nx

TYPE_STRING = {
	ct.REGULAR: "REGULAR",
	ct.MINIMUM: "MINIMA",
	ct.SADDLE: "SADDLE",
	ct.MAXIMUM: "MAXIMA"
}

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
		

def compute_arc_features(ctree_path, simpl: float):        
	topo = ct.TopologicalFeatures()
	topo.loadData(ctree_path)

	data = topo.ctdata
	features = [RichFeature(id, f, data) for id, f in enumerate(topo.getArcFeatures(-1, simpl)[0])]

	return features, data

def make_arc_map(features: list[RichFeature]):
    arc_map = {}
    for feat in features:
        for arc_id in feat.arcs:
            assert arc_id not in arc_map, "Arc belongs to multiple features!"
            arc_map[arc_id] = feat
    
    return arc_map

def get_adjlist_from_graph(G: nx.Graph):
	"""
	Converts a NetworkX graph to an adjacency list format suitable for GraphScalarFunction.

	Parameters:
	- G: nx.Graph
		The input graph.

	Returns:
	- adjlist: list of lists
		The adjacency list representation of the graph.
	"""

	adjlist = [[] for _ in range(G.number_of_nodes())]

	for node, nbr_dict in G.adjacency():
		adj = list(map(int, nbr_dict.keys()))
		adjlist[node] = adj

	return adjlist

def make_graph_from_feats(features: list[RichFeature], colors) -> nx.DiGraph:
	useful_nodes = set()

	for i, f in enumerate(features):
		useful_nodes.add((f.frm, f.type_frm, f.fn_frm, colors[f.frm]))
		useful_nodes.add((f.to, f.type_to, f.fn_to, colors[f.to]))
	
	nxg = nx.DiGraph()
	for node_id, node_type, node_fn, node_color in useful_nodes:
		nxg.add_node(node_id, color=node_color, cp_type=node_type, fn_val=node_fn)
  
	for i, f in enumerate(features):
		nxg.add_edge(
			f.frm,
			f.to,
			color="#888888",
			width=2,
			feature_id=f.id,
		)
  
	return nxg