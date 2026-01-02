import pyct as ct
from utils import RichFeature

def make_layout_off(path: str, feats: list[RichFeature], top_k: int, topofeats: ct.TopologicalFeatures):
	layoutCT = ct.LayoutCT(topofeats)
	layoutCT.layoutTree(top_k)
	locations = layoutCT.getNodeLocations()
	
	reindex = {}
	nodeids = {}
	ctr = 0
	for l in locations:
		reindex[l] = ctr
		nodeids[ctr] = l
		ctr += 1
  
	f = open(path, 'w')
	f.write("OFF\n")
	f.write(f"{len(locations)} {len(feats)} 0\n")
	
	for i in range(len(locations)):
		f.write(f"{locations[nodeids[i]].x} {locations[nodeids[i]].y} {locations[nodeids[i]].z}\n")
	
	for feat in feats:
		from_idx = reindex[feat.frm]
		to_idx = reindex[feat.to]
		f.write(f"2 {from_idx} {to_idx} {feat.fn_frm} {feat.fn_to} {feat.type_string.replace('-', ' ')}\n")
	

def make_layouts(ctree_path: str, topk: int = -1, thresh: float = 1e-6):
	feats = ct.TopologicalFeatures()
	feats.loadData(ctree_path)
	
	arc_feats, top_k = feats.getArcFeatures(topk, thresh)
	arc_feats = [RichFeature(id, f, feats.ctdata) for id, f in enumerate(arc_feats)]    
	make_layout_off(ctree_path + "_arc.off", arc_feats, top_k, feats)
	
	partitioned_feats, top_k = feats.getPartitionedExtremaFeatures(topk, thresh)
	partitioned_feats = [RichFeature(id, f, feats.ctdata) for id, f in enumerate(partitioned_feats)]
	make_layout_off(ctree_path + "_part.off", partitioned_feats, top_k, feats)