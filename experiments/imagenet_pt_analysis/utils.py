import pyct as ct

from experiment import LossLandscapeExperiment, find_all_datasets, find_all_experiments

TYPE_STRING = {
	ct.REGULAR: "REGULAR",
	ct.MINIMUM: "MINIMA",
	ct.SADDLE: "SADDLE",
	ct.MAXIMUM: "MAXIMA"
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
		

def compute_arc_features(exp, simpl: float):        
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

datasets = find_all_datasets("datasets/")
experiments = find_all_experiments(datasets, data_path, stree_path)