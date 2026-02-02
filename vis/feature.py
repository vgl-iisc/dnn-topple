import pyct as ct

TYPE_STRING = {
    ct.REGULAR: "regular", # type: ignore
    ct.MINIMUM: "minima", # type: ignore
    ct.SADDLE: "saddle", # type: ignore
    ct.MAXIMUM: "maxima" # type: ignore
}

class RichFeature:
    def __init__(self, id: int, feature: ct.Feature, data: ct.ContourTreeData): # type: ignore
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