import pyct as ct

TYPE_STRING = {
    ct.REGULAR: "regular", # type: ignore
    ct.MINIMUM: "minima", # type: ignore
    ct.SADDLE: "saddle", # type: ignore
    ct.MAXIMUM: "maxima" # type: ignore
}

def get_type_string(feat):
    """Helper function to get type string for a RichFeature"""
    return TYPE_STRING[feat.type_frm] + "-" + TYPE_STRING[feat.type_to]