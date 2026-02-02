from basic_utils import RichFeature
from experiment import LossLandscapeExperiment

import streamlit as st

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

@st.cache_data(hash_funcs={LossLandscapeExperiment: LossLandscapeExperiment.__hash__})
def get_class_coverage(exp: LossLandscapeExperiment, count: int, class_idx: int) -> float:
    ds = exp.dataset
    total = ds.class_size_by_split[exp.split][ds.classes[int(class_idx)]]
    return round(count / total if total > 0 else 0.0, 4)

@st.cache_data(hash_funcs={LossLandscapeExperiment: LossLandscapeExperiment.__hash__, list: lambda x: hash(tuple(f.id for f in x)) if x and isinstance(x[0], RichFeature) else hash(tuple(x))})
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