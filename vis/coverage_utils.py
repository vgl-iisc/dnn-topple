from experiment import LossLandscapeExperiment, BertExperiment

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

@st.cache_data(hash_funcs={LossLandscapeExperiment: LossLandscapeExperiment.__hash__, BertExperiment: BertExperiment.__hash__})
def get_class_coverage(exp: LossLandscapeExperiment, count: int, class_idx: int) -> float:
    ds = exp.dataset
    total = ds.class_size_by_split[exp.split][ds.classes[int(class_idx)]]
    return round(count / total if total > 0 else 0.0, 4)