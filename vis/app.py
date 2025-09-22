"""
Contains a streamlit app allowing exploration of the contour tree's arcs and how well they cover the underlying dataset.

Usage:
streamlit run app.py <ct_dir> <landscapes_dir> <datasets_dir>
"""

from sys import argv
import os

import pyct as ct
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import streamlit as st
import pandas as pd
import altair as alt
import numpy as np

from experiment import Dataset, LossLandscapeExperiment, find_all_datasets, find_all_experiments

@st.cache_data
def find_available_experiments(ct_dir, landscapes_dir, datasets_dir):
    datasets = find_all_datasets(datasets_dir)
    experiments = find_all_experiments(datasets, landscapes_dir, ct_dir)
    return datasets, experiments

def render_experiment(experiment: LossLandscapeExperiment):
    arc_col, cov_col = st.columns([3, 2])
    with arc_col:
        st.subheader("Arc Explorer")
        
        
    with cov_col:
        st.subheader("Coverage Map")

def main():
    if len(argv) != 4:
        print("Usage: streamlit run vis/app.py <ct_dir> <landscapes_dir> <datasets_dir>")
        exit(1)

    st.set_page_config(layout="wide")

    ct_dir = argv[1].strip("/\\")
    landscapes_dir = argv[2].strip("/\\")
    datasets_dir = argv[3].strip("/\\")

    datasets, experiments = find_available_experiments(ct_dir, landscapes_dir, datasets_dir)

    if "state_setup_done" not in st.session_state:
        st.session_state.state_setup_done = True
        st.session_state.selected_experiments = []
        
        print(f"Found {len(datasets)} datasets and {len(experiments)} experiments")
        print("\n".join([repr(exp) for exp in experiments]))

    st.title("Loss Landscape Arc Explorer")
    st.divider()

    st.session_state.selected_experiments = st.multiselect("Select experiments", options=experiments, format_func=lambda d: repr(d), key="experiment_selector", max_selections=8)

    for exp in st.session_state.selected_experiments:
        with st.expander(repr(exp)):
            render_experiment(exp)

    # for i in range(0, len(st.session_state.selected_experiments), 2):
    #     first = st.session_state.selected_experiments[i]
    #     second = st.session_state.selected_experiments[i+1] if i+1 < len(st.session_state.selected_experiments) else None

    #     title_text = f"{repr(first)}" if second is None else f"{repr(first)} | {repr(second)}"
    #     with st.expander(title_text):
    #         if second is None:
    #             with st.container():
    #                 render_experiment(first)
    #         else:
    #             cols = st.columns([1, 1])
    #             with cols[0]:
    #                 render_experiment(first)
    #             with cols[1]:
    #                 render_experiment(second)

if __name__ == "__main__":
    main()
