"""
Contains a streamlit app allowing exploration of the contour tree's arcs and how well they cover the underlying dataset.

Usage:
streamlit run app.py <ct_dir> <landscapes_dir> <datasets_dir>
"""

from sys import argv, path
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
from vis_utils import find_steady_simplification_states, compute_arc_features, compute_feature_map, load_preds, compute_feature_coverage_data
from components import render_arc_explorer, render_coverage_map

path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../scripts')))
from chart_simplification_valleys import get_valley_vs_thresh

@st.cache_data
def find_available_experiments():
    datasets = find_all_datasets(st.session_state.datasets_dir)
    experiments = find_all_experiments(datasets, st.session_state.landscapes_dir, st.session_state.ct_dir)
    return datasets, experiments

def clear_features(exp: LossLandscapeExperiment):
    st.session_state[f"feats_{repr(exp)}"] = None

def compute_arcs_and_coverage(exp: LossLandscapeExperiment, simpl: float):
    clear_features(exp)
    
    st.session_state[f"computed_simpl_{repr(exp)}"] = simpl
    
    with st.spinner(f"Computing arcs and coverage at simplification {simpl}..."):
        feats = compute_arc_features(exp, simpl)
        st.session_state[f"feats_{repr(exp)}"] = feats
        node2feat = compute_feature_map(exp, feats)
        st.session_state[f"node2feat_{repr(exp)}"] = node2feat
        preds = load_preds(exp)
        st.session_state[f"preds_{repr(exp)}"] = preds
        compute_feature_coverage_data(exp, feats, preds)
        
# TODO: one idea would be to lift all keys into lambda functions so they are only evaluated when needed, and consistent throughout

def render_experiment(exp: LossLandscapeExperiment, half_width: bool):

    with st.expander("Steady State Finder", expanded=False):
        state_tol = st.slider("Steady Simplification State Coverage Threshold", min_value=0.0, max_value=100.0, value=0.002, step=0.001, key=f"tol_slider_{repr(exp)}", format="%0.3f%%",
                                help="Minimum proportion of the range of thresholds for which the number of minima should remain constant for a simplification to be considered steady.")
        steady_thresh, steady_minima = get_steady_simplification_states(exp, state_tol / 100.0)
        
        if f"feats_{repr(exp)}" not in st.session_state:
            st.session_state[f"feats_{repr(exp)}"] = None
        
        max_wt = get_steady_simplification_states(exp, 0.0)[0][-1][1]
                
        st.dataframe(pd.DataFrame({"Steady Threshold Start": [t[0] for t in steady_thresh], "Steady Threshold End": [t[1] for t in steady_thresh], "Number of Valleys": steady_minima}))
        
    # pick start+eps of first steady state as default simplification ("denoising" justification) 
    simpl_def = steady_thresh[0][0] + (steady_thresh[0][1] - steady_thresh[0][0]) / 100 if len(steady_thresh) > 0 else max_wt / 2.0
    simpl_key = f"simpl_thresh_{repr(exp)}"

    if simpl_key not in st.session_state:
        st.session_state[simpl_key] = simpl_def

    simpl = st.number_input("Select Simplification Threshold", max_value=max_wt, step=0.0001, key=simpl_key, format="%0.32f")

    with st.container(horizontal=True, horizontal_alignment="center") as c:
        st.button("Reset Simplification", on_click=lambda: st.session_state.update({simpl_key: simpl_def}), key=f"reset_button_{repr(exp)}")

        if st.button("Compute Arcs and Coverage", key=f"compute_button_{repr(exp)}"):
            compute_arcs_and_coverage(exp, simpl)

    if half_width:
        arc_container = st.container()
        cov_container = st.container()
    else:
        arc_container, cov_container = st.columns([3, 2])

    with arc_container:
        st.subheader("Arc Explorer")

        render_arc_explorer(exp)

    with cov_container:
        st.subheader("Coverage Map")

        render_coverage_map(exp)

def get_steady_simplification_states(exp: LossLandscapeExperiment, tol: float) -> tuple[list[tuple[float, float]], list[int]]:
    paths = exp.get_paths(st.session_state.landscapes_dir, st.session_state.ct_dir)
    
    thresh, num_min = get_valley_vs_thresh(paths["ctree"])
    steady_thresh, steady_minima = find_steady_simplification_states(thresh, num_min, tol)
    
    return steady_thresh, steady_minima
    
def main():
    if len(argv) != 4:
        print("Usage: streamlit run vis/app.py <ct_dir> <landscapes_dir> <datasets_dir>")
        exit(1)

    st.set_page_config(layout="wide")

    st.session_state.ct_dir = argv[1].strip("/\\")
    st.session_state.landscapes_dir = argv[2].strip("/\\")
    st.session_state.datasets_dir = argv[3].strip("/\\")
    
    datasets, experiments = find_available_experiments()

    if "state_setup_done" not in st.session_state:
        st.session_state.state_setup_done = True
        st.session_state.selected_experiments = []
        
        print(f"Found {len(datasets)} datasets and {len(experiments)} experiments")
        print("\n".join([repr(exp) for exp in experiments]))

    st.title("Topological Loss Landscape Explorer")
    st.divider()

    col1, col2 = st.columns([5, 1], vertical_alignment="bottom")
    
    with col1:
        st.session_state.selected_experiments = st.multiselect("Select experiments", options=experiments, format_func=lambda d: repr(d), key="experiment_selector", max_selections=8)
    
    with col2:
        st.session_state.comparison_mode = st.checkbox("Comparison Mode", value=True, help="Render groups of experiments side-by-side for easier comparison.", key="comparison_mode_checkbox")

    if st.session_state.selected_experiments is None or len(st.session_state.selected_experiments) == 0:
        st.info("Select one or more experiments to begin exploring the corresponding loss landscapes.")
        return
    
    if st.session_state.comparison_mode:
        for i in range(0, len(st.session_state.selected_experiments), 2):
            first = st.session_state.selected_experiments[i]
            second = st.session_state.selected_experiments[i+1] if i+1 < len(st.session_state.selected_experiments) else None

            title_text = f"{repr(first)}" if second is None else f"{repr(first)} **||** {repr(second)}"
            with st.expander(title_text, expanded=True):
                if second is None:
                    with st.container():
                        render_experiment(first, half_width=False)
                else:
                    cols = st.columns([1, 1])
                    with cols[0]:
                        render_experiment(first, half_width=True)
                    with cols[1]:
                        render_experiment(second, half_width=True)
    else:
        for exp in st.session_state.selected_experiments:
            with st.expander(repr(exp), expanded=True):
                render_experiment(exp, half_width=False)


if __name__ == "__main__":
    main()
