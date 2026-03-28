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

from pyinstrument import Profiler

import argparse as ap

from experiment import Dataset, LossLandscapeExperiment, find_all_datasets, find_all_experiments, BertExperiment, find_all_bert_experiments
from basic_utils import get_filtered_cps_vs_thresh, get_order_and_weights, get_tree, get_labels, get_preds, get_partition, compute_arc_features, get_valley_vs_thresh
from tree_explorer import render_tree_explorer
from coverage_view import render_coverage_map
from components import render_experiment_selector

path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../scripts')))

@st.cache_resource
def find_available_experiments():
    if st.session_state.get("is_bert", False):
        experiments = find_all_bert_experiments(
            st.session_state.landscapes_dir, st.session_state.ct_dir
        )
        return {}, experiments
    datasets = find_all_datasets(st.session_state.datasets_dir)
    experiments = find_all_experiments(datasets, st.session_state.landscapes_dir, st.session_state.ct_dir)
    return datasets, experiments

def clear_features(id: int):
    st.session_state[f"feats_{id}"] = None

def load_tree(exp: LossLandscapeExperiment):
    get_tree(exp)
    get_preds(exp)
    get_labels(exp)
    get_order_and_weights(exp)
    get_partition(exp)

def compute_arcs_and_coverage(id: int, simpl: float):
    exp = st.session_state.get(f"selected_experiment_{id}", None)
    clear_features(id)
    st.session_state[f"computed_simpl_{id}"] = simpl
    
    with st.spinner(f"Computing arcs and coverage at simplification {simpl}..."):
        feats = compute_arc_features(exp, simpl)
        st.session_state[f"feats_{id}"] = feats
        
def render_experiment(id: int, half_width: bool):

    @st.fragment
    def valley_simpl_chart(exp: LossLandscapeExperiment):
        use_old = st.checkbox("Use Old Valley Computation", value=False, help="Use the previous method for computing valleys vs simplification thresholds.", key=f"use_old_valley_comp_{id}")
        
        if use_old:        
            threshs, num_min = get_valley_vs_thresh(exp)
                        
            data = pd.DataFrame({"Simplification Threshold": threshs, "Number of Minima": num_min}).sort_values(by="Number of Minima", ascending=False)

            interval = alt.selection_interval(encodings=['x'], bind='scales')
            chart = alt.Chart(data).mark_line(interpolate='step-after').encode(
                x="Simplification Threshold",
                y=alt.Y("Number of Minima", scale=alt.Scale(type="log")),
                tooltip=["Simplification Threshold", "Number of Minima"],
            ).add_params(interval)
                    
            st.altair_chart(chart, use_container_width=True)
        else:
            with st.container(horizontal=True, horizontal_alignment="center"):
                fn_start_min = st.number_input("Function Start Min", value=0.0, key=f"valley_fnstart_min_{id}")
                fn_start_max = st.number_input("Function Start Max", value=10000.0, key=f"valley_fnstart_max_{id}")
                fn_end_min = st.number_input("Function End Min", value=0.0, key=f"valley_fnend_min_{id}")
                fn_end_max = st.number_input("Function End Max", value=10000.0, key=f"valley_fnend_max_{id}")
                
            with st.container(horizontal=True, horizontal_alignment="center", vertical_alignment="bottom"):
                types = st.multiselect("Arc Types", options=[ct.MINIMUM, ct.SADDLE, ct.MAXIMUM], format_func=lambda t: {ct.MINIMUM: "Minima", ct.SADDLE: "Saddles", ct.MAXIMUM: "Maxima"}[t], default=[ct.MINIMUM], key=f"valley_cp_types_{id}") # type: ignore
                log_scale = st.checkbox("Log Scale", value=True, key=f"valley_log_scale_{id}")
            
            threshs, counts = get_filtered_cps_vs_thresh(exp, (fn_start_min, fn_start_max), (fn_end_min, fn_end_max), types)
            
            data = pd.DataFrame({"Simplification Threshold": threshs, "Count": counts}).sort_values(by="Count", ascending=False)

            interval = alt.selection_interval(encodings=['x'], bind='scales')
            chart = alt.Chart(data).mark_line(interpolate='step-after').encode(
                x="Simplification Threshold",
                y=alt.Y("Count", scale=alt.Scale(type="log" if log_scale else "linear")),
                tooltip=["Simplification Threshold", "Count"],
            ).add_params(interval)
                    
            st.altair_chart(chart, use_container_width=True)
            

    def remove_experiment():
        st.session_state.exp_ids.remove(id)
        st.rerun(scope="app")

    _, experiments = find_available_experiments()
    
    a, b = st.columns([20, 1], vertical_alignment="bottom", gap="small")
    with a:
        # exp = st.selectbox("Select Experiment", options=[None] + experiments, format_func=lambda e: repr(e) if e is not None else "Select an experiment", key=f"selected_experiment_{id}")
        exp = render_experiment_selector(id, experiments)
    with b:
        with st.container(horizontal_alignment="right"):
            if st.button("X", key=f"remove_experiment_button_{id}"):
                remove_experiment()

    if exp is None:
        st.warning("No experiment selected.")
        return
    
    load_tree(exp)
    st.session_state[f"selected_experiment_{id}"] = exp

    with st.expander("Steady State Finder", expanded=False):
        valley_simpl_chart(exp)
    
    simpl_key = f"simpl_thresh_{id}"
    simpl = st.number_input("Simplification Threshold", value=0.0, max_value=500.0, step=0.0001, key=simpl_key, format="%0.32f")

    with st.container(horizontal=True, horizontal_alignment="center") as c:
        if st.button("Compute Tree and Coverage", key=f"compute_button_{id}"):
            compute_arcs_and_coverage(id, simpl)

    @st.fragment
    def main_body():
        if half_width:
            tree_container = st.container()
            cov_container = st.container()
        else:
            tree_container, cov_container = st.columns([3, 2])

        with tree_container:
            st.subheader("Tree Explorer")

            render_tree_explorer(id)

        with cov_container:
            st.subheader("Coverage Map")

            render_coverage_map(id)
            
    main_body()

def _detect_bert_from_ct_dir(ct_dir: str) -> bool:
    """Auto-detect BERT mode: ctree files live at depth 1 inside ct_dir (not depth 2)."""
    if not os.path.isdir(ct_dir):
        return False
    for name in os.listdir(ct_dir):
        sub = os.path.join(ct_dir, name)
        if os.path.isdir(sub):
            for fname in os.listdir(sub):
                if fname.endswith(".order.dat"):
                    return True  # ctree file found one level deep → BERT layout
    return False


def main():
    parser = ap.ArgumentParser(description="Streamlit app for exploring contour trees and coverage.", prefix_chars="+")
    parser.add_argument("ct_dir", type=str, help="Directory containing contour tree files.")
    parser.add_argument("landscapes_dir", type=str, help="Directory containing landscape data files.")
    parser.add_argument("datasets_dir", type=str, help="Directory containing dataset files.")
    
    # streamlit doesn't allow flags, so we use "+flagname" to indicate boolean flags
    parser.add_argument("+model_eq_dataset", action="store_true", help="Whether to treat model name as equivalent to dataset name when loading experiments.")
    parser.add_argument("+no_preds", action="store_true", help="Whether to skip loading predictions, which can speed up loading but disable coverage computations.")
    parser.add_argument("+bert", action="store_true", help="Force BERT mode. Auto-detected from ct_dir structure when omitted.")
    parser.add_argument("+complexes_dir", type=str, default="", help="Directory containing BERT KNN complexes (token_coords_*.pt files). Required for BERT mode.")
    parser.add_argument("+conll_dir", type=str, default="", help="Directory containing CoNLL-2003 eng.{train,testa,testb} files. Used for sentence context in the BERT data explorer.")

    args = parser.parse_args()
    
    st.set_page_config(layout="wide")

    st.session_state.ct_dir = args.ct_dir.strip("/\\")
    st.session_state.landscapes_dir = args.landscapes_dir.strip("/\\")
    st.session_state.datasets_dir = args.datasets_dir.strip("/\\")
    
    st.session_state.model_eq_dataset = args.model_eq_dataset
    st.session_state.no_preds = args.no_preds

    # Determine BERT mode: explicit flag takes priority; fall back to auto-detection.
    is_bert = args.bert or _detect_bert_from_ct_dir(st.session_state.ct_dir)
    st.session_state.is_bert = is_bert
    st.session_state.complexes_dir = args.complexes_dir.strip("/\\") if args.complexes_dir else ""
    if args.conll_dir:
        conll_dir = args.conll_dir.strip("/\\")
    else:
        conll_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../datasets/conll/data/conll2003')
    st.session_state.conll_dir = conll_dir
    
    datasets, experiments = find_available_experiments()

    if "state_setup_done" not in st.session_state:
        st.session_state.state_setup_done = True
        st.session_state.selected_experiments = []
        
        print(f"Found {len(datasets)} datasets and {len(experiments)} functions")

    st.title("TOPPLE: Topology-powered Latent-space Exploration")
    st.divider()

    # render_save()

    col1, col2 = st.columns([5, 1], vertical_alignment="bottom")
    
    if st.session_state.get("last_exp_id", None) is None:
        st.session_state.last_exp_id = -1
    
    if st.session_state.get("exp_ids", None) is None:
        st.session_state.exp_ids = []
   
    def add_exp():
        st.session_state.last_exp_id += 1
        st.session_state.exp_ids.append(st.session_state.last_exp_id)
    
    with col1:
        st.button("Add Function", on_click=add_exp, key="add_experiment_button")
    
    with col2:
        st.session_state.comparison_mode = st.checkbox("Comparison Mode", value=True, help="Render groups of experiments side-by-side for easier comparison.", key="comparison_mode_checkbox")

    if len(st.session_state.exp_ids) == 0:
        st.info("Add functions to begin exploring the corresponding latent spaces.")
        return
    
    def id2title(id: int) -> str:
        exp = st.session_state.get(f"selected_experiment_{id}", None)
        
        return repr(exp) if exp is not None else f"Unselected Function"
    
    if st.session_state.comparison_mode:
        for i in range(0, len(st.session_state.exp_ids), 2):
            first = st.session_state.exp_ids[i]
            second = st.session_state.exp_ids[i+1] if i+1 < len(st.session_state.exp_ids) else None
            
            title_text = f"{id2title(first)}" if second is None else f"{id2title(first)} **||** {id2title(second)}"
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
        for exp in st.session_state.exp_ids:
            with st.expander(id2title(exp), expanded=True):
                render_experiment(exp, half_width=False)

if __name__ == "__main__":
    # profiler = Profiler()
    # profiler.start()
    main()
    # profiler.stop()
    # 
    # run_id = st.session_state.get("run_id", 0)
    # with open(f"app_profile_{run_id}.html", "w") as f:
    #     f.write(profiler.output_html())
    
    # st.session_state["run_id"] = run_id + 1    
