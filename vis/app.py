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

import pickle

from experiment import Dataset, LossLandscapeExperiment, find_all_datasets, find_all_experiments
from vis_utils import find_steady_simplification_states, compute_arc_features, compute_feature_map, load_preds, compute_feature_coverage_data
from components import render_tree_explorer, render_coverage_map, render_experiment_selector

path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../scripts')))
from chart_simplification_valleys import get_valley_vs_thresh

@st.cache_data
def find_available_experiments():
    datasets = find_all_datasets(st.session_state.datasets_dir)
    experiments = find_all_experiments(datasets, st.session_state.landscapes_dir, st.session_state.ct_dir)
    return datasets, experiments

def clear_features(id: int):
    st.session_state[f"feats_{id}"] = None

def compute_arcs_and_coverage(id: int, simpl: float):
    
    exp = st.session_state.get(f"selected_experiment_{id}", None)
    clear_features(id)
    st.session_state[f"computed_simpl_{id}"] = simpl
    
    with st.spinner(f"Computing arcs and coverage at simplification {simpl}..."):
        feats, data = compute_arc_features(exp, simpl)
        st.session_state[f"feats_{id}"] = feats
        st.session_state[f"ctdata_{id}"] = data
        node2feat = compute_feature_map(exp, feats, data)
        st.session_state[f"node2feat_{id}"] = node2feat
        preds = load_preds(exp)
        st.session_state[f"preds_{id}"] = preds
        compute_feature_coverage_data(exp, feats, preds)
        
# TODO: one idea would be to lift all keys into lambda functions so they are only evaluated when needed, and consistent throughout

def float_input(key: str, label: str, default: float, min_value = None, max_value = None, step = None, help: str = "") -> float:    
    val = st.text_input(label, value=str(default), key=key, help=help)

    if val is None or val.strip() == "":
        st.error("Value cannot be empty.")
        return default

    try:
        fval = float(val)
    except ValueError:
        st.error(f"Value must be a valid number.")
        return default

    if min_value is not None and fval < min_value:
        st.error(f"Value must be at least {min_value}.")
        return default

    if max_value is not None and fval > max_value:
        st.error(f"Value must be at most {max_value}.")
        return default

    if step is not None:
        # snap to nearest step
        fval = round((fval - (min_value if min_value is not None else 0)) / step) * step + (min_value if min_value is not None else 0)

    return fval

# TODO: we'd really like to fragment, but there's a really strange bug that makes the app unusable, so for now we just use a normal function
@st.fragment
def render_experiment(id: int, half_width: bool):

    _, experiments = find_available_experiments()
    
    a, b = st.columns([20, 1], vertical_alignment="bottom", gap="small")
    with a:
        # exp = st.selectbox("Select Experiment", options=[None] + experiments, format_func=lambda e: repr(e) if e is not None else "Select an experiment", key=f"selected_experiment_{id}")
        exp = render_experiment_selector(id, experiments)
    with b:
        with st.container(horizontal_alignment="right"):
            st.button("X", on_click=lambda: st.session_state.exp_ids.remove(id), key=f"remove_experiment_button_{id}")
    
    if exp is None:
        st.warning("No experiment selected.")
        return
    
    st.session_state[f"selected_experiment_{id}"] = exp

    with st.expander("Steady State Finder", expanded=False):
        threshs, num_min = valley_vs_thresh_data(exp, 0.0)
        
        data = pd.DataFrame({"Simplification Threshold": threshs, "Number of Minima": num_min}).sort_values(by="Number of Minima")

        chart = alt.Chart(data).mark_line().encode(
            x="Simplification Threshold",
            y="Number of Minima",
            tooltip=["Simplification Threshold", "Number of Minima"]
        ).interactive()
        
        st.altair_chart(chart, use_container_width=True)
        
        # state_tol = float_input(key=f"tol_steady_{id}", label="Steady Simplification State Coverage Threshold", default=0.002, min_value=0.0, max_value=100.0, step=0.001,
        #                         help="Minimum proportion of the range of thresholds for which the number of minima should remain constant for a simplification to be considered steady.")
        
        # try:
        #     state_tol = float(state_tol)
        # except ValueError:
        #     st.error("Tolerance must be a valid number.")
        #     st.session_state[f"tol_steady_{id}"] = "0.002"
        #     state_tol = 0.002
            
        # if state_tol < 0.0 or state_tol > 100.0:
        #     st.error("Tolerance must be between 0 and 100.")
        #     st.session_state[f"tol_steady_{id}"] = "0.002"
        #     state_tol = 0.002
        
        # steady_thresh, steady_minima = get_steady_simplification_states(exp, state_tol / 100.0)
        
        # if f"feats_{id}" not in st.session_state:
        #     st.session_state[f"feats_{id}"] = None
        
        # max_wt = get_steady_simplification_states(exp, 0.0)[0][-1][1]

        # df = pd.DataFrame({"Steady Threshold Start": [t[0] for t in steady_thresh], "Steady Threshold End": [t[1] for t in steady_thresh],
        #                    "Persistence": [t[1] - t[0] for t in steady_thresh], "Extremity": [steady_minima[i - 1] - n if i > 0 else 0 for i, n in enumerate(steady_minima)], 
        #                    "Number of Valleys": steady_minima})
        
        # st.dataframe(df)

    # pick start+eps of first steady state as default simplification ("denoising" justification)
    # simpl_def = steady_thresh[0][0] + (steady_thresh[0][1] - steady_thresh[0][0]) / 100 if len(steady_thresh) > 0 else max_wt / 2.0
    simpl_def = 0.0
    simpl_key = f"simpl_thresh_{id}"

    if simpl_key not in st.session_state:
        st.session_state[simpl_key] = simpl_def

    simpl = st.number_input("Simplification Threshold", max_value=500.0, step=0.0001, key=simpl_key, format="%0.32f")

    with st.container(horizontal=True, horizontal_alignment="center") as c:
        st.button("Reset Simplification", on_click=lambda: st.session_state.update({simpl_key: simpl_def}), key=f"reset_button_{id}")

        if st.button("Compute Tree and Coverage", key=f"compute_button_{id}"):
            compute_arcs_and_coverage(id, simpl)

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

def valley_vs_thresh_data(exp: LossLandscapeExperiment, tol: float) -> tuple[list[float], list[int]]:
    paths = exp.get_paths(st.session_state.landscapes_dir, st.session_state.ct_dir)
    
    thresh, num_min = get_valley_vs_thresh(paths["ctree"])
    # steady_thresh, steady_minima = find_steady_simplification_states(thresh, num_min, tol)

    return thresh, num_min

def main():
    if len(argv) < 4:
        print("Usage: streamlit run vis/app.py <ct_dir> <landscapes_dir> <datasets_dir>")
        exit(1)
        
    # if len(argv) < 4:
    #     print("Usage: streamlit run vis/app.py <ct_dir> <landscapes_dir> <datasets_dir> [state_file_to_load]")
    #     exit(1)

    # if len(argv) > 4:
    #     state_file = argv[4]
    #     if not os.path.exists(state_file):
    #         print(f"State file {state_file} does not exist.")
    #         exit(1)
    #     with open(state_file, "rb") as f:
    #         loaded_state = pickle.load(f)
    #         for k in loaded_state:
    #             st.session_state[k] = loaded_state[k]

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
        st.button("Add Experiment", on_click=add_exp, key="add_experiment_button")
    
    with col2:
        st.session_state.comparison_mode = st.checkbox("Comparison Mode", value=True, help="Render groups of experiments side-by-side for easier comparison.", key="comparison_mode_checkbox")

    if len(st.session_state.exp_ids) == 0:
        st.info("Add experiments to begin exploring the corresponding loss landscapes.")
        return
    
    def id2title(id: int) -> str:
        exp = st.session_state.get(f"selected_experiment_{id}", None)
        
        return repr(exp) if exp is not None else f"Unselected Experiment"
    
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
    main()
