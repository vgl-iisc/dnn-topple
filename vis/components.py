from vis_utils import RichFeature, class2color, get_class_coverage, load_preds, compute_tree_graph
from dataset_loader import load_dataset

import pyvis.network as net

import os

from experiment import LossLandscapeExperiment, Dataset
import streamlit as st
import streamlit.components.v1 as components

import networkx as nx
import altair as alt
import pandas as pd

import numpy as np
import pyct as ct

import pickle
import datetime

ARC_X_AXIS_TYPE = {
    "id": "Feature ID",
    "index": "Index",
    "sorted": "Sorted Index",
}

ARC_X_AXIS_SORTING = {
    "fstart": "Feature Start",
    "fend": "Feature End",
    "pers": "Feature Persistence",
    "volume": "Feature Volume",
    "majority class": "Majority Class",
    "major class coverage": "Major Class Coverage"
}

ALLOWED_TYPE_SETS = {
    "valleys": {(ct.MINIMUM, ct.SADDLE), (ct.MINIMUM, ct.MAXIMUM), (ct.MINIMUM, ct.REGULAR)},
    "peaks": {(ct.SADDLE, ct.MAXIMUM), (ct.MINIMUM, ct.MAXIMUM), (ct.REGULAR, ct.MAXIMUM)},
    "saddle-saddle": {(ct.SADDLE, ct.SADDLE)},
    "regular": {(ct.REGULAR, ct.REGULAR), (ct.REGULAR, ct.SADDLE), (ct.REGULAR, ct.MINIMUM), (ct.REGULAR, ct.MAXIMUM), 
                (ct.SADDLE, ct.REGULAR), (ct.MINIMUM, ct.REGULAR), (ct.MAXIMUM, ct.REGULAR)},
}

def render_tree_explorer(id: int):
    
    exp = st.session_state.get(f"selected_experiment_{id}", None)
    
    def arc_explorer_plot(features: list[RichFeature], axis_type: str | list[str]) -> alt.Chart:
        
        max_pers = max([f.pers for f in features]) if len(features) > 0 else 1.0
        
        df = pd.DataFrame([{
            "id": f.id,
            "fstart": f.fn_frm,
            "fend": f.fn_to if f.pers > max_pers / 50 else f.fn_frm + max_pers / 50,
            "ftype": f.type_string,
            "pers": f.pers,
            "volume": f.size,
            "logvol": np.log1p(f.size),
            "majority class": exp.dataset.classes[f.majority_class],
            "major class size": f.major_class_size,
            "major class coverage": get_class_coverage(exp, f.major_class_size, f.majority_class)
        } for i, f in enumerate(features)])

        x_axis_item = "idx"
        x_title = "Index"

        if axis_type == "id":
            x_axis_item = "id"
            x_title = "Feature ID"
            
        if isinstance(axis_type, list) and len(axis_type) > 0:
            df = df.sort_values(by=axis_type + ["id"]).reset_index(drop=True)
            x_title = "Sorted Index"

        df["idx"] = df.index + 1
        x_axis = alt.X(x_axis_item, title=x_title)

        brush = alt.selection_point(name="arcs", fields=['id'], on="click")
        
        zoom = alt.selection_interval(bind='scales')

        # selection = alt.selection_interval(
        #     name="drag",
        #     on="[mousedown[event.shiftKey], mouseup] > mousemove",
        #     translate="[mousedown[event.shiftKey], mouseup] > mousemove!",
        #     encodings=['x']
        # )
        
        # Create a bar chart using Altair
        plot = alt.Chart(df).mark_line().encode(
            x=x_axis,
            y=alt.Y('fstart'+":Q", title="Loss Start"),
            y2=alt.Y2('fend'+":Q", title="Loss End"),
            tooltip=[alt.Tooltip('id', title="Feature ID"), alt.Tooltip('fstart', title="Loss Start"), alt.Tooltip('fend', title="Loss End"), 
                     alt.Tooltip('ftype', title="Feature Type"), alt.Tooltip('pers', title="Persistence"), alt.Tooltip('volume', title="Volume"),
                     alt.Tooltip('majority class', title="Majority Class"), alt.Tooltip('major class size', title="Major Class Size"), alt.Tooltip('major class coverage', title="Majority Class Coverage")],
            opacity=alt.condition(brush, alt.value(1), alt.value(0.2)),
            color=alt.Color('logvol', scale=alt.Scale(scheme='yelloworangered')).title(None).legend(None),
            strokeWidth=alt.value(5)
        ).add_params(
            zoom, brush
        )
        
        return plot
    
    def arc_explorer(features, allowed_types):
        simpl: float = st.session_state[f"computed_simpl_{id}"]
                        
        with st.container(horizontal=True, horizontal_alignment="center") as c:
            x_type = st.selectbox("X-Axis", options=list(ARC_X_AXIS_TYPE.keys()), index=2, format_func=lambda x: ARC_X_AXIS_TYPE[x], key=f"x_axis_selector_{id}")
        
        if x_type == "sorted":
            x_type = st.multiselect("Sort by", options=list(ARC_X_AXIS_SORTING.keys()), default=["majority class", "fstart"], key=f"sort_type_selector_{id}")
        
        filtered_features = [f for f in features if (f.type_frm, f.type_to) in allowed_types]
        st.session_state[f"filtered_feats_{id}"] = filtered_features
        
        if len(filtered_features) == 0:
            st.warning("No features of the selected types.")
            return

        st.write(f"Rendering {len(filtered_features)} features at simplification {simpl}")
        
        plot = arc_explorer_plot(filtered_features, x_type)
        
        selection_dict = st.altair_chart(plot, use_container_width=True, on_select="rerun", key=f"arc_explorer_plot_{id}")
        
        if selection_dict is not None and "selection" in selection_dict and "arcs" in selection_dict["selection"]:
            st.session_state[f"explorer_selected_arcs_{id}"] = [rec["id"] for rec in selection_dict["selection"]["arcs"]]
        else:
            st.session_state[f"explorer_selected_arcs_{id}"] = []
            
    def tree_view(features, allowed_types):
        
        steiner_mode = "None"
        ego_origin = "None"
        ego_radius = 0
        
        with st.container(horizontal=True, horizontal_alignment="center", vertical_alignment="bottom", gap="medium"):
            simpl_mode = st.selectbox("Simplification Mode", key=f"simpl_mode_selector_{id}", options=["Feature Types", "Steiner", "Ego"], index=0, help="Choose how to simplify the tree for visualization.")
            
            if simpl_mode == "Steiner":
                steiner_mode = st.selectbox("Steiner Tree", key=f"steiner_selector_{id}", options=["Minima", "Maxima"], index=0, help="Use Steiner tree to include important critical points in the tree view.")
            if simpl_mode == "Ego":
                ego_origin = st.selectbox("Ego Origin", key=f"ego_origin_selector_{id}", options=["Minima", "Maxima"], index=0, help="Choose the type of critical point to center the ego tree around.")
                ego_radius = st.slider("Ego Radius", key=f"ego_radius_slider_{id}", min_value=1, max_value=50, value=2, help="Radius of the ego tree to display.")
            
            # TODO: saddle simplification is killing arcs, need to fix that
            # saddle_simpl = st.toggle(f"Saddle Simplification", key=f"saddle_simpl_toggle_{id}", value=False, help="Remove chains of saddle-saddle connections for a cleaner tree view.")
            saddle_simpl = False
        
        valid_features = features

        if simpl_mode == "Feature Types":
            valid_features = [f for f in features if (f.type_frm, f.type_to) in allowed_types]

        if len(valid_features) == 0:
            st.warning("No features of the selected types.")
            return
        
        if st.button("Recompute Tree Graph", key=f"recompute_tree_graph_{id}"):
            st.session_state[f"tree_graph_{id}"] = compute_tree_graph(exp, valid_features, steiner_mode, ego_origin, ego_radius, saddle_simpl)

        gnx = st.session_state.get(f"tree_graph_{id}", None)

        if gnx is None:
            st.text("Compute tree graph to begin.")
            return

        st.text(f"{len(valid_features)} features selected. Rendering {len(gnx.edges)} features after processing. Connected: {nx.is_connected(gnx.to_undirected())}")
        
        g = net.Network(height="600px", width="100%", directed=True)
        g.from_nx(gnx)
        
        html = g.generate_html()
        components.html(html, height=600)
        
    features: list[RichFeature] | None = st.session_state.get(f"feats_{id}", None)

    if features is None:
        st.info("Compute tree and coverage to begin.")
        return
    
    types = st.multiselect("Feature types", options=list(ALLOWED_TYPE_SETS.keys()), default=["valleys"], key=f"feature_type_selector_{id}")
    
    allowed_types = set()
    for t in types:
        allowed_types = allowed_types.union(ALLOWED_TYPE_SETS.get(t, set()))
    
    arcs, tree = st.tabs(["Arc Explorer", "Tree View"])
    
    with arcs:
        arc_explorer(features, allowed_types)
    
    with tree:
        tree_view(features, allowed_types)

F2C_VIEWS = ["plain", "correctness", "class-wise confusion"]

def render_coverage_map(id: int):
    
    exp = st.session_state.get(f"selected_experiment_{id}", None)
    
    def f2c_plot(feats: list[RichFeature], view: str, show_what: str, freeze_top: bool) -> alt.Chart | None:
        classes = exp.dataset.classes
        colors = {cls: class2color(i) for i, cls in enumerate(classes)}
        node2label = exp.dataset.labels_by_split[exp.split]
        preds = st.session_state.get(f"preds_{id}", {})
    
        if len(feats) == 0:
            return None
    
        node_feats = set.union(*[set(zip(f.members, [f] * f.size)) for f in feats])
        
        max_class_size = max([exp.dataset.class_size_by_split[exp.split][cls] for cls in exp.dataset.classes])
        
        data = []
        for (node, feat) in node_feats:
            true_label = node2label[node]
            pred_label = preds[node]
            
            data.append({
                "Feature ID": feat.id,
                "True Label": true_label,
                "Predicted Label": pred_label,
                "True Class": exp.dataset.classes[true_label],
                "Predicted Class": exp.dataset.classes[pred_label],
                "Correct": true_label == pred_label,
            })
                
        if len(data) == 0:
            return None
                
        df = pd.DataFrame(data)
        
        grouping = ["True Label", "True Class"]
        color = None
        pred_tooltip = None
        
        if view == "correctness":
            grouping += ["Correct"]
            color = alt.Color('Correct:N', scale=alt.Scale(domain=[True, False], range=["#4CAF50", "#F44336"]), title="Correctness")
            pred_tooltip = alt.Tooltip('Correct', title="Prediction Correctness")
        elif view == "class-wise confusion":
            grouping += ["Predicted Label", "Predicted Class"]
            color = alt.Color('Predicted Class:N', scale=alt.Scale(domain=list(colors.keys()), range=list(colors.values())), title="Prediction")
            pred_tooltip = alt.Tooltip('Predicted Class', title="Predicted Class")

        df = df.groupby(grouping).size().reset_index(name='count').sort_values(by=["count"], ascending=False)

        df["Proportion"] = df['count'] / df['count'].sum()
        df["Coverage"] = [get_class_coverage(exp, entry["count"], entry["True Label"]) for _, entry in df.iterrows()]
        
        scale = alt.Scale()
        
        if show_what == "Coverage":
            y = alt.Y('Coverage', title="Class Coverage", axis=alt.Axis(format='%'))
            if freeze_top:
                scale = alt.Scale(domain=[0, 1.0])
        elif show_what == "Proportions":
            y = alt.Y('Proportion', title="Proportion", axis=alt.Axis(format='%'))
            if freeze_top:
                scale = alt.Scale(domain=[0, 1.0])
        else:
            y = alt.Y('count', title="Number of Points")
            if freeze_top:
                scale = alt.Scale(domain=[0, max_class_size])

        y = y.scale(scale)
        
        plot = alt.Chart(df).mark_bar().encode(
            x=alt.X('True Class:N', title="True Class"),
            y=y,
            color=color if color is not None else alt.Color('True Class:N', scale=alt.Scale(domain=list(colors.keys()), range=list(colors.values())), title="True Class"),
            tooltip=[alt.Tooltip('count', title="Number of Points"), alt.Tooltip('True Class', title="True Class"), 
                     alt.Tooltip('Coverage', title="Class Coverage").format('.3%'), alt.Tooltip('Proportion', title="Proportion").format('.3%')] + ([pred_tooltip] if pred_tooltip is not None else []),
            order=alt.Order('True Class:N', sort='ascending')
        ).properties(
            width=600,
            height=400
        )
        
        return plot
        
    
    features: list[RichFeature] | None = st.session_state.get(f"feats_{id}", None)
    
    if features is None:
        st.info("Compute arcs and coverage to explore coverage map.")
        return
    
    filtered_features: list[RichFeature] = st.session_state.get(f"filtered_feats_{id}", [])
    selection: list[int] = st.session_state.get(f"explorer_selected_arcs_{id}", [])
    st.session_state[f"dataset_obj_{id}"] = st.session_state.get(f"dataset_obj_{id}", load_dataset(exp))
    ds = st.session_state[f"dataset_obj_{id}"]

    if len(selection) == 0:
        selection = [f.id for f in filtered_features]

    f2c, acc, c2f, f2m, datex = st.tabs(["Feature to Class", "Accuracy", "Class to Feature", "Feature to Members", "Data Explorer"])
    selected_features = [features[fid] for fid in selection]
    
    with f2c:        
        with st.container(horizontal=True, vertical_alignment="top", horizontal_alignment="left"):
            fine_grained = st.selectbox("View", key=f"fine_grained_{id}", options=F2C_VIEWS, format_func=lambda x: x.title())
            show_what = st.selectbox("Y-axis", key=f"show_what_{id}", options=["Counts", "Proportions", "Coverage"], index=2)
            freeze_top = st.checkbox("Freeze top", key=f"freeze_top_{id}", value=True)
        
        plot = f2c_plot(selected_features, fine_grained, show_what, freeze_top)
        
        if plot is None:
            st.warning("No datapoints in selection.")
        else:
            st.altair_chart(plot, use_container_width=True, key=f"f2c_plot_{id}")
        
    with acc:
        total_points = sum([f.size for f in selected_features])
        correct_points = 0
        preds = st.session_state.get(f"preds_{id}", {})
        node2label = exp.dataset.labels_by_split[exp.split]
        
        for f in selected_features:
            correct_points += sum([1 for n in f.members if preds[n] == node2label[n]])
                    
        accuracy = correct_points / total_points if total_points > 0 else 0.0
        
        st.write(f"Accuracy over selected features: **{accuracy * 100.0:0.3f}%** ({correct_points} / {total_points})")
        
        confusion = pd.DataFrame(0, index=exp.dataset.classes, columns=exp.dataset.classes)
        for f in selected_features:
            for n in f.members:
                true_cls = exp.dataset.classes[node2label[n]]
                pred_cls = exp.dataset.classes[preds[n]]
                confusion.at[true_cls, pred_cls] += 1
                    
        heat = alt.Chart(confusion.reset_index().melt(id_vars='index')).mark_rect().encode(
            x=alt.X('variable:N', title="Predicted Class"),
            y=alt.Y('index:N', title="True Class"),
            color=alt.Color('value:Q', scale=alt.Scale(scheme='blues', type="symlog"), title="Number of Points"),
            tooltip=[alt.Tooltip('value:Q', title="Number of Points"), alt.Tooltip('index:N', title="True Class"), alt.Tooltip('variable:N', title="Predicted Class")]
        ).properties(
            width=400,
            height=400
        )
        
        st.altair_chart(heat, use_container_width=True, key=f"confusion_heatmap_{id}")
        
    with c2f:
        st.info("Class to Feature view not implemented yet.")
        
    with f2m:
        with st.container(height=600):
            for f in selected_features:
                with st.expander(f"**Feature ID {f.id}** (Type: {f.type_string}, Persistence: {f.pers:0.4f}, Size: {f.size}, Majority Class: {exp.dataset.classes[f.majority_class]})"):
                    st.text(', '.join([str(n) for n in f.members]))
        
    with datex:
        cs_idx = st.text_input("Data Indices (comma-separated)", key=f"data_explorer_indices_{id}", value="")
        if st.button("Load Data Points", key=f"load_data_points_{id}"):
            indices = []
            for part in cs_idx.split(","):
                part = part.strip()
                if part.isdigit():
                    indices.append(int(part))
                    
            if any([i < 0 or i >= len(ds) for i in indices]):
                st.error("One or more indices are out of bounds.")    
            else:
                st.session_state[f"loaded_indices_{id}"] = indices
                st.session_state[f"loaded_data_points_{id}"] = [ds[i] for i in indices]

        indices = st.session_state.get(f"loaded_indices_{id}", [])
        datapoints = st.session_state.get(f"loaded_data_points_{id}", [])

        st.write(f"Loaded {len(datapoints)} data points.")
        with st.container(height=600):
            cols = st.columns(4)
            for i, (idx, data_point) in enumerate(zip(indices, datapoints)):
                lbl = exp.dataset.classes[data_point["label"]]
                with cols[i % 4]:
                    st.image(data_point['image'].numpy().transpose(1, 2, 0), caption=f"Idx {idx}: {lbl}")

def render_experiment_selector(id: int, experiments: list[LossLandscapeExperiment]):
    
    possible_experiments = experiments
    possible_datasets = list(set(map(lambda exp: exp.dataset.name, possible_experiments)))
    
    with st.container(horizontal=True, vertical_alignment="center", gap="medium") as c:
        selected_dataset = st.selectbox("Dataset", options=["None"] + possible_datasets, key=f"experiment_dataset_selector_{id}")
        
        if selected_dataset == "None":
            return 
        
        possible_splits = list(set([exp.split for exp in possible_experiments if exp.dataset.name == selected_dataset]))
        selected_split = st.selectbox("Split", options=["None"] + possible_splits, key=f"experiment_split_selector_{id}")
        
        if selected_split == "None":
            return
        
        possible_models = list(set([exp.model for exp in possible_experiments if exp.dataset.name == selected_dataset and exp.split == selected_split]))
        selected_model = st.selectbox("Model", options=["None"] + possible_models, key=f"experiment_model_selector_{id}")
        
        if selected_model == "None":
            return
        
        # possible_ks = list(set([exp.k for exp in possible_experiments if exp.dataset.name == selected_dataset and exp.split == selected_split and exp.model == selected_model]))
        # selected_k = st.selectbox("k", options=["None"] + possible_ks, key=f"experiment_k_selector_{id}")
        selected_k = 20
        
        # if selected_k == "None":
        #     return

        possible_epochs = sorted(list(set([exp.epoch for exp in possible_experiments if exp.dataset.name == selected_dataset and exp.split == selected_split and
                                    exp.model == selected_model and exp.k == selected_k])))
        selected_epoch = st.selectbox("Epoch", options=["None"] + possible_epochs, key=f"experiment_epoch_selector_{id}")
        
        if selected_epoch == "None":
            return

        possible_layers = sorted(list(set([exp.layer for exp in possible_experiments if exp.dataset.name == selected_dataset and exp.split == selected_split and
                                    exp.model == selected_model and exp.k == selected_k and exp.epoch == selected_epoch])))
        selected_layer = st.selectbox("Layer", options=["None"] + possible_layers, key=f"experiment_layer_selector_{id}")
        
        if selected_layer == "None":
            return
        
        selected_experiment = next((exp for exp in possible_experiments if exp.dataset.name == selected_dataset and exp.split == selected_split and 
                                    exp.model == selected_model and exp.k == selected_k and exp.epoch == selected_epoch and exp.layer == selected_layer), None)
    
    return selected_experiment
        
def render_save():
    def save_state():
        os.makedirs("saved_states", exist_ok=True)
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        dump_dict = {k: st.session_state[k] for k in st.session_state if k not in ("preds_cache", "preds_cache_info") and not k.endswith("_button")}
        with open(f"saved_states/state_{now}.pkl", "wb") as f:
            pickle.dump(dump_dict, f)

    with st.container(horizontal=True):
        
        if st.button("Save State", key="save_state_button"):
            save_state()

        st.markdown("<div id='save_header'></div>", unsafe_allow_html=True)        
    
    st.markdown("""
<style>
    div[data-testid="stVerticalBlock"] div:has(div#save_header) {
        position: fixed;
        top: 3.875rem;
        right: 4rem;
        padding: 1rem 1rem;
        width: fit-content;
        z-index: 999;
    }
</style>
    """, unsafe_allow_html=True)