from experiment import LossLandscapeExperiment, BertExperiment
from coverage_utils import get_class_coverage
from graph_utils import compute_tree_graph
from feature import get_type_string

import pyct as ct
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import altair as alt
import numpy as np
import networkx as nx
from pyvis import network as net

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
    "valleys": {(ct.MINIMUM, ct.SADDLE), (ct.MINIMUM, ct.MAXIMUM), (ct.MINIMUM, ct.REGULAR)}, # type: ignore
    "peaks": {(ct.SADDLE, ct.MAXIMUM), (ct.MINIMUM, ct.MAXIMUM), (ct.REGULAR, ct.MAXIMUM)}, # type: ignore
    "saddle-saddle": {(ct.SADDLE, ct.SADDLE)}, # type: ignore
    "regular": {(ct.REGULAR, ct.REGULAR), (ct.REGULAR, ct.SADDLE), (ct.REGULAR, ct.MINIMUM), (ct.REGULAR, ct.MAXIMUM),  # type: ignore
                (ct.SADDLE, ct.REGULAR), (ct.MINIMUM, ct.REGULAR), (ct.MAXIMUM, ct.REGULAR)}, # type: ignore
}

def render_tree_explorer(id: int):
    
    exp = st.session_state.get(f"selected_experiment_{id}", None)
    
    @st.cache_data(hash_funcs={LossLandscapeExperiment: LossLandscapeExperiment.__hash__, BertExperiment: BertExperiment.__hash__, list: lambda x: hash(tuple(f.id for f in x)) if x and isinstance(x[0], ct.RichFeature) else hash(tuple(x))})
    def arc_explorer_plot(exp: LossLandscapeExperiment, features: list[ct.RichFeature], axis_type: str | list[str]) -> alt.Chart:
        
        max_pers = max([f.pers for f in features]) if len(features) > 0 else 1.0
        
        df = pd.DataFrame([{
            "id": f.id,
            "fstart": f.fn_frm,
            "fend": f.fn_to if f.pers > max_pers / 50 else f.fn_frm + max_pers / 50,
            "ftype": get_type_string(f),
            "pers": f.pers,
            "volume": f.size,
            "logvol": np.log1p(f.size),
            "majority class": exp.dataset.classes[f.majority_class],
            "homogeneity": f.homogeneity,
            "major class size": f.major_class_size,
            "major class coverage": get_class_coverage(exp, f.major_class_size, f.majority_class),
            "fnode": f.frm,
            "tnode": f.to,
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
                     alt.Tooltip('ftype', title="Feature Type"), alt.Tooltip('pers', title="Persistence"), alt.Tooltip('volume', title="Volume"), alt.Tooltip('homogeneity', title="Homogeneity"),
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
            x_type = st.selectbox("X-Axis", options=list(ARC_X_AXIS_TYPE.keys()), index=0, format_func=lambda x: ARC_X_AXIS_TYPE[x], key=f"x_axis_selector_{id}")
        
        if x_type == "sorted":
            x_type = st.multiselect("Sort by", options=list(ARC_X_AXIS_SORTING.keys()), default=["majority class", "fstart"], key=f"sort_type_selector_{id}")
        
        with st.container(horizontal=True, horizontal_alignment="center", gap="medium"):
            filter_low_vol = st.number_input("Minimum Volume", min_value=0, value=0, step=1, key=f"min_volume_input_{id}")
            filter_high_vol = st.number_input("Maximum Volume", min_value=0, value=1_000_000_000, step=1, key=f"max_volume_input_{id}")
            
            filter_low_loss_interval = st.number_input("Minimum Loss Interval", min_value=0.0, value=0.0, step=0.001, key=f"min_loss_interval_input_{id}")
            filter_high_loss_interval = st.number_input("Maximum Loss Interval", max_value=1e12, value=1e12, step=0.001, key=f"max_loss_interval_input_{id}")
            
            filter_low_homo = st.number_input("Minimum Homogeneity", min_value=0.0, max_value=1.0, value=0.0, step=0.001, key=f"min_homogeneity_input_{id}")
            filter_high_homo = st.number_input("Maximum Homogeneity", min_value=0.0, max_value=1.0, value=1.0, step=0.001, key=f"max_homogeneity_input_{id}")
        
        criteria = ct.FilterCriteria() # type: ignore
        criteria.min_size = filter_low_vol
        criteria.max_size = filter_high_vol
        criteria.min_fn_from = filter_low_loss_interval
        criteria.max_fn_from = filter_high_loss_interval
        criteria.min_homogeneity = filter_low_homo
        criteria.max_homogeneity = filter_high_homo
        criteria.allowed_types = allowed_types
        
        if features is None:
            return
        
        filtered_indices = ct.filterFeatures(features, criteria) # type: ignore
        filtered_features = [features[i] for i in filtered_indices]
        st.session_state[f"filtered_feats_{id}"] = filtered_features
        
        if len(filtered_features) == 0:
            st.warning("No features of the selected types.")
            return

        # st.write(f"Rendering {len(filtered_features)} features at simplification {simpl}")
        st.write(f"Rendering {len(filtered_features)} features")
        
        plot = arc_explorer_plot(exp, filtered_features, x_type)
        
        selection_dict = st.altair_chart(plot, use_container_width=True, on_select="rerun", key=f"arc_explorer_plot_{id}")
        
        if selection_dict is not None and "selection" in selection_dict and "arcs" in selection_dict["selection"]:
            st.session_state[f"explorer_selected_arcs_{id}"] = [rec["id"] for rec in selection_dict["selection"]["arcs"]]
        else:
            st.session_state[f"explorer_selected_arcs_{id}"] = []
    
    @st.fragment
    def tree_view(features, allowed_types):
        
        steiner_mode = "None"
        ego_origin = "None"
        ego_radius = 0
        
        with st.container(horizontal=True, horizontal_alignment="center", vertical_alignment="bottom", gap="medium"):
            simpl_mode = st.selectbox("Simplification Mode", key=f"simpl_mode_selector_{id}", options=["None", "Feature Types", "Steiner", "Ego", "Ego From"], index=0, help="Choose how to simplify the tree for visualization.")
            
            if simpl_mode == "Steiner":
                steiner_mode = st.selectbox("Steiner Tree", key=f"steiner_selector_{id}", options=["Minima", "Maxima"], index=0, help="Use Steiner tree to include important critical points in the tree view.")
            if simpl_mode == "Ego":
                ego_origin = st.selectbox("Ego Origin", key=f"ego_origin_selector_{id}", options=["Minima", "Maxima"], index=0, help="Choose the type of critical point to center the ego tree around.")
                ego_radius = st.slider("Ego Radius", key=f"ego_radius_slider_{id}", min_value=1, max_value=50, value=2, help="Radius of the ego tree to display.")
            if simpl_mode == "Ego From":
                ego_origin = st.text_input("Ego Origin ID", key=f"ego_origin_id_input_{id}", value="0", help="Comma separated list of critical point IDs to center the ego tree around.")
                ego_origin = tuple(int(x.strip()) for x in ego_origin.split(",") if x.strip().isdigit())
                ego_radius = int(st.number_input("Ego Radius", key=f"ego_radius_input_{id}", min_value=2, value=2, help="Radius of the ego tree to display."))
            
            # TODO: saddle simplification is killing arcs, need to fix that
            # saddle_simpl = st.toggle(f"Saddle Simplification", key=f"saddle_simpl_toggle_{id}", value=False, help="Remove chains of saddle-saddle connections for a cleaner tree view.")
            saddle_simpl = False
        
        valid_features = features

        if simpl_mode == "Feature Types":
            criteria = ct.FilterCriteria() # type: ignore
            criteria.allowed_types = allowed_types
            filtered_indices = ct.filterFeatures(features, criteria) # type: ignore
            valid_features = [features[i] for i in filtered_indices]

        if len(valid_features) == 0:
            st.warning("No features of the selected types.")
            return

        with st.container(horizontal=True, horizontal_alignment="left", vertical_alignment="center"):
            if st.button("Recompute Tree Graph", key=f"recompute_tree_graph_{id}"):
                st.session_state[f"tree_graph_{id}"] = compute_tree_graph(exp, valid_features, steiner_mode, ego_origin, ego_radius, saddle_simpl)
            
            display_tree = st.checkbox("Display Tree Graph", key=f"display_tree_checkbox_{id}", value=False)
            use_phys = st.checkbox("Use Physics", key=f"use_phys_checkbox_{id}", value=False)

        gnx = st.session_state.get(f"tree_graph_{id}", None)

        if gnx is None:
            st.text("Compute tree graph to begin.")
            return

        if not display_tree:
            st.text(f"Tree graph computed with {len(gnx.nodes)} nodes and {len(gnx.edges)} edges. Connected: {nx.is_connected(gnx.to_undirected())}. Not displayed.")
            return

        st.text(f"{len(valid_features)} features selected. Rendering {len(gnx.edges)} features after processing. Connected: {nx.is_connected(gnx.to_undirected())}")

        # imbalance_metrics = compute_tree_imbalance_metrics(gnx)
        
        g = net.Network(height="600px", width="100%", directed=True)
        g.from_nx(gnx)
        g.toggle_physics(use_phys)
        
        html = g.generate_html()
        components.html(html, height=600)
        # st.write(imbalance_metrics)
        
    features: list[ct.RichFeature] | None = st.session_state.get(f"feats_{id}", None)

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