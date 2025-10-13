from vis_utils import RichFeature, class2color, get_class_coverage, load_preds

from experiment import LossLandscapeExperiment
import streamlit as st

import altair as alt
import pandas as pd

import numpy as np
import pyct as ct

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

def render_arc_explorer(exp: LossLandscapeExperiment):
    
    def arc_explorer_plot(features: list[RichFeature], axis_type: str | list[str]) -> alt.Chart:
        
        df = pd.DataFrame([{
            "id": f.id,
            "fstart": f.fn_frm,
            "fend": f.fn_to,
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
    
    features: list[RichFeature] | None = st.session_state.get(f"feats_{repr(exp)}", None)

    if features is None:
        st.info("Compute arcs and coverage to explore arcs.")
        return
    
    simpl: float = st.session_state[f"computed_simpl_{repr(exp)}"]
        
    types = st.multiselect("Feature types", options=list(ALLOWED_TYPE_SETS.keys()), default=["valleys"], key=f"feature_type_selector_{repr(exp)}")
    
    allowed_types = set()
    for t in types:
        allowed_types = allowed_types.union(ALLOWED_TYPE_SETS.get(t, set()))
            
    with st.container(horizontal=True, horizontal_alignment="center") as c:
        x_type = st.selectbox("X-Axis", options=list(ARC_X_AXIS_TYPE.keys()), index=2, format_func=lambda x: ARC_X_AXIS_TYPE[x], key=f"x_axis_selector_{repr(exp)}")
    
    if x_type == "sorted":
        x_type = st.multiselect("Sort by", options=list(ARC_X_AXIS_SORTING.keys()), default=["majority class", "fstart"], key=f"sort_type_selector_{repr(exp)}")
    
    filtered_features = [f for f in features if (f.type_frm, f.type_to) in allowed_types]
    st.session_state[f"filtered_feats_{repr(exp)}"] = filtered_features
    
    if len(filtered_features) == 0:
        st.warning("No features of the selected types.")
        return

    st.write(f"Rendering {len(filtered_features)} features at simplification {simpl}")
    
    plot = arc_explorer_plot(filtered_features, x_type)
    
    selection_dict = st.altair_chart(plot, use_container_width=True, on_select="rerun")
    
    if selection_dict is not None and "selection" in selection_dict and "arcs" in selection_dict["selection"]:
        st.session_state[f"arc_explorer_selection_{repr(exp)}"] = [rec["id"] for rec in selection_dict["selection"]["arcs"]]
    else:
        st.session_state[f"arc_explorer_selection_{repr(exp)}"] = []

F2C_VIEWS = ["plain", "correctness", "class-wise confusion"]

def render_coverage_map(exp: LossLandscapeExperiment):
    
    def f2c_plot(feats: list[RichFeature], view: str, show_proportions: bool, freeze_top: bool) -> alt.Chart | None:
        classes = exp.dataset.classes
        colors = {cls: class2color(i) for i, cls in enumerate(classes)}
        node2label = exp.dataset.labels_by_split[exp.split]
        preds = st.session_state.get(f"preds_{repr(exp)}", {})
    
        node_feats = set.union(*[set(zip(f.members, [f] * f.size)) for f in feats])
        
        max_class_size = max([exp.dataset.class_size_by_split[exp.split][cls] for cls in exp.dataset.classes])
        max_class_prop = max_class_size / len(node2label)
        
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

        # TODO: proportion and coverage are improperly set up

        df["Proportion"] = df['count'] / len(node2label)
        df["Coverage"] = [get_class_coverage(exp, entry["count"], entry["True Label"]) for _, entry in df.iterrows()]
        
        scale = alt.Scale()
        if freeze_top:
            scale = alt.Scale(domain=[0, max_class_prop if show_proportions else max_class_size])
        
        y = alt.Y('count', title="Number of Points") if not show_proportions else alt.Y('Proportion', title="Proportion", axis=alt.Axis(format='%'))
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
        
    
    features: list[RichFeature] | None = st.session_state.get(f"feats_{repr(exp)}", None)
    
    if features is None:
        st.info("Compute arcs and coverage to explore coverage map.")
        return
    
    filtered_features: list[RichFeature] = st.session_state.get(f"filtered_feats_{repr(exp)}", [])
    selection: list[int] = st.session_state.get(f"arc_explorer_selection_{repr(exp)}", [])

    if len(selection) == 0:
        selection = [f.id for f in filtered_features]
        
    # UI Controls
    with st.container(horizontal=True, vertical_alignment="top", horizontal_alignment="left"):
        fine_grained = st.selectbox("View", key=f"fine_grained_{repr(exp)}", options=F2C_VIEWS, format_func=lambda x: x.title())
        
        with st.container():
            freeze_top = st.checkbox("Freeze top", key=f"freeze_top_{repr(exp)}")
            show_proportions = st.checkbox("Proportions instead of counts", key=f"show_proportions_{repr(exp)}")
        
    f2c, c2f = st.tabs(["Feature to Class", "Class to Feature"])
    
    selected_features = [features[fid] for fid in selection]
    
    with f2c:        
        plot = f2c_plot(selected_features, fine_grained, show_proportions, freeze_top)
        
        if plot is None:
            st.warning("No datapoints in selection.")
        else:
            st.altair_chart(plot, use_container_width=True)
        
    with c2f:
        st.info("Class to Feature view not implemented yet.")