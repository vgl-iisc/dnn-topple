from vis_utils import RichFeature, class2color, get_class_coverage

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

        labels = exp.dataset.labels_by_split[exp.split]
        
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
        zoom = alt.selection_interval(name="view", bind='scales', encodings=['x', 'y'])
        
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
            zoom,
            brush
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
        x_type = st.selectbox("X-Axis", options=list(ARC_X_AXIS_TYPE.keys()), index=1, format_func=lambda x: ARC_X_AXIS_TYPE[x], key=f"x_axis_selector_{repr(exp)}")
    
    if x_type == "sorted":
        x_type = st.multiselect("Sort by", options=list(ARC_X_AXIS_SORTING.keys()), default=["fstart"], key=f"sort_type_selector_{repr(exp)}")
    
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

def render_coverage_map(exp: LossLandscapeExperiment):
    features: list[RichFeature] | None = st.session_state.get(f"feats_{repr(exp)}", None)
    
    if features is None:
        st.info("Compute arcs and coverage to explore coverage map.")
        return
    
    filtered_features: list[RichFeature] = st.session_state.get(f"filtered_feats_{repr(exp)}", [])
    node2feat: list[int] = st.session_state.get(f"node2feat_{repr(exp)}", [])
    selection: list[int] = st.session_state.get(f"arc_explorer_selection_{repr(exp)}", [])

    if len(selection) == 0:
        selection = [f.id for f in filtered_features]
        
    f2c, c2f = st.tabs(["Feature to Class", "Class to Feature"])
    
    selected_features = [features[fid] for fid in selection]
    classes = exp.dataset.classes
    colors = {cls: class2color(i) for i, cls in enumerate(classes)}
    class_counts = {cls: 0 for cls in classes}
    
    node2class = exp.dataset.labels_by_split[exp.split]
    
    for feat in selected_features:
        for node in feat.members:
            cls = node2class[node]
            class_counts[classes[cls]] += 1
    
    # TODO: convert this to a stacked bar chart
    with f2c:
        f2c_data = pd.DataFrame({
            "class": list(class_counts.keys()),
            "count": list(class_counts.values()),
            "share": [round(count / sum(f.size for f in selected_features)) if len(selected_features) > 0 else 0 for count in class_counts.values()],
            "class coverage": [round(count / node2class.count(i) if node2class.count(i) > 0 else 0.0, 4) for i, count in enumerate(class_counts.values())],
            "color": [colors[cls] for cls in class_counts.keys()]
        })
                
        f2c_plot = alt.Chart(f2c_data).mark_bar().encode(
            x=alt.X('class:N', title="Class"),
            y=alt.Y('count:Q', title="Number of points covered"),
            color=alt.Color('class:N', scale=alt.Scale(domain=list(colors.keys()), range=list(colors.values())), title="Class").legend(None),
            tooltip=[alt.Tooltip('class', title="Class"), alt.Tooltip('count', title="Count"), alt.Tooltip('share', title="Share"), alt.Tooltip('class coverage', title="Class Coverage")]
        )
        
        st.altair_chart(f2c_plot)
        
    with c2f:
        pass