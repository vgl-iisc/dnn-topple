from vis_utils import RichFeature, class2color

from experiment import LossLandscapeExperiment
import streamlit as st

import altair as alt
import pandas as pd

import pyct as ct

def render_arc_explorer(exp: LossLandscapeExperiment):
    
    def arc_explorer_plot(df: pd.DataFrame) -> alt.Chart:
        brush = alt.selection_point(encodings=['x'])
        zoom = alt.selection_interval(bind='scales', encodings=['x', 'y'])
        
        # Create a bar chart using Altair
        plot = alt.Chart(df).mark_line().encode(
            x=alt.X('id', title="Feature ID"),
            y=alt.Y('fstart'+":Q", title="Loss Start"),
            y2=alt.Y2('fend'+":Q", title="Loss End"),
            tooltip=['id', 'fstart', 'fend', 'ftype'],
            opacity=alt.condition(brush, alt.value(1), alt.value(0.2)),
            color=alt.value('steelblue'),
            # color=alt.Color(AVol, scale=alt.Scale(scheme='yelloworangered')).title(None),
            strokeWidth=alt.value(5)
        ).add_params(
            brush,
            zoom
        )
        
        return plot
    
    features: list[RichFeature] | None = st.session_state.get(f"feats_{repr(exp)}", None)

    if features is None:
        st.info("Compute arcs and coverage to explore arcs.")
        return
    
    simpl: float = st.session_state[f"computed_simpl_{repr(exp)}"]
        
    types = st.multiselect("Feature types", options=["valleys", "peaks", "saddle-saddle", "regular"], default=["valleys"], key=f"feature_type_selector_{repr(exp)}")
    
    allowed_types = set()
    if "valleys" in types:
        allowed_types.add((ct.MINIMUM, ct.SADDLE))
        allowed_types.add((ct.MINIMUM, ct.MAXIMUM))
        allowed_types.add((ct.MINIMUM, ct.REGULAR))
    if "peaks" in types:
        allowed_types.add((ct.SADDLE, ct.MAXIMUM))
        allowed_types.add((ct.MINIMUM, ct.MAXIMUM))
        allowed_types.add((ct.REGULAR, ct.MAXIMUM))
    if "saddle-saddle" in types:
        allowed_types.add((ct.SADDLE, ct.SADDLE))
    if "regular" in types:
        allowed_types.add((ct.REGULAR, ct.REGULAR))
        allowed_types.add((ct.REGULAR, ct.SADDLE))
        allowed_types.add((ct.REGULAR, ct.MINIMUM))
        allowed_types.add((ct.REGULAR, ct.MAXIMUM))
        
        allowed_types.add((ct.SADDLE, ct.REGULAR))
        allowed_types.add((ct.MINIMUM, ct.REGULAR))
        allowed_types.add((ct.MAXIMUM, ct.REGULAR))
        
    
    filtered_features = [f for f in features if (f.type_frm, f.type_to) in allowed_types]
    st.session_state[f"filtered_feats_{repr(exp)}"] = filtered_features
    
    if len(filtered_features) == 0:
        st.warning("No features of the selected types.")
        return

    df = pd.DataFrame([{
        "id": f.id,
        "fstart": f.fn_frm,
        "fend": f.fn_to,
        "ftype": f.type_string,
    } for i, f in enumerate(filtered_features)])

    st.write(f"Rendering {len(df)} features at simplification {simpl}")
    
    plot = arc_explorer_plot(df)
    st.session_state[f"arc_explorer_selection_{repr(exp)}"] = st.altair_chart(plot, use_container_width=True, on_select="rerun")

def render_coverage_map(exp: LossLandscapeExperiment):
    features: list[RichFeature] | None = st.session_state.get(f"feats_{repr(exp)}", None)
    
    if features is None:
        st.info("Compute arcs and coverage to explore coverage map.")
        return
    
    filtered_features: list[RichFeature] = st.session_state.get(f"filtered_feats_{repr(exp)}", [])
    node2feat: list[int] = st.session_state.get(f"node2feat_{repr(exp)}", [])
    feat2nodes: dict[RichFeature, set[int]] = st.session_state.get(f"feat2nodes_{repr(exp)}", {})
    selection = st.session_state.get(f"arc_explorer_selection_{repr(exp)}", None)
    
    if selection is None or len(selection["selection"]["param_1"]) == 0:
        selection = [f.id for f in filtered_features]
    else:
        selection = [r["id"] for r in selection["selection"]["param_1"]]


    f2c, c2f = st.tabs(["Feature to Class", "Class to Feature"])
    
    classes = exp.dataset.classes
    colors = {cls: class2color(i) for i, cls in enumerate(classes)}
    class_counts = {cls: 0 for cls in classes}
    
    node2class = exp.dataset.get_split_labels(exp.split)
    
    for arc_id in selection:        
        feat = features[arc_id]
                
        for node in feat2nodes[feat]:
            cls = node2class[node]
            class_counts[classes[cls]] += 1
    
    # TODO: convert this to a stacked bar chart
    with f2c:
        f2c_data = pd.DataFrame({
            "class": list(class_counts.keys()),
            "count": list(class_counts.values()),
            "share": [round(count / sum(class_counts.values()) if sum(class_counts.values()) > 0 else 0.0, 3) for count in class_counts.values()],
            "share of class": [round(count / node2class.count(i) if node2class.count(i) > 0 else 0.0, 3) for i, count in enumerate(class_counts.values())],
            "color": [colors[cls] for cls in class_counts.keys()]
        })
                
        f2c_plot = alt.Chart(f2c_data).mark_bar().encode(
            x=alt.X('class:N', title="Class"),
            y=alt.Y('count:Q', title="Number of points covered"),
            color=alt.Color('class:N', scale=alt.Scale(domain=list(colors.keys()), range=list(colors.values())), title="Class"),
            tooltip=['class', 'count', 'share', 'share of class']
        )
        
        st.altair_chart(f2c_plot)
        
    with c2f:
        pass