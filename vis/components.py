from vis_utils import RichFeature

from experiment import LossLandscapeExperiment
import streamlit as st

import altair as alt
import pandas as pd

import pyct as ct

def render_arc_explorer(exp: LossLandscapeExperiment, simpl: float):
    features: list[RichFeature] | None = st.session_state.get(f"feats_{repr(exp)}", None)
    
    if features is None:
        st.info("Compute arcs and coverage to explore arcs.")
        return
    
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
    
    if len(filtered_features) == 0:
        st.warning("No features of the selected types.")
        return

    df = pd.DataFrame([{
        "index": i,
        "fstart": f.fn_frm,
        "fend": f.fn_to,
        "ftype": f.type_string,
    } for i, f in enumerate(filtered_features)])

    st.write(f"Rendering {len(df)} features at simplification {simpl}")
    
    # setup brushing to select multiple bars
    brush = alt.selection_point(encodings=['x'])
    # Create a bar chart using Altair
    plot = alt.Chart(df).mark_line().encode(
        x=alt.X('index', title="Feature Index"),
        y=alt.Y('fstart'+":Q", title="Loss Start"),
        y2=alt.Y2('fend'+":Q", title="Loss End"),
        tooltip=['index', 'fstart', 'fend', 'ftype'],
        opacity=alt.condition(brush, alt.value(1), alt.value(0.2)),
        color=alt.value('steelblue'),
        # color=alt.Color(AVol, scale=alt.Scale(scheme='yelloworangered')).title(None),
        strokeWidth=alt.value(5)
    ).add_params(
        brush
    )
    
    st.altair_chart(plot, use_container_width=True)

def render_coverage_map(exp: LossLandscapeExperiment, simpl: float):
    features: list[RichFeature] | None = st.session_state.get(f"feats_{repr(exp)}", None)
    
    if features is None:
        st.info("Compute arcs and coverage to explore coverage map.")
        return

    c2v, v2c = st.tabs(["Class to Valley", "Valley to Class"])
        
    with c2v:
        pass
    
    with v2c:
        pass