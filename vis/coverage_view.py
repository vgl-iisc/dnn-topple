from basic_utils import get_preds, get_labels, get_bert_sentences
from coverage_utils import class2color, get_class_coverage
from feature import get_type_string
from dataset_loader import load_dataset, load_bert_flat_losses

import pyvis.network as net

import sys

import os

from experiment import LossLandscapeExperiment, Dataset, BertExperiment
import streamlit as st
import streamlit.components.v1 as components

import torch
import networkx as nx
import altair as alt
import pandas as pd

import numpy as np
import pyct as ct

import pickle
import datetime

F2C_VIEWS = ["plain", "correctness", "class-wise confusion"]

@st.fragment
def render_coverage_map(id: int):
    
    exp = st.session_state.get(f"selected_experiment_{id}", None)
    
    features: list[ct.RichFeature] | None = st.session_state.get(f"feats_{id}", None)
    
    if features is None:
        st.info("Compute arcs and coverage to explore coverage map.")
        return
        
    filtered_features: list[ct.RichFeature] = st.session_state.get(f"filtered_feats_{id}", [])
    selection: list[int] = st.session_state.get(f"explorer_selected_arcs_{id}", [])
    ds = load_dataset(exp)

    if len(selection) == 0:
        selection = [f.id for f in filtered_features]

    classes = exp.dataset.classes
    n_classes = len(classes)
    many_classes = n_classes > 30

    focused_classes = list(range(n_classes))

    with st.container(horizontal=True, vertical_alignment="center"):
        focused_feats = st.text_input("Focus Feature IDs (comma-separated)", key=f"add_feat_ids_{id}", value="")        
        focused_feat_ids = [int(fid.strip()) for fid in focused_feats.split(",") if fid.strip().isdigit()]
        
        if many_classes:
            focused_classes_input = st.text_input("Focus Class Indices (comma-separated, hyphenated)", key=f"add_class_indices_{id}", value="")
            if focused_classes_input.strip() != "":
                focused_classes = []
                for part in focused_classes_input.split(","):
                    part = part.strip()
                    if part.isdigit():
                        focused_classes.append(int(part))
                    elif '-' in part:
                        try:
                            start, end = map(int, part.split('-'))
                            focused_classes.extend(list(range(start, end + 1)))
                        except ValueError:
                            st.warning(f"Invalid range format: '{part}'")
                    else:
                        st.warning(f"Invalid class index: '{part}'")
    
    if len(focused_feat_ids) > 0:
        selection = focused_feat_ids
    selection = list(set(selection))

    f2c, acc, c2f, cc, datex = st.tabs(["Feature to Class", "Accuracy", "Class to Features", "Class Co-Occurrences", "Data Explorer"])
    selected_features = [features[fid] for fid in selection]
    node2feat = st.session_state.get(f"node2feat_{id}") or []
   
    @st.fragment
    def f2c_viewer():
        @st.cache_data(hash_funcs={LossLandscapeExperiment: LossLandscapeExperiment.__hash__, BertExperiment: BertExperiment.__hash__, list: lambda x: hash(tuple(f.id for f in x)) if x and isinstance(x[0], ct.RichFeature) else hash(tuple(x))})
        def f2c_plot(exp: LossLandscapeExperiment, feats: list[ct.RichFeature], preds: list[int], view: str, show_what: str, freeze_top: bool, focused_classes: list[int]) -> alt.Chart | None:
            focused_class_set = set(focused_classes)
            classes = [exp.dataset.classes[i] for i in focused_classes]
            colors = {exp.dataset.classes[i]: class2color(i) for i in focused_classes}
            node2label = exp.dataset.labels_by_split[exp.split]
        
            if len(feats) == 0:
                return None
        
            node_feats = set.union(*[set(zip(f.members, [f] * f.size)) for f in feats])
            
            max_class_size = max([exp.dataset.class_size_by_split[exp.split][exp.dataset.classes[i]] for i in focused_classes]) if focused_classes else 1
            
            data = []
            for (node, feat) in node_feats:
                true_label = node2label[node]

                # Filter by focused classes
                if true_label not in focused_class_set:
                    continue

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
                x=alt.X('True Class:N', title="True Class", sort=alt.EncodingSortField(field='True Label', order='ascending')),
                y=y,
                color=color if color is not None else alt.Color('True Class:N', scale=alt.Scale(domain=list(colors.keys()), range=list(colors.values())), title="True Class"),
                tooltip=[alt.Tooltip('count', title="Number of Points"), alt.Tooltip('True Label', title="Label Index"), alt.Tooltip('True Class', title="True Class"), 
                        alt.Tooltip('Coverage', title="Class Coverage").format('.3%'), alt.Tooltip('Proportion', title="Proportion").format('.3%')] + ([pred_tooltip] if pred_tooltip is not None else []),
                order=alt.Order('True Label:Q', sort='ascending')
            ).properties(
                width=600,
                height=400
            )
            
            return plot

        with st.container(horizontal=True, vertical_alignment="top", horizontal_alignment="left"):
            fine_grained = st.selectbox("View", key=f"fine_grained_{id}", options=F2C_VIEWS, format_func=lambda x: x.title())
            show_what = st.selectbox("Y-axis", key=f"show_what_{id}", options=["Counts", "Proportions", "Coverage"], index=2)
            freeze_top = st.checkbox("Freeze top", key=f"freeze_top_{id}", value=True)
        
        preds = get_preds(exp)
        plot = f2c_plot(exp, selected_features, preds, fine_grained, show_what, freeze_top, focused_classes)
        
        if plot is None:
            st.warning("No datapoints in selection.")
        else:
            st.altair_chart(plot, use_container_width=True, key=f"f2c_plot_{id}")
    
    @st.fragment
    def acc_viewer():
        focused_class_set = set(focused_classes)
        total_points = 0
        correct_points = 0
        preds = get_preds(exp)
        node2label = exp.dataset.labels_by_split[exp.split]
        
        for f in selected_features:
            for n in f.members:
                if node2label[n] not in focused_class_set:
                    continue
                total_points += 1
                if preds[n] == node2label[n]:
                    correct_points += 1
                    
        accuracy = correct_points / total_points if total_points > 0 else 0.0
        
        st.write(f"Accuracy over selected features: **{accuracy * 100.0:0.3f}%** ({correct_points} / {total_points})")
        
        focused_class_names = [exp.dataset.classes[i] for i in focused_classes]
        confusion = pd.DataFrame(0, index=focused_class_names, columns=focused_class_names)
        for f in selected_features:
            for n in f.members:
                if node2label[n] not in focused_class_set:
                    continue
                true_cls = exp.dataset.classes[node2label[n]]
                pred_cls = exp.dataset.classes[preds[n]]
                if true_cls in focused_class_names and pred_cls in focused_class_names:
                    confusion.at[true_cls, pred_cls] += 1 # type: ignore
                    
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

    @st.fragment
    def c2f_viewer():   
        selected_labels = focused_classes
        
        sub_view = st.radio(
            "View",
            options=["Class to Feature Mappings", "Class to Feature Counts"],
            horizontal=True,
            key=f"c2f_subview_{id}"
        )
        
        if sub_view == "Class to Feature Mappings":
            sort_by = st.selectbox(
                "Sort by",
                options=["Coverage", "Proportion"],
                index=0,
                key=f"c2f_sortby_{id}"
            )
            
            rows = []
            for cls in selected_labels:
                class_name = exp.dataset.classes[cls]
                for f in selected_features:
                    cnt = f.class_counts.get(cls, 0)
                    if cnt <= 0:
                        continue
                    proportion = f.class_proportions.get(cls, 0.0)
                    coverage = f.class_coverage.get(cls, 0.0)
                    
                    rows.append({
                        "Class": class_name,
                        "Class Index": cls,
                        "Feature ID": f.id,
                        "Type": get_type_string(f),
                        "Size": f.size,
                        "Count": cnt,
                        "Proportion": proportion,
                        "Coverage": coverage
                    })
            
            if len(rows) == 0:
                st.warning("No features contain the selected class(es).")
            else:
                df = pd.DataFrame(rows)
                df = df.sort_values(by=["Class Index", sort_by, "Feature ID"], ascending=[True, False, True])
                df = df.drop(columns=["Class Index"])
                st.dataframe(
                    df,
                    width="content",
                    hide_index=True,
                    key=f"c2f_mapping_table_{id}"
                )
        
        else:  # Class to Feature Counts
            data_total = []
            
            for cls in selected_labels:
                class_name = exp.dataset.classes[cls]
                features_with_class = 0
                features_majority = 0
                
                for f in selected_features:
                    if f.class_counts.get(cls, 0) > 0:
                        features_with_class += 1
                        if f.majority_class == cls:
                            features_majority += 1
                
                data_total.append({
                    "Class": class_name,
                    "Count": features_with_class,
                    "Count Majority": features_majority
                })
            
            df = pd.DataFrame(data_total)
            
            # Background bar (features containing class)
            base = alt.Chart(df).mark_bar(color='steelblue', opacity=1.0).encode(
                x=alt.X("Class:N", title="Class"),
                y=alt.Y("Count:Q", title="Number of Features"),
                tooltip=[
                    alt.Tooltip("Class:N", title="Class"),
                    alt.Tooltip("Count:Q", title="Features containing class")
                
                ]
            )
            
            # Foreground bar (features where class is majority)
            overlay = alt.Chart(df).mark_bar(color='blue', opacity=1.0).encode(
                x=alt.X("Class:N", title="Class"),
                y=alt.Y("Count Majority:Q", title="Number of Features"),
                tooltip=[
                    alt.Tooltip("Class:N", title="Class"),
                    alt.Tooltip("Count Majority:Q", title="Features where class is majority")
                ]
            )
            
            chart = (base + overlay).properties(
                width=600,
                height=400
            )
            st.altair_chart(chart, use_container_width=True, key=f"c2f_counts_chart_{id}")
    
    @st.fragment
    def cc_viewer():
        focused_class_names = [exp.dataset.classes[i] for i in focused_classes]
        selected_classes = st.multiselect("Co-Occurrences", options=focused_class_names, default=[], key=f"class_selector_c2f_{id}")
        selected_labels = set(exp.dataset.classes.index(c) for c in selected_classes)
        
        entries = []
        
        for f in features:
            if not all([lab in f.class_counts for lab in selected_labels]) or len(selected_labels) == 0:
                continue
            
            share = sum([f.class_counts[lab] for lab in selected_labels]) / f.size
            coverage = sum([f.class_counts[lab] for lab in selected_labels]) / sum([exp.dataset.class_size_by_split[exp.split][exp.dataset.classes[lab]] for lab in selected_labels])
            
            top_5_classes = sorted(f.class_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            
            entries.append({
                "ID": f.id,
                "Type": get_type_string(f),
                "Size": f.size,
                "Share": share,
                "Coverage": coverage,
                "Top 5": ', '.join([f"{exp.dataset.classes[cid]} ({cnt})" for cid, cnt in top_5_classes])
            })
                
        if len(entries) == 0:
            st.warning("No features contain the selected class(es).")
        else:
            df = pd.DataFrame(entries).sort_values(by=["Coverage", "ID"], ascending=[False, True])
            df["Share"] = df["Share"]
            df["Coverage"] = df["Coverage"]
            st.dataframe(df, width="content", key=f"class_to_feature_table_{id}", hide_index=True)

    @st.fragment
    def datex_viewer():
        node2label = exp.dataset.labels_by_split[exp.split]
        
        @st.cache_data(hash_funcs={LossLandscapeExperiment: LossLandscapeExperiment.__hash__, BertExperiment: BertExperiment.__hash__})
        def load_datapoints_by_indices(exp: LossLandscapeExperiment, indices: tuple[int, ...]) -> list:
            """Load datapoints from dataset by indices. Cached to avoid repeated expensive access."""
            ds = load_dataset(exp)
            if ds is None:
                return []
            return [ds[i] for i in indices]

        def image_view(indices_fids: list[tuple[int, int]], datapoints: list, siamese: bool = False):
            st.write(f"Loaded {len(datapoints)} data points.")
            
            key = f"data_explorer_image_view_{id}"
            key += "_siamese" if siamese else ""
            
            flip = exp.dataset.name == "emnist_letters"
            
            per_page = 16
            N = len(datapoints)
            if N > 0 and N > per_page:
                page_count = (len(datapoints) + per_page - 1) // per_page
                page = st.slider("Page", min_value=1, max_value=max(1, page_count), value=1, key=key) - 1
                
                with st.container(height=700):
                    cols = st.columns(4)
                    for i, ((idx, fid), data_point) in enumerate(list(zip(indices_fids, datapoints))[page * per_page:(page + 1) * per_page]):
                        lbl = exp.dataset.classes[data_point["label"]]
                        with cols[i % 4]:
                            image_np = data_point['image'].numpy().transpose(1, 2, 0)
                            if flip:
                                image_np = np.flip(image_np, axis=0)
                            st.image(image_np, caption=f"Idx {idx} ({fid}): {lbl}")
            elif N > 0:
                with st.container(height=700):
                    cols = st.columns(4)
                    for i, ((idx, fid), data_point) in enumerate(zip(indices_fids, datapoints)):
                        lbl = exp.dataset.classes[data_point["label"]]
                        with cols[i % 4]:
                            image_np = data_point['image'].numpy().transpose(1, 2, 0)
                            if flip:
                                image_np = np.flip(image_np, axis=0)
                            st.image(image_np, caption=f"Idx {idx} ({fid}): {lbl}")
            else:
                st.info("No data points loaded.")
        
        with st.container(horizontal=True, vertical_alignment="center"):
            mode = st.selectbox("Load Mode", options=["From Indices", "From Selected Features"], index=1, key=f"data_explorer_mode_selector_{id}")
            if mode == "From Selected Features":
                focused_class_names = [exp.dataset.classes[i] for i in focused_classes]
                selected_classes = st.multiselect("Filter Classes", options=focused_class_names, default=focused_class_names, key=f"member_classes_selector_datex_{id}")
                selected_labels = set(classes.index(c) for c in selected_classes)
            elif mode == "From Indices":
                selected_labels = set(focused_classes)
                cs_idx = st.text_input("Data Indices (comma-separated)", key=f"data_explorer_indices_{id}", value="")
        
        def load_from_indices_fids(indices_fids: list[tuple[int, int]], siamese: bool = False):
            st.session_state[f"loaded_indices_{id}"] = indices_fids
            # Filter indices by selected labels
            filtered_indices_fids = [(i, fid) for i, fid in indices_fids if node2label[i] in selected_labels]
            filtered_indices = tuple(i for i, _ in filtered_indices_fids)
            # Use cached loading function
            st.session_state[f"loaded_data_points_{id}"] = load_datapoints_by_indices(exp, filtered_indices) if filtered_indices else []
            
            st.session_state[f"data_explorer_siamese_{id}"] = siamese
            
            if not siamese:
                return
            
            indices_fids_set = set(indices_fids)
            siamese_indices_fids = [(i, f.id) for f in features for i in f.members if node2label[i] in selected_labels]
            siamese_indices_fids = [x for x in siamese_indices_fids if x not in indices_fids_set]
            
            st.session_state[f"siamese_indices_{id}"] = siamese_indices_fids
            siamese_indices = tuple(i for i, _ in siamese_indices_fids)
            # Use cached loading function
            st.session_state[f"siamese_data_points_{id}"] = load_datapoints_by_indices(exp, siamese_indices) if siamese_indices else []

        if mode == "From Indices" and cs_idx.strip() != "":
            indices_fids = []
            for part in cs_idx.split(","):
                part = part.strip()
                if part.isdigit():
                    idx = int(part)
                    fid = node2feat[idx].id if idx < len(node2feat) else -1
                    indices_fids.append((idx, fid))
                    
            ds = load_dataset(exp)
            if ds is not None and any([i[0] < 0 or i[0] >= len(ds) for i in indices_fids]):
                st.error("One or more indices are out of bounds.")    
            else:
                load_from_indices_fids(indices_fids)
        elif mode == "From Selected Features":
            with st.container(horizontal=True):
                if st.button("Load from Selected Features", key=f"load_from_features_{id}"):
                    indices_fids = set()
                    for f in selected_features:
                        indices_fids = indices_fids.union(set([(m, f.id) for m in f.members]))
                    indices_fids = sorted(list(indices_fids))
                    load_from_indices_fids(indices_fids)
                if st.button("Load in Siamese View", key=f"load_siamese_view_{id}"):
                    indices_fids = set()
                    for f in selected_features:
                        indices_fids = indices_fids.union(set([(m, f.id) for m in f.members]))
                    indices_fids = sorted(list(indices_fids))
                    load_from_indices_fids(indices_fids, True)

        indices_fids = st.session_state.get(f"loaded_indices_{id}", [])
        datapoints = st.session_state.get(f"loaded_data_points_{id}", [])

        if st.session_state.get(f"data_explorer_siamese_{id}", False):
            l, r = st.columns(2)
            with l:
                st.text(f"Inliers ({len(datapoints)})")
                image_view(indices_fids, datapoints)
            
            siamese_indices_fids = st.session_state.get(f"siamese_indices_{id}", [])
            siamese_datapoints = st.session_state.get(f"siamese_data_points_{id}", [])
            
            with r:
                st.text(f"Outliers ({len(siamese_datapoints)})")
                image_view(siamese_indices_fids, siamese_datapoints, True)
        else:
            image_view(indices_fids, datapoints)
    
    def show_utility(container, viewer_func, name):
        with container:
            show = True
            
            if many_classes:
                show = st.checkbox(f"Show {name} View", key=f"show_cov_view_{name}", value=False)

            if show:
                viewer_func()
            else:
                st.info(f"{name} view is hidden")    

    show_utility(f2c, f2c_viewer, "Feature to Class")
    show_utility(acc, acc_viewer, "Accuracy")
    show_utility(c2f, c2f_viewer, "Class to Features")
    show_utility(cc, cc_viewer, "Class to Feature")

    # ---------- Data Explorer: branch on model type ----------
    @st.fragment
    def bert_datex_viewer():
        """Token-level data explorer for BERT NER experiments."""
        node2label = exp.dataset.labels_by_split[exp.split]
        labels_arr = np.array(get_labels(exp))
        preds_arr  = np.array(get_preds(exp))
        ner_classes = exp.dataset.classes
        sentences = get_bert_sentences(exp.split)  # sentences[dataset_idx] = list[str]

        try:
            losses_arr = load_bert_flat_losses(exp)
        except Exception as e:
            st.warning(f"Could not load flat losses: {e}")
            losses_arr = np.zeros(len(node2label), dtype=np.float32)

        coords_path = os.path.join(
            st.session_state.complexes_dir,
            exp.split,
            f"token_coords_a{exp.tag}_{exp.epoch_tag}.pt",
        )
        try:
            coords = torch.load(coords_path, weights_only=True).numpy()  # (N_valid, 2)
        except Exception:
            coords = np.zeros((len(node2label), 2), dtype=np.int64)

        def token_table_view(indices_fids: list[tuple[int, int]]):
            per_page = 100
            rows = []
            for node_idx, fid in indices_fids:
                if node_idx < 0 or node_idx >= len(node2label):
                    continue
                sent_idx = int(coords[node_idx, 0]) if node_idx < len(coords) else -1
                word_idx = int(coords[node_idx, 1]) if node_idx < len(coords) else -1
                true_lbl = ner_classes[int(labels_arr[node_idx])] if node_idx < len(labels_arr) else "?"
                pred_lbl = ner_classes[int(preds_arr[node_idx])]  if node_idx < len(preds_arr)  else "?"
                loss = float(losses_arr[node_idx]) if node_idx < len(losses_arr) else float("nan")
                if sentences and 0 <= sent_idx < len(sentences):
                    words = sentences[sent_idx]
                    parts = [f"[{w.upper()}]" if i == word_idx else w for i, w in enumerate(words)]
                    sentence_str = " ".join(parts)
                else:
                    sentence_str = f"sent={sent_idx}, word={word_idx}"
                rows.append({
                    "Node": node_idx,
                    "Feature": fid,
                    "Sentence": sentence_str,
                    "True": true_lbl,
                    "Pred": pred_lbl,
                    "Loss": round(loss, 5),
                    "Correct": true_lbl == pred_lbl,
                })
            if not rows:
                st.info("No valid tokens in selection.")
                return
            df = pd.DataFrame(rows)
            N = len(df)
            if N > per_page:
                page_count = (N + per_page - 1) // per_page
                page = st.slider(
                    "Page", min_value=1, max_value=max(1, page_count), value=1,
                    key=f"bert_datex_page_{id}",
                ) - 1
                st.dataframe(
                    df.iloc[page * per_page:(page + 1) * per_page],
                    hide_index=True, width="stretch",
                    key=f"bert_token_table_{id}",
                )
            else:
                st.dataframe(df, hide_index=True, width="stretch",
                             key=f"bert_token_table_{id}")

        with st.container(horizontal=True, vertical_alignment="center"):
            mode = st.selectbox(
                "Load Mode",
                options=["From Indices", "From Selected Features"],
                index=1,
                key=f"bert_datex_mode_{id}",
            )
            if mode == "From Indices":
                cs_idx = st.text_input(
                    "Token Indices (comma-separated)",
                    key=f"bert_datex_indices_{id}", value=""
                )

        if mode == "From Indices" and cs_idx.strip() != "":
            indices_fids = []
            for part in cs_idx.split(","):
                part = part.strip()
                if part.isdigit():
                    idx = int(part)
                    fid = node2feat[idx].id if idx < len(node2feat) else -1
                    indices_fids.append((idx, fid))
            st.session_state[f"bert_loaded_indices_{id}"] = indices_fids
        elif mode == "From Selected Features":
            if st.button("Load from Selected Features", key=f"bert_load_from_feats_{id}"):
                indices_fids = set()
                for f in selected_features:
                    indices_fids.update((m, f.id) for m in f.members)
                st.session_state[f"bert_loaded_indices_{id}"] = sorted(indices_fids)

        token_table_view(st.session_state.get(f"bert_loaded_indices_{id}", []))

    if isinstance(exp, BertExperiment) or st.session_state.is_bert:
        show_utility(datex, bert_datex_viewer, "Data Explorer")
    else:
        show_utility(datex, datex_viewer, "Data Explorer")