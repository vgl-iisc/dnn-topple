from experiment import LossLandscapeExperiment
import streamlit as st

def render_experiment_selector(id: int, experiments: list[LossLandscapeExperiment]):
    
    def invalidate_experiment_view():
        st.session_state[f"feats_{id}"] = None
        st.session_state[f"computed_simpl_{id}"] = 0.0
        st.session_state[f"node2feat_{id}"] = None
        st.session_state[f"tree_graph_{id}"] = None
        st.session_state.pop(f"loaded_indices_{id}", None)
        st.session_state.pop(f"loaded_data_points_{id}", None)
    
    possible_experiments = experiments
    possible_datasets = list(set(map(lambda exp: exp.dataset.name, possible_experiments)))
    
    with st.container(horizontal=True, vertical_alignment="center", gap="medium") as c:
        selected_dataset = st.selectbox("Dataset", options=possible_datasets, key=f"experiment_dataset_selector_{id}", on_change=invalidate_experiment_view)
                
        possible_splits = list(set([exp.split for exp in possible_experiments if exp.dataset.name == selected_dataset]))
        selected_split = st.selectbox("Split", options=possible_splits, key=f"experiment_split_selector_{id}", on_change=invalidate_experiment_view)
                
        possible_models = list(set([exp.model for exp in possible_experiments if exp.dataset.name == selected_dataset and exp.split == selected_split]))
        selected_model = st.selectbox("Model", options=possible_models, key=f"experiment_model_selector_{id}", on_change=invalidate_experiment_view)
                
        # possible_ks = list(set([exp.k for exp in possible_experiments if exp.dataset.name == selected_dataset and exp.split == selected_split and exp.model == selected_model]))
        # selected_k = st.selectbox("k", options=["None"] + possible_ks, key=f"experiment_k_selector_{id}")
        selected_k = 20
        
        # if selected_k == "None":
        #     return

        possible_epochs = sorted(list(set([exp.epoch for exp in possible_experiments if exp.dataset.name == selected_dataset and exp.split == selected_split and
                                    exp.model == selected_model and exp.k == selected_k])))
        selected_epoch = st.selectbox("Epoch", options=possible_epochs, key=f"experiment_epoch_selector_{id}", on_change=invalidate_experiment_view)

        possible_layers = sorted(list(set([exp.pretty_layer for exp in possible_experiments if exp.dataset.name == selected_dataset and exp.split == selected_split and
                                    exp.model == selected_model and exp.k == selected_k and exp.epoch == selected_epoch])))
        selected_layer = st.selectbox("Layer", options=possible_layers, key=f"experiment_layer_selector_{id}", on_change=invalidate_experiment_view)

        selected_experiment = next((exp for exp in possible_experiments if exp.dataset.name == selected_dataset and exp.split == selected_split and
                                    exp.model == selected_model and exp.k == selected_k and exp.epoch == selected_epoch and exp.pretty_layer == selected_layer), None)
    
    return selected_experiment