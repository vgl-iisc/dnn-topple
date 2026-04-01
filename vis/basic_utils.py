import os
import streamlit as st
import numpy as np
import torch
import pyct as ct
from experiment import LossLandscapeExperiment, BertExperiment, NER_LABELS


# ---------------------------------------------------------------------------
# BERT helpers (not cached — called from within cached functions)
# ---------------------------------------------------------------------------

def _bert_token_coords(exp: BertExperiment):
    """Load (N_valid, 2) int64 token-coords tensor from complexes_dir."""
    path = os.path.join(
        st.session_state.complexes_dir,
        exp.split,
        f"token_coords_a{exp.tag}_{exp.epoch_tag}.pt",
    )
    return torch.load(path, weights_only=True)


def _bert_labels_and_update_dataset(exp: BertExperiment) -> list[int]:
    """Load flat token-level NER labels and populate BertDataset fields."""
    label_path = exp.get_paths(
        st.session_state.landscapes_dir, st.session_state.ct_dir
    )["labels"]
    labels_2d = torch.load(label_path, weights_only=True)   # (N_sent, max_words)
    coords = _bert_token_coords(exp)                         # (N_valid, 2)
    flat = labels_2d[coords[:, 0], coords[:, 1]].numpy()    # (N_valid,)
    flat = np.where(flat < 0, 0, flat).astype(int)          # map -100 → 0 ('O')
    flat_list = flat.tolist()

    split = exp.split
    ds = exp.dataset
    counts: dict[str, int] = {cls: 0 for cls in NER_LABELS}
    for lbl in flat_list:
        counts[NER_LABELS[lbl]] += 1

    ds.labels_by_split[split] = flat_list
    ds.size_by_split[split] = len(flat_list)
    ds.class_size_by_split[split] = counts
    ds.largest_class_size_by_split[split] = max(counts.values()) if counts else 0

    return flat_list

_CACHE_HASH_FUNCS = {
    LossLandscapeExperiment: LossLandscapeExperiment.__hash__,
    BertExperiment: BertExperiment.__hash__,
}

_CONLL_SPLIT_FILES = {
    "train": "eng.train",
    "val": "eng.testa",
    "valid": "eng.testa",
    "testa": "eng.testa",
    "test": "eng.testb",
    "testb": "eng.testb",
}
_CONLL_SEED = 17591432379

def _parse_conll_file(filepath: str) -> list[list[str]]:
    """Parse a CoNLL-2003 file into a list of word lists (unpermuted)."""
    sentences: list[list[str]] = []
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    for block in content.strip().split("\n\n"):
        block = block.strip()
        if not block:
            continue
        lines = block.split("\n")
        first_word = lines[0].split()[0] if lines[0].strip() else ""
        if first_word == "-DOCSTART-":
            continue
        words = [line.strip().split()[0] for line in lines if line.strip() and len(line.strip().split()) >= 1]
        if words:
            sentences.append(words)
    return sentences

def _permute_sentences(sentences: list[list[str]]) -> list[list[str]]:
    """Apply the deterministic permutation used by CoNLLDataset (seed=_CONLL_SEED)."""
    N = len(sentences)
    rng_state = torch.get_rng_state()
    torch.manual_seed(_CONLL_SEED)
    perm = torch.randperm(N).tolist()
    torch.set_rng_state(rng_state)
    return [sentences[perm[i]] for i in range(N)]

@st.cache_resource
def get_bert_sentences(split: str) -> list[list[str]]:
    """Return sentences[dataset_idx] = list of words, permuted to match CoNLLDataset order.

    'trainUval' mirrors the training pipeline: train sentences (permuted) followed by
    val sentences (permuted), with val indices offset by train_size.
    """
    conll_dir = st.session_state.get("conll_dir", "")
    if not conll_dir:
        return []

    if split.lower() == "trainuval":
        train_file = os.path.join(conll_dir, "eng.train")
        val_file   = os.path.join(conll_dir, "eng.testa")
        if not os.path.exists(train_file) or not os.path.exists(val_file):
            return []
        return (_permute_sentences(_parse_conll_file(train_file)) +
                _permute_sentences(_parse_conll_file(val_file)))

    fname = _CONLL_SPLIT_FILES.get(split.lower())
    if not fname:
        return []
    filepath = os.path.join(conll_dir, fname)
    if not os.path.exists(filepath):
        return []
    return _permute_sentences(_parse_conll_file(filepath))


@st.cache_resource(hash_funcs=_CACHE_HASH_FUNCS)
def get_preds(exp) -> list[int]:
    if isinstance(exp, BertExperiment) or st.session_state.is_bert:
        if st.session_state.no_preds:
            return get_labels(exp)
        pred_path = exp.get_paths(
            st.session_state.landscapes_dir, st.session_state.ct_dir
        )["predictions"]
        preds_2d = torch.load(pred_path, weights_only=True)   # (N_sent, max_words)
        coords = _bert_token_coords(exp)
        flat = preds_2d[coords[:, 0], coords[:, 1]].numpy()
        flat = np.where(flat < 0, 0, flat).astype(int)
        return flat.tolist()
    # CNN path
    if st.session_state.no_preds:
        return get_labels(exp)
    pred_path = exp.get_paths(
        data_dir=st.session_state.landscapes_dir,
        ct_dir=st.session_state.ct_dir,
    )["predictions"]
    with open(pred_path, "rb") as f:
        preds = np.loadtxt(f, dtype=np.int32).reshape(-1)
    return preds.tolist()

@st.cache_resource(hash_funcs=_CACHE_HASH_FUNCS)
def get_labels(exp) -> list[int]:
    if isinstance(exp, BertExperiment) or st.session_state.is_bert:
        return _bert_labels_and_update_dataset(exp)
    return exp.dataset.labels_by_split[exp.split]

@st.cache_resource(hash_funcs=_CACHE_HASH_FUNCS)
def get_tree(exp):
    ctree_name = exp.get_paths(
        data_dir=st.session_state.landscapes_dir, ct_dir=st.session_state.ct_dir
    )["ctree"]
    topo = ct.TopologicalFeatures()  # type: ignore
    topo.loadData(ctree_name)
    return topo.ctdata, topo

@st.cache_resource(hash_funcs=_CACHE_HASH_FUNCS)
def get_order_and_weights(exp) -> tuple[list[int], list[float]]:
    tree_files = exp.get_paths(
        data_dir=st.session_state.landscapes_dir, ct_dir=st.session_state.ct_dir
    )["ctree"]
    with open(f"{tree_files}.order.dat", "r") as f:
        no_simpl = int(f.readline().strip())
    with open(f"{tree_files}.order.bin", "rb") as file:
        order = [int(f) for f in np.fromfile(file, dtype=np.uint32, count=no_simpl)]
        wts = [float(f) for f in np.fromfile(file, dtype=np.float32, count=no_simpl)]
    return order, wts

@st.cache_resource(hash_funcs=_CACHE_HASH_FUNCS)
def get_partition(exp) -> list[int]:
    if isinstance(exp, BertExperiment) or st.session_state.is_bert:
        labels = get_labels(exp)
        n_valid = len(labels)
        part_path = (
            exp.get_paths(st.session_state.landscapes_dir, st.session_state.ct_dir)["ctree"]
            + ".part.raw"
        )
        with open(part_path, "rb") as f:
            parts = np.fromfile(f, dtype=np.uint32, count=n_valid)
        return parts.tolist()
    # CNN path
    count = len(exp.dataset.labels_by_split[exp.split])
    ctree = exp.get_paths(
        data_dir=st.session_state.landscapes_dir,
        ct_dir=st.session_state.ct_dir,
    )["ctree"]
    with open(f"{ctree}.part.raw", "rb") as f:
        parts = np.fromfile(f, dtype=np.uint32, count=count)
    return parts.tolist()

@st.cache_resource(hash_funcs=_CACHE_HASH_FUNCS)
def get_valley_vs_thresh(exp) -> tuple[list[float], list[int]]:
    data, _ = get_tree(exp)
    simpl = ct.SimplifyCT() # type: ignore
    simpl.setInput(data)

    order, wts = get_order_and_weights(exp)
    fns, num_min, _ = simpl.getSimplificationPlot(order, wts)

    return fns, num_min

def get_filtered_cps_vs_thresh(exp, fnstart_range, fnend_range, types):
    data, _ = get_tree(exp)
    simpl = ct.SimplifyCT() # type: ignore
    simpl.setInput(data)

    order, wts = get_order_and_weights(exp)
    fns, counts = simpl.getFilteredSimplificationPlot(order, wts, fnstart_range[0], fnstart_range[1], fnend_range[0], fnend_range[1], types)

    return fns, counts

@st.cache_resource(hash_funcs=_CACHE_HASH_FUNCS)
def compute_arc_features(exp, simpl: float):
    """
    Computes rich features using the C++ implementation.
    Returns a list of RichFeature objects with all metadata populated.
    """
    if isinstance(exp, BertExperiment) or st.session_state.is_bert:
        # Ensure BertDataset.class_size_by_split is populated before we read it
        get_labels(exp)
    _, topo = get_tree(exp)
    partition = get_partition(exp)
    labels = get_labels(exp)
    preds = get_preds(exp)
    # Get class sizes for computing class coverage
    class_sizes = [exp.dataset.class_size_by_split[exp.split][cls] for cls in exp.dataset.classes]
    # Use C++ implementation for fast computation
    features = ct.computeRichFeatures(topo, -1, simpl, partition, labels, preds, class_sizes)  # type: ignore
    return features