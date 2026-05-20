"""GPT inference pipeline for paramgolf models.

Mirrors bert_ner/run_train_inference.py, adapted for next-token-prediction LMs.
No training is performed; all outputs are labelled epoch 0.
Model weights are loaded from a frozen state dict (.pt / .npz / .npy / .np).

Run:
    python run_inference.py --config runs/smeargate_bos.yaml -o logs/run.log

Saved layout (mirrors bert_ner):
    Losses/{split}/losses_e0.pt              (N, seq_len) float32
    Predictions/{split}/predictions_e0.pt    (N, seq_len) int64
    Labels/{split}/labels_e0.pt             (N, seq_len) int64
    Tensors/{split}/mask_e0.pt              (N, seq_len) bool   [all True: every token is valid]
    Tensors/{split}/a{tag}_e0.pt            (N, seq_len, hidden_size) float32
    Indices/{split}/selected_dataset_indices.txt  1-D sorted global sequence indices

trainUval offsets val sequence indices by train_total_seqs so the combined index
space is contiguous, matching the bert_ner convention.

When ds_subsample is set the selected sequences are drawn *before* inference so
that only those sequences are actually processed (efficient for large datasets).
When ds_subsample is None the full split is processed.

Token-chunk layout
------------------
Each logical sequence i consumes seq_len + 1 consecutive raw tokens:
    x[i]   = tokens[i*seq_len   : i*seq_len + seq_len]      (seq_len inputs)
    y[i]   = tokens[i*seq_len+1 : i*seq_len + seq_len + 1]  (seq_len targets)
count_total_seqs() uses (total_tokens - 1) // seq_len, which is the exact upper
bound on valid i.

Activation / loss correspondence
---------------------------------
This model is a causal decoder-only LM.  The "encoder" and "decoder" labels in
GPT._forward_hidden refer only to a U-Net-style skip-connection arrangement
within a single causal forward pass — NOT to separate source/target sequences.
Every token position t attends only to positions <= t at every block, and every
position has both a well-defined causal activation and a next-token prediction
loss.  The correspondence activation[t] <-> loss[t] is strict 1-to-1 for all
block layers.

Parallel-block hook limitation
------------------------------
Blocks with index >= parallel_start_layer (default: 8) are processed via
GPT._parallel_block(), which calls block.attn(...) and block.mlp(...) directly
and BYPASSES block.forward().  Consequently:
  * A hook on  blocks[N]       (N >= parallel_start_layer) NEVER fires.
  * A hook on  blocks[N].attn  always fires; i[0] is the post-attn-norm input.
  * A hook on  blocks[N].mlp   always fires; i[0] is the post-mlp-norm input.
  * A hook on  final_norm      always fires; i[0] is the merged hidden state
                               after all blocks, before the LM projection.
For blocks below parallel_start_layer, blocks[N] fires normally and i[0] is the
pre-norm residual stream x entering that block.
"""

import argparse
import gc
import glob
import importlib.util
import os
import sys
import time
from pathlib import Path
from logging import FileHandler, Formatter, Logger, StreamHandler
from typing import Dict, List, Optional, Tuple

import numpy as np
import psutil
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

logger = Logger(__name__)

BATCH_LOG_INTERVAL = 10


# ---------------------------------------------------------------------------
# Memory utilities (verbatim from bert_ner/run_train_inference.py)
# ---------------------------------------------------------------------------

def get_memory_usage(device=None):
    proc = psutil.Process(os.getpid())
    cpu_rss = proc.memory_info().rss
    cpu_vms = proc.memory_info().vms
    gpu_stats = {}
    if torch.cuda.is_available():
        if device is None:
            device = torch.device('cuda')
        gpu_stats = {
            'gpu_allocated':     torch.cuda.memory_allocated(device),
            'gpu_reserved':      torch.cuda.memory_reserved(device),
            'gpu_max_allocated': torch.cuda.max_memory_allocated(device),
            'gpu_max_reserved':  torch.cuda.max_memory_reserved(device),
        }
    return {'cpu_rss': cpu_rss, 'cpu_vms': cpu_vms, **gpu_stats}


def log_memory_stats(device, step, phase='', prev_mem=None):
    mem = get_memory_usage(device)
    parts = [f'Memory [{phase}]:' if phase else 'Memory:']
    parts.append(f'CPU RSS={mem["cpu_rss"]/1e9:.2f}GB')
    parts.append(f'CPU VMS={mem["cpu_vms"]/1e9:.2f}GB')
    if 'gpu_allocated' in mem:
        parts.append(f'GPU alloc={mem["gpu_allocated"]/1e9:.2f}GB')
        parts.append(f'GPU reserved={mem["gpu_reserved"]/1e9:.2f}GB')
    if prev_mem is not None:
        parts.append(f'(CPU delta={(mem["cpu_rss"] - prev_mem["cpu_rss"])/1e6:.1f}MB)')
        if 'gpu_allocated' in mem and 'gpu_allocated' in prev_mem:
            parts.append(f'(GPU delta={(mem["gpu_allocated"] - prev_mem["gpu_allocated"])/1e6:.1f}MB)')
    logger.info(' '.join(parts))
    return mem


def log_collected_shapes(collected_act, tag=''):
    prefix = f'[{tag}] ' if tag else ''
    for k, v in collected_act.items():
        if isinstance(v, list) and len(v) > 0:
            logger.info(
                f'{prefix}Collected activations for tag {k!r}: '
                f'{len(v)} batches, example shape: {v[0].shape}'
            )
        else:
            logger.info(f'{prefix}No activations collected for tag {k!r} yet.')


# ---------------------------------------------------------------------------
# Hook attachment (verbatim from bert_ner/run_train_inference.py)
# Captures i[0]: the first positional input to the hooked module.
#
# Valid hook targets and what they capture (all shapes (B, seq_len, model_dim)):
#   blocks[N]       N < parallel_start_layer: pre-norm residual stream x.
#                   N >= parallel_start_layer: HOOK NEVER FIRES (see module docstring).
#   blocks[N].attn  all N: post-attn-norm input; fires for both regular and parallel blocks.
#   blocks[N].mlp   all N: post-mlp-norm input;  fires for both regular and parallel blocks.
#   final_norm      merged hidden state after all blocks, before LM projection.
# ---------------------------------------------------------------------------

def _get_parallel_block_indices(model) -> set:
    """Return the set of block indices processed via _parallel_block().

    These are the decoder-half blocks with index >= parallel_start_layer.
    With looping_active=False (inference default):
        encoder  : range(0, num_encoder_layers)
        decoder  : range(num_encoder_layers, num_encoder_layers + num_decoder_layers)
        parallel : {i for i in decoder_range if i >= parallel_start_layer}
    """
    psl = getattr(model, 'parallel_start_layer', 0)
    if psl <= 0:
        return set()
    num_enc = getattr(model, 'num_encoder_layers', 0)
    num_dec = getattr(model, 'num_decoder_layers', 0)
    return {i for i in range(num_enc, num_enc + num_dec) if i >= psl}


def _check_hook_collection(model, collection):
    """Warn if any collection item targets a parallel block at the block level.

    A hook on blocks[N] where N is in the parallel zone will never fire because
    _parallel_block() bypasses Block.forward().  Use blocks[N].attn or
    blocks[N].mlp instead, or final_norm for the merged post-block state.
    """
    parallel_indices = _get_parallel_block_indices(model)
    if not parallel_indices:
        return
    import re
    block_re = re.compile(r'^blocks\[(\d+)\]$')
    for item in collection:
        path = item.get('path', '')
        m = block_re.match(path)
        if m:
            idx = int(m.group(1))
            if idx in parallel_indices:
                logger.warning(
                    f"Hook target {path!r} (tag={item.get('tag')!r}) targets a PARALLEL block "
                    f"(index {idx} >= parallel_start_layer={getattr(model,'parallel_start_layer',0)}). "
                    f"This hook will NEVER fire because _parallel_block() bypasses Block.forward(). "
                    f"Use '{path}.attn' to capture the post-attn-norm input, or 'final_norm' for "
                    f"the merged output after all blocks."
                )


def attach_collection_hooks(model, collection, collected_activations):
    """Register forward hooks on named submodules; capture i[0] (input to module)."""
    def find_module(path):
        parts = path.split('.')
        indexed_parts = [
            tuple(p.replace(']', '').split('[')) if '[' in p else (p, None)
            for p in parts
        ]
        mod = model
        for part, idx in indexed_parts:
            mod = getattr(mod, part)
            if idx is not None:
                mod = mod[int(idx)]
        return mod

    _check_hook_collection(model, collection)
    hooks = []
    for item in collection:
        path = item['path']
        tag  = item['tag']
        collected_activations[tag] = []
        mod = find_module(path)
        logger.info(f'Attaching hook to {path!r} (tag={tag!r}): {mod.__class__.__name__}')

        def hook_fn(m, i, o, tag=tag):
            collected_activations[tag].append(i[0].detach().cpu())

        hooks.append(mod.register_forward_hook(hook_fn))
    return hooks


# ---------------------------------------------------------------------------
# Reordering utilities
# ---------------------------------------------------------------------------

def get_total_examples(batches):
    return sum(int(b.shape[0]) for b in batches)


def build_selection_lookup(total_size, selected_positions=None):
    if selected_positions is None:
        selected_positions = torch.arange(total_size, dtype=torch.int64)
    else:
        selected_positions = selected_positions.detach().cpu().to(dtype=torch.int64).reshape(-1)
    lookup = torch.full((total_size,), -1, dtype=torch.int64)
    lookup[selected_positions] = torch.arange(selected_positions.numel(), dtype=torch.int64)
    return selected_positions, lookup


def reorder_2d_batches(value_batches, idx_batches, selection_lookup, selected_size):
    """Reorder (B, seq_len) output batches into canonical sequence-index order.

    value_batches : list of (B, seq_len) tensors  (losses, preds, labels, masks)
    idx_batches   : list of (B,) int64 global sequence indices
    selection_lookup : 1-D tensor; lookup[global_idx] = row in output or -1
    """
    seq_len = value_batches[0].shape[1]
    dtype   = value_batches[0].dtype
    if dtype == torch.bool:
        reordered = torch.zeros((selected_size, seq_len), dtype=torch.bool)
    elif dtype in (torch.int64, torch.long):
        reordered = torch.full((selected_size, seq_len), -1, dtype=torch.int64)
    else:
        reordered = torch.zeros((selected_size, seq_len), dtype=torch.float32)

    filled = 0
    for batch_vals, batch_idx in zip(value_batches, idx_batches):
        vals  = batch_vals.detach().cpu()
        slots = selection_lookup[batch_idx]
        mask  = slots >= 0
        if torch.any(mask):
            reordered[slots[mask]] = vals[mask]
            filled += int(mask.sum().item())

    if filled != selected_size:
        raise RuntimeError(
            f'Reordering mismatch (2D): expected {selected_size} rows, filled {filled}.'
        )
    return reordered


def reorder_3d_activation_batches(act_batches, idx_batches, selection_lookup, selected_size):
    """Reorder (B, seq_len, hidden_size) activation batches into canonical order.

    Since GPT hooks capture the full-resolution token tensor (no word-level
    extraction like BERT NER), this is a direct index-based scatter.

    act_batches : list of (B, seq_len, hidden_size) tensors
    idx_batches : list of (B,) int64 global sequence indices
    Returns (selected_size, seq_len, hidden_size) float32.
    """
    seq_len     = act_batches[0].shape[1]
    hidden_size = act_batches[0].shape[2]
    reordered   = torch.zeros((selected_size, seq_len, hidden_size), dtype=torch.float32)
    filled = 0

    for batch_acts, batch_idx in zip(act_batches, idx_batches):
        acts  = batch_acts.detach().cpu().float()
        slots = selection_lookup[batch_idx]
        mask  = slots >= 0
        if torch.any(mask):
            reordered[slots[mask]] = acts[mask]
            filled += int(mask.sum().item())

    if filled != selected_size:
        raise RuntimeError(
            f'Reordering mismatch (3D activations): expected {selected_size} rows, filled {filled}.'
        )
    return reordered


def clear_collected_tensors(collection):
    if isinstance(collection, dict):
        for v in collection.values():
            if isinstance(v, list):
                v.clear()
        collection.clear()


# ---------------------------------------------------------------------------
# State dict loading
# ---------------------------------------------------------------------------

def load_state_dict_from_file(path: str, device: str = 'cpu') -> dict:
    """Load a model state dict from several formats.

    .pt / .pth   PyTorch save (state_dict, checkpoint dict, or raw model dict).
    .npz         numpy.savez archive; keys must match model state dict keys.
    .npy / .np   numpy.save with allow_pickle=True containing a plain Python
                 dict {name: ndarray}.  (Less common; prefer .npz for state dicts.)
    Any other    Attempted via torch.load as a fallback.
    """
    ext = os.path.splitext(path)[1].lower()
    logger.info(f'Loading state dict from {path!r} (ext={ext!r})...')

    if ext in ('.pt', '.pth'):
        state = torch.load(path, map_location=device, weights_only=False)
        if isinstance(state, dict):
            for key in ('model_state_dict', 'state_dict', 'model'):
                if key in state:
                    logger.info(f'  Unwrapped checkpoint key {key!r}.')
                    return state[key]
        return state

    elif ext == '.npz':
        data = np.load(path, allow_pickle=False)
        return {k: torch.from_numpy(np.array(data[k])) for k in data.files}

    elif ext in ('.npy', '.np'):
        data = np.load(path, allow_pickle=True)
        # np.save of a dict produces a 0-d object array.
        obj = data.item() if (data.ndim == 0) else data
        if isinstance(obj, dict):
            return {k: torch.from_numpy(np.array(v)) for k, v in obj.items()}
        raise ValueError(
            f'Cannot load a model state dict from a plain ndarray file {path!r}. '
            'Save it with np.savez(**state_dict) and use the .npz extension.'
        )

    else:
        logger.warning(f'Unknown extension {ext!r}; attempting torch.load as fallback.')
        state = torch.load(path, map_location=device, weights_only=False)
        if isinstance(state, dict):
            for key in ('model_state_dict', 'state_dict', 'model'):
                if key in state:
                    return state[key]
        return state


# ---------------------------------------------------------------------------
# Token file helpers
# ---------------------------------------------------------------------------

def resolve_token_files(pattern: str) -> List[Path]:
    """Expand glob, excluding CaseOps per-token byte sidecar shards."""
    return sorted(
        Path(p) for p in glob.glob(pattern)
        if '_bytes_' not in Path(p).name
    )


def count_total_seqs(files: List[Path], seq_len: int, tg_module) -> int:
    """Count total complete sequences across all shards.

    Each sequence i requires tokens[i*seq_len : i*seq_len + seq_len + 1], i.e. seq_len+1
    consecutive tokens (seq_len inputs + 1 extra for the last target token).  The formula
    (total_tokens - 1) // seq_len is the exact number of such non-overlapping windows.
    """
    total_tokens = sum(tg_module._read_num_tokens(f) for f in files)
    return max(0, (total_tokens - 1) // seq_len)


# ---------------------------------------------------------------------------
# Subsampling (random; no class-balance because LM has no per-sequence label)
# ---------------------------------------------------------------------------

def build_random_subsample_indices(
    total_seqs: int,
    ds_subsample: Optional[int],
    split_name: str,
    seed: int = 42,
) -> Optional[torch.Tensor]:
    """Return a sorted 1-D tensor of selected sequence indices, or None (full split)."""
    if ds_subsample is None:
        return None
    ds_subsample = int(ds_subsample)
    if ds_subsample <= 0 or ds_subsample >= total_seqs:
        if ds_subsample > total_seqs:
            logger.warning(
                f'ds_subsample={ds_subsample} exceeds split size {total_seqs} '
                f'for {split_name!r}; using full split.'
            )
        return None
    rng = torch.Generator().manual_seed(seed)
    indices = torch.randperm(total_seqs, generator=rng)[:ds_subsample]
    return indices.sort().values


# ---------------------------------------------------------------------------
# Inference loop
# ---------------------------------------------------------------------------

def run_inference(
    model,
    token_files: List[Path],
    seq_len: int,
    batch_seqs: int,
    device: torch.device,
    acts: dict,
    tg_module,
    selected_positions: Optional[torch.Tensor] = None,
    split_name: str = '',
) -> dict:
    """Forward-pass inference over token shards, collecting per-token outputs.

    When selected_positions is given only those global sequence indices are
    forwarded; all other sequences in the shards are skipped.  Hooks attached
    to the model fire only for processed batches, so collected_activations
    will contain exactly the batches processed here.

    Returns a dict with list-of-batch tensors ready for reorder_*_batches().
    """
    model.eval()

    # Build a sorted list of global indices to process (or None = all).
    if selected_positions is not None:
        sel_list = selected_positions.sort().values.tolist()
    else:
        sel_list = None

    sel_ptr = 0  # pointer into sel_list (advances shard by shard)
    global_seq_offset = 0

    collected_losses  = []
    collected_preds   = []
    collected_labels  = []
    collected_masks   = []
    collected_indices = []

    prev_mem  = None
    step_num  = 0
    use_autocast = device.type == 'cuda'

    with torch.no_grad():
        for shard_path in tqdm(token_files, desc=f'Inference [{split_name}] shards'):
            shard_tokens = tg_module.load_data_shard(shard_path).to(torch.int64)
            num_shard_seqs = max(0, (shard_tokens.numel() - 1) // seq_len)

            if num_shard_seqs == 0:
                global_seq_offset += num_shard_seqs
                continue

            shard_global_start = global_seq_offset
            shard_global_end   = global_seq_offset + num_shard_seqs

            # Determine which local (shard-relative) indices to process.
            if sel_list is not None:
                # Advance pointer to first index in this shard.
                while sel_ptr < len(sel_list) and sel_list[sel_ptr] < shard_global_start:
                    sel_ptr += 1
                # Collect all selected indices in [shard_global_start, shard_global_end).
                local_global_pairs: List[Tuple[int, int]] = []
                tmp_ptr = sel_ptr
                while tmp_ptr < len(sel_list) and sel_list[tmp_ptr] < shard_global_end:
                    gi = sel_list[tmp_ptr]
                    local_global_pairs.append((gi - shard_global_start, gi))
                    tmp_ptr += 1

                if not local_global_pairs:
                    global_seq_offset += num_shard_seqs
                    continue

                # Advance sel_ptr past indices consumed from this shard.
                sel_ptr = tmp_ptr

                # Process in batches of batch_seqs.
                for b_start in range(0, len(local_global_pairs), batch_seqs):
                    batch_pairs = local_global_pairs[b_start: b_start + batch_seqs]
                    local_idxs  = [p[0] for p in batch_pairs]
                    global_idxs = torch.tensor([p[1] for p in batch_pairs], dtype=torch.int64)

                    x_list = [
                        shard_tokens[li * seq_len: li * seq_len + seq_len]
                        for li in local_idxs
                    ]
                    y_list = [
                        shard_tokens[li * seq_len + 1: li * seq_len + seq_len + 1]
                        for li in local_idxs
                    ]
                    x = torch.stack(x_list).to(device, non_blocking=True)
                    y = torch.stack(y_list).to(device, non_blocking=True)
                    B = x.size(0)

                    with torch.autocast(
                        device_type=device.type, dtype=torch.bfloat16,
                        enabled=use_autocast,
                    ):
                        logits = model.forward_logits(x)  # (B, seq_len, vocab_size)

                    per_token_loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)).float(),
                        y.reshape(-1),
                        reduction='none',
                    ).view(B, seq_len).detach().cpu()

                    preds = logits.argmax(-1).detach().cpu()
                    mask  = torch.ones(B, seq_len, dtype=torch.bool)

                    collected_losses.append(per_token_loss)
                    collected_preds.append(preds)
                    collected_labels.append(y.cpu())
                    collected_masks.append(mask)
                    collected_indices.append(global_idxs)

                    step_num += 1
                    if step_num % BATCH_LOG_INTERVAL == 0:
                        prev_mem = log_memory_stats(
                            device, step_num,
                            f'{split_name}_step{step_num}', prev_mem,
                        )
                        log_collected_shapes(acts, tag=split_name)

            else:
                # Process all sequences in this shard.
                for b_start in tqdm(
                    range(0, num_shard_seqs, batch_seqs),
                    desc=f'  shard {shard_path.name}', leave=False,
                ):
                    b_end = min(b_start + batch_seqs, num_shard_seqs)
                    B = b_end - b_start
                    global_idxs = torch.arange(
                        shard_global_start + b_start,
                        shard_global_start + b_end,
                        dtype=torch.int64,
                    )

                    x_list = [
                        shard_tokens[li * seq_len: li * seq_len + seq_len]
                        for li in range(b_start, b_end)
                    ]
                    y_list = [
                        shard_tokens[li * seq_len + 1: li * seq_len + seq_len + 1]
                        for li in range(b_start, b_end)
                    ]
                    x = torch.stack(x_list).to(device, non_blocking=True)
                    y = torch.stack(y_list).to(device, non_blocking=True)

                    with torch.autocast(
                        device_type=device.type, dtype=torch.bfloat16,
                        enabled=use_autocast,
                    ):
                        logits = model.forward_logits(x)

                    per_token_loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)).float(),
                        y.reshape(-1),
                        reduction='none',
                    ).view(B, seq_len).detach().cpu()

                    preds = logits.argmax(-1).detach().cpu()
                    mask  = torch.ones(B, seq_len, dtype=torch.bool)

                    collected_losses.append(per_token_loss)
                    collected_preds.append(preds)
                    collected_labels.append(y.cpu())
                    collected_masks.append(mask)
                    collected_indices.append(global_idxs)

                    step_num += 1
                    if step_num % BATCH_LOG_INTERVAL == 0:
                        prev_mem = log_memory_stats(
                            device, step_num,
                            f'{split_name}_step{step_num}', prev_mem,
                        )
                        log_collected_shapes(acts, tag=split_name)

            global_seq_offset += num_shard_seqs

    return {
        'losses':  collected_losses,
        'preds':   collected_preds,
        'labels':  collected_labels,
        'masks':   collected_masks,
        'indices': collected_indices,
        'seq_len': seq_len,
    }


# ---------------------------------------------------------------------------
# Save inference outputs
# ---------------------------------------------------------------------------

def save_inference_outputs(
    epoch: int,
    train_outputs: dict,
    val_outputs: dict,
    train_activations: dict,
    val_activations: dict,
    output_root: str,
    train_total_seqs: int,
    val_total_seqs: int,
) -> None:
    """Save per-token per-sequence inference outputs mirroring bert_ner conventions.

    The selected sequence indices are already encoded in output['indices'] —
    no further subsampling happens here.  trainUval offsets val indices by
    train_total_seqs for a contiguous joint index space.
    """
    logger.info(f'Saving inference outputs for epoch {epoch}...')
    os.makedirs(output_root, exist_ok=True)

    def write_output(split: str, output: dict, activations: dict, total_seqs_split: int):
        loss_dir   = os.path.join(output_root, 'Losses',      split)
        preds_dir  = os.path.join(output_root, 'Predictions', split)
        labels_dir = os.path.join(output_root, 'Labels',      split)
        tens_dir   = os.path.join(output_root, 'Tensors',     split)
        idxs_dir   = os.path.join(output_root, 'Indices',     split)
        for d in (loss_dir, preds_dir, labels_dir, tens_dir, idxs_dir):
            os.makedirs(d, exist_ok=True)

        # Build global-index → output-slot lookup.
        all_indices = torch.cat(output['indices'])  # (N_processed,)
        N           = int(all_indices.numel())
        seq_len     = output['seq_len']

        # Guard against indices that exceed the declared total (e.g. truncation).
        max_gi = int(all_indices.max().item()) + 1 if N > 0 else 0
        lookup_size = max(total_seqs_split, max_gi)
        selected_positions, selection_lookup = build_selection_lookup(
            lookup_size, all_indices,
        )
        selected_size = N

        # Reorder 2-D tensors into global-index order.
        reordered_losses = reorder_2d_batches(
            output['losses'], output['indices'], selection_lookup, selected_size,
        )
        reordered_preds  = reorder_2d_batches(
            output['preds'],  output['indices'], selection_lookup, selected_size,
        )
        reordered_labels = reorder_2d_batches(
            output['labels'], output['indices'], selection_lookup, selected_size,
        )
        reordered_masks  = reorder_2d_batches(
            output['masks'],  output['indices'], selection_lookup, selected_size,
        )

        torch.save(reordered_losses.float(), os.path.join(loss_dir,   f'losses_e{epoch}.pt'))
        torch.save(reordered_preds,          os.path.join(preds_dir,  f'predictions_e{epoch}.pt'))
        torch.save(reordered_labels,         os.path.join(labels_dir, f'labels_e{epoch}.pt'))
        torch.save(reordered_masks,          os.path.join(tens_dir,   f'mask_e{epoch}.pt'))
        np.savetxt(
            os.path.join(idxs_dir, 'selected_dataset_indices.txt'),
            selected_positions.numpy(), fmt='%d',
        )
        logger.info(
            f'Saved {selected_size} sequences × {seq_len} tokens for split {split!r}.'
        )

        del reordered_losses, reordered_preds, reordered_labels, reordered_masks
        gc.collect()

        # Activations: (B, seq_len, hidden_size) hooks → (N, seq_len, hidden_size) tensors.
        for tag, act_batches in activations.items():
            if not act_batches:
                logger.warning(
                    f'No activations collected for tag {tag!r}, split {split!r}.'
                )
                continue
            logger.info(f'Reordering activations for tag {tag!r} (split={split!r})...')
            reordered_acts = reorder_3d_activation_batches(
                act_batches, output['indices'], selection_lookup, selected_size,
            )
            logger.info(
                f'Saving activations tag={tag!r} shape={tuple(reordered_acts.shape)}'
            )
            torch.save(
                reordered_acts.float(),
                os.path.join(tens_dir, f'a{tag}_e{epoch}.pt'),
            )
            del reordered_acts
            gc.collect()

        del selection_lookup, selected_positions, all_indices
        gc.collect()

    write_output('train', train_outputs, train_activations, train_total_seqs)
    write_output('val',   val_outputs,   val_activations,   val_total_seqs)

    # trainUval: offset val indices by train_total_seqs (bert_ner convention).
    val_indices_offset = [
        idx + train_total_seqs for idx in val_outputs['indices']
    ]
    combined_outputs = {
        'losses':  train_outputs['losses']  + val_outputs['losses'],
        'preds':   train_outputs['preds']   + val_outputs['preds'],
        'labels':  train_outputs['labels']  + val_outputs['labels'],
        'masks':   train_outputs['masks']   + val_outputs['masks'],
        'indices': list(train_outputs['indices']) + val_indices_offset,
        'seq_len': train_outputs['seq_len'],
    }
    combined_acts = {
        tag: train_activations.get(tag, []) + val_activations.get(tag, [])
        for tag in set(list(train_activations) + list(val_activations))
    }
    trainUval_total = train_total_seqs + val_total_seqs

    write_output('trainUval', combined_outputs, combined_acts, trainUval_total)

    clear_collected_tensors(combined_outputs)
    clear_collected_tensors(combined_acts)
    gc.collect()

    logger.info(f'Inference outputs saved for epoch {epoch}.')


# ---------------------------------------------------------------------------
# Model import and construction
# ---------------------------------------------------------------------------

def import_train_gpt(model_dir: str):
    """Import train_gpt.py from the given directory as a module."""
    model_dir = os.path.abspath(model_dir)
    spec = importlib.util.spec_from_file_location(
        'train_gpt', os.path.join(model_dir, 'train_gpt.py'),
    )
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot find train_gpt.py in {model_dir!r}')
    tg = importlib.util.module_from_spec(spec)
    sys.modules.setdefault('train_gpt', tg)  # avoid double-load if already imported
    spec.loader.exec_module(tg)
    return tg


def build_model_from_cfg(cfg: dict, device: torch.device, tg):
    """Instantiate GPT and load frozen state dict.

    cfg['model_env'] is a dict of env-var overrides consumed by Hyperparameters.
    cfg['state_dict'] is the path to the frozen weights file.

    .ptz files are quantized + compressed (brotli or lzma) and require weight
    rebanking.  tg.deserialize(h, device) handles all of that internally, so
    the model is built and loaded in one step when the path ends in .ptz.
    """
    # Apply env-var overrides before constructing Hyperparameters.
    for k, v in cfg.get('model_env', {}).items():
        os.environ[str(k)] = str(v)

    # model_env was already applied in do_run before the import; the loop
    # below is a no-op at this point but kept for safety (idempotent).
    for k, v in cfg.get('model_env', {}).items():
        os.environ[str(k)] = str(v)

    h = tg.Hyperparameters()

    state_dict_path = cfg['state_dict']
    if state_dict_path.lower().endswith('.ptz'):
        # deserialize() creates the model, decompresses (brotli/lzma),
        # dequantizes, and rebanks weights — all in one call.
        logger.info(f'Loading .ptz quantized model from {state_dict_path!r}')
        h.quantized_model_path = state_dict_path
        model = tg.deserialize(h, device)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f'Model loaded from .ptz: {n_params:,} parameters, device={device}')
        return model, h

    model = tg.GPT(h).to(device).bfloat16()
    tg.restore_fp32_params(model)

    sd = load_state_dict_from_file(state_dict_path, device='cpu')
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        logger.warning(f'Missing keys in state dict ({len(missing)}): {missing[:8]}...')
    if unexpected:
        logger.warning(f'Unexpected keys in state dict ({len(unexpected)}): {unexpected[:8]}...')
    if not missing and not unexpected:
        logger.info('State dict loaded with strict match.')

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f'Model loaded: {n_params:,} parameters, device={device}')
    return model, h


# ---------------------------------------------------------------------------
# Main run orchestration
# ---------------------------------------------------------------------------

def do_run(cfg: dict, device: torch.device) -> None:
    # ------------------------------------------------------------------
    # Apply model_env overrides BEFORE importing train_gpt.py.
    # Hyperparameters attributes are class-level (evaluated at class-definition
    # time, i.e. at import time via exec_module).  Setting os.environ after the
    # import has no effect on already-evaluated class attributes.
    # ------------------------------------------------------------------
    for k, v in cfg.get('model_env', {}).items():
        os.environ[str(k)] = str(v)

    # ------------------------------------------------------------------
    # Locate and import train_gpt.py.
    # ------------------------------------------------------------------
    default_model_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '../../paramgolf/SmearGateBOSFix_3Seed',
    )
    train_gpt_dir = cfg.get('train_gpt_dir', default_model_dir)
    logger.info(f'Importing train_gpt from {train_gpt_dir!r}')
    tg = import_train_gpt(train_gpt_dir)

    # ------------------------------------------------------------------
    # Build model.
    # ------------------------------------------------------------------
    model, h = build_model_from_cfg(cfg, device, tg)

    # ------------------------------------------------------------------
    # Resolve inference settings.
    # ------------------------------------------------------------------
    infer_cfg    = cfg['infer']
    collection   = infer_cfg.get('collect', [])
    seq_len      = int(infer_cfg.get('seq_len', h.eval_seq_len))
    batch_seqs   = int(infer_cfg.get('batch_seqs', 16))
    output_root  = infer_cfg['output_root']
    ds_subsample = infer_cfg.get('ds_subsample', None)
    seed         = int(cfg.get('seed', 42))

    # ------------------------------------------------------------------
    # Resolve data files.  DATA_PATH / TOKENIZER_PATH may be overridden
    # in cfg['model_env'] before Hyperparameters() is constructed above.
    # ------------------------------------------------------------------
    train_files = resolve_token_files(h.train_files)
    val_files   = resolve_token_files(h.val_files)

    if not train_files:
        raise FileNotFoundError(
            f'No train shards found for pattern: {h.train_files}'
        )
    if not val_files:
        raise FileNotFoundError(
            f'No val shards found for pattern: {h.val_files}'
        )

    logger.info(f'Train shards: {len(train_files)}  |  Val shards: {len(val_files)}')

    train_total_seqs = count_total_seqs(train_files, seq_len, tg)
    val_total_seqs   = count_total_seqs(val_files,   seq_len, tg)
    logger.info(
        f'Total sequences — train: {train_total_seqs:,}  val: {val_total_seqs:,}'
    )

    # ------------------------------------------------------------------
    # Pre-compute subsampled indices (drawn before inference so only
    # those sequences are forwarded through the model).
    # ------------------------------------------------------------------
    if ds_subsample is not None:
        train_selected = build_random_subsample_indices(
            train_total_seqs, ds_subsample, 'train', seed,
        )
        val_selected = build_random_subsample_indices(
            val_total_seqs, ds_subsample, 'val', seed + 1,
        )
        n_train = int(train_selected.numel()) if train_selected is not None else train_total_seqs
        n_val   = int(val_selected.numel())   if val_selected   is not None else val_total_seqs
        logger.info(
            f'Subsampled — train: {n_train:,}  val: {n_val:,}  (ds_subsample={ds_subsample})'
        )
    else:
        train_selected = None
        val_selected   = None
        logger.info('Processing full splits (ds_subsample=null).')

    prev_mem = get_memory_usage(device)

    # ------------------------------------------------------------------
    # Train split inference.
    # ------------------------------------------------------------------
    logger.info('--- Running inference on training split ---')
    train_activations: dict = {}
    train_hooks = attach_collection_hooks(model, collection, train_activations)

    train_outputs = run_inference(
        model, train_files, seq_len, batch_seqs, device,
        train_activations, tg,
        selected_positions=train_selected,
        split_name='train',
    )

    for h_hook in train_hooks:
        h_hook.remove()
    prev_mem = log_memory_stats(device, 0, 'post_train_inference', prev_mem)
    gc.collect()

    # ------------------------------------------------------------------
    # Val split inference.
    # ------------------------------------------------------------------
    logger.info('--- Running inference on validation split ---')
    val_activations: dict = {}
    val_hooks = attach_collection_hooks(model, collection, val_activations)

    val_outputs = run_inference(
        model, val_files, seq_len, batch_seqs, device,
        val_activations, tg,
        selected_positions=val_selected,
        split_name='val',
    )

    for h_hook in val_hooks:
        h_hook.remove()
    prev_mem = log_memory_stats(device, 0, 'post_val_inference', prev_mem)
    gc.collect()

    # ------------------------------------------------------------------
    # Save.
    # ------------------------------------------------------------------
    logger.info('--- Saving inference outputs ---')
    try:
        save_inference_outputs(
            epoch=0,
            train_outputs=train_outputs,
            val_outputs=val_outputs,
            train_activations=train_activations,
            val_activations=val_activations,
            output_root=output_root,
            train_total_seqs=train_total_seqs,
            val_total_seqs=val_total_seqs,
        )
    finally:
        clear_collected_tensors(train_outputs)
        clear_collected_tensors(val_outputs)
        clear_collected_tensors(train_activations)
        clear_collected_tensors(val_activations)
        gc.collect()

    post_mem = get_memory_usage(device)
    logger.info(
        f'Run complete.  CPU RSS={post_mem["cpu_rss"]/1e9:.2f}GB  '
        f'GPU alloc={post_mem.get("gpu_allocated", 0)/1e9:.2f}GB'
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description='GPT inference pipeline — frozen state dict, epoch 0 only',
    )
    parser.add_argument('--config', '-c', required=True, help='YAML config file path')
    parser.add_argument('--output_file', '-o', help='Log file path (timestamped)')
    args = parser.parse_args(argv)

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    log_formatter = Formatter('%(asctime)s - %(levelname)s - %(message)s')
    if args.output_file:
        fname        = os.path.splitext(args.output_file)[0]
        log_filename = f'{fname}_{time.strftime("%Y%m%d_%H%M%S")}.log'
        fh = FileHandler(log_filename, mode='w')
        fh.setFormatter(log_formatter)
        logger.addHandler(fh)
    sh = StreamHandler()
    sh.setFormatter(log_formatter)
    logger.addHandler(sh)
    logger.setLevel('INFO')

    device   = torch.device(config.get('device', 'cuda:0'))
    run_name = config.get('name', 'gpt_inference')
    datestr  = time.strftime('%Y%m%d_%H%M%S')

    # Stamp a unique output sub-directory so repeated runs don't overwrite.
    cfg = dict(config)
    cfg['infer'] = dict(config['infer'])
    cfg['infer']['output_root'] = os.path.join(
        config['infer']['output_dir'], run_name, datestr,
    )

    logger.info(f'=== Starting run: {run_name} ===')
    logger.info(f'Output root: {cfg["infer"]["output_root"]}')

    do_run(cfg, device)

    logger.info(f'=== Finished run: {run_name} ===')


if __name__ == '__main__':
    main()
