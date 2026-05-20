"""GPT inference pipeline for the naive baseline model (paramgolf/naive_base).

Mirrors bert_ner/run_train_inference.py and run_inference.py, adapted for the
naive baseline's different constructor and lack of forward_logits().

Key differences from run_inference.py (SmearGateBOSFix_3Seed):
  * GPT constructor takes explicit kwargs, not a Hyperparameters object.
  * GPT.forward(input_ids, target_ids) returns a scalar loss; there is no
    forward_logits().  A forward_logits() method is monkey-patched onto the
    model after construction (same causal logic, returns (B, T, V) logits).
  * restore_low_dim_params_to_fp32() is used instead of restore_fp32_params().
  * _read_num_tokens() does not exist; shard token counts are read from the
    512-byte binary header instead (same format as load_data_shard).
  * There are NO parallel blocks — all blocks[N] hooks fire correctly.
  * Default seq_len comes from h.train_seq_len (not h.eval_seq_len).

Run:
    python run_inference_naive.py --config runs/naive_base.yaml -o logs/run.log

Saved layout (mirrors bert_ner):
    Losses/{split}/losses_e0.pt              (N, seq_len) float32
    Predictions/{split}/predictions_e0.pt    (N, seq_len) int64
    Labels/{split}/labels_e0.pt             (N, seq_len) int64
    Tensors/{split}/mask_e0.pt              (N, seq_len) bool   [all True]
    Tensors/{split}/a{tag}_e0.pt            (N, seq_len, hidden_size) float32
    Indices/{split}/selected_dataset_indices.txt

trainUval offsets val sequence indices by train_total_seqs.

Token-chunk layout
------------------
Each logical sequence i consumes seq_len + 1 consecutive raw tokens:
    x[i]   = tokens[i*seq_len   : i*seq_len + seq_len]
    y[i]   = tokens[i*seq_len+1 : i*seq_len + seq_len + 1]
count_total_seqs() uses (total_tokens - 1) // seq_len.

Activation / loss correspondence
---------------------------------
All token positions t have a 1-to-1 correspondence between causal activations
and next-token prediction losses.  There are no parallel blocks in the naive
baseline — every blocks[N] hook fires normally.
"""

import argparse
import gc
import glob
import importlib.util
import io
import os
import sys
import time
import types
import zlib
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
# Memory utilities
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
# Hook attachment
# Captures i[0]: the first positional input to the hooked module.
#
# All blocks[N] hooks fire for the naive baseline (no parallel blocks).
# Valid hook targets (shapes all (B, seq_len, model_dim)):
#   blocks[N]    pre-norm residual stream x entering block N.
#   final_norm   hidden state after all blocks, before LM projection.
# ---------------------------------------------------------------------------

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
    """Load a model state dict from .pt/.pth, .npz, .npy/.np, or torch.load fallback."""
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


def _read_num_tokens_from_header(file: Path) -> int:
    """Read token count from the 1024-byte binary shard header (int32 at offset 8).

    Header format (same as naive_base/train_gpt.py load_data_shard):
        header[0] == 20240520  (magic)
        header[1] == 1         (version)
        header[2]              (num_tokens)
    """
    header = np.fromfile(file, dtype='<i4', count=256)
    if header.size < 3 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f'Unexpected shard header in {file}')
    return int(header[2])


def count_total_seqs(files: List[Path], seq_len: int) -> int:
    """Count total complete sequences across all shards.

    Uses (total_tokens - 1) // seq_len — the exact number of non-overlapping
    windows of length seq_len+1 (seq_len inputs + 1 target).
    """
    total_tokens = sum(_read_num_tokens_from_header(f) for f in files)
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
    load_data_shard,
    selected_positions: Optional[torch.Tensor] = None,
    split_name: str = '',
) -> dict:
    """Forward-pass inference over token shards, collecting per-token outputs.

    load_data_shard is the function from train_gpt (tg.load_data_shard).
    model must have a forward_logits(input_ids) method returning (B, T, V).
    """
    model.eval()

    if selected_positions is not None:
        sel_list = selected_positions.sort().values.tolist()
    else:
        sel_list = None

    sel_ptr = 0
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
            shard_tokens = load_data_shard(shard_path).to(torch.int64)
            num_shard_seqs = max(0, (shard_tokens.numel() - 1) // seq_len)

            if num_shard_seqs == 0:
                global_seq_offset += num_shard_seqs
                continue

            shard_global_start = global_seq_offset
            shard_global_end   = global_seq_offset + num_shard_seqs

            if sel_list is not None:
                while sel_ptr < len(sel_list) and sel_list[sel_ptr] < shard_global_start:
                    sel_ptr += 1

                local_global_pairs: List[Tuple[int, int]] = []
                tmp_ptr = sel_ptr
                while tmp_ptr < len(sel_list) and sel_list[tmp_ptr] < shard_global_end:
                    gi = sel_list[tmp_ptr]
                    local_global_pairs.append((gi - shard_global_start, gi))
                    tmp_ptr += 1

                if not local_global_pairs:
                    global_seq_offset += num_shard_seqs
                    continue

                sel_ptr = tmp_ptr

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

        all_indices = torch.cat(output['indices'])
        N           = int(all_indices.numel())
        seq_len     = output['seq_len']

        max_gi = int(all_indices.max().item()) + 1 if N > 0 else 0
        lookup_size = max(total_seqs_split, max_gi)
        selected_positions, selection_lookup = build_selection_lookup(
            lookup_size, all_indices,
        )
        selected_size = N

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
        'train_gpt_naive', os.path.join(model_dir, 'train_gpt.py'),
    )
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot find train_gpt.py in {model_dir!r}')
    tg = importlib.util.module_from_spec(spec)
    sys.modules.setdefault('train_gpt_naive', tg)
    spec.loader.exec_module(tg)
    return tg


def _make_forward_logits(model):
    """Return a forward_logits(input_ids) method for the naive baseline GPT.

    Replicates GPT.forward() but returns (B, seq_len, vocab_size) logits
    instead of a scalar cross-entropy loss.  All sub-module calls are
    identical, so hooks fire exactly as they would during training.
    """
    def forward_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips = []
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[self.num_encoder_layers + i](x, x0)
        # Do NOT reshape before final_norm so that hooks on final_norm receive (B, T, D).
        x = self.final_norm(x)           # (B, seq_len, model_dim)
        B, T, D = x.shape
        x_flat = x.reshape(-1, D)
        if self.tie_embeddings:
            logits_proj = F.linear(x_flat, self.tok_emb.weight)
        else:
            logits_proj = self.lm_head(x_flat)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        return logits.view(B, T, -1)     # (B, seq_len, vocab_size)

    return types.MethodType(forward_logits, model)


def build_model_from_cfg(cfg: dict, device: torch.device, tg):
    """Instantiate naive baseline GPT and load frozen state dict.

    cfg['model_env'] overrides Hyperparameters env vars before construction.
    cfg['state_dict'] is the path to the frozen weights file.
    """
    for k, v in cfg.get('model_env', {}).items():
        os.environ[str(k)] = str(v)

    h = tg.Hyperparameters()

    model = tg.GPT(
        vocab_size=h.vocab_size,
        num_layers=h.num_layers,
        model_dim=h.model_dim,
        num_heads=h.num_heads,
        num_kv_heads=h.num_kv_heads,
        mlp_mult=h.mlp_mult,
        tie_embeddings=h.tie_embeddings,
        tied_embed_init_std=h.tied_embed_init_std,
        logit_softcap=h.logit_softcap,
        rope_base=h.rope_base,
        qk_gain_init=h.qk_gain_init,
    ).to(device).bfloat16()

    tg.restore_low_dim_params_to_fp32(model)

    # Monkey-patch forward_logits so the inference loop can call it uniformly.
    model.forward_logits = _make_forward_logits(model)

    state_dict_path = cfg['state_dict']
    if state_dict_path.lower().endswith('.ptz'):
        # .ptz files are zlib-compressed int8-quantized state dicts produced by
        # quantize_state_dict_int8() + zlib.compress(..., level=9).
        logger.info(f'Loading .ptz quantized model from {state_dict_path!r}')
        with open(state_dict_path, 'rb') as f:
            quant_blob = f.read()
        quant_state = torch.load(
            io.BytesIO(zlib.decompress(quant_blob)), map_location='cpu', weights_only=False,
        )
        sd = tg.dequantize_state_dict_int8(quant_state)
        missing, unexpected = model.load_state_dict(sd, strict=True)
    else:
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
    # Apply model_env overrides BEFORE importing train_gpt.py.
    # Hyperparameters attributes are class-level and evaluated at import time.
    for k, v in cfg.get('model_env', {}).items():
        os.environ[str(k)] = str(v)

    default_model_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '../../paramgolf/naive_base',
    )
    train_gpt_dir = cfg.get('train_gpt_dir', default_model_dir)
    logger.info(f'Importing train_gpt from {train_gpt_dir!r}')
    tg = import_train_gpt(train_gpt_dir)

    model, h = build_model_from_cfg(cfg, device, tg)

    infer_cfg    = cfg['infer']
    collection   = infer_cfg.get('collect', [])
    seq_len      = int(infer_cfg.get('seq_len', h.train_seq_len))
    batch_seqs   = int(infer_cfg.get('batch_seqs', 16))
    output_root  = infer_cfg['output_root']
    ds_subsample = infer_cfg.get('ds_subsample', None)
    seed         = int(cfg.get('seed', 42))

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

    train_total_seqs = count_total_seqs(train_files, seq_len)
    val_total_seqs   = count_total_seqs(val_files,   seq_len)
    logger.info(
        f'Total sequences — train: {train_total_seqs:,}  val: {val_total_seqs:,}'
    )

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

    logger.info('--- Running inference on training split ---')
    train_activations: dict = {}
    train_hooks = attach_collection_hooks(model, collection, train_activations)

    train_outputs = run_inference(
        model, train_files, seq_len, batch_seqs, device,
        train_activations, tg.load_data_shard,
        selected_positions=train_selected,
        split_name='train',
    )

    for h_hook in train_hooks:
        h_hook.remove()
    prev_mem = log_memory_stats(device, 0, 'post_train_inference', prev_mem)
    gc.collect()

    logger.info('--- Running inference on validation split ---')
    val_activations: dict = {}
    val_hooks = attach_collection_hooks(model, collection, val_activations)

    val_outputs = run_inference(
        model, val_files, seq_len, batch_seqs, device,
        val_activations, tg.load_data_shard,
        selected_positions=val_selected,
        split_name='val',
    )

    for h_hook in val_hooks:
        h_hook.remove()
    prev_mem = log_memory_stats(device, 0, 'post_val_inference', prev_mem)
    gc.collect()

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
        description='GPT inference pipeline — naive baseline, frozen state dict, epoch 0 only',
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
    run_name = config.get('name', 'gpt_inference_naive')
    datestr  = time.strftime('%Y%m%d_%H%M%S')

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
