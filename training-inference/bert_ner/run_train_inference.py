"""
BERT NER training + inference pipeline for CoNLL-2003.
Mirrors the imagenet/run_train_inference.py pipeline, adapted for sequence labeling.

Run:
	python run_train_inference.py --config runs/conll.yaml -o logs/run.log
"""

import argparse
import copy
import gc
import os
import time
import yaml

import numpy as np
import psutil
import torch
import torch.nn as nn
from logging import FileHandler, Formatter, Logger, StreamHandler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import bert_ner_model
import conll_loader
from conll_loader import IGNORE_INDEX

logger = Logger(__name__)

BATCH_LOG_INTERVAL = 10


# ---------------------------------------------------------------------------
# Memory utilities (verbatim from imagenet/run_train_inference.py)
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


def log_memory_stats(device, writer, epoch, step=None, phase='', prev_mem=None):
	mem = get_memory_usage(device)
	parts = [f'Memory [{phase}]:' if phase else 'Memory:']
	parts.append(f'CPU RSS={mem["cpu_rss"]/1e9:.2f}GB')
	parts.append(f'CPU VMS={mem["cpu_vms"]/1e9:.2f}GB')
	if 'gpu_allocated' in mem:
		parts.append(f'GPU alloc={mem["gpu_allocated"]/1e9:.2f}GB')
		parts.append(f'GPU reserved={mem["gpu_reserved"]/1e9:.2f}GB')
	if prev_mem is not None:
		parts.append(f'(CPU delta={( mem["cpu_rss"] - prev_mem["cpu_rss"])/1e6:.1f}MB)')
		if 'gpu_allocated' in mem and 'gpu_allocated' in prev_mem:
			parts.append(f'(GPU delta={(mem["gpu_allocated"] - prev_mem["gpu_allocated"])/1e6:.1f}MB)')
	logger.info(' '.join(parts))
	if writer is not None:
		gs = step if step is not None else epoch
		prefix = f'mem/{phase}/' if phase else 'mem/'
		writer.add_scalar(f'{prefix}cpu_rss', mem['cpu_rss'], gs)
		writer.add_scalar(f'{prefix}cpu_vms', mem['cpu_vms'], gs)
		if 'gpu_allocated' in mem:
			writer.add_scalar(f'{prefix}gpu_allocated',     mem['gpu_allocated'],     gs)
			writer.add_scalar(f'{prefix}gpu_reserved',      mem['gpu_reserved'],      gs)
			writer.add_scalar(f'{prefix}gpu_max_allocated', mem['gpu_max_allocated'], gs)
			writer.add_scalar(f'{prefix}gpu_max_reserved',  mem['gpu_max_reserved'],  gs)
	return mem


def log_collected_shapes(collected_act, tag=''):
	prefix = f'[{tag}] ' if tag else ''
	for k, v in collected_act.items():
		if isinstance(v, list) and len(v) > 0:
			logger.info(f'{prefix}Collected activations for tag {k!r}: {len(v)} batches, example shape: {v[0].shape}')
		else:
			logger.info(f'{prefix}No activations collected for tag {k!r} yet.')


# ---------------------------------------------------------------------------
# Hook attachment (verbatim from imagenet, spatial_subsample omitted)
# ---------------------------------------------------------------------------

def attach_collection_hooks(model, collection, collected_activations):
	"""Register forward hooks on named submodules; capture i[0] (input to layer)."""
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
# Reordering utilities (NER-specific: 2-D word-level outputs and 3-D activations)
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


def reorder_2d_batches(value_batches, perm_batches, selection_lookup, selected_size):
	"""Reorder padded-sequence outputs of shape (B, max_words) into canonical sentence order.

	Works for float32 (losses), int64 (labels, preds), and bool (masks).
	"""
	max_words = value_batches[0].shape[1]
	first = value_batches[0].detach().cpu()
	reordered = torch.empty((selected_size, max_words), dtype=first.dtype)
	if first.dtype == torch.bool:
		reordered.fill_(False)
	elif first.dtype in (torch.int64, torch.long):
		reordered.fill_(IGNORE_INDEX)
	else:
		reordered.fill_(0.0)

	filled = 0
	for batch_vals, perm_idxs in zip(value_batches, perm_batches):
		vals  = batch_vals.detach().cpu()
		slots = selection_lookup[perm_idxs]
		mask  = slots >= 0
		if torch.any(mask):
			reordered[slots[mask]] = vals[mask]
			filled += int(mask.sum().item())

	if filled != selected_size:
		raise RuntimeError(f'Reordering mismatch (2D): expected {selected_size} rows, filled {filled}.')
	return reordered


def reorder_word_activation_batches(activation_batches, fsp_batches, perm_batches,
                                     selection_lookup, selected_size, max_words):
	"""Extract word-level activations from subword-level hook outputs and reorder.

	activation_batches : list of (B, max_subword_length, hidden_size)  – raw hook output
	fsp_batches        : list of (B, max_words) int64                   – first_subword_positions
	perm_batches       : list of (B,) int64                             – sentence perm indices
	Returns (selected_size, max_words, hidden_size) float32.
	"""
	hidden_size = activation_batches[0].shape[2]
	reordered = torch.zeros((selected_size, max_words, hidden_size), dtype=torch.float32)
	filled = 0

	for batch_acts, fsp, perm_idxs in zip(activation_batches, fsp_batches, perm_batches):
		acts = batch_acts.detach().cpu().float()   # (B, max_subword_length, hidden_size)
		fsp  = fsp.detach().cpu()                  # (B, max_words)

		fsp_safe     = fsp.clamp(min=0)
		fsp_expanded = fsp_safe.unsqueeze(-1).expand(-1, -1, hidden_size)  # (B, max_words, hidden_size)
		word_acts    = torch.gather(acts, 1, fsp_expanded)                 # (B, max_words, hidden_size)

		# Zero out positions where fsp was -1 (truncated or padding words).
		invalid = (fsp < 0).unsqueeze(-1).expand_as(word_acts)
		word_acts[invalid] = 0.0

		slots = selection_lookup[perm_idxs]
		mask  = slots >= 0
		if torch.any(mask):
			reordered[slots[mask]] = word_acts[mask]
			filled += int(mask.sum().item())

	if filled != selected_size:
		raise RuntimeError(f'Reordering mismatch (activations): expected {selected_size} rows, filled {filled}.')
	return reordered


def clear_collected_tensors(collection):
	if isinstance(collection, dict):
		for v in collection.values():
			if isinstance(v, list):
				v.clear()
		collection.clear()


# ---------------------------------------------------------------------------
# Dataset-level subsampling (adapted from imagenet; sentence-level for NER)
# ---------------------------------------------------------------------------

def compute_sentence_labels(word_labels):
	"""Summarise per-sentence NER distribution into a single integer label.

	Returns the most frequent entity class (label ID 1-8) per sentence,
	or 0 (O) when the sentence contains no named entities.  Used only by
	build_fixed_balanced_indices when ds_subsample is requested.

	word_labels : (N, max_words) int64, IGNORE_INDEX for padding.
	"""
	N = word_labels.shape[0]
	sent_labels = torch.zeros(N, dtype=torch.int64)
	best_counts = torch.zeros(N, dtype=torch.int64)
	for cls in range(1, len(conll_loader.NER_LABELS)):
		counts = (word_labels == cls).sum(dim=1)  # (N,)
		update = counts > best_counts
		sent_labels[update] = cls
		best_counts[update] = counts[update]
	return sent_labels


def build_fixed_balanced_indices(labels, ds_subsample, split_name):
	"""Build a fixed class-balanced random subset of sentence indices.

	Identical to the imagenet implementation except it operates on per-sentence
	summary labels produced by compute_sentence_labels.
	"""
	if ds_subsample is None:
		return None

	ds_subsample = int(ds_subsample)
	n = int(labels.shape[0])
	if ds_subsample <= 0 or ds_subsample >= n:
		if ds_subsample > n:
			logger.warning(f"ds_subsample={ds_subsample} exceeds split size {n} for '{split_name}'. Using full split.")
		return None

	unique_classes, class_counts = torch.unique(labels, sorted=True, return_counts=True)
	n_classes = int(unique_classes.numel())
	if n_classes == 0:
		return None

	base      = ds_subsample // n_classes
	remainder = ds_subsample % n_classes
	if remainder != 0:
		logger.warning(f"ds_subsample={ds_subsample} not divisible by {n_classes} classes for '{split_name}'.")

	targets  = torch.full((n_classes,), base, dtype=torch.int64)
	targets[:remainder] += 1

	available          = class_counts.to(dtype=torch.int64)
	selected_per_class = torch.minimum(targets, available)
	deficit            = int(ds_subsample - selected_per_class.sum().item())

	if deficit > 0:
		capacity = available - selected_per_class
		for idx in torch.argsort(capacity, descending=True).tolist():
			if deficit <= 0:
				break
			take = min(int(capacity[idx].item()), deficit)
			if take > 0:
				selected_per_class[idx] += take
				deficit -= take

	if deficit > 0:
		logger.warning(f"Could not reach ds_subsample={ds_subsample} for '{split_name}'. "
		               f"Using {int(selected_per_class.sum().item())} samples.")

	selected_positions = []
	for cls_idx, cls in enumerate(unique_classes):
		k = int(selected_per_class[cls_idx].item())
		if k <= 0:
			continue
		cls_pos = torch.where(labels == cls)[0]
		if cls_pos.numel() > k:
			perm   = torch.randperm(int(cls_pos.numel()))
			chosen = cls_pos[perm[:k]]
		else:
			chosen = cls_pos
		selected_positions.append(chosen)

	if not selected_positions:
		return None

	selected_positions, _ = torch.sort(torch.cat(selected_positions))
	return selected_positions


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def set_seed(seed: int):
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	import random; random.seed(seed)
	np.random.seed(seed)


def save_checkpoint(state, ckpt_dir, epoch):
	os.makedirs(ckpt_dir, exist_ok=True)
	path = os.path.join(ckpt_dir, f'checkpoint_e{epoch}.pt')
	torch.save(state, path)
	return path


# ---------------------------------------------------------------------------
# Pipeline components
# ---------------------------------------------------------------------------

def get_dataloaders(cfg):
	assert cfg['dataset'] == 'conll', f"Only 'conll' dataset supported, got {cfg['dataset']!r}"
	return conll_loader.make_conll_dataloaders(
		data_root=cfg['data_root'],
		tokenizer_name=cfg.get('model', 'bert-base-uncased'),
		max_subword_length=cfg.get('max_subword_length', 256),
		max_words=cfg.get('max_words', 128),
		batch_size=cfg['batch_size'],
		shuffle=True,
		random_label_prop=cfg.get('random_prop', 0.0),
	)


def build_model(cfg, num_labels, device):
	return bert_ner_model.create_bert_ner(
		model_name=cfg.get('model', 'bert-base-uncased'),
		num_labels=num_labels,
		dropout=cfg.get('dropout', 0.1),
		checkpoint=cfg.get('checkpoint', None),
		device=str(device),
	)


def make_optimizer(model, cfg):
	opt_cfg = cfg.get('opt', {})
	typ     = opt_cfg.get('type', 'adamw').lower()
	lr      = float(cfg['lr'])
	decay   = float(cfg.get('weight_decay', 0.01))
	if typ == 'adamw':
		return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=decay)
	elif typ == 'adam':
		betas = tuple(cfg.get('betas', (0.9, 0.999)))
		return torch.optim.Adam(model.parameters(), lr=lr, betas=betas, weight_decay=decay)
	elif typ == 'sgd':
		momentum = opt_cfg.get('momentum', 0.9)
		return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=decay)
	else:
		raise ValueError(f'Unsupported optimizer type: {typ!r}')


def make_lr_scheduler(optimizer, cfg, num_training_steps=None):
	"""Return a scheduler or None.

	For linear_warmup_decay the scheduler steps per batch; the caller must step
	it inside the training loop.  For cosineWR / step it steps per epoch.
	"""
	schedule_cfg = cfg.get('schedule', {})
	stype = schedule_cfg.get('type', '')
	if stype == 'cosineWR':
		return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
			optimizer, T_0=schedule_cfg.get('T0', 10), T_mult=schedule_cfg.get('T_mult', 2)
		)
	elif stype == 'step':
		return torch.optim.lr_scheduler.StepLR(
			optimizer,
			step_size=schedule_cfg.get('step_size', 30),
			gamma=schedule_cfg.get('gamma', 0.1),
		)
	elif stype == 'linear_warmup_decay':
		if num_training_steps is None:
			raise ValueError("num_training_steps required for linear_warmup_decay scheduler.")
		warmup_steps = int(schedule_cfg.get('warmup_steps', 0))

		def lr_lambda(current_step):
			if current_step < warmup_steps:
				return float(current_step) / float(max(1, warmup_steps))
			remaining = num_training_steps - current_step
			total     = max(1, num_training_steps - warmup_steps)
			return max(0.0, float(remaining) / float(total))

		return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
	else:
		return None


def _is_batch_level_scheduler(cfg):
	return cfg.get('schedule', {}).get('type', '') == 'linear_warmup_decay'


# ---------------------------------------------------------------------------
# Scalar field helpers
# ---------------------------------------------------------------------------

EXTRA_METRIC_NAMES = ('gradnorm_importance',)


def _compute_per_sentence_gradnorm(per_sent_losses, model):
	"""Compute per-sentence gradient norm ||∇_θ L_i||_2 w.r.t. all model parameters.

	per_sent_losses : (B,) tensor with grad_fn (one summed loss per sentence).
	                  Caller must ensure the computation graph is still alive
	                  (i.e. has not been freed by a prior backward() call).
	Returns         : (B,) float32 CPU tensor.

	Uses retain_graph=True on every call so the same graph can later be used
	for the training loss backward() without an extra forward pass.
	"""
	params = [p for p in model.parameters() if p.requires_grad]
	gnorms = []
	for loss_i in per_sent_losses:
		grads = torch.autograd.grad(
			loss_i, params,
			retain_graph=True,   # kept alive for the subsequent loss.backward()
			create_graph=False,
			allow_unused=True,
		)
		gnorm_sq = sum(g.detach().norm() ** 2 for g in grads if g is not None)
		gnorms.append(gnorm_sq.item() ** 0.5)
	return torch.tensor(gnorms, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Train / validate
# ---------------------------------------------------------------------------

def _get_max_words(loader):
	"""Traverse Subset/ConcatDataset wrappers to find the underlying CoNLLDataset.max_words."""
	ds = loader.dataset
	while not hasattr(ds, 'max_words') and hasattr(ds, 'dataset'):
		ds = ds.dataset
	return ds.max_words


def train_epoch(model, loader, criterion, criterion_collect, optimizer,
                scheduler, step_per_batch, device, epoch, writer, acts,
                max_grad_norm=1.0, extra_metrics=frozenset()):
	model.train()
	running_loss    = 0.0
	total_valid     = 0
	total_correct   = 0
	max_words       = _get_max_words(loader)

	collected_word_losses       = []
	collected_word_labels       = []
	collected_word_preds        = []
	collected_word_masks        = []
	collected_fsp               = []
	collected_perm_indices      = []
	collected_gradnorm_importance = []

	need_gradnorm = 'gradnorm_importance' in extra_metrics
	prev_mem = None

	for step, batch in tqdm(enumerate(loader), total=len(loader), desc=f'Epoch {epoch}'):
		if step % BATCH_LOG_INTERVAL == 0:
			global_step = epoch * len(loader) + step
			prev_mem = log_memory_stats(device, writer, epoch, global_step, f'train_batch_{step}', prev_mem)
			log_collected_shapes(acts, tag='train')

		input_ids      = batch['input_ids'].to(device)
		attention_mask = batch['attention_mask'].to(device)
		token_type_ids = batch['token_type_ids'].to(device)
		labels         = batch['labels'].to(device)         # (B, max_subword_length) subword-level
		word_labels    = batch['word_labels']               # (B, max_words) CPU
		word_mask      = batch['word_mask']                 # (B, max_words) bool CPU
		fsp            = batch['first_subword_positions']   # (B, max_words) int64 CPU
		perm_idxs      = batch['perm_index']                # (B,) int64 CPU

		logits = model(input_ids, attention_mask, token_type_ids)  # (B, seq_len, num_labels)
		B, seq_len, num_labels = logits.shape

		# Training loss: mean CE over all first-subword positions (label != -100).
		loss = criterion(logits.view(-1, num_labels), labels.view(-1))

		# Per-sentence gradnorm: reuse the live computation graph before zero_grad.
		# retain_graph=True on every call keeps the graph alive for loss.backward().
		if need_gradnorm:
			per_token_loss_g = criterion_collect(
				logits.view(-1, num_labels), labels.view(-1)
			).view(B, seq_len)  # (B, seq_len), keeps grad_fn
			valid_float      = (labels != IGNORE_INDEX).float().detach()
			per_sent_losses  = (per_token_loss_g * valid_float).sum(dim=1)  # (B,) with grad_fn
			batch_gnorms     = _compute_per_sentence_gradnorm(per_sent_losses, model)
			per_token_loss   = per_token_loss_g.detach().cpu()
		else:
			batch_gnorms = None
			# Per-token loss for collection (no gradient needed).
			with torch.no_grad():
				per_token_loss = criterion_collect(
					logits.view(-1, num_labels), labels.view(-1)
				).view(B, seq_len).detach().cpu()

		with torch.no_grad():
			preds = logits.argmax(-1).detach().cpu()   # (B, seq_len)

			fsp_safe     = fsp.clamp(min=0)            # (B, max_words)
			word_losses  = torch.gather(per_token_loss, 1, fsp_safe) * word_mask.float()
			word_preds   = torch.gather(preds, 1, fsp_safe)          # (B, max_words)
			word_preds_stored        = word_preds.clone()
			word_preds_stored[~word_mask] = IGNORE_INDEX

			if batch_gnorms is not None:
				# Broadcast sentence-level gradnorm to word level; 0 at padding positions.
				gnorm_word = batch_gnorms.unsqueeze(1).expand(B, max_words) * word_mask.float()

		optimizer.zero_grad()
		loss.backward()
		torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
		optimizer.step()
		if step_per_batch and scheduler is not None:
			scheduler.step()

		if writer is not None and step % BATCH_LOG_INTERVAL == 0:
			writer.add_scalar('train/batch_loss', loss.item(), epoch * len(loader) + step)

		valid_count   = int(word_mask.sum().item())
		running_loss += loss.item() * valid_count
		total_valid  += valid_count
		total_correct += int(((word_preds == word_labels) & word_mask).sum().item())

		collected_word_losses.append(word_losses.detach())
		collected_word_labels.append(word_labels.detach())
		collected_word_preds.append(word_preds_stored.detach())
		collected_word_masks.append(word_mask.detach())
		collected_fsp.append(fsp.detach())
		collected_perm_indices.append(perm_idxs.to(dtype=torch.int64))
		if batch_gnorms is not None:
			collected_gradnorm_importance.append(gnorm_word.detach())

	epoch_loss = running_loss / max(total_valid, 1)
	epoch_acc  = total_correct / max(total_valid, 1)

	return epoch_loss, epoch_acc, {
		'word_losses':              collected_word_losses,
		'word_labels':              collected_word_labels,
		'word_preds':               collected_word_preds,
		'word_masks':               collected_word_masks,
		'first_subword_positions':  collected_fsp,
		'perm_indices':             collected_perm_indices,
		'max_words':                max_words,
		'gradnorm_importance':      collected_gradnorm_importance,
	}


def validate(model, loader, criterion, criterion_collect, device, epoch=0, writer=None, acts={},
             extra_metrics=frozenset()):
	model.eval()
	running_loss  = 0.0
	total_valid   = 0
	total_correct = 0
	max_words     = _get_max_words(loader)

	collected_word_losses         = []
	collected_word_labels         = []
	collected_word_preds          = []
	collected_word_masks          = []
	collected_fsp                 = []
	collected_perm_indices        = []
	collected_gradnorm_importance = []

	need_gradnorm = 'gradnorm_importance' in extra_metrics
	prev_mem = None

	with torch.no_grad():
		for step, batch in enumerate(tqdm(loader, total=len(loader), desc='Validation')):
			if step % BATCH_LOG_INTERVAL == 0:
				global_step = epoch * len(loader) + step
				prev_mem = log_memory_stats(device, writer, epoch, global_step, f'val_batch_{step}', prev_mem)
				log_collected_shapes(acts, tag='val')

			input_ids      = batch['input_ids'].to(device)
			attention_mask = batch['attention_mask'].to(device)
			token_type_ids = batch['token_type_ids'].to(device)
			labels         = batch['labels'].to(device)
			word_labels    = batch['word_labels']
			word_mask      = batch['word_mask']
			fsp            = batch['first_subword_positions']
			perm_idxs      = batch['perm_index']

			logits = model(input_ids, attention_mask, token_type_ids)
			B, seq_len, num_labels = logits.shape

			loss           = criterion(logits.view(-1, num_labels), labels.view(-1))
			per_token_loss = criterion_collect(
				logits.view(-1, num_labels), labels.view(-1)
			).view(B, seq_len).cpu()

			preds        = logits.argmax(-1).cpu()
			fsp_safe     = fsp.clamp(min=0)
			word_losses  = torch.gather(per_token_loss, 1, fsp_safe) * word_mask.float()
			word_preds   = torch.gather(preds, 1, fsp_safe)
			word_preds_stored           = word_preds.clone()
			word_preds_stored[~word_mask] = IGNORE_INDEX

			valid_count   = int(word_mask.sum().item())
			running_loss += loss.item() * valid_count
			total_valid  += valid_count
			total_correct += int(((word_preds == word_labels) & word_mask).sum().item())

			collected_word_losses.append(word_losses.detach())
			collected_word_labels.append(word_labels.detach())
			collected_word_preds.append(word_preds_stored.detach())
			collected_word_masks.append(word_mask.detach())
			collected_fsp.append(fsp.detach())
			collected_perm_indices.append(perm_idxs.to(dtype=torch.int64))

			if need_gradnorm:
				# Second forward pass with grad tracking to compute per-sentence gradient norms.
				with torch.enable_grad():
					logits_g = model(input_ids, attention_mask, token_type_ids)
					ptl_g    = criterion_collect(
						logits_g.view(-1, num_labels), labels.view(-1)
					).view(B, seq_len)
					valid_float     = (labels != IGNORE_INDEX).float().detach()
					per_sent_losses = (ptl_g * valid_float).sum(dim=1)
					batch_gnorms    = _compute_per_sentence_gradnorm(per_sent_losses, model)
					del logits_g, ptl_g, per_sent_losses
				gnorm_word = batch_gnorms.unsqueeze(1).expand(B, max_words) * word_mask.float()
				collected_gradnorm_importance.append(gnorm_word.detach())

	epoch_loss = running_loss / max(total_valid, 1)
	epoch_acc  = total_correct / max(total_valid, 1)

	return epoch_loss, epoch_acc, {
		'word_losses':             collected_word_losses,
		'word_labels':             collected_word_labels,
		'word_preds':              collected_word_preds,
		'word_masks':              collected_word_masks,
		'first_subword_positions': collected_fsp,
		'perm_indices':            collected_perm_indices,
		'max_words':               max_words,
		'gradnorm_importance':     collected_gradnorm_importance,
	}


# ---------------------------------------------------------------------------
# Save inference outputs
# ---------------------------------------------------------------------------

def save_inference_outputs(
	epoch,
	train_outputs,
	val_outputs,
	train_activations,
	val_activations,
	output_root,
	ds_subsample=None,
	fixed_indices_by_split=None,
):
	"""Save per-token per-sentence inference outputs.

	Saved tensors:
	  Losses/{split}/losses_e{epoch}.pt           (N_sent, max_words) float32
	  Predictions/{split}/predictions_e{epoch}.pt (N_sent, max_words) int64
	  Labels/{split}/labels_e{epoch}.pt           (N_sent, max_words) int64
	  Tensors/{split}/mask_e{epoch}.pt            (N_sent, max_words) bool
	  Tensors/{split}/a{tag}_e{epoch}.pt          (N_sent, max_words, hidden_size) float32
	  Indices/{split}/selected_dataset_indices.txt 1D sentence perm indices
	"""
	logger.info(f'Saving inference outputs for epoch {epoch}...')
	os.makedirs(output_root, exist_ok=True)
	if fixed_indices_by_split is None:
		fixed_indices_by_split = {}

	def write_output(split, output, collected_activations):
		loss_dir    = os.path.join(output_root, 'Losses',      split)
		preds_dir   = os.path.join(output_root, 'Predictions', split)
		labels_dir  = os.path.join(output_root, 'Labels',      split)
		tens_dir    = os.path.join(output_root, 'Tensors',     split)
		idxs_dir    = os.path.join(output_root, 'Indices',     split)
		gnorm_dir   = os.path.join(output_root, 'GradNorm',    split)
		for d in (loss_dir, preds_dir, labels_dir, tens_dir, idxs_dir):
			os.makedirs(d, exist_ok=True)

		max_words      = output['max_words']
		total_examples = get_total_examples(output['perm_indices'])
		perm_batches   = [idx.detach().cpu().to(dtype=torch.int64).reshape(-1)
		                   for idx in output['perm_indices']]
		fsp_batches    = [f.detach().cpu() for f in output['first_subword_positions']]

		# Build full lookup to compute sentence-level labels for ds_subsample.
		all_positions, full_lookup = build_selection_lookup(total_examples)
		reordered_labels_full = reorder_2d_batches(
			output['word_labels'], perm_batches, full_lookup, total_examples,
		)  # (N, max_words)

		# Determine (and optionally build) the fixed sentence subset.
		selected_positions = fixed_indices_by_split.get(split)
		if ds_subsample is not None and selected_positions is None:
			sent_labels       = compute_sentence_labels(reordered_labels_full)
			selected_positions = build_fixed_balanced_indices(sent_labels, ds_subsample, split)
			fixed_indices_by_split[split] = selected_positions

		if selected_positions is not None:
			selected_positions = selected_positions.to(dtype=torch.int64)
			max_idx = int(selected_positions.max().item()) if selected_positions.numel() > 0 else -1
			if max_idx >= total_examples:
				logger.warning(f"Stored ds_subsample indices for {split!r} out of range; recomputing.")
				sent_labels        = compute_sentence_labels(reordered_labels_full)
				selected_positions = build_fixed_balanced_indices(sent_labels, ds_subsample, split)
				fixed_indices_by_split[split] = selected_positions

		selected_perm_indices, selection_lookup = build_selection_lookup(total_examples, selected_positions)
		selected_size = selected_perm_indices.numel()

		reordered_losses = reorder_2d_batches(output['word_losses'], perm_batches, selection_lookup, selected_size)
		reordered_preds  = reorder_2d_batches(output['word_preds'],  perm_batches, selection_lookup, selected_size)
		reordered_masks  = reorder_2d_batches(output['word_masks'],  perm_batches, selection_lookup, selected_size)
		reordered_labels = reordered_labels_full[selected_perm_indices]

		torch.save(reordered_losses, os.path.join(loss_dir,   f'losses_e{epoch}.pt'))
		torch.save(reordered_preds,  os.path.join(preds_dir,  f'predictions_e{epoch}.pt'))
		torch.save(reordered_labels, os.path.join(labels_dir, f'labels_e{epoch}.pt'))
		torch.save(reordered_masks,  os.path.join(tens_dir,   f'mask_e{epoch}.pt'))
		np.savetxt(os.path.join(idxs_dir, 'selected_dataset_indices.txt'),
		           selected_perm_indices.numpy(), fmt='%d')

		# Activations: extract word-level from subword-level hook outputs.
		for tag, activations in collected_activations.items():
			logger.info(f'Reordering activations for tag {tag!r} (split={split})...')
			reordered_acts = reorder_word_activation_batches(
				activations, fsp_batches, perm_batches,
				selection_lookup, selected_size, max_words,
			)
			logger.info(f'Saving activations for tag {tag!r}, shape: {reordered_acts.shape}')
			torch.save(reordered_acts, os.path.join(tens_dir, f'a{tag}_e{epoch}.pt'))

			# L2 norm of each word's activation vector.
			l2 = reordered_acts.norm(dim=-1)  # (N, max_words)
			torch.save(l2, os.path.join(tens_dir, f'l2_{tag}_e{epoch}.pt'))
			del l2, reordered_acts
			gc.collect()

		# Gradient-norm importance field.
		gradnorm_batches = output.get('gradnorm_importance', [])
		if gradnorm_batches:
			os.makedirs(gnorm_dir, exist_ok=True)
			reordered_gradnorm = reorder_2d_batches(
				gradnorm_batches, perm_batches, selection_lookup, selected_size,
			)
			torch.save(reordered_gradnorm, os.path.join(gnorm_dir, f'gradnorm_importance_e{epoch}.pt'))
			del reordered_gradnorm
			gc.collect()

		del reordered_losses, reordered_preds, reordered_masks, reordered_labels
		del reordered_labels_full, perm_batches, selection_lookup
		del full_lookup, all_positions, selected_perm_indices
		gc.collect()

	write_output('train', train_outputs, train_activations)
	write_output('val',   val_outputs,   val_activations)

	# trainUval: offset val perm_indices by train_size (same as imagenet pipeline).
	train_size       = get_total_examples(train_outputs['perm_indices'])
	combined_outputs = {}
	for key in ('word_losses', 'word_labels', 'word_preds', 'word_masks', 'first_subword_positions'):
		combined_outputs[key] = train_outputs[key] + val_outputs[key]
	train_perm = [idx.to(dtype=torch.int64) for idx in train_outputs['perm_indices']]
	val_perm   = [idx.to(dtype=torch.int64) + train_size for idx in val_outputs['perm_indices']]
	combined_outputs['perm_indices']        = train_perm + val_perm
	combined_outputs['max_words']           = train_outputs['max_words']
	combined_outputs['gradnorm_importance'] = (
		train_outputs.get('gradnorm_importance', []) +
		val_outputs.get('gradnorm_importance', [])
	)

	combined_acts = {tag: train_activations[tag] + val_activations[tag]
	                 for tag in train_activations}

	write_output('trainUval', combined_outputs, combined_acts)
	clear_collected_tensors(combined_outputs)
	clear_collected_tensors(combined_acts)
	gc.collect()

	logger.info(f'Inference outputs saved for epoch {epoch}.')


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def do_run(
	cfg,
	device,
	tboard_base,
	checkpoints_base,
	inference_cfg,
	only_last_best=False,
	start_checkpoint=None,
	start_epoch=1,
):
	train_loader, val_loader = get_dataloaders(cfg)
	num_labels = train_loader.dataset.num_labels

	logger.info(f'Using config: {cfg}')

	model = build_model(cfg, num_labels, device)

	criterion         = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
	criterion_collect = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, reduction='none')
	optimizer         = make_optimizer(model, cfg)
	checkpoint_data   = None

	if start_checkpoint is not None:
		logger.info(f'Loading start checkpoint: {start_checkpoint}')
		checkpoint_data = torch.load(start_checkpoint, map_location=device)
		model.load_state_dict(checkpoint_data['model_state_dict'])
		if 'optimizer_state_dict' in checkpoint_data:
			optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
		ckpt_epoch = checkpoint_data.get('epoch')
		if isinstance(ckpt_epoch, int) and start_epoch != (ckpt_epoch + 1):
			raise ValueError(
				f'start_epoch={start_epoch} but checkpoint epoch={ckpt_epoch}. '
				f'Set --start_epoch {ckpt_epoch + 1}.'
			)
		logger.info(f'Loaded checkpoint epoch={checkpoint_data.get("epoch", "unknown")}.')

	epochs             = cfg['epochs']
	num_training_steps = epochs * len(train_loader)
	scheduler          = make_lr_scheduler(optimizer, cfg, num_training_steps=num_training_steps)
	step_per_batch     = _is_batch_level_scheduler(cfg)

	if scheduler is not None:
		if checkpoint_data is not None and 'scheduler_state_dict' in checkpoint_data:
			scheduler.load_state_dict(checkpoint_data['scheduler_state_dict'])
			logger.info('Restored scheduler state from checkpoint.')
		elif start_epoch > 1:
			logger.warning('No scheduler state in checkpoint; fast-forwarding scheduler.')
			if step_per_batch:
				logger.warning(
					'linear_warmup_decay is a batch-level scheduler; fast-forward by epoch count '
					'is approximate. Exact recovery requires scheduler_state_dict.'
				)
			for _ in range(1, start_epoch):
				scheduler.step()
			logger.info(f'Scheduler fast-forwarded by {start_epoch - 1} steps.')

	writer = SummaryWriter(tboard_base)
	set_seed(cfg['seed'])

	try:
		writer.add_text('config', yaml.dump(cfg))
		writer.add_text('inference_config', yaml.dump(inference_cfg))
	except Exception:
		pass

	best_val_acc  = 0.0
	best_epoch    = 0
	ckpt_dir      = checkpoints_base
	prev_mem      = get_memory_usage(device)
	ds_subsample  = inference_cfg.get('ds_subsample', None)
	max_grad_norm = float(cfg.get('max_grad_norm', 1.0))
	extra_metrics = frozenset(inference_cfg.get('metrics', []))
	fixed_ds_subsample_indices = {}

	collection = inference_cfg.get('collect', [])

	for epoch in range(start_epoch, epochs + 1):
		t0 = time.time()
		logger.info(f'--- Epoch {epoch}/{epochs} ---')
		prev_mem = log_memory_stats(device, writer, epoch, None, f'epoch_{epoch}_start', prev_mem)

		# Epoch collection schedule (verbatim from imagenet / autoencoder pipelines).
		should_collect_inference = False
		epochs_config = inference_cfg.get('epochs', 'all')
		if epochs_config == 'all' or 'all' in (epochs_config if isinstance(epochs_config, list) else []):
			should_collect_inference = True
		elif 'last' in (epochs_config if isinstance(epochs_config, list) else [epochs_config]) and epoch == epochs:
			should_collect_inference = True
		elif isinstance(epochs_config, list) and len(epochs_config) == 3 and all(isinstance(x, int) for x in epochs_config):
			should_collect_inference = (epoch - 1) in list(range(epochs_config[0], epochs_config[1], epochs_config[2]))
		elif isinstance(epochs_config, list):
			should_collect_inference = (epoch - 1) in epochs_config
		elif isinstance(epochs_config, int):
			should_collect_inference = ((epoch - 1) % epochs_config == 0)

		logger.info(f'Collecting inference outputs this epoch: {should_collect_inference}')

		train_activations = {}
		val_activations   = {}
		train_hooks       = []
		val_hooks         = []

		if should_collect_inference:
			train_hooks = attach_collection_hooks(model, collection, train_activations)

		train_loss, train_acc, train_outputs = train_epoch(
			model, train_loader, criterion, criterion_collect, optimizer,
			scheduler, step_per_batch, device, epoch, writer, train_activations,
			max_grad_norm=max_grad_norm, extra_metrics=extra_metrics,
		)

		if should_collect_inference:
			for h in train_hooks:
				h.remove()
			val_hooks = attach_collection_hooks(model, collection, val_activations)

		if scheduler is not None and not step_per_batch:
			scheduler.step()

		logger.info('Evaluating on validation set...')
		val_loss, val_acc, val_outputs = validate(
			model, val_loader, criterion, criterion_collect,
			device, epoch, writer, val_activations,
			extra_metrics=extra_metrics,
		)

		if should_collect_inference:
			for h in val_hooks:
				h.remove()

		gc.collect()
		prev_mem = log_memory_stats(device, writer, epoch, None, f'epoch_{epoch}_end', prev_mem)

		writer.add_scalar('train/loss',      train_loss, epoch)
		writer.add_scalar('train/token_acc', train_acc,  epoch)
		writer.add_scalar('val/loss',        val_loss,   epoch)
		writer.add_scalar('val/token_acc',   val_acc,    epoch)
		for i, pg in enumerate(optimizer.param_groups):
			writer.add_scalar(f'optimizer/group_{i}_lr', pg.get('lr', 0.0), epoch)

		epoch_time = time.time() - t0
		logger.info(
			f'Epoch {epoch}/{epochs} - train_loss={train_loss:.4f} train_acc={train_acc:.4f} '
			f'val_loss={val_loss:.4f} val_acc={val_acc:.4f} ({epoch_time:.1f}s)'
		)

		if should_collect_inference:
			try:
				save_inference_outputs(
					epoch - 1,
					train_outputs, val_outputs,
					train_activations, val_activations,
					inference_cfg['output_root'],
					ds_subsample=ds_subsample,
					fixed_indices_by_split=fixed_ds_subsample_indices,
				)
			except Exception as e:
				logger.error(f'Error saving inference outputs for epoch {epoch}: {e}')
			finally:
				clear_collected_tensors(train_outputs)
				clear_collected_tensors(val_outputs)
				clear_collected_tensors(train_activations)
				clear_collected_tensors(val_activations)
				gc.collect()
				post_save_mem = get_memory_usage(device)
				logger.info(
					f'Memory after inference save epoch {epoch}: '
					f'CPU RSS={post_save_mem["cpu_rss"]/1e9:.2f}GB '
					f'GPU alloc={post_save_mem.get("gpu_allocated", 0)/1e9:.2f}GB'
				)

		state = {
			'epoch':               epoch,
			'model_state_dict':    model.state_dict(),
			'optimizer_state_dict': optimizer.state_dict(),
			'cfg':                 cfg,
		}
		if scheduler is not None:
			state['scheduler_state_dict'] = scheduler.state_dict()

		save_checkpoint(state, ckpt_dir, epoch)
		if only_last_best and epoch > start_epoch:
			try:
				os.remove(os.path.join(ckpt_dir, f'checkpoint_e{epoch - 1}.pt'))
			except FileNotFoundError:
				pass

		if val_acc > best_val_acc:
			best_val_acc = val_acc
			if best_epoch > 0:
				try:
					os.remove(os.path.join(ckpt_dir, f'best_e{best_epoch}.pt'))
				except FileNotFoundError:
					pass
			best_epoch = epoch
			torch.save(state, os.path.join(ckpt_dir, f'best_e{epoch}.pt'))

	logger.info(f'Best validation token accuracy: {best_val_acc:.4f} @ epoch {best_epoch}')
	writer.close()


def main(argv=None):
	parser = argparse.ArgumentParser(description='BERT NER training + inference pipeline')
	parser.add_argument('--config',         '-c', required=True,  help='YAML config file')
	parser.add_argument('--output_file',    '-o',                  help='log file base path')
	parser.add_argument('--tensorboard_dir', default=None,         help='override tensorboard dir')
	parser.add_argument('--checkpoint_dir', default=None,          help='override checkpoint dir')
	parser.add_argument('--start_checkpoint', default=None,        help='checkpoint to resume from')
	parser.add_argument('--start_epoch',    type=int, default=1,   help='epoch to resume at')
	args = parser.parse_args(argv)

	if args.start_epoch < 1:
		raise ValueError(f'--start_epoch must be >= 1, got {args.start_epoch}')

	with open(args.config, 'r') as f:
		config = yaml.safe_load(f)

	log_formatter = Formatter('%(asctime)s - %(levelname)s - %(message)s')
	if args.output_file:
		fname       = os.path.splitext(args.output_file)[0]
		log_filename = f'{fname}_{time.strftime("%Y%m%d_%H%M%S")}.log'
		fh = FileHandler(log_filename, mode='w')
		fh.setFormatter(log_formatter)
		logger.addHandler(fh)
	sh = StreamHandler()
	sh.setFormatter(log_formatter)
	logger.addHandler(sh)
	logger.setLevel('INFO')

	train_cfg = config['train']
	infer_cfg = config['infer']

	dataset_name        = train_cfg['dataset']
	train_cfg['data_root'] = config['datasets'][dataset_name]

	device         = torch.device(config['device'])
	tensorboard_dir = args.tensorboard_dir or train_cfg.get('tensorboard_dir')
	checkpoint_dir  = args.checkpoint_dir  or train_cfg.get('checkpoints_dir')

	run_name = config['name']
	datestr  = time.strftime('%Y%m%d_%H%M%S')
	tb_dir   = os.path.join(tensorboard_dir, run_name, datestr)
	ckpt_dir = os.path.join(checkpoint_dir,  run_name, datestr)

	run_infer_cfg               = copy.deepcopy(infer_cfg)
	run_infer_cfg['model']      = train_cfg['model']
	run_infer_cfg['output_root'] = os.path.join(infer_cfg['output_dir'], run_name, datestr)

	random_values = config.get('random', [])
	runs = []
	if random_values:
		for r in random_values:
			tcfg               = copy.deepcopy(train_cfg)
			tcfg['random_prop'] = r
			name               = f'{run_name}-r{r}'
			runs.append((
				name, tcfg, device,
				os.path.join(tensorboard_dir, name, datestr),
				os.path.join(checkpoint_dir,  name, datestr),
				{**copy.deepcopy(run_infer_cfg),
				 'output_root': os.path.join(infer_cfg['output_dir'], name, datestr)},
			))
	else:
		runs = [(run_name, train_cfg, device, tb_dir, ckpt_dir, run_infer_cfg)]

	logger.info(f'Using device: {device}')

	for (name, tcfg, dev, tb, ckpt, icfg) in runs:
		os.makedirs(tb,   exist_ok=True)
		os.makedirs(ckpt, exist_ok=True)

		logger.info(f'Inference output root: {icfg["output_root"]}')
		logger.info(f'Inference epochs schedule: {icfg.get("epochs", "all")}')
		logger.info(f'Inference collection layers: {icfg.get("collect", [])}')

		only_last_best = tcfg.get('only_last_best', False)

		logger.info(f'\n=== Starting run: {name} ===')
		do_run(
			tcfg, dev, tb, ckpt, icfg,
			only_last_best=only_last_best,
			start_checkpoint=args.start_checkpoint,
			start_epoch=args.start_epoch,
		)
		logger.info(f'=== Finished run: {name} ===\n')


if __name__ == '__main__':
	main()
