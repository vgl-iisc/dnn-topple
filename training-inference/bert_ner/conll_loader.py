import os
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizerFast

SEED = 17591432379

NER_LABELS = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
LABEL2ID = {l: i for i, l in enumerate(NER_LABELS)}
ID2LABEL = {i: l for i, l in enumerate(NER_LABELS)}
IGNORE_INDEX = -100


def parse_conll_file(filepath: str) -> List[Tuple[List[str], List[str]]]:
	"""Parse a CoNLL-2003 file into a list of (words, ner_tags) sentence pairs.

	Format: 4 whitespace-separated columns per token (word, POS, chunk, NER).
	Sentences are separated by blank lines. Blocks starting with -DOCSTART- are skipped.
	"""
	with open(filepath, 'r', encoding='utf-8') as f:
		content = f.read()

	sentences = []
	for block in content.strip().split('\n\n'):
		block = block.strip()
		if not block:
			continue
		lines = block.split('\n')
		first_token = lines[0].split()[0] if lines[0].strip() else ''
		if first_token == '-DOCSTART-':
			continue
		words, ner_tags = [], []
		for line in lines:
			parts = line.strip().split()
			if len(parts) < 4:
				continue
			words.append(parts[0])
			ner_tags.append(parts[3])
		if words:
			sentences.append((words, ner_tags))
	return sentences


class CoNLLDataset(Dataset):
	"""CoNLL-2003 NER dataset with fixed-seed sentence permutation.

	Every __getitem__ call returns a dict of fixed-length tensors so that the
	default PyTorch collation (tensor stacking) works without a custom collate_fn.

	perm_index is the canonical sentence position (0..N-1).  orig_idx (stored
	under key 'index') is the underlying sentence position after applying the
	permutation, mirroring the convention used by the image dataset loaders.
	"""

	def __init__(
		self,
		filepath: str,
		tokenizer_name: str = 'bert-base-uncased',
		max_subword_length: int = 256,
		max_words: int = 128,
		permute: bool = True,
		random_label_prop: float = 0.0,
	):
		self.sentences = parse_conll_file(filepath)
		self.max_subword_length = max_subword_length
		self.max_words = max_words
		self.num_labels = len(NER_LABELS)
		self.num_classes = self.num_labels  # alias for compatibility with build_model

		self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name)

		if random_label_prop > 0.0:
			rng = np.random.RandomState(SEED)
			for _, ner_tags in self.sentences:
				n = len(ner_tags)
				corrupt = rng.binomial(1, random_label_prop, n).astype(bool)
				for i in range(n):
					if corrupt[i]:
						ner_tags[i] = NER_LABELS[rng.randint(0, len(NER_LABELS))]

		torch.random.manual_seed(SEED)
		N = len(self.sentences)
		self.perm = torch.randperm(N) if permute else torch.arange(N)

	def __len__(self) -> int:
		return len(self.sentences)

	def __getitem__(self, idx: int) -> dict:
		perm_idx = idx                      # canonical sentence position
		orig_idx = int(self.perm[idx])      # underlying sentence after permutation

		words, ner_tags = self.sentences[orig_idx]

		encoding = self.tokenizer(
			words,
			is_split_into_words=True,
			truncation=True,
			max_length=self.max_subword_length,
			padding='max_length',
			return_tensors='pt',
		)

		input_ids      = encoding['input_ids'].squeeze(0)       # (max_subword_length,)
		attention_mask = encoding['attention_mask'].squeeze(0)  # (max_subword_length,)
		token_type_ids = encoding['token_type_ids'].squeeze(0)  # (max_subword_length,)

		word_ids = encoding.word_ids(batch_index=0)  # List[Optional[int]], length = max_subword_length

		# Subword-level labels: first subword of word j → LABEL2ID[ner_tags[j]].
		# CLS, SEP, padding and non-first subwords get IGNORE_INDEX so CE loss skips them.
		subword_labels = torch.full((self.max_subword_length,), IGNORE_INDEX, dtype=torch.long)
		seen = set()
		for token_pos, word_id in enumerate(word_ids):
			if word_id is None:
				continue
			if word_id not in seen:
				seen.add(word_id)
				if word_id < len(ner_tags):
					subword_labels[token_pos] = LABEL2ID[ner_tags[word_id]]

		# first_subword_positions[j] = subword token index of the first subword of word j.
		# -1 if word j is beyond max_words or was truncated.
		first_subword_positions = torch.full((self.max_words,), -1, dtype=torch.long)
		seen_fsp = set()
		for token_pos, word_id in enumerate(word_ids):
			if word_id is None:
				continue
			if word_id not in seen_fsp and word_id < self.max_words:
				first_subword_positions[word_id] = token_pos
				seen_fsp.add(word_id)

		# Word-level labels used for metric computation and storage.
		# IGNORE_INDEX for padding or truncated words.
		word_labels = torch.full((self.max_words,), IGNORE_INDEX, dtype=torch.long)
		for j in range(min(len(ner_tags), self.max_words)):
			if first_subword_positions[j].item() >= 0:
				word_labels[j] = LABEL2ID[ner_tags[j]]

		# True at positions belonging to an actual word (not padding/truncated).
		word_mask = first_subword_positions >= 0  # (max_words,) bool

		return {
			'input_ids':               input_ids,               # (max_subword_length,) int64
			'attention_mask':          attention_mask,          # (max_subword_length,) int64
			'token_type_ids':          token_type_ids,          # (max_subword_length,) int64
			'labels':                  subword_labels,          # (max_subword_length,) int64
			'word_labels':             word_labels,             # (max_words,) int64
			'word_mask':               word_mask,               # (max_words,) bool
			'first_subword_positions': first_subword_positions, # (max_words,) int64
			'index':                   torch.tensor(orig_idx, dtype=torch.long),
			'perm_index':              torch.tensor(perm_idx, dtype=torch.long),
		}


def make_conll_dataloaders(
	data_root: str,
	tokenizer_name: str = 'bert-base-uncased',
	max_subword_length: int = 256,
	max_words: int = 128,
	batch_size: int = 16,
	shuffle: bool = True,
	random_label_prop: float = 0.0,
	num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
	"""Create train (eng.train) and validation (eng.testa) DataLoaders for CoNLL-2003."""
	train_ds = CoNLLDataset(
		os.path.join(data_root, 'eng.train'),
		tokenizer_name=tokenizer_name,
		max_subword_length=max_subword_length,
		max_words=max_words,
		permute=True,
		random_label_prop=random_label_prop,
	)
	val_ds = CoNLLDataset(
		os.path.join(data_root, 'eng.testa'),
		tokenizer_name=tokenizer_name,
		max_subword_length=max_subword_length,
		max_words=max_words,
		permute=True,
		random_label_prop=0.0,  # never corrupt validation labels
	)
	train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle,
	                          num_workers=num_workers, pin_memory=True)
	val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
	                          num_workers=num_workers, pin_memory=True)
	return train_loader, val_loader
