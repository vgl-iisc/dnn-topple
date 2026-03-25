from typing import Optional

import torch
import torch.nn as nn
from transformers import BertModel


class BertNER(nn.Module):
	"""BERT with a linear token-classification head for Named Entity Recognition.

	forward() returns raw logits of shape (B, seq_len, num_labels).

	Hook paths compatible with attach_collection_hooks:
	  bert.encoder.layer[N]  ->  i[0] is (B, seq_len, 768), input to transformer block N
	  classifier             ->  i[0] is (B, seq_len, 768), BERT output fed into the head
	"""

	def __init__(
		self,
		model_name: str = 'bert-base-uncased',
		num_labels: int = 9,
		dropout: float = 0.1,
	):
		super().__init__()
		self.bert = BertModel.from_pretrained(model_name)
		self.dropout = nn.Dropout(dropout)
		self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
		self.num_labels = num_labels
		self.hidden_size = self.bert.config.hidden_size

	def forward(
		self,
		input_ids: torch.Tensor,       # (B, seq_len)
		attention_mask: torch.Tensor,  # (B, seq_len)
		token_type_ids: torch.Tensor,  # (B, seq_len)
	) -> torch.Tensor:                 # (B, seq_len, num_labels)
		outputs = self.bert(
			input_ids=input_ids,
			attention_mask=attention_mask,
			token_type_ids=token_type_ids,
		)
		sequence_output = outputs.last_hidden_state      # (B, seq_len, hidden_size)
		sequence_output = self.dropout(sequence_output)
		logits = self.classifier(sequence_output)        # (B, seq_len, num_labels)
		return logits


def create_bert_ner(
	model_name: str = 'bert-base-uncased',
	num_labels: int = 9,
	dropout: float = 0.1,
	checkpoint: Optional[str] = None,
	device: str = 'cpu',
) -> BertNER:
	"""Factory for BertNER.  Mirrors the create_autoencoder / create_model convention.

	Checkpoint loading supports 'model_state_dict', 'state_dict', or a raw state dict.
	"""
	model = BertNER(model_name, num_labels, dropout)
	if checkpoint is not None:
		state = torch.load(checkpoint, map_location=device)
		if isinstance(state, dict) and 'model_state_dict' in state:
			model.load_state_dict(state['model_state_dict'])
		elif isinstance(state, dict) and 'state_dict' in state:
			model.load_state_dict(state['state_dict'])
		else:
			model.load_state_dict(state)
	model.to(torch.device(device))
	return model
