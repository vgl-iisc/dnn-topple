#!/bin/bash

pip install brotli python-minifier flash-attn

# python3 prepare_case_ops_data.py \
	# --docs ../data/docs_selected.jsonl \
	# --out  ./data/datasets/fineweb10B_sp8192_caseops/datasets \
	# --sp   ./tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model

SEED=42 \
CASEOPS_ENABLED=1 \
EMBED_BITS=7 \
SMEAR_GATE_ENABLED=1 \
SPARSE_ATTN_GATE_ENABLED=1 \
MIN_LR=0.1 \
EMBED_CLIP_SIGMAS=15.0 \
MLP_CLIP_SIGMAS=12.0 \
GPTQ_RESERVE_SECONDS=8.0 \
PHASED_TTT_NUM_PHASES=3 \
TOKENIZER_PATH=./tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model \
torchrun --standalone --nproc_per_node=4 train_gpt.py