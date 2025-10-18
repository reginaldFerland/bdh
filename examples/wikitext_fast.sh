#!/usr/bin/env bash
# Fast training script for testing purposes
# Uses minimal settings and only 100 iterations for quick validation

python train.py \
  --dataset-name wikitext \
  --dataset-config wikitext-103-raw-v1 \
  --text-column text \
  --tokenizer-path tokenizers/bpe-32k \
  --batch-size 4 \
  --block-size 256 \
  --max-iters 100 \
  --log-freq 10 \
  --eval-freq 50 \
  --eval-iters 5 \
  --save-freq 50 \
  --keep-last-n 2 \
  --log-dir logs/fast_test
