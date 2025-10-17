#!/usr/bin/env bash

python train.py \
  --dataset-name wikitext \
  --dataset-config wikitext-103-raw-v1 \
  --text-column text \
  --tokenizer-path tokenizers/bpe-32k \
  --batch-size 8 \
  --block-size 512 \
  --max-iters 30000
