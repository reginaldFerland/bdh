#!/usr/bin/env bash

python train.py \
  --dataset-name wikitext \
  --dataset-config wikitext-2-raw-v1 \
  --text-column text \
  --tokenizer-path tokenizers/bpe-16k \
  --batch-size 2 \
  --block-size 512 \
  --max-iters 60000
