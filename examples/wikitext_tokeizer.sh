#!/usr/bin/env bash

python train_tokenizer.py \
  --dataset-name wikitext \
  --dataset-config wikitext-103-raw-v1 \
  --text-column text \
  --tokenizer-type bpe \
  --vocab-size 32000 \
  --output-dir tokenizers/bpe-32k
