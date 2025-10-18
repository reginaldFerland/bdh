ls #!/bin/bash
# Train a BPE tokenizer on WikiText-2 with 16k vocabulary

set -e

# Configuration
TOKENIZER_TYPE="bpe"
VOCAB_SIZE=16000
OUTPUT_DIR="tokenizers/bpe-16k"
DATASET_NAME="wikitext"
DATASET_CONFIG="wikitext-2-raw-v1"

echo "Training BPE tokenizer with 16k vocabulary on WikiText-2..."
echo "Dataset: ${DATASET_NAME} (${DATASET_CONFIG})"
echo "Output directory: ${OUTPUT_DIR}"

python train_tokenizer.py \
    --tokenizer_type "${TOKENIZER_TYPE}" \
    --vocab_size "${VOCAB_SIZE}" \
    --dataset_name "${DATASET_NAME}" \
    --dataset_config "${DATASET_CONFIG}" \
    --output_dir "${OUTPUT_DIR}" 

echo "Tokenizer training complete!"
echo "Saved to: ${OUTPUT_DIR}"
