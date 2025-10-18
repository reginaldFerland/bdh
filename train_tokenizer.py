import argparse
from pathlib import Path
from typing import Iterable, Iterator, Optional

from datasets import load_dataset

from tokenizer_utils import TokenizerManager, iter_texts_from_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a tokenizer for BDH models.")
    parser.add_argument("--dataset_name", "--dataset-name", required=True, help="HuggingFace dataset name.")
    parser.add_argument(
        "--dataset_config",
        "--dataset-config",
        default=None,
        help="Optional dataset configuration or subset.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to use for tokenizer training.",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Stream dataset from source instead of loading into memory.",
    )
    parser.add_argument(
        "--text_column",
        "--text-column",
        default=None,
        help="Column containing text (defaults to heuristic selection).",
    )
    parser.add_argument(
        "--tokenizer_type",
        "--tokenizer-type",
        default="bpe",
        help="Tokenizer type: bpe, wordpiece, unigram, or byte.",
    )
    parser.add_argument(
        "--vocab_size",
        "--vocab-size",
        type=int,
        default=32000,
        help="Target vocabulary size for the tokenizer.",
    )
    parser.add_argument(
        "--output_dir",
        "--output-dir",
        required=True,
        help="Directory to store the trained tokenizer files.",
    )
    parser.add_argument(
        "--sample_limit",
        "--sample-limit",
        type=int,
        default=None,
        help="Optional limit on the number of text samples to use.",
    )
    parser.add_argument(
        "--skip-if-exists",
        "--skip_if_exists",
        action="store_true",
        help="Skip training if tokenizer already exists in output directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Check if tokenizer already exists and skip if requested
    if args.skip_if_exists:
        config_path = Path(args.output_dir) / "tokenizer_config.json"
        if config_path.exists():
            print(f"✓ Tokenizer already exists at {args.output_dir}")
            print("  Skipping training (use --no-skip-if-exists to force retrain)")
            
            # Load and display existing tokenizer info
            try:
                manager = TokenizerManager.from_directory(args.output_dir)
                print(f"  Type: {manager.tokenizer_type}")
                print(f"  Vocab size: {manager.vocab_size}")
                print(f"  Special tokens: PAD={manager.pad_token_id}, BOS={manager.bos_token_id}, "
                      f"EOS={manager.eos_token_id}, UNK={manager.unk_token_id}")
            except Exception as e:
                print(f"  Warning: Could not load tokenizer info: {e}")
            
            return

    if args.tokenizer_type.lower() == "byte":
        manager = TokenizerManager(tokenizer_type="byte")
        manager.save(args.output_dir)
        print(f"Saved byte-level tokenizer metadata to {args.output_dir}")
        return

    print(f"Loading dataset '{args.dataset_name}'...")
    if args.dataset_config:
        print(f"  Config: {args.dataset_config}")
    print(f"  Split: {args.split}")
    print(f"  Streaming: {args.streaming}")
    
    dataset = load_dataset(
        args.dataset_name,
        args.dataset_config,
        split=args.split,
        streaming=args.streaming,
    )

    print(f"\nExtracting texts from dataset...")
    texts = iter_texts_from_dataset(dataset, args.text_column, args.sample_limit)
    
    print(f"\nTraining {args.tokenizer_type} tokenizer with vocab_size={args.vocab_size}...")
    
    manager = TokenizerManager(
        tokenizer_type=args.tokenizer_type,
        vocab_size=args.vocab_size,
    )
    manager.train_tokenizer(texts, args.output_dir)
    print(f"\n✓ Tokenizer ({manager.tokenizer_type}) saved to {args.output_dir}")
    print(f"  Final vocab size: {manager.vocab_size}")
    print(f"  Special tokens: PAD={manager.pad_token_id}, BOS={manager.bos_token_id}, "
          f"EOS={manager.eos_token_id}, UNK={manager.unk_token_id}")


if __name__ == "__main__":
    main()
