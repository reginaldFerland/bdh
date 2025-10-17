import argparse
from pathlib import Path
from typing import Iterable, Iterator, Optional

from datasets import load_dataset

from tokenizer_utils import TokenizerManager


def extract_text(record: dict, text_column: Optional[str]) -> str:
    if text_column:
        value = record.get(text_column, "")
        return value if isinstance(value, str) else ""

    for key in ("text", "content", "article", "body"):
        value = record.get(key)
        if isinstance(value, str):
            return value

    for value in record.values():
        if isinstance(value, str):
            return value
    return ""


def iter_texts(dataset, text_column: Optional[str], limit: Optional[int] = None) -> Iterator[str]:
    count = 0
    for record in dataset:
        text = extract_text(record, text_column)
        if not text:
            continue
        yield text
        count += 1
        if limit is not None and count >= limit:
            break


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.tokenizer_type.lower() == "byte":
        manager = TokenizerManager(tokenizer_type="byte")
        manager.save(args.output_dir)
        print(f"Saved byte-level tokenizer metadata to {args.output_dir}")
        return

    dataset = load_dataset(
        args.dataset_name,
        args.dataset_config,
        split=args.split,
        streaming=args.streaming,
    )

    texts = iter_texts(dataset, args.text_column, args.sample_limit)
    manager = TokenizerManager(
        tokenizer_type=args.tokenizer_type,
        vocab_size=args.vocab_size,
    )
    manager.train_tokenizer(texts, args.output_dir, limit=args.sample_limit)
    print(f"Tokenizer ({manager.tokenizer_type}) saved to {args.output_dir}")


if __name__ == "__main__":
    main()
