# Copyright Pathway Technology, Inc.

import argparse
from contextlib import nullcontext
from pathlib import Path

import torch

import bdh
from data import DatasetLoader, DatasetLoaderConfig
from tokenizer_utils import TokenizerManager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the BDH model.")
    parser.add_argument(
        "--dataset_name",
        "--dataset-name",
        default="shakespeare",
        help="Dataset identifier (e.g. shakespeare, wikitext, openwebtext, c4).",
    )
    parser.add_argument(
        "--dataset_config",
        "--dataset-config",
        default=None,
        help="Optional dataset configuration/subset name (e.g. wikitext-103-raw-v1, en).",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Enable datasets streaming mode for large corpora.",
    )
    parser.add_argument(
        "--text_column",
        "--text-column",
        default=None,
        help="Column containing text (defaults to common names like 'text').",
    )
    parser.add_argument(
        "--block_size",
        "--block-size",
        type=int,
        default=512,
        help="Context window size for language modelling.",
    )
    parser.add_argument(
        "--train_split",
        "--train-split",
        type=float,
        default=0.9,
        help="Fraction of data to use for training when splitting automatically.",
    )
    parser.add_argument(
        "--batch_size",
        "--batch-size",
        type=int,
        default=8,
        help="Number of sequences per optimisation step.",
    )
    parser.add_argument(
        "--max_iters",
        "--max-iters",
        type=int,
        default=30000,
        help="Total optimisation steps.",
    )
    parser.add_argument(
        "--log_freq",
        "--log-freq",
        type=int,
        default=100,
        help="Logging frequency measured in optimisation steps.",
    )
    parser.add_argument(
        "--learning_rate",
        "--learning-rate",
        type=float,
        default=1e-3,
        help="AdamW learning rate.",
    )
    parser.add_argument(
        "--weight_decay",
        "--weight-decay",
        type=float,
        default=0.1,
        help="AdamW weight decay.",
    )
    parser.add_argument(
        "--no_compile",
        action="store_true",
        help="Disable torch.compile for environments where it is unavailable.",
    )
    parser.add_argument(
        "--tokenizer_type",
        "--tokenizer-type",
        default="byte",
        help="Tokenizer type: byte, bpe, wordpiece, or unigram.",
    )
    parser.add_argument(
        "--tokenizer_vocab_size",
        "--tokenizer-vocab-size",
        type=int,
        default=256,
        help="Desired tokenizer vocabulary size (ignored for byte-level tokenizers).",
    )
    parser.add_argument(
        "--tokenizer_path",
        "--tokenizer-path",
        default=None,
        help="Path to a trained tokenizer directory created with train_tokenizer.py.",
    )
    parser.add_argument(
        "--save_dir",
        "--save-dir",
        default=None,
        help="Optional directory to persist the trained model and tokenizer.",
    )
    return parser.parse_args()


def setup_precision(device: torch.device):
    if device.type == "cuda":
        dtype = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
    else:
        dtype = "float32"
    torch_dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype]
    ctx = (
        torch.amp.autocast(device_type=device.type, dtype=torch_dtype)
        if device.type == "cuda"
        else nullcontext()
    )
    scaler = torch.amp.GradScaler(device=device.type, enabled=(dtype == "float16"))
    return dtype, ctx, scaler


def init_tokenizer_manager(args: argparse.Namespace) -> TokenizerManager:
    if args.tokenizer_path:
        manager = TokenizerManager.from_directory(args.tokenizer_path)
    else:
        manager = TokenizerManager(
            tokenizer_type=args.tokenizer_type,
            vocab_size=args.tokenizer_vocab_size,
        )
        if manager.tokenizer_type != "byte":
            raise ValueError(
                "Non-byte tokenizers require --tokenizer_path. "
                "Train one with train_tokenizer.py before launching training."
            )
    return manager


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(1337)

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    dtype, ctx, scaler = setup_precision(device)
    print(f"Using device: {device} with dtype {dtype}")

    tokenizer_manager = init_tokenizer_manager(args)

    bdh_config = bdh.BDHConfig()
    bdh_config.vocab_size = tokenizer_manager.vocab_size
    bdh_config.tokenizer_type = tokenizer_manager.tokenizer_type
    bdh_config.tokenizer_vocab_size = tokenizer_manager.vocab_size
    bdh_config.tokenizer_path = args.tokenizer_path or tokenizer_manager.tokenizer_path
    bdh_config.bos_token_id = tokenizer_manager.bos_token_id
    bdh_config.eos_token_id = tokenizer_manager.eos_token_id
    bdh_config.pad_token_id = tokenizer_manager.pad_token_id
    bdh_config.unk_token_id = tokenizer_manager.unk_token_id

    model = bdh.BDH(bdh_config, tokenizer=tokenizer_manager).to(device)
    if hasattr(torch, "compile") and not args.no_compile:
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    loader_config = DatasetLoaderConfig(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        streaming=args.streaming,
        text_column=args.text_column,
        tokenizer_manager=tokenizer_manager,
        block_size=args.block_size,
        train_split=args.train_split,
        data_dir=Path(__file__).resolve().parent,
        device=device,
    )
    dataset_loader = DatasetLoader(loader_config)
    dataset_loader.load_dataset()

    model.train()
    x, y = dataset_loader.get_batch("train", args.batch_size)

    loss_acc = 0.0
    loss_steps = 0
    for step in range(args.max_iters):
        with ctx:
            logits, loss = model(x, y)
        x, y = dataset_loader.get_batch("train", args.batch_size)
        loss_acc += loss.item()
        loss_steps += 1

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        if step % args.log_freq == 0:
            avg_loss = loss_acc / loss_steps if loss_steps else float("nan")
            print(f"Step: {step}/{args.max_iters} loss {avg_loss:.3f}")
            loss_acc = 0.0
            loss_steps = 0

    if args.save_dir:
        model.save_pretrained(args.save_dir, tokenizer=tokenizer_manager)

    print("Training done, now generating a sample")
    model.eval()
    prompt_text = "To be or "
    prompt_tokens = tokenizer_manager.encode(prompt_text, add_special_tokens=False)
    prompt_tensor = torch.tensor(
        prompt_tokens,
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)
    ret = model.generate(prompt_tensor, max_new_tokens=100, top_k=3)
    generated_tokens = ret[0].tolist()
    prompt_length = len(prompt_tokens)
    continuation_tokens = generated_tokens[prompt_length:]
    continuation = tokenizer_manager.decode(continuation_tokens)
    print(prompt_text + continuation)


if __name__ == "__main__":
    main()
