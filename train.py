# Copyright Pathway Technology, Inc.

import argparse
from contextlib import nullcontext
from pathlib import Path

import torch

import bdh
from data import DatasetLoader, DatasetLoaderConfig


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
        default=300,
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
    return parser.parse_args()


def setup_precision(device: torch.device):
    if device.type == "cuda":
        dtype = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
    else:
        dtype = "float32"
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype]
    ctx = (
        torch.amp.autocast(device_type=device.type, dtype=ptdtype)
        if device.type == "cuda"
        else nullcontext()
    )
    scaler = torch.amp.GradScaler(device=device.type, enabled=(dtype == "float16"))
    return dtype, ctx, scaler


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(1337)

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    dtype, ctx, scaler = setup_precision(device)
    print(f"Using device: {device} with dtype {dtype}")

    bdh_config = bdh.BDHConfig()
    model = bdh.BDH(bdh_config).to(device)
    if hasattr(torch, "compile") and not args.no_compile:
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    loader_config = DatasetLoaderConfig(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        streaming=args.streaming,
        text_column=args.text_column,
        block_size=args.block_size,
        train_split=args.train_split,
        data_dir=Path(__file__).resolve().parent,
        device=device,
    )
    dataset_loader = DatasetLoader(loader_config)
    # Example CLI usage:
    #   python train.py --dataset_name shakespeare
    #   python train.py --dataset_name wikitext --dataset_config wikitext-103-raw-v1 --text_column text
    #   python train.py --dataset_name openwebtext --text_column text --streaming
    #   python train.py --dataset_name c4 --dataset_config en --text_column text --streaming
    dataset_loader.load_dataset()

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

    print("Training done, now generating a sample")
    model.eval()
    prompt = torch.tensor(
        bytearray("To be or ", "utf-8"), dtype=torch.long, device=device
    ).unsqueeze(0)
    ret = model.generate(prompt, max_new_tokens=100, top_k=3)
    ret_decoded = bytes(ret.to(torch.uint8).to("cpu").squeeze(0)).decode(
        errors="backslashreplace"
    )
    print(ret_decoded)


if __name__ == "__main__":
    main()
