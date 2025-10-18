# Copyright Pathway Technology, Inc.

import argparse
import dataclasses
import logging
import sys
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

import bdh
from checkpoint import CheckpointManager, CheckpointState
from data import DatasetLoader, DatasetLoaderConfig
from tokenizer_utils import TokenizerManager


def setup_logging(log_dir: Path = Path("logs")) -> logging.Logger:
    """Setup logging to both console and a date-stamped log file."""
    log_dir.mkdir(exist_ok=True)
    
    # Create log filename with current date and time
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = log_dir / f"{timestamp}.log"
    
    # Configure logging
    logger = logging.getLogger("bdh_training")
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # File handler
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"Logging to file: {log_file}")
    
    return logger


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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the latest checkpoint in --checkpoint_dir.",
    )
    parser.add_argument(
        "--checkpoint_path",
        "--checkpoint-path",
        default=None,
        help="Path to a specific checkpoint file to resume from.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        "--checkpoint-dir",
        default="checkpoints",
        help="Directory for saving training checkpoints.",
    )
    parser.add_argument(
        "--save_freq",
        "--save-freq",
        type=int,
        default=1000,
        help="Save a checkpoint every N steps.",
    )
    parser.add_argument(
        "--keep_last_n",
        "--keep-last-n",
        type=int,
        default=5,
        help="Number of recent checkpoints to keep.",
    )
    parser.add_argument(
        "--eval_freq",
        "--eval-freq",
        type=int,
        default=500,
        help="Run validation every N steps (requires validation split).",
    )
    parser.add_argument(
        "--eval_iters",
        "--eval-iters",
        type=int,
        default=50,
        help="Number of validation batches per evaluation.",
    )
    parser.add_argument(
        "--log_dir",
        "--log-dir",
        default="logs",
        help="Directory for saving training logs.",
    )
    return parser.parse_args()


def _strip_compiled_prefix(state_dict: Dict) -> Dict:
    """Remove _orig_mod. prefix from compiled model state dict keys."""
    return {
        k.replace("_orig_mod.", ""): v 
        for k, v in state_dict.items()
    }


def _build_checkpoint_state(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    global_step: int,
    bdh_config: bdh.BDHConfig,
    args: argparse.Namespace,
    best_val_loss: float,
    train_loss_history: List[Dict[str, float]],
    training_start_time: float,
) -> Dict:
    """Build checkpoint state dictionary with versioning and metadata."""
    # Handle torch.compile by stripping _orig_mod. prefix
    model_state = model.state_dict()
    if any(k.startswith("_orig_mod.") for k in model_state.keys()):
        model_state = _strip_compiled_prefix(model_state)
    
    return {
        "checkpoint_version": 1,
        "timestamp": datetime.now().isoformat(),
        "elapsed_time": time.time() - training_start_time,
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "step": global_step,
        "config": dataclasses.asdict(bdh_config),
        "train_args": vars(args),
        "best_val_loss": best_val_loss,
        "loss_history": train_loss_history,
        "rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }


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


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataset_loader: DatasetLoader,
    batch_size: int,
    ctx,
    eval_iters: int,
) -> Optional[float]:
    if not dataset_loader.has_split("val"):
        return None

    device = next(model.parameters()).device
    losses: List[float] = []
    model.eval()
    try:
        for _ in range(eval_iters):
            x_val, y_val = dataset_loader.get_batch("val", batch_size)
            x_val = x_val.to(device)
            y_val = y_val.to(device)
            with ctx:
                _, loss = model(x_val, y_val)
            losses.append(loss.item())
    finally:
        model.train()

    if not losses:
        return None
    return float(np.mean(losses))


def main() -> None:
    args = parse_args()
    
    # Setup logging
    logger = setup_logging(log_dir=Path(args.log_dir))

    device = torch.device("cuda" if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu')
    torch.manual_seed(1337)

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    dtype, ctx, scaler = setup_precision(device)
    logger.info(f"Using device: {device} with dtype {dtype}")

    checkpoint_manager = CheckpointManager(
        checkpoint_dir=Path(args.checkpoint_dir),
        keep_last_n=args.keep_last_n,
    )
    resume_state: Optional[Dict] = None
    checkpoint_state: Optional[CheckpointState] = None
    if args.resume or args.checkpoint_path:
        checkpoint_state = checkpoint_manager.load_checkpoint(args.checkpoint_path)
        if checkpoint_state is not None:
            resume_state = checkpoint_state.state
            logger.info(f"Resuming from checkpoint: {checkpoint_state.path}")
        else:
            logger.info("No checkpoint found for resume request; starting fresh.")

    # Determine tokenizer configuration (checkpoint values take precedence).
    resume_config: Optional[Dict] = None
    if resume_state and "config" in resume_state:
        resume_config = resume_state["config"]

    tokenizer_path = args.tokenizer_path or (resume_config.get("tokenizer_path") if resume_config else None)
    tokenizer_type = resume_config.get("tokenizer_type") if resume_config else args.tokenizer_type
    tokenizer_vocab_size = resume_config.get("tokenizer_vocab_size") if resume_config else args.tokenizer_vocab_size

    if tokenizer_path:
        tokenizer_manager = TokenizerManager.from_directory(tokenizer_path)
    else:
        tokenizer_manager = TokenizerManager(
            tokenizer_type=tokenizer_type,
            vocab_size=tokenizer_vocab_size,
        )
        if tokenizer_manager.tokenizer_type != "byte":
            raise ValueError(
                "A trained tokenizer path is required for non-byte tokenizers. "
                "Provide --tokenizer_path or resume from a checkpoint with tokenizer metadata."
            )

    if resume_config:
        bdh_config = bdh.BDHConfig(**resume_config)
    else:
        bdh_config = bdh.BDHConfig()
        bdh_config.vocab_size = tokenizer_manager.vocab_size
        bdh_config.tokenizer_type = tokenizer_manager.tokenizer_type
        bdh_config.tokenizer_vocab_size = tokenizer_manager.vocab_size
        bdh_config.tokenizer_path = tokenizer_path or tokenizer_manager.tokenizer_path
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

    start_step = 0
    best_val_loss = float("inf")
    train_loss_history: List[Dict[str, float]] = []
    training_start_time = time.time()
    if resume_state:
        if "model_state_dict" in resume_state:
            model.load_state_dict(resume_state["model_state_dict"])
        if "optimizer_state_dict" in resume_state:
            optimizer.load_state_dict(resume_state["optimizer_state_dict"])
        if "scaler_state_dict" in resume_state:
            scaler.load_state_dict(resume_state["scaler_state_dict"])
        start_step = resume_state.get("step", 0)
        best_val_loss = resume_state.get("best_val_loss", float("inf"))
        train_loss_history = resume_state.get("loss_history", [])
        # Preserve elapsed time from previous run
        if "elapsed_time" in resume_state:
            training_start_time = time.time() - resume_state["elapsed_time"]
        if "rng_state" in resume_state:
            torch.set_rng_state(resume_state["rng_state"])
        cuda_rng_state = resume_state.get("cuda_rng_state")
        if cuda_rng_state is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state(cuda_rng_state)
        logger.info(f"Resumed training from step {start_step}")

    model.train()
    x, y = dataset_loader.get_batch("train", args.batch_size)

    loss_acc = 0.0
    loss_steps = 0
    last_checkpoint_step = start_step
    for step in range(start_step, args.max_iters):
        with ctx:
            logits, loss = model(x, y)
        x, y = dataset_loader.get_batch("train", args.batch_size)
        loss_acc += loss.item()
        loss_steps += 1

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        global_step = step + 1

        if global_step % args.log_freq == 0:
            avg_loss = loss_acc / loss_steps if loss_steps else float("nan")
            logger.info(f"Step: {global_step}/{args.max_iters} loss {avg_loss:.3f}")
            train_loss_history.append({"step": int(global_step), "loss": float(avg_loss)})
            if len(train_loss_history) > 1000:
                train_loss_history = train_loss_history[-1000:]
            loss_acc = 0.0
            loss_steps = 0

        should_eval = args.eval_freq > 0 and global_step % args.eval_freq == 0
        val_loss: Optional[float] = None
        improved = False
        if should_eval:
            val_loss = evaluate_model(
                model=model,
                dataset_loader=dataset_loader,
                batch_size=args.batch_size,
                ctx=ctx,
                eval_iters=args.eval_iters,
            )
            if val_loss is not None:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    improved = True
            else:
                logger.info("Validation split not available; skipping evaluation.")
            
            # Generate a sample prompt
            model.eval()
            prompt_text = "Who was the first US president?"
            prompt_tokens = tokenizer_manager.encode(prompt_text, add_special_tokens=False)
            prompt_tensor = torch.tensor(
                prompt_tokens,
                dtype=torch.long,
                device=device,
            ).unsqueeze(0)
            with torch.no_grad():
                ret = model.generate(prompt_tensor, max_new_tokens=50, top_k=3)
            generated_tokens = ret[0].tolist()
            prompt_length = len(prompt_tokens)
            continuation_tokens = generated_tokens[prompt_length:]
            continuation = tokenizer_manager.decode(continuation_tokens)
            logger.info(f"Sample generation: {prompt_text + continuation}")
            model.train()

        should_save = args.save_freq > 0 and global_step % args.save_freq == 0
        if should_save or improved:
            state = _build_checkpoint_state(
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                global_step=global_step,
                bdh_config=bdh_config,
                args=args,
                best_val_loss=best_val_loss,
                train_loss_history=train_loss_history,
                training_start_time=training_start_time,
            )
            is_best = improved
            checkpoint_manager.save_checkpoint(state, global_step, is_best=is_best)
            last_checkpoint_step = global_step

    if args.save_dir:
        model.save_pretrained(args.save_dir, tokenizer=tokenizer_manager)

    # Ensure a final checkpoint exists for the completed run.
    if last_checkpoint_step != args.max_iters:
        final_state = _build_checkpoint_state(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            global_step=args.max_iters,
            bdh_config=bdh_config,
            args=args,
            best_val_loss=best_val_loss,
            train_loss_history=train_loss_history,
            training_start_time=training_start_time,
        )
        checkpoint_manager.save_checkpoint(final_state, args.max_iters, is_best=False)

    logger.info("Training done, now generating a sample")
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
    logger.info(f"Generated sample: {prompt_text + continuation}")


if __name__ == "__main__":
    main()
