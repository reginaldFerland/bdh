import argparse
from pathlib import Path
from typing import List

import torch


def average_checkpoints(checkpoint_paths: List[Path]) -> dict:
    if not checkpoint_paths:
        raise ValueError("No checkpoints provided for averaging.")

    checkpoints = [torch.load(path, map_location="cpu") for path in checkpoint_paths]
    model_keys = checkpoints[0]["model_state_dict"].keys()
    averaged_state = {}

    for key in model_keys:
        tensors = [ckpt["model_state_dict"][key].float() for ckpt in checkpoints]
        stacked = torch.stack(tensors, dim=0)
        averaged_state[key] = stacked.mean(dim=0)

    template = dict(checkpoints[0])
    template["model_state_dict"] = averaged_state
    return template


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Average BDH checkpoints.")
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        help="List of checkpoint paths to average.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write the averaged checkpoint.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_paths = [Path(path) for path in args.checkpoints]
    averaged_state = average_checkpoints(checkpoint_paths)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(averaged_state, output_path)
    print(f"Averaged checkpoint saved to {output_path}")


if __name__ == "__main__":
    main()
