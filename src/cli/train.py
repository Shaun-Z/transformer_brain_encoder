from __future__ import annotations

import argparse

from src.config import load_config
from src.train.trainer import train


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a paper-faithful brain encoder run")
    parser.add_argument("--config", default="configs/tben/base.yaml")
    parser.add_argument("--subject", type=int)
    parser.add_argument("--output-root")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--distributed", action="store_true")
    args = parser.parse_args()

    overrides = {}
    if args.subject is not None:
        overrides["subject"] = args.subject
    if args.output_root is not None:
        overrides["output_root"] = args.output_root
    if args.distributed:
        overrides["distributed"] = True

    checkpoint_path = train(
        load_config(args.config, overrides=overrides),
        epochs=args.epochs,
        max_train_steps=args.max_train_steps,
    )
    print(checkpoint_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
