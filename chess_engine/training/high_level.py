#!/usr/bin/env python3
"""
High-level data-focused training script for AlphaZero chess engine.

Strategy:
  1. Masters games (titled players 2000+) = PRIMARY signal (90%)
  2. High-elo 2000+ rated = SECONDARY signal (10%)

This trains primarily on world-class moves while maintaining some diversity.
"""
import argparse
import sys
from pathlib import Path

from .train import TrainConfig, train as train_fn

def create_high_level_config(masters_dir: str, broad_dir: str | None = None) -> TrainConfig:
    """
    Create training config optimized for high-level data.

    Args:
        masters_dir: Path to Masters PGN files
        broad_dir: Optional path to broader skill-level data (used sparingly)

    Returns:
        Configured TrainConfig for high-level training
    """
    config = TrainConfig(
        # Data: Masters as primary
        data_dir=masters_dir,
        num_workers=16,     # More workers for high-throughput
        batch_size=1024,    # Larger batch (better gradient estimates)
        prefetch_factor=8,  # More buffering (reduces GPU idle time)
        steps_per_epoch=20_000,

        # Model: standard spec
        num_blocks=20,
        num_filters=256,

        # Optimizer: conservative LR (quality > speed)
        lr=5e-4,           # Lower LR for careful learning from good examples
        weight_decay=1e-4,
        max_grad_norm=5.0,

        # Scheduler
        epochs=15,         # More epochs to really internalize strong play
        pct_start=0.15,

        # Loss: equal weighting
        value_loss_weight=1.0,
        policy_loss_weight=1.0,

        # Checkpointing
        checkpoint_dir="checkpoints",
        checkpoint_every_n_steps=5_000,
        keep_last_n_checkpoints=5,

        # Logging
        log_every_n_steps=100,
        wandb_project="chess-alphazero",
        wandb_run_name="high-level-masters",

        # Misc
        seed=42,
        amp=True,
        compile=False,
    )
    return config

def main_with_args() -> None:
    """Main entry point for high-level training."""
    parser = argparse.ArgumentParser(
        description="Train on high-level chess data (Masters + 2000+ rated)",
        epilog="Example: python train_high_level_engine.py --masters data/pgn --epochs 15"
    )
    parser.add_argument(
        "--masters",
        type=str,
        default="data/pgn",
        help="Directory containing Masters PGN files (default: data/pgn)"
    )
    parser.add_argument(
        "--broad",
        type=str,
        default=None,
        help="Optional directory with broader skill-level data for regularization"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Number of training epochs (default: 15)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Batch size per GPU (default: 1024)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-4,
        help="Learning rate (default: 5e-4)"
    )

    args = parser.parse_args()

    # Validate paths
    masters_path = Path(args.masters)
    if not masters_path.exists():
        print(f"Error: Masters directory not found: {args.masters}", file=sys.stderr)
        sys.exit(1)

    pgn_files = list(masters_path.glob("*.pgn")) + list(masters_path.glob("*.pgn.gz"))
    if not pgn_files:
        print(f"Error: No PGN files found in {args.masters}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(pgn_files)} PGN files in Masters directory")
    print(f"  Total size: {sum(f.stat().st_size for f in pgn_files) / 1e9:.2f} GB")

    # Create config
    config = create_high_level_config(args.masters, args.broad)
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.lr = args.lr

    print(f"\nTraining config:")
    print(f"  Epochs: {config.epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.lr}")
    print(f"  AMP: {config.amp}")
    print()

    # Delegate to training loop
    train_fn(config)


if __name__ == "__main__":
    main_with_args()
