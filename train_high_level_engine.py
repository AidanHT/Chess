#!/usr/bin/env python3
"""
Entry point for high-level training on strong player games only.

Usage:
    python train_high_level_engine.py --masters data/pgn --epochs 15
    torchrun --nproc_per_node=4 train_high_level_engine.py --masters data/pgn --epochs 15
"""

if __name__ == "__main__":
    from chess_engine.training.high_level import main_with_args

    main_with_args()
