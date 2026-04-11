#!/usr/bin/env python3
"""
Entry point for training the AlphaZero-style chess engine.

Usage:
    python train_engine.py --data_dir data/pgn --epochs 10
    torchrun --nproc_per_node=4 train_engine.py --data_dir data/pgn
"""

if __name__ == "__main__":
    from chess_engine.training.train import _build_parser, TrainConfig, train

    args = _build_parser().parse_args()
    cfg = TrainConfig(
        data_dir=args.data_dir,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        prefetch_factor=args.prefetch_factor,
        steps_per_epoch=args.steps_per_epoch,
        num_blocks=args.num_blocks,
        num_filters=args.num_filters,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        epochs=args.epochs,
        pct_start=args.pct_start,
        value_loss_weight=args.value_loss_weight,
        policy_loss_weight=args.policy_loss_weight,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_every_n_steps=args.checkpoint_every_n_steps,
        keep_last_n_checkpoints=args.keep_last_n_checkpoints,
        puzzle_dir=args.puzzle_dir,
        puzzle_ratio=args.puzzle_ratio,
        min_puzzle_rating=args.min_puzzle_rating,
        log_every_n_steps=args.log_every_n_steps,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        wandb_disabled=args.wandb_disabled,
        seed=args.seed,
        fine_tune=args.fine_tune,
        amp=not args.no_amp,
        compile=args.compile,
    )
    train(cfg, resume_from=args.resume)
