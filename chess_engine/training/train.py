"""
Distributed supervised training for the AlphaZero-style chess engine.

Launch (multi-GPU with torchrun):
    torchrun --nproc_per_node=<N> train.py --data_dir /path/to/pgn

Single-GPU / CPU:
    python train.py --data_dir /path/to/pgn
"""
from __future__ import annotations

import argparse
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

try:
    from tqdm import tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    wandb = None  # type: ignore[assignment]
    _WANDB_AVAILABLE = False

from ..data import find_pgn_files, make_dataloader, make_combined_dataloader
from ..models import ChessResNet

log = logging.getLogger(__name__)


# ─── Config ───────────────────────────────────────────────────────────────────

@dataclass
class TrainConfig:
    # ── Data ──────────────────────────────────────────────────────────────────
    data_dir:        str = "data/pgn"
    num_workers:     int = 8         # Increase workers to keep GPU fed
    batch_size:      int = 512       # per-GPU batch size
    prefetch_factor: int = 4         # Increase buffering to reduce GPU idle
    # IterableDataset has no __len__; this estimate sizes OneCycleLR.
    steps_per_epoch: int = 50_000

    # ── Model ─────────────────────────────────────────────────────────────────
    num_blocks:  int = 20
    num_filters: int = 256

    # ── Optimiser ─────────────────────────────────────────────────────────────
    lr:            float = 1e-3
    weight_decay:  float = 1e-4
    max_grad_norm: float = 5.0

    # ── Scheduler (OneCycleLR) ────────────────────────────────────────────────
    epochs:    int   = 10
    pct_start: float = 0.1   # fraction of training spent in the warm-up ramp

    # ── Loss weights ──────────────────────────────────────────────────────────
    value_loss_weight:  float = 1.0
    policy_loss_weight: float = 1.0

    # ── Checkpointing ─────────────────────────────────────────────────────────
    checkpoint_dir:           str = "checkpoints"
    checkpoint_every_n_steps: int = 1_000   # Save every 1000 steps (0 = epoch-only)
    keep_last_n_checkpoints:  int = 10      # Keep last 10 checkpoints

    # ── Puzzle mixing ─────────────────────────────────────────────────────────
    puzzle_dir:    Optional[str] = None   # Path to lichess_db_puzzle.csv; None = no puzzles
    puzzle_ratio:  float         = 0.2   # Fraction of samples drawn from puzzles
    min_puzzle_rating: int       = 1200  # Skip trivially easy puzzles

    # ── Fine-tune mode ────────────────────────────────────────────────────────
    # When True, --resume loads model weights only (fresh optimizer + scheduler).
    # Use this for Phase 2 fine-tuning on elite/puzzle data after Phase 1.
    fine_tune: bool = False

    # ── Logging ───────────────────────────────────────────────────────────────
    log_every_n_steps: int          = 100
    wandb_project:     str          = "chess-alphazero"
    wandb_run_name:    Optional[str] = None
    wandb_disabled:    bool          = False

    # ── Misc ──────────────────────────────────────────────────────────────────
    seed:    int  = 42
    amp:     bool = True   # Automatic Mixed Precision (requires CUDA)
    compile: bool = False  # torch.compile (requires PyTorch ≥ 2.0)


# ─── Distributed helpers ──────────────────────────────────────────────────────

def _setup_ddp() -> tuple[int, int, int]:
    """
    Initialise DDP from env-vars set by ``torchrun``.
    Returns (rank, local_rank, world_size).
    Falls back to (0, 0, 1) when not launched with torchrun.
    """
    if "RANK" not in os.environ:
        return 0, 0, 1
    dist.init_process_group(backend="nccl")
    rank       = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    return rank, local_rank, world_size


def _cleanup_ddp() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


# ─── Loss ─────────────────────────────────────────────────────────────────────

def compute_loss(
    policy_logits:   torch.Tensor,                    # (B, 4672)
    value_pred:      torch.Tensor,                    # (B,)
    policy_target:   torch.Tensor,                    # (B, 4672) float32 one-hot / soft
    value_target:    torch.Tensor,                    # (B,)      float32 in {-1, 0, 1}
    legal_move_mask: Optional[torch.Tensor] = None,  # (B, 4672) bool; True = legal
    *,
    value_weight:  float = 1.0,
    policy_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns (total_loss, policy_loss, value_loss).

    Illegal-move masking
    --------------------
    When ``legal_move_mask`` is provided, illegal positions in ``policy_logits``
    are filled with ``-inf`` before the softmax so they receive exactly zero
    probability.  Expert PGN one-hot labels already constrain the target to the
    played (legal) move, but masking also prevents the network from wasting
    representational capacity on logically impossible actions.

    If ``legal_move_mask`` is None (default) — e.g. the dataset does not yet
    provide masks — the raw logits are used as-is and the model is supervised
    purely by the one-hot target distribution.
    """
    # ── Policy loss ───────────────────────────────────────────────────────────
    if legal_move_mask is not None:
        # -inf fills survive fp16: torch.finfo(torch.float16).min ≈ -65504.
        fill_val = torch.finfo(policy_logits.dtype).min
        policy_logits = policy_logits.masked_fill(~legal_move_mask, fill_val)

    # PyTorch ≥ 1.10 accepts float targets of shape (B, C) as soft labels,
    # equivalent to KL-divergence with the one-hot distribution.
    policy_loss = F.cross_entropy(policy_logits, policy_target)

    # ── Value loss ────────────────────────────────────────────────────────────
    value_loss = F.mse_loss(value_pred, value_target)

    total_loss = policy_weight * policy_loss + value_weight * value_loss
    return total_loss, policy_loss, value_loss


# ─── Checkpointing ────────────────────────────────────────────────────────────

def _ckpt_path(directory: Path, step: int) -> Path:
    return directory / f"step_{step:010d}.pt"


def save_checkpoint(
    directory:   Path,
    global_step: int,
    epoch:       int,
    model:       torch.nn.Module,
    optimizer:   torch.optim.Optimizer,
    scheduler:   object,
    scaler:      torch.cuda.amp.GradScaler,
    keep_last_n: int = 3,
) -> None:
    """Persist full training state; prune old checkpoints to ``keep_last_n``."""
    directory.mkdir(parents=True, exist_ok=True)
    path = _ckpt_path(directory, global_step)

    # Unwrap DDP so the checkpoint is portable (loadable without DDP).
    raw_model = model.module if isinstance(model, DDP) else model
    torch.save(
        {
            "global_step": global_step,
            "epoch":       epoch,
            "model":       raw_model.state_dict(),
            "optimizer":   optimizer.state_dict(),
            "scheduler":   scheduler.state_dict(),
            "scaler":      scaler.state_dict(),
        },
        path,
    )
    log.info("Checkpoint saved → %s", path)

    if keep_last_n > 0:
        for old in sorted(directory.glob("step_*.pt"))[:-keep_last_n]:
            old.unlink(missing_ok=True)


def load_checkpoint(
    path:      Path,
    model:     torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: object,
    scaler:    torch.cuda.amp.GradScaler,
    *,
    model_only: bool = False,
) -> tuple[int, int]:
    """
    Restore training state in-place. Returns (global_step, epoch).

    Parameters
    ----------
    model_only :
        When True, load only model weights and skip optimizer / scheduler /
        scaler state.  Use this for fine-tuning (Phase 2) so the new training
        phase starts with a fresh learning-rate schedule.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    raw  = model.module if isinstance(model, DDP) else model
    raw.load_state_dict(ckpt["model"])
    if not model_only:
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
    log.info(
        "Loaded weights from %s  (step %d, epoch %d)%s",
        path, ckpt["global_step"], ckpt["epoch"],
        " [model weights only — fresh optimizer]" if model_only else "",
    )
    return ckpt["global_step"], ckpt["epoch"]


# ─── Training ─────────────────────────────────────────────────────────────────

def train(cfg: TrainConfig, resume_from: Optional[str] = None) -> None:
    # ── Distributed setup ─────────────────────────────────────────────────────
    rank, local_rank, world_size = _setup_ddp()
    is_main = rank == 0

    torch.manual_seed(cfg.seed + rank)

    # ── Device ────────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
        cfg.amp = False  # GradScaler requires CUDA

    if is_main:
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
        )
        log.info("")
        log.info("┌" + "─" * 78 + "┐")
        log.info("│" + " AlphaZero Chess Engine — Training Started ".center(78) + "│")
        log.info("└" + "─" * 78 + "┘")
        log.info("")

    # ── Data — shard PGN files across DDP ranks ───────────────────────────────
    # Each rank trains on a disjoint, round-robin slice of PGN files, preventing
    # duplicate gradient signals in multi-GPU runs.
    all_files  = find_pgn_files(cfg.data_dir)
    if not all_files:
        raise FileNotFoundError(f"No PGN files found under {cfg.data_dir!r}")

    rank_files = all_files[rank::world_size]
    if not rank_files:
        raise RuntimeError(
            f"Rank {rank}: no files assigned "
            f"(total={len(all_files)}, world_size={world_size}). "
            "Add more PGN files or reduce --nproc_per_node."
        )

    if is_main:
        log.info(f"  📁 Loaded {len(all_files)} PGN files ({len(rank_files)} for rank 0)")
        log.info(f"  🎯 Batch size: {cfg.batch_size} │ Workers: {cfg.num_workers} │ Prefetch: {cfg.prefetch_factor}")
        log.info("")

    if cfg.puzzle_dir is not None:
        puzzle_path = cfg.puzzle_dir
        if is_main:
            log.info(f"  🧩 Puzzle mixing enabled — CSV: {puzzle_path}  ratio: {cfg.puzzle_ratio}")
        loader = make_combined_dataloader(
            rank_files,
            puzzle_path,
            puzzle_ratio=cfg.puzzle_ratio,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            prefetch_factor=cfg.prefetch_factor,
            shuffle_files=True,
            seed=cfg.seed + rank,
            pin_memory=torch.cuda.is_available(),
            min_puzzle_rating=cfg.min_puzzle_rating,
        )
    else:
        loader = make_dataloader(
            rank_files,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            prefetch_factor=cfg.prefetch_factor,
            shuffle_files=True,
            seed=cfg.seed + rank,
            pin_memory=torch.cuda.is_available(),
        )

    # ── Model ─────────────────────────────────────────────────────────────────
    model: torch.nn.Module = ChessResNet(
        num_blocks=cfg.num_blocks,
        num_filters=cfg.num_filters,
    ).to(device)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    # Compile after DDP so torch.compile traces the full distributed graph,
    # enabling communication-compute overlap and correct AllReduce fusion.
    if cfg.compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    # ── Optimiser ─────────────────────────────────────────────────────────────
    # Fused AdamW kernel (PyTorch ≥ 2.0, CUDA only) reduces kernel-launch
    # overhead by merging all parameter updates into a single CUDA kernel.
    major = int(torch.__version__.split(".")[0])
    use_fused = torch.cuda.is_available() and major >= 2
    optimizer = AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        **({"fused": True} if use_fused else {}),
    )

    # OneCycleLR needs total_steps upfront; we use a user-supplied estimate.
    total_steps = cfg.epochs * cfg.steps_per_epoch
    scheduler = OneCycleLR(
        optimizer,
        max_lr=cfg.lr,
        total_steps=total_steps,
        pct_start=cfg.pct_start,
        anneal_strategy="cos",
    )

    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)

    # ── Resume ────────────────────────────────────────────────────────────────
    global_step = 0
    start_epoch = 0
    if resume_from:
        global_step, start_epoch = load_checkpoint(
            Path(resume_from), model, optimizer, scheduler, scaler,
            model_only=cfg.fine_tune,
        )
        if cfg.fine_tune:
            # Fine-tune starts a fresh LR schedule from step 0.
            global_step = 0
            start_epoch = 0
            if is_main:
                log.info("Fine-tune mode: optimizer and LR schedule reset to epoch 0.")

    # ── wandb (rank 0 only) ───────────────────────────────────────────────────
    if is_main and _WANDB_AVAILABLE and not cfg.wandb_disabled:
        wandb.init(
            project=cfg.wandb_project,
            name=cfg.wandb_run_name,
            config=vars(cfg),
            resume="allow",
        )
        raw_for_watch = model.module if isinstance(model, DDP) else model
        wandb.watch(raw_for_watch, log="gradients", log_freq=cfg.log_every_n_steps * 10)

    checkpoint_dir = Path(cfg.checkpoint_dir)

    # ── Epoch loop ────────────────────────────────────────────────────────────
    try:
        for epoch in range(start_epoch, cfg.epochs):
            if is_main:
                log.info(f"\n🔄 Starting Epoch {epoch+1}/{cfg.epochs}...")

            model.train()
            epoch_p_loss = 0.0
            epoch_v_loss = 0.0
            epoch_steps  = 0
            t0           = time.perf_counter()

            # Wrap dataloader with progress bar if available
            dataloader_iter = loader
            use_pbar = is_main and _TQDM_AVAILABLE
            if use_pbar:
                dataloader_iter = tqdm(
                    loader,
                    desc=f"  Epoch {epoch+1:2d}/{cfg.epochs}",
                    unit="batch",
                    ncols=120,
                    bar_format="{desc} │ {bar:40} │ {postfix}",
                    dynamic_ncols=True,
                )

            for board, policy_target, value_target in dataloader_iter:
                # Non-blocking transfers overlap with previous kernel execution.
                board         = board.to(device, non_blocking=True)          # (B, 119, 8, 8)
                policy_target = policy_target.to(device, non_blocking=True)  # (B, 4672)
                value_target  = value_target.to(device, non_blocking=True)   # (B,)

                optimizer.zero_grad(set_to_none=True)

                # AMP: forward + loss in lower precision; backward accumulates in fp32.
                with autocast(device_type=device.type, enabled=cfg.amp):
                    policy_logits, value_pred = model(board)
                    total_loss, p_loss, v_loss = compute_loss(
                        policy_logits, value_pred,
                        policy_target, value_target,
                        value_weight=cfg.value_loss_weight,
                        policy_weight=cfg.policy_loss_weight,
                    )

                scaler.scale(total_loss).backward()

                # Unscale before clipping so the threshold is in real-gradient space.
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)

                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                # Only advance the LR schedule when the optimizer step was actually
                # taken. If overflow was detected, scaler.update() reduces the scale,
                # and advancing the scheduler without a real update would drift the
                # LR curve ahead of actual training progress.
                if scaler.get_scale() >= scale_before:
                    scheduler.step()

                global_step  += 1
                epoch_steps  += 1
                epoch_p_loss += p_loss.item()
                epoch_v_loss += v_loss.item()

                # ── Real-time progress bar updates ─────────────────────────────────
                if is_main and use_pbar:
                    # Update progress bar every step with quick stats
                    if global_step % 10 == 0:  # Update every 10 steps for responsiveness
                        lr        = scheduler.get_last_lr()[0]
                        elapsed   = time.perf_counter() - t0
                        throughput = (
                            cfg.batch_size * world_size * 10 / elapsed if elapsed > 0 else 0
                        )
                        postfix_str = (
                            f"Step {global_step:7d} │ "
                            f"Loss: {total_loss.item():.4f} │ "
                            f"P: {p_loss.item():.4f} │ "
                            f"V: {v_loss.item():.4f} │ "
                            f"LR: {lr:.2e} │ "
                            f"{throughput:.0f} pos/s"
                        )
                        dataloader_iter.set_postfix_str(postfix_str)
                        t0 = time.perf_counter()

                # ── Step-level console logging ─────────────────────────────────────
                if is_main and global_step % cfg.log_every_n_steps == 0:

                    if _WANDB_AVAILABLE and not cfg.wandb_disabled:
                        wandb.log(
                            {
                                "train/total_loss":  total_loss.item(),
                                "train/policy_loss": p_loss.item(),
                                "train/value_loss":  v_loss.item(),
                                "train/lr":          lr,
                                "train/throughput":  throughput,
                                "epoch":             epoch,
                            },
                            step=global_step,
                        )
                    t0 = time.perf_counter()

                # ── Step checkpoint ──────────────────────────────────────────────
                if (
                    is_main
                    and cfg.checkpoint_every_n_steps > 0
                    and global_step % cfg.checkpoint_every_n_steps == 0
                ):
                    save_checkpoint(
                        checkpoint_dir, global_step, epoch,
                        model, optimizer, scheduler, scaler,
                        keep_last_n=cfg.keep_last_n_checkpoints,
                    )

            # ── End-of-epoch ──────────────────────────────────────────────────────
            if is_main:
                avg_p = epoch_p_loss / max(epoch_steps, 1)
                avg_v = epoch_v_loss / max(epoch_steps, 1)
                log.info("")
                log.info(
                    f"  ✓ Epoch {epoch + 1}/{cfg.epochs} complete │ "
                    f"Avg Policy Loss: {avg_p:.4f} │ Avg Value Loss: {avg_v:.4f}"
                )
                if _WANDB_AVAILABLE and not cfg.wandb_disabled:
                    wandb.log(
                        {
                            "epoch/avg_policy_loss": avg_p,
                            "epoch/avg_value_loss":  avg_v,
                        },
                        step=global_step,
                    )
                save_checkpoint(
                    checkpoint_dir, global_step, epoch + 1,
                    model, optimizer, scheduler, scaler,
                    keep_last_n=cfg.keep_last_n_checkpoints,
                )

        if is_main:
            log.info("")
            log.info("┌" + "─" * 78 + "┐")
            log.info("│" + " Training Complete! ".center(78) + "│")
            log.info("│" + f"Final checkpoint: {str(_ckpt_path(checkpoint_dir, global_step))[:70]}".ljust(79) + "│")
            log.info("└" + "─" * 78 + "┘")
            log.info("")

    finally:
        # Always shut down DataLoader workers before DDP teardown.
        # Without this, worker processes become zombies on exception exit.
        del loader
        # Close progress bar if it exists
        try:
            if is_main and _TQDM_AVAILABLE:
                dataloader_iter.close()
        except (NameError, AttributeError, TypeError):
            pass
        if is_main and _WANDB_AVAILABLE and not cfg.wandb_disabled:
            wandb.finish()
        _cleanup_ddp()


# ─── CLI ─────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="AlphaZero chess supervised training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    d = TrainConfig()

    g = p.add_argument_group("data")
    g.add_argument("--data_dir",        default=d.data_dir)
    g.add_argument("--num_workers",     type=int, default=d.num_workers)
    g.add_argument("--batch_size",      type=int, default=d.batch_size)
    g.add_argument("--prefetch_factor", type=int, default=d.prefetch_factor)
    g.add_argument("--steps_per_epoch", type=int, default=d.steps_per_epoch,
                   help="Estimated optimiser steps/epoch (used to size OneCycleLR).")

    g = p.add_argument_group("model")
    g.add_argument("--num_blocks",  type=int, default=d.num_blocks)
    g.add_argument("--num_filters", type=int, default=d.num_filters)

    g = p.add_argument_group("optimiser")
    g.add_argument("--lr",            type=float, default=d.lr)
    g.add_argument("--weight_decay",  type=float, default=d.weight_decay)
    g.add_argument("--max_grad_norm", type=float, default=d.max_grad_norm)
    g.add_argument("--epochs",        type=int,   default=d.epochs)
    g.add_argument("--pct_start",     type=float, default=d.pct_start)

    g = p.add_argument_group("loss")
    g.add_argument("--value_loss_weight",  type=float, default=d.value_loss_weight)
    g.add_argument("--policy_loss_weight", type=float, default=d.policy_loss_weight)

    g = p.add_argument_group("checkpointing")
    g.add_argument("--checkpoint_dir",           default=d.checkpoint_dir)
    g.add_argument("--checkpoint_every_n_steps", type=int, default=d.checkpoint_every_n_steps)
    g.add_argument("--keep_last_n_checkpoints",  type=int, default=d.keep_last_n_checkpoints)

    g = p.add_argument_group("puzzle mixing")
    g.add_argument("--puzzle_dir",        default=d.puzzle_dir,
                   help="Path to lichess_db_puzzle.csv (enables puzzle mixing).")
    g.add_argument("--puzzle_ratio",      type=float, default=d.puzzle_ratio,
                   help="Fraction of samples drawn from puzzles (default 0.2).")
    g.add_argument("--min_puzzle_rating", type=int, default=d.min_puzzle_rating,
                   help="Ignore puzzles below this Lichess rating (default 1200).")

    g = p.add_argument_group("logging")
    g.add_argument("--log_every_n_steps", type=int, default=d.log_every_n_steps)
    g.add_argument("--wandb_project",     default=d.wandb_project)
    g.add_argument("--wandb_run_name",    default=d.wandb_run_name)
    g.add_argument("--wandb_disabled",    action="store_true")

    g = p.add_argument_group("misc")
    g.add_argument("--seed",       type=int, default=d.seed)
    g.add_argument("--no_amp",     action="store_true", help="Disable Automatic Mixed Precision.")
    g.add_argument("--compile",    action="store_true", help="torch.compile the model (PyTorch ≥ 2).")
    g.add_argument("--resume",     default=None, help="Path to a checkpoint to resume from.")
    g.add_argument("--fine_tune",  action="store_true",
                   help="Load model weights only from --resume; reset optimizer and LR schedule.")

    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    cfg  = TrainConfig(
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
        fine_tune=args.fine_tune,
        log_every_n_steps=args.log_every_n_steps,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        wandb_disabled=args.wandb_disabled,
        seed=args.seed,
        amp=not args.no_amp,
        compile=args.compile,
    )
    train(cfg, resume_from=args.resume)
