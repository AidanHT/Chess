"""
High-performance data pipeline for chess PGN datasets.

Architecture
────────────
Streaming parser   Reads PGN files lazily with yield generators — no file is
                   loaded into RAM in full.  python-chess handles one game at a
                   time; positions within each game are yielded one by one.

Multiprocessing    DataLoader spawns N worker processes (PyTorch multiprocessing).
                   Each worker owns a round-robin shard of the PGN file list so
                   no position is duplicated across workers.  A separate
                   ProcessPoolExecutor-based helper is also provided for
                   offline pre-processing of very large datasets.

IterableDataset    ChessDataset consumes the per-worker file shard and converts
                   raw numpy arrays into float32 tensors on the fly.

DataLoader         Configured with pin_memory=True and prefetch_factor ≥ 2 to
                   keep the GPU fed with minimal stall time.

Sample format
─────────────
board_tensor : torch.Tensor, shape (119, 8, 8), dtype float32
               AlphaZero board encoding (channels-first, current-player POV).
policy       : torch.Tensor, shape (4672,), dtype float32
               One-hot over the expert move in the AlphaZero action space.
value        : torch.Tensor, shape (), dtype float32
               Game result from the current player's perspective:
               +1.0 (win), −1.0 (loss), 0.0 (draw).
"""

from __future__ import annotations

import logging
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

import chess
import chess.pgn
import numpy as np
import torch
import torch.utils.data

from core_types import ACTION_SPACE_SIZE
from state_encoding import encode_board, encode_move_perspective

log = logging.getLogger(__name__)

# ─── Result helpers ────────────────────────────────────────────────────────────

_RESULT_MAP = {"1-0": 1.0, "0-1": -1.0, "1/2-1/2": 0.0}


def _result_to_white_value(result_str: str) -> Optional[float]:
    """Return the game value from White's perspective, or None for unknown results."""
    return _RESULT_MAP.get(result_str)


# ─── Position-level encoder ────────────────────────────────────────────────────

def _encode_position(
    board: chess.Board, move: chess.Move, white_value: float
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Encode a single (board, move) pair into network targets.

    Returns
    -------
    board_tensor : np.ndarray, shape (119, 8, 8), float32   (channels-first)
    policy       : np.ndarray, shape (4672,),    float32   (one-hot)
    value        : float  — current player's perspective
    """
    turn = board.turn

    # (8, 8, 119) → (119, 8, 8); .copy() ensures C-contiguous layout for
    # torch.from_numpy and efficient IPC transfer to the main process.
    board_tensor: np.ndarray = np.ascontiguousarray(
        encode_board(board).transpose(2, 0, 1)
    )

    policy = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
    policy[encode_move_perspective(move, turn)] = 1.0

    value = white_value if turn == chess.WHITE else -white_value

    return board_tensor, policy, float(value)


# ─── Streaming PGN parser ──────────────────────────────────────────────────────

def stream_pgn_file(
    path: os.PathLike | str,
) -> Iterator[Tuple[np.ndarray, np.ndarray, float]]:
    """
    Lazily stream ``(board_tensor, policy, value)`` samples from one PGN file.

    The file is read incrementally with ``chess.pgn.read_game``; only a single
    game occupies memory at any moment.  Corrupt or unparseable games are
    logged at WARNING level and skipped; parsing resumes at the next game.

    Parameters
    ----------
    path :
        Path to a PGN file (may contain any number of games).

    Yields
    ------
    board_tensor : np.ndarray, shape (119, 8, 8), float32
    policy       : np.ndarray, shape (4672,),    float32
    value        : float in {−1.0, 0.0, 1.0}
    """
    path = Path(path)
    try:
        fh = open(path, encoding="utf-8", errors="replace")
    except OSError as exc:
        log.error("Cannot open %s: %s", path, exc)
        return

    game_idx = 0
    sample_count = 0

    with fh:
        while True:
            # --- read one game ---
            try:
                game = chess.pgn.read_game(fh)
            except Exception as exc:  # noqa: BLE001  (broad catch is intentional)
                log.warning("Corrupt PGN near game %d in %s: %s", game_idx, path, exc)
                continue  # try to keep reading — python-chess may have recovered

            if game is None:
                break  # clean EOF

            game_idx += 1

            result_str = game.headers.get("Result", "*")
            white_value = _result_to_white_value(result_str)
            if white_value is None:
                log.debug(
                    "Skipping game %d in %s: unrecognised result %r",
                    game_idx, path, result_str,
                )
                continue

            # --- iterate positions within the game ---
            board = game.board()
            try:
                for move in game.mainline_moves():
                    try:
                        board_tensor, policy, value = _encode_position(
                            board, move, white_value
                        )
                        sample_count += 1
                        yield board_tensor, policy, value
                    except KeyError:
                        # encode_move_perspective raises KeyError for geometrically
                        # invalid moves (should not occur for legal PGN, but guard anyway).
                        log.warning(
                            "Unencodable move %s in game %d of %s; skipping position",
                            move.uci(), game_idx, path,
                        )
                    finally:
                        board.push(move)  # advance board whether or not we yielded
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "Error iterating game %d in %s: %s", game_idx, path, exc
                )

    log.debug("Done %s — %d games, %d samples", path, game_idx, sample_count)


# ─── IterableDataset ───────────────────────────────────────────────────────────

class ChessDataset(torch.utils.data.IterableDataset):
    """
    Streaming PyTorch dataset over an arbitrary number of PGN files.

    **DataLoader worker safety**: when ``DataLoader(num_workers=N)`` spawns N
    worker processes, each worker receives a non-overlapping round-robin shard
    of ``pgn_files`` via ``torch.utils.data.get_worker_info()``.  No position
    is emitted twice within a single epoch regardless of N.

    Parameters
    ----------
    pgn_files :
        Ordered collection of PGN file paths.
    shuffle_files :
        Shuffle each worker's file shard at the start of every ``__iter__``
        call.  Different workers use different seeds (base seed + worker id)
        so they don't end up with identical orderings.
    seed :
        Base integer seed for file-list shuffling.  Set to ``None`` for
        non-deterministic shuffling.
    """

    def __init__(
        self,
        pgn_files: Sequence[os.PathLike | str],
        *,
        shuffle_files: bool = True,
        seed: Optional[int] = 42,
    ) -> None:
        super().__init__()
        if not pgn_files:
            raise ValueError("pgn_files must be a non-empty sequence.")
        self.pgn_files: List[Path] = [Path(p) for p in pgn_files]
        self.shuffle_files = shuffle_files
        self.seed = seed

    def __iter__(
        self,
    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        worker_info = torch.utils.data.get_worker_info()

        files: List[Path] = list(self.pgn_files)

        # Shard BEFORE shuffling so the partition is deterministic and
        # non-overlapping regardless of the random seed.  round-robin slicing
        # on a fixed list is always disjoint and exhaustive across all workers.
        if worker_info is not None:
            files = files[worker_info.id :: worker_info.num_workers]

        # Shuffle within the shard for within-worker processing variety.
        if self.shuffle_files:
            worker_id = worker_info.id if worker_info is not None else 0
            rng = random.Random(
                (self.seed if self.seed is not None else 0) + worker_id
            )
            rng.shuffle(files)

        # --- stream positions from each file ---
        for path in files:
            for board_arr, policy_arr, value in stream_pgn_file(path):
                yield (
                    torch.from_numpy(board_arr),                   # (119, 8, 8) float32
                    torch.from_numpy(policy_arr),                  # (4672,)     float32
                    torch.tensor(value, dtype=torch.float32),      # scalar      float32
                )


# ─── DataLoader factory ────────────────────────────────────────────────────────

def make_dataloader(
    pgn_files: Sequence[os.PathLike | str],
    *,
    batch_size: int = 512,
    num_workers: int = 4,
    prefetch_factor: int = 2,
    shuffle_files: bool = True,
    seed: Optional[int] = 42,
    pin_memory: bool = True,
) -> torch.utils.data.DataLoader:
    """
    Build a DataLoader optimised for GPU training on chess positions.

    Each ``num_workers`` subprocess streams its own shard of PGN files
    (PyTorch multiprocessing) and pre-fetches ``prefetch_factor`` batches
    ahead.  Pinned memory is activated when CUDA is available, enabling
    asynchronous DMA transfers that overlap with GPU compute.

    Parameters
    ----------
    pgn_files :
        PGN files to include.  Pass the result of ``find_pgn_files()`` or
        any sequence of path-like objects.
    batch_size :
        Samples per batch delivered to the training loop.
    num_workers :
        Parallel parser processes.  0 = single-process (good for debugging).
        For GPU training, 4–8 workers are typical.
    prefetch_factor :
        Batches each worker pre-fetches beyond the current demand.
        Must be ≥ 2; higher values increase RAM usage but reduce stall time.
    shuffle_files :
        Shuffle file order at the start of each epoch.
    seed :
        Base random seed for file shuffling.
    pin_memory :
        Use pinned (page-locked) CPU memory for faster GPU DMA transfers.
        Automatically disabled when CUDA is not available.

    Returns
    -------
    torch.utils.data.DataLoader
        Yields ``(board_tensor, policy, value)`` batches of shape
        ``(B, 119, 8, 8)``, ``(B, 4672)``, ``(B,)``.

    Raises
    ------
    ValueError
        If ``prefetch_factor < 2`` or ``pgn_files`` is empty.
    """
    if prefetch_factor < 2:
        raise ValueError(f"prefetch_factor must be ≥ 2, got {prefetch_factor}.")

    dataset = ChessDataset(pgn_files, shuffle_files=shuffle_files, seed=seed)

    effective_pin_memory = pin_memory and torch.cuda.is_available()

    loader_kwargs: dict = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=effective_pin_memory,
        # Keep workers alive across batches — avoids fork overhead per epoch.
        persistent_workers=(num_workers > 0),
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor

    return torch.utils.data.DataLoader(dataset, **loader_kwargs)


# ─── Offline pre-processing with ProcessPoolExecutor ──────────────────────────

def _parse_file_worker(path: Path) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    """
    Worker function: parse an entire PGN file and return all samples as a list.

    Intended for offline pre-processing (e.g. caching to disk) where the full
    dataset fits in RAM.  For streaming training, use ``ChessDataset`` instead.
    """
    return list(stream_pgn_file(path))


def preprocess_parallel(
    pgn_files: Sequence[os.PathLike | str],
    *,
    max_workers: Optional[int] = None,
) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    """
    Parse all PGN files in parallel using ``ProcessPoolExecutor`` and return
    a flat list of all samples.

    This is useful for datasets small enough to fit in RAM.  For datasets with
    millions of games, use ``make_dataloader`` (streaming) instead.

    Parameters
    ----------
    pgn_files :
        PGN files to process.
    max_workers :
        Maximum parallel processes.  Defaults to ``os.cpu_count()``.

    Returns
    -------
    List of ``(board_tensor, policy, value)`` tuples.
    """
    paths = [Path(p) for p in pgn_files]
    samples: List[Tuple[np.ndarray, np.ndarray, float]] = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {executor.submit(_parse_file_worker, p): p for p in paths}
        for future in as_completed(future_to_path):
            path = future_to_path[future]
            try:
                samples.extend(future.result())
            except Exception as exc:  # noqa: BLE001
                log.error("Failed to parse %s: %s", path, exc)

    return samples


# ─── File discovery ────────────────────────────────────────────────────────────

def find_pgn_files(
    root: os.PathLike | str,
    *,
    recursive: bool = True,
) -> List[Path]:
    """
    Return all ``*.pgn`` files under *root*, sorted for reproducibility.

    Parameters
    ----------
    root :
        Directory to search.
    recursive :
        Search subdirectories recursively (default True).
    """
    root = Path(root)
    pattern = "**/*.pgn" if recursive else "*.pgn"
    return sorted(root.glob(pattern))
