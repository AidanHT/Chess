"""
Core type aliases and constants for the AlphaZero-style chess engine.

All constants define the exact geometry of the 8×8×119 board encoding and the
4672-action action space, matching the specification in Silver et al. (2018).
"""

from __future__ import annotations

from typing import Final, List, TypeAlias

import chess
import numpy as np

# ─── Board Encoding Dimensions ────────────────────────────────────────────────

BOARD_SIZE: Final[int] = 8

NUM_HISTORY_FRAMES: Final[int] = 8
"""Number of board positions retained in the history window (T = 8)."""

NUM_PIECE_TYPES: Final[int] = 6
"""Pawn, Knight, Bishop, Rook, Queen, King."""

PIECE_PLANES_PER_FRAME: Final[int] = NUM_PIECE_TYPES * 2  # 12
"""12 binary planes per frame: 6 for P1 pieces + 6 for P2 pieces."""

REPETITION_PLANES_PER_FRAME: Final[int] = 2
"""
Two repetition-indicator planes per history frame:
  offset 12 — position has been seen ≥ 1 time before in the game
  offset 13 — position has been seen ≥ 2 times before in the game
"""

PLANES_PER_FRAME: Final[int] = PIECE_PLANES_PER_FRAME + REPETITION_PLANES_PER_FRAME  # 14

HISTORY_PLANES: Final[int] = NUM_HISTORY_FRAMES * PLANES_PER_FRAME  # 112

SCALAR_PLANES: Final[int] = 7
"""
Seven scalar (8×8 constant-value) planes appended after the history block:
  index 112 — colour of current player       (0.0 = white, 1.0 = black)
  index 113 — en passant indicator           (1.0 at the target square file-column; else 0)
  index 114 — current-player kingside castling right
  index 115 — current-player queenside castling right
  index 116 — opponent kingside castling right
  index 117 — opponent queenside castling right
  index 118 — half-move clock, normalised by 100
"""

TOTAL_PLANES: Final[int] = HISTORY_PLANES + SCALAR_PLANES  # 119

# ─── Action Space ─────────────────────────────────────────────────────────────

NUM_QUEEN_DIRS: Final[int] = 8
"""Eight sliding directions: N, NE, E, SE, S, SW, W, NW."""

MAX_QUEEN_DIST: Final[int] = 7
"""Maximum squares a sliding piece can travel in one direction."""

NUM_QUEEN_MOVES: Final[int] = NUM_QUEEN_DIRS * MAX_QUEEN_DIST  # 56

NUM_KNIGHT_MOVES: Final[int] = 8
"""Eight distinct L-shaped knight offsets."""

NUM_UNDERPROMO_DIRS: Final[int] = 3
"""Underpromotion directions: capture-left, straight push, capture-right."""

NUM_UNDERPROMO_PIECES: Final[int] = 3
"""Knight, Bishop, Rook (queen promotions reuse the queen-move slot)."""

NUM_UNDERPROMO_MOVES: Final[int] = NUM_UNDERPROMO_DIRS * NUM_UNDERPROMO_PIECES  # 9

MOVE_TYPES_PER_SQUARE: Final[int] = (
    NUM_QUEEN_MOVES + NUM_KNIGHT_MOVES + NUM_UNDERPROMO_MOVES
)  # 73

ACTION_SPACE_SIZE: Final[int] = BOARD_SIZE * BOARD_SIZE * MOVE_TYPES_PER_SQUARE  # 4672
"""Total action space: 64 source squares × 73 move types."""

# ─── Piece Ordering ───────────────────────────────────────────────────────────

PIECE_ORDER: Final[List[chess.PieceType]] = [
    chess.PAWN,    # plane offset 0
    chess.KNIGHT,  # plane offset 1
    chess.BISHOP,  # plane offset 2
    chess.ROOK,    # plane offset 3
    chess.QUEEN,   # plane offset 4
    chess.KING,    # plane offset 5
]
"""
Canonical piece-type ordering within each player's 6-plane block.
Consistent across all history frames and both P1/P2 groups.
"""

# ─── Type Aliases ─────────────────────────────────────────────────────────────

BoardTensor: TypeAlias = np.ndarray
"""
np.ndarray, shape (8, 8, 119), dtype float32.
Channels-last layout; transpose to (119, 8, 8) before feeding PyTorch Conv2d.
"""

PolicyArray: TypeAlias = np.ndarray
"""np.ndarray, shape (4672,), dtype float32. Raw logits or probabilities."""

ValueScalar: TypeAlias = float
"""Scalar in [-1, 1] representing the expected game outcome from P1's perspective."""
