"""
Board-state and move encoding for the AlphaZero-style chess engine.

Board Encoding — shape (8, 8, 119), dtype float32
══════════════════════════════════════════════════
Planes 0 – 111  History block  (8 frames × 14 planes, newest first).

  Each frame f occupies planes [f*14 : f*14+14]:
    offset  0 – 5   P1 piece planes  (Pawn, Knight, Bishop, Rook, Queen, King)
    offset  6 – 11  P2 piece planes  (same ordering)
    offset 12       Repetition ≥ 1   (1.0 if the position was seen before)
    offset 13       Repetition ≥ 2   (1.0 if the position was seen twice)

  P1 is always the player whose turn it is at the ROOT position (frame 0).

Planes 112 – 118  Scalar block (each plane is a constant 8×8 value):
    112  Colour of current player       (0.0 = white, 1.0 = black)
    113  En passant indicator           (1.0 column = ep-target-file; else 0.0)
    114  Current-player kingside castling
    115  Current-player queenside castling
    116  Opponent kingside castling
    117  Opponent queenside castling
    118  Half-move clock / 100

All planes are encoded from the CURRENT PLAYER'S perspective:
  when Black is to move, all bitboards are rank-mirrored before assignment,
  equivalent to viewing the board from the Black side of the table.

──────────────────────────────────────────────────────────────────────────────
Action Space — 4672 actions
══════════════════════════════════════════════════
  index = from_square × 73 + move_type

  move_type  0 – 55   Queen/sliding:    dir_idx × 7 + (dist − 1)
                       dir_idx: N=0, NE=1, E=2, SE=3, S=4, SW=5, W=6, NW=7
  move_type 56 – 63   Knight moves      (8 L-shaped offsets, fixed ordering)
  move_type 64 – 72   Underpromotions:  up_dir × 3 + piece_idx
                       up_dir:    capture-left=0, straight=1, capture-right=2
                       piece_idx: knight=0, bishop=1, rook=2

Queen promotions are encoded as the equivalent queen/sliding move (same index as
the non-promotion move with the same from_square → to_square geometry).

All indices assume moves are presented from WHITE'S absolute perspective.
For Black's moves, mirror from/to squares via chess.square_mirror() before
encoding, or use the perspective-aware helpers encode_move_perspective() and
decode_move_perspective().
"""

from __future__ import annotations

from typing import Dict, Final, List, Optional, Tuple

import chess
import numpy as np

from core_types import (
    ACTION_SPACE_SIZE,
    BOARD_SIZE,
    MAX_QUEEN_DIST,
    MOVE_TYPES_PER_SQUARE,
    NUM_HISTORY_FRAMES,
    NUM_KNIGHT_MOVES,
    NUM_PIECE_TYPES,
    NUM_QUEEN_MOVES,
    NUM_UNDERPROMO_PIECES,
    PIECE_ORDER,
    PLANES_PER_FRAME,
    SCALAR_PLANES,
    TOTAL_PLANES,
)

# ─── Direction Tables (module-level constants) ────────────────────────────────

_QUEEN_DIRS: Final[Tuple[Tuple[int, int], ...]] = (
    ( 1,  0),  # N
    ( 1,  1),  # NE
    ( 0,  1),  # E
    (-1,  1),  # SE
    (-1,  0),  # S
    (-1, -1),  # SW
    ( 0, -1),  # W
    ( 1, -1),  # NW
)
"""(rank-delta, file-delta) for the eight sliding directions in AlphaZero order."""

_KNIGHT_DIRS: Final[Tuple[Tuple[int, int], ...]] = (
    ( 2,  1), ( 2, -1),
    ( 1,  2), ( 1, -2),
    (-1,  2), (-1, -2),
    (-2,  1), (-2, -1),
)
"""(rank-delta, file-delta) for the eight knight offsets in AlphaZero order."""

# ─── Internal helpers ─────────────────────────────────────────────────────────

def _bitboards_to_planes(bbs: np.ndarray, mirror: bool) -> np.ndarray:
    """
    Vectorised conversion of N bitboard integers to N 8×8 float32 arrays.

    Uses explicit little-endian byte layout so results are identical on both
    little-endian (x86/ARM) and big-endian systems.

    Parameters
    ----------
    bbs : np.ndarray
        Shape (N,), dtype uint64 (or castable to uint64).  Each element is a
        python-chess bitboard where bit i is set iff a piece occupies square i
        (a1=0, h1=7, a2=8, …, h8=63).
    mirror : bool
        If True, flip rank order (row 0 ↔ row 7) to obtain Black's perspective.

    Returns
    -------
    np.ndarray
        Shape (N, 8, 8), dtype float32.  planes[n, rank, file] = 1.0 iff the
        corresponding bit is set in bbs[n].
    """
    n = len(bbs)
    # Cast to explicit little-endian uint64 ('<u8') then reinterpret as bytes.
    # Vectorised — no Python loop per element — and portable: '<u8' forces
    # little-endian layout on any host so bit 0 (a1) always lands in byte 0.
    buf = np.asarray(bbs, dtype="<u8").view(np.uint8).reshape(n, 8)

    # Unpack to (N, 64) with bitorder='little' so index 0 = bit 0 = square a1.
    planes = (
        np.unpackbits(buf, axis=1, bitorder="little")
        .reshape(n, 8, 8)
        .astype(np.float32)
    )
    # planes[n, rank, file] where rank 0 = rank-1 (a1-h1), rank 7 = rank-8.

    if mirror:
        planes = planes[:, ::-1, :].copy()  # rank-flip for Black's perspective

    return planes  # shape (N, 8, 8)


# ─── Action-space LUTs (built once at import time) ────────────────────────────

def _build_encode_lut() -> Dict[Tuple[int, int, Optional[int]], int]:
    """
    Build a mapping (from_sq, to_sq, promotion) → action_index.

    Queen promotions (promotion=chess.QUEEN) share the index of the geometrically
    equivalent queen/sliding move (promotion=None).  Non-queen promotions get their
    own indices in the underpromotion range [64, 72].
    """
    lut: Dict[Tuple[int, int, Optional[int]], int] = {}

    for from_sq in range(64):
        fr = chess.square_rank(from_sq)
        ff = chess.square_file(from_sq)
        base = from_sq * MOVE_TYPES_PER_SQUARE

        # ── Queen / sliding moves ──────────────────────────────────────────
        for dir_idx, (dr, dc) in enumerate(_QUEEN_DIRS):
            for dist in range(1, BOARD_SIZE):
                to_r = fr + dr * dist
                to_c = ff + dc * dist
                if not (0 <= to_r < BOARD_SIZE and 0 <= to_c < BOARD_SIZE):
                    break  # ray exits board; no farther squares in this direction
                to_sq = chess.square(to_c, to_r)
                move_type = dir_idx * MAX_QUEEN_DIST + (dist - 1)
                idx = base + move_type
                lut[(from_sq, to_sq, None)] = idx
                lut[(from_sq, to_sq, chess.QUEEN)] = idx  # queen promo → same slot

        # ── Knight moves ───────────────────────────────────────────────────
        for k_idx, (dr, dc) in enumerate(_KNIGHT_DIRS):
            to_r = fr + dr
            to_c = ff + dc
            if 0 <= to_r < BOARD_SIZE and 0 <= to_c < BOARD_SIZE:
                to_sq = chess.square(to_c, to_r)
                lut[(from_sq, to_sq, None)] = base + NUM_QUEEN_MOVES + k_idx

        # ── Underpromotions ────────────────────────────────────────────────
        # One rank ahead (dist=1); legality (piece is a pawn, destination is
        # back rank) is enforced by python-chess, not by this encoding.
        for up_dir, dc in enumerate((-1, 0, 1)):
            to_r = fr + 1
            to_c = ff + dc
            if 0 <= to_r < BOARD_SIZE and 0 <= to_c < BOARD_SIZE:
                to_sq = chess.square(to_c, to_r)
                for piece_idx, promo in enumerate(
                    (chess.KNIGHT, chess.BISHOP, chess.ROOK)
                ):
                    move_type = (
                        NUM_QUEEN_MOVES
                        + NUM_KNIGHT_MOVES
                        + up_dir * NUM_UNDERPROMO_PIECES
                        + piece_idx
                    )
                    lut[(from_sq, to_sq, promo)] = base + move_type

    return lut


def _build_decode_arrays() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build flat arrays for O(1) index → move decoding.

    Returns
    -------
    from_sqs : np.ndarray, shape (4672,), dtype int8
    to_sqs   : np.ndarray, shape (4672,), dtype int8
    promos   : np.ndarray, shape (4672,), dtype int8
        0 = no promotion; otherwise a chess.PieceType value.
        Queen promotions are stored as chess.QUEEN (5).
    """
    from_sqs = np.zeros(ACTION_SPACE_SIZE, dtype=np.int8)
    to_sqs   = np.zeros(ACTION_SPACE_SIZE, dtype=np.int8)
    promos   = np.zeros(ACTION_SPACE_SIZE, dtype=np.int8)

    for from_sq in range(64):
        fr   = chess.square_rank(from_sq)
        ff   = chess.square_file(from_sq)
        base = from_sq * MOVE_TYPES_PER_SQUARE

        # Queen / sliding
        for dir_idx, (dr, dc) in enumerate(_QUEEN_DIRS):
            for dist in range(1, BOARD_SIZE):
                to_r = fr + dr * dist
                to_c = ff + dc * dist
                if not (0 <= to_r < BOARD_SIZE and 0 <= to_c < BOARD_SIZE):
                    break
                to_sq     = chess.square(to_c, to_r)
                move_type = dir_idx * MAX_QUEEN_DIST + (dist - 1)
                idx = base + move_type
                from_sqs[idx] = from_sq
                to_sqs[idx]   = to_sq
                promos[idx]   = chess.QUEEN if to_r == 7 else 0

        # Knight
        for k_idx, (dr, dc) in enumerate(_KNIGHT_DIRS):
            to_r = fr + dr
            to_c = ff + dc
            if 0 <= to_r < BOARD_SIZE and 0 <= to_c < BOARD_SIZE:
                to_sq = chess.square(to_c, to_r)
                idx = base + NUM_QUEEN_MOVES + k_idx
                from_sqs[idx] = from_sq
                to_sqs[idx]   = to_sq
                # promos[idx] already 0

        # Underpromotions
        for up_dir, dc in enumerate((-1, 0, 1)):
            to_r = fr + 1
            to_c = ff + dc
            if 0 <= to_r < BOARD_SIZE and 0 <= to_c < BOARD_SIZE:
                to_sq = chess.square(to_c, to_r)
                for piece_idx, promo in enumerate(
                    (chess.KNIGHT, chess.BISHOP, chess.ROOK)
                ):
                    move_type = (
                        NUM_QUEEN_MOVES
                        + NUM_KNIGHT_MOVES
                        + up_dir * NUM_UNDERPROMO_PIECES
                        + piece_idx
                    )
                    idx = base + move_type
                    from_sqs[idx] = from_sq
                    to_sqs[idx]   = to_sq
                    promos[idx]   = promo

    return from_sqs, to_sqs, promos


# Build LUTs at import time (fast; runs once)
_ENCODE_LUT: Final[Dict[Tuple[int, int, Optional[int]], int]] = _build_encode_lut()
_DECODE_FROM, _DECODE_TO, _DECODE_PROMO = _build_decode_arrays()

# ─── Public API ───────────────────────────────────────────────────────────────

def encode_board(board: chess.Board) -> np.ndarray:
    """
    Convert a chess board state into an (8, 8, 119) float32 feature tensor.

    The encoding follows the AlphaZero specification:
      * T = 8 history frames, each with 14 planes (12 piece + 2 repetition).
      * 7 scalar planes appended at the end.
      * Board is always presented from the CURRENT PLAYER'S perspective
        (rank-mirrored when it is Black's turn so P1 pieces are always
        at the "bottom" of the spatial grid).
      * History is reconstructed from board.move_stack; frames with no
        preceding position are left as zero planes.

    Parameters
    ----------
    board : chess.Board
        Current board state.  Previous positions are extracted from
        board.move_stack.  The original board is NOT modified.

    Returns
    -------
    np.ndarray
        Shape (8, 8, 119), dtype float32.
        Transpose to (119, 8, 8) before passing to a PyTorch Conv2d layer.
    """
    planes = np.zeros((BOARD_SIZE, BOARD_SIZE, TOTAL_PLANES), dtype=np.float32)

    is_black: bool = board.turn == chess.BLACK
    p1_color: chess.Color = chess.BLACK if is_black else chess.WHITE
    p2_color: chess.Color = chess.WHITE if is_black else chess.BLACK

    # ── 2. Encode history frames ──────────────────────────────────────────────
    # Single mutable copy serves both piece-plane extraction and repetition
    # queries — eliminates a second full board.copy() and the intermediate
    # list of scratch.copy(stack=False) snapshots.
    scratch = board.copy()

    for frame_idx in range(NUM_HISTORY_FRAMES):
        base = frame_idx * PLANES_PER_FRAME  # 0, 14, 28, …

        # ── 2a. Piece planes (vectorised over 12 bitboards) ───────────────
        bbs = np.array(
            [scratch.pieces_mask(pt, p1_color) for pt in PIECE_ORDER]
            + [scratch.pieces_mask(pt, p2_color) for pt in PIECE_ORDER],
            dtype=np.uint64,
        )  # shape (12,)

        piece_planes = _bitboards_to_planes(bbs, mirror=is_black)
        # piece_planes shape: (12, 8, 8) → assign to (8, 8, 12) slice
        planes[:, :, base : base + 12] = piece_planes.transpose(1, 2, 0)

        # ── 2b. Repetition planes ─────────────────────────────────────────
        # is_repetition(n) returns True if the position at this ply appeared
        # at least n times earlier in the game (move_stack intact on scratch).
        if scratch.is_repetition(2):
            planes[:, :, base + 12] = 1.0
        if scratch.is_repetition(3):
            planes[:, :, base + 13] = 1.0

        # Advance to the previous position; stop when history is exhausted.
        # Remaining frames are implicitly zero (array was zeroed at allocation).
        if frame_idx + 1 < NUM_HISTORY_FRAMES:
            if not scratch.move_stack:
                break
            scratch.pop()

    # ── 3. Scalar planes (broadcast constants) ────────────────────────────────
    scalar_base = TOTAL_PLANES - SCALAR_PLANES  # 112

    # 112: colour (0 = white to move, 1 = black to move)
    planes[:, :, scalar_base + 0] = float(is_black)

    # 113: en passant indicator — 1.0 in the entire column of the ep target square
    ep_sq = board.ep_square
    if ep_sq is not None:
        ep_file = chess.square_file(ep_sq)
        ep_rank = chess.square_rank(ep_sq)
        # Apply the same perspective flip as the piece planes
        if is_black:
            ep_rank = 7 - ep_rank
        planes[ep_rank, ep_file, scalar_base + 1] = 1.0

    # 114 – 117: castling rights
    planes[:, :, scalar_base + 2] = float(board.has_kingside_castling_rights(p1_color))
    planes[:, :, scalar_base + 3] = float(board.has_queenside_castling_rights(p1_color))
    planes[:, :, scalar_base + 4] = float(board.has_kingside_castling_rights(p2_color))
    planes[:, :, scalar_base + 5] = float(board.has_queenside_castling_rights(p2_color))

    # 118: half-move clock normalised to [0, 1] (50-move rule triggers at 100 half-moves)
    planes[:, :, scalar_base + 6] = board.halfmove_clock / 100.0

    return planes


def encode_move(move: chess.Move) -> int:
    """
    Encode a chess move into an action index in [0, 4671].

    The encoding is from WHITE'S absolute perspective.  For Black's moves,
    mirror both squares with chess.square_mirror() before calling, or use
    encode_move_perspective() instead.

    Queen promotions and the corresponding no-promotion move with the same
    from/to squares share a single index.

    Parameters
    ----------
    move : chess.Move
        A chess move.  move.promotion should be set for promotion moves
        (None or chess.QUEEN for queen promotions; chess.KNIGHT / BISHOP /
        ROOK for underpromotions).

    Returns
    -------
    int
        Action index in [0, ACTION_SPACE_SIZE).

    Raises
    ------
    KeyError
        If the move geometry is not representable in the 4672-action space
        (i.e., the move is geometrically illegal in chess).
    """
    return _ENCODE_LUT[(move.from_square, move.to_square, move.promotion)]


def decode_move(index: int) -> chess.Move:
    """
    Decode an action index into a chess.Move.

    The returned move is from WHITE'S absolute perspective.  For moves that
    were encoded from Black's perspective, mirror the squares afterwards with
    chess.square_mirror(), or use decode_move_perspective() instead.

    For queen/sliding moves that land on rank 8 (rank index 7), the returned
    move has promotion=chess.QUEEN, consistent with encode_move behaviour for
    queen promotions.  Non-pawn pieces that happen to land on rank 8 will also
    carry this promotion flag; legality checking (board.legal_moves) removes
    such spurious entries at play-time.

    Parameters
    ----------
    index : int
        Action index in [0, ACTION_SPACE_SIZE).

    Returns
    -------
    chess.Move
        Decoded move.  May be illegal on a given board position.

    Raises
    ------
    IndexError
        If index is out of [0, ACTION_SPACE_SIZE).
    """
    if not (0 <= index < ACTION_SPACE_SIZE):
        raise IndexError(
            f"Action index {index} out of range [0, {ACTION_SPACE_SIZE})."
        )
    from_sq = int(_DECODE_FROM[index])
    to_sq   = int(_DECODE_TO[index])
    promo   = int(_DECODE_PROMO[index])
    return chess.Move(from_sq, to_sq, promotion=promo if promo != 0 else None)


def encode_move_perspective(move: chess.Move, turn: chess.Color) -> int:
    """
    Encode a move using the current player's perspective.

    When turn is chess.BLACK, from/to squares are rank-mirrored before
    encoding so that the action index is consistent with the perspective-
    corrected board tensor produced by encode_board().

    Parameters
    ----------
    move : chess.Move
    turn : chess.Color
        chess.WHITE or chess.BLACK — the side that is making the move.

    Returns
    -------
    int
        Action index in [0, ACTION_SPACE_SIZE).
    """
    if turn == chess.BLACK:
        mirrored = chess.Move(
            chess.square_mirror(move.from_square),
            chess.square_mirror(move.to_square),
            promotion=move.promotion,
        )
        return encode_move(mirrored)
    return encode_move(move)


def decode_move_perspective(index: int, turn: chess.Color) -> chess.Move:
    """
    Decode an action index back to an absolute chess.Move.

    Inverse of encode_move_perspective().

    Parameters
    ----------
    index : int
        Action index in [0, ACTION_SPACE_SIZE).
    turn : chess.Color
        The side whose perspective was used during encoding.

    Returns
    -------
    chess.Move
        Move in absolute (python-chess) square coordinates.
    """
    move = decode_move(index)
    if turn == chess.BLACK:
        return chess.Move(
            chess.square_mirror(move.from_square),
            chess.square_mirror(move.to_square),
            promotion=move.promotion,
        )
    return move
