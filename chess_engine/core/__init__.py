"""Core types and encoding for chess board representation."""

from .encoding import decode_move, decode_move_perspective, encode_board, encode_move, encode_move_perspective
from .types import (
    ACTION_SPACE_SIZE,
    BOARD_SIZE,
    PIECE_ORDER,
    TOTAL_PLANES,
    BoardTensor,
    PolicyArray,
    ValueScalar,
)

__all__ = [
    "ACTION_SPACE_SIZE",
    "BOARD_SIZE",
    "TOTAL_PLANES",
    "PIECE_ORDER",
    "BoardTensor",
    "PolicyArray",
    "ValueScalar",
    "encode_board",
    "encode_move",
    "encode_move_perspective",
    "decode_move",
    "decode_move_perspective",
]
