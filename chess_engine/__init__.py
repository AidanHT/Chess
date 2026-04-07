"""AlphaZero-style chess engine implementation."""

from .core.encoding import decode_move, decode_move_perspective, encode_board, encode_move, encode_move_perspective
from .core.types import (
    ACTION_SPACE_SIZE,
    BOARD_SIZE,
    TOTAL_PLANES,
    BoardTensor,
    PolicyArray,
    ValueScalar,
)
from .data.pipeline import ChessDataset, find_pgn_files, make_dataloader, stream_pgn_file
from .engine.uci import UCIEngine
from .mcts.search import MCTS
from .models.resnet import ChessResNet

__all__ = [
    # Types
    "BoardTensor",
    "PolicyArray",
    "ValueScalar",
    # Constants
    "ACTION_SPACE_SIZE",
    "BOARD_SIZE",
    "TOTAL_PLANES",
    # Core functions
    "encode_board",
    "encode_move",
    "encode_move_perspective",
    "decode_move",
    "decode_move_perspective",
    # Data
    "stream_pgn_file",
    "ChessDataset",
    "find_pgn_files",
    "make_dataloader",
    # Models
    "ChessResNet",
    # MCTS
    "MCTS",
    # Engine
    "UCIEngine",
]
