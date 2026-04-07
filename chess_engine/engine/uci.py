"""
uci_engine.py
─────────────
Universal Chess Interface (UCI) wrapper for the AlphaZero-style chess engine.

Usage:
    python uci_engine.py [--checkpoint_dir CHECKPOINTS] [--num_blocks N] [--num_filters F]

Protocol flow (all traffic via stdin/stdout; diagnostics to stderr):
    GUI → engine : uci, isready, position, go, stop, quit, setoption
    engine → GUI : id, uciok, readyok, bestmove, info

State-encoding contract:
    encode_board() reads the board's move_stack to reconstruct up to 8 history
    frames.  We preserve that stack by using board.push_uci() for every move in
    the position command and by passing a board copy (with stack=True, the
    default) to the search thread.
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
from pathlib import Path
from typing import Optional

import chess
import numpy as np
import torch

from ..core import decode_move_perspective, encode_move_perspective
from ..mcts import MCTS
from ..models import ChessResNet

# ── Logging (stderr only — UCI traffic must stay on stdout) ───────────────────

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── UCI output helper ─────────────────────────────────────────────────────────

def _send(line: str) -> None:
    """Write one UCI response line to stdout, flushed immediately."""
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


# ── Constants ─────────────────────────────────────────────────────────────────

ENGINE_NAME   = "AlphaZeroChess"
ENGINE_AUTHOR = "Aidan"

# Simulations/second used to convert movetime → node count.
# GPU baseline: ~800 sims/s; CPU is considerably slower.  Tunable at runtime
# via the NPS UCI option so users can calibrate to their hardware.
_DEFAULT_NPS = 800

# For "go infinite": simulations per MCTS.run() call.  The engine loops until
# "stop" arrives, checking the stop-event between calls.  Each call starts a
# fresh tree (MCTS.run() is stateless between invocations), so quality per
# response is fixed at this value regardless of how long "infinite" has run.
_INFINITE_CHUNK = 800


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def _latest_checkpoint(directory: str) -> Optional[Path]:
    """Return the newest step_*.pt file in *directory*, or None."""
    files = sorted(Path(directory).glob("step_*.pt"))
    return files[-1] if files else None


def _load_model(
    directory: str,
    num_blocks: int,
    num_filters: int,
    device: torch.device,
) -> ChessResNet:
    """
    Instantiate ChessResNet, load the latest checkpoint, move to *device*, and
    lock into eval mode.  Falls back to random weights if no checkpoint exists.
    """
    model = ChessResNet(num_blocks=num_blocks, num_filters=num_filters)

    ckpt_path = _latest_checkpoint(directory)
    if ckpt_path is None:
        log.warning(
            "No checkpoint found in '%s'. Running with random weights.", directory
        )
    else:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model"])
        step = ckpt.get("global_step", "?")
        log.info("Loaded checkpoint: %s  (step %s)", ckpt_path, step)

    model.to(device)
    model.eval()
    return model


# ── UCI Engine ────────────────────────────────────────────────────────────────

class UCIEngine:
    """
    Implements the UCI protocol loop.

    Model loading is deferred until the first ``isready`` command so that
    ``uci`` / ``setoption`` handshakes can complete before the (potentially
    slow) checkpoint load.

    Search runs in a daemon thread so the main loop remains responsive to
    ``stop`` and ``quit`` at all times.  For bounded searches (movetime, nodes,
    wtime/btime) the thread runs a single MCTS.run() call and emits bestmove
    when done.  For ``go infinite`` the thread loops in chunks of
    _INFINITE_CHUNK simulations, exiting when the stop-event is set.
    """

    # ── UCI options (name → (type, default, min, max)) ────────────────────────
    _OPTIONS: dict[str, tuple] = {
        "NPS":           ("spin",   _DEFAULT_NPS, 1,    1_000_000),
        "CheckpointDir": ("string", "checkpoints", None, None),
        "NumBlocks":     ("spin",   20,            1,    100),
        "NumFilters":    ("spin",   256,           32,   1024),
    }

    def __init__(
        self,
        checkpoint_dir: str = "checkpoints",
        num_blocks: int = 20,
        num_filters: int = 256,
    ) -> None:
        # UCI option values (may be overridden by setoption before isready)
        self._opt_checkpoint_dir: str = checkpoint_dir
        self._opt_num_blocks:     int = num_blocks
        self._opt_num_filters:    int = num_filters
        self._opt_nps:            int = _DEFAULT_NPS

        # Lazily initialised on first isready / go
        self._device: Optional[torch.device] = None
        self._model:  Optional[ChessResNet]  = None
        self._mcts:   Optional[MCTS]         = None

        # Current board position (move_stack preserved for encode_board history)
        self._board = chess.Board()

        # Search thread bookkeeping
        self._search_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    # ── Lazy initialisation ───────────────────────────────────────────────────

    def _ensure_model(self) -> None:
        """Load model + MCTS on first call; no-op on subsequent calls."""
        if self._model is not None:
            return

        if torch.cuda.is_available():
            self._device = torch.device("cuda")
            log.info("CUDA device: %s", torch.cuda.get_device_name(0))
        else:
            self._device = torch.device("cpu")
            log.info("CUDA unavailable — falling back to CPU (search will be slow).")

        self._model = _load_model(
            self._opt_checkpoint_dir,
            self._opt_num_blocks,
            self._opt_num_filters,
            self._device,
        )
        self._mcts = MCTS(self._model, self._device)

    # ── Position parsing ──────────────────────────────────────────────────────

    def _apply_position(self, tokens: list[str]) -> None:
        """
        Parse the token tail of a UCI ``position`` command and update
        ``self._board``, preserving the full move_stack for history encoding.

        Accepted forms:
            startpos [moves m1 m2 …]
            fen <fen_string> [moves m1 m2 …]

        FEN strings may contain spaces (they always do); the parser locates the
        ``moves`` keyword to split the FEN from the move list.
        """
        if not tokens:
            return

        if tokens[0] == "startpos":
            self._board = chess.Board()
            # Move list starts after optional "moves" keyword
            move_start = 2 if len(tokens) > 1 and tokens[1] == "moves" else len(tokens)

        elif tokens[0] == "fen":
            try:
                moves_kw_idx = tokens.index("moves")
                fen_str      = " ".join(tokens[1:moves_kw_idx])
                move_start   = moves_kw_idx + 1
            except ValueError:
                # No "moves" keyword — entire tail is the FEN
                fen_str    = " ".join(tokens[1:])
                move_start = len(tokens)
            try:
                self._board = chess.Board(fen_str)
            except ValueError as exc:
                log.error("Invalid FEN '%s': %s", fen_str, exc)
                return

        else:
            log.warning("Unknown position type: '%s'", tokens[0])
            return

        # Push moves — each push_uci call extends move_stack, which is what
        # encode_board() needs to reconstruct history frames.
        for uci_move in tokens[move_start:]:
            try:
                self._board.push_uci(uci_move)
            except (ValueError, AssertionError) as exc:
                log.error("Illegal move '%s' in position command: %s", uci_move, exc)
                break

    # ── Move selection helpers ────────────────────────────────────────────────

    @staticmethod
    def _best_move_from_policy(policy: np.ndarray, board: chess.Board) -> chess.Move:
        """
        Convert a policy probability vector to the best legal chess.Move.

        Primary path:  argmax of the policy → decode → legality check.
        Fallback path: iterate legal moves, pick the one with the highest
                       policy probability.  This handles the rare case where the
                       argmax action decodes to a move that is illegal on this
                       board (e.g., a spurious queen-promotion flag on a
                       non-pawn move).
        """
        best_action = int(np.argmax(policy))
        candidate   = decode_move_perspective(best_action, board.turn)
        if candidate in board.legal_moves:
            return candidate

        # Fallback — scan legal moves for highest policy probability
        best_move: Optional[chess.Move] = None
        best_prob = -1.0
        for move in board.legal_moves:
            idx  = encode_move_perspective(move, board.turn)
            prob = float(policy[idx])
            if prob > best_prob:
                best_prob = prob
                best_move = move

        if best_move is not None:
            return best_move

        # Last resort — first legal move
        return next(iter(board.legal_moves))

    # ── Search workers ────────────────────────────────────────────────────────

    def _worker_bounded(self, board: chess.Board, num_sims: int) -> None:
        """
        Run MCTS for exactly *num_sims* simulations, then emit bestmove.
        Called in a daemon thread for bounded searches (movetime / nodes).
        """
        try:
            policy    = self._mcts.run(board, num_sims, temperature=0.0)
            best_move = self._best_move_from_policy(policy, board)
        except Exception:
            log.exception("Error in bounded search worker")
            legal     = list(board.legal_moves)
            best_move = legal[0] if legal else chess.Move.null()

        _send(f"bestmove {best_move.uci()}")

    def _worker_infinite(self, board: chess.Board) -> None:
        """
        Loop MCTS in _INFINITE_CHUNK-simulation batches until stop_event fires.

        MCTS.run() is synchronous and not interruptible mid-call, so the
        stop_event is only checked between batches.  Each batch starts a fresh
        MCTS tree (no cross-batch tree reuse), keeping per-response quality
        constant at _INFINITE_CHUNK simulations.
        """
        policy: Optional[np.ndarray] = None
        try:
            while not self._stop_event.is_set():
                policy = self._mcts.run(board, _INFINITE_CHUNK, temperature=0.0)
        except Exception:
            log.exception("Error in infinite search worker")

        if policy is not None:
            try:
                best_move = self._best_move_from_policy(policy, board)
            except Exception:
                log.exception("Error selecting best move after infinite search")
                legal     = list(board.legal_moves)
                best_move = legal[0] if legal else chess.Move.null()
        else:
            # stop_event was set before a single iteration completed
            legal     = list(board.legal_moves)
            best_move = legal[0] if legal else chess.Move.null()

        _send(f"bestmove {best_move.uci()}")

    def _start_search(self, num_sims: Optional[int]) -> None:
        """
        Abort any running search, then launch a new search thread.

        Parameters
        ----------
        num_sims
            Number of simulations for bounded mode, or ``None`` for infinite.
        """
        # Abort an existing search before starting a new one
        if self._search_thread is not None and self._search_thread.is_alive():
            self._stop_event.set()
            self._search_thread.join(timeout=5.0)

        self._ensure_model()
        self._stop_event.clear()

        # Snapshot board with full move_stack (stack=True is the default, but
        # stated explicitly for clarity — encode_board() depends on it).
        board_snap = self._board.copy(stack=True)

        if num_sims is None:
            target, args = self._worker_infinite, (board_snap,)
        else:
            target, args = self._worker_bounded, (board_snap, num_sims)

        self._search_thread = threading.Thread(
            target=target, args=args, daemon=True, name="search"
        )
        self._search_thread.start()

    # ── Time management ───────────────────────────────────────────────────────

    def _sims_from_ms(self, ms: int) -> int:
        """Convert a time budget in milliseconds to a simulation count."""
        return max(1, round(self._opt_nps * ms / 1000))

    def _movetime_from_clock(
        self,
        wtime: Optional[int],
        btime: Optional[int],
        winc: int,
        binc: int,
    ) -> int:
        """
        Simple time-management heuristic.

        Formula: movetime = remaining / 40 + increment
        Clamped so we never use more than (remaining − 50 ms).
        """
        if self._board.turn == chess.WHITE:
            remaining = wtime if wtime is not None else 60_000
            increment = winc
        else:
            remaining = btime if btime is not None else 60_000
            increment = binc

        movetime = remaining // 40 + increment
        return max(50, min(movetime, remaining - 50))

    # ── Command handlers ──────────────────────────────────────────────────────

    def _handle_uci(self) -> None:
        _send(f"id name {ENGINE_NAME}")
        _send(f"id author {ENGINE_AUTHOR}")
        for name, spec in self._OPTIONS.items():
            opt_type, default, min_val, max_val = spec
            if opt_type == "spin":
                _send(
                    f"option name {name} type spin"
                    f" default {default} min {min_val} max {max_val}"
                )
            elif opt_type == "string":
                _send(f"option name {name} type string default {default}")
        _send("uciok")

    def _handle_setoption(self, tokens: list[str]) -> None:
        """
        Parse: setoption name <Name> value <Value>

        Option names and values may contain spaces, so we locate the "name" and
        "value" keywords by position rather than by index.
        """
        try:
            name_pos  = tokens.index("name")
            value_pos = tokens.index("value")
        except ValueError:
            log.warning("Malformed setoption — missing 'name' or 'value': %s", tokens)
            return

        opt_name  = " ".join(tokens[name_pos + 1 : value_pos])
        opt_value = " ".join(tokens[value_pos + 1 :])

        if opt_name == "NPS":
            self._opt_nps = int(opt_value)
            log.info("NPS set to %d", self._opt_nps)
        elif opt_name == "CheckpointDir":
            self._opt_checkpoint_dir = opt_value
            # Invalidate cached model so it reloads from the new directory
            self._model = None
            self._mcts  = None
            log.info("CheckpointDir set to '%s'", opt_value)
        elif opt_name == "NumBlocks":
            self._opt_num_blocks = int(opt_value)
            self._model = None
            self._mcts  = None
        elif opt_name == "NumFilters":
            self._opt_num_filters = int(opt_value)
            self._model = None
            self._mcts  = None
        else:
            log.warning("Unknown option: '%s'", opt_name)

    def _handle_go(self, tokens: list[str]) -> None:
        """
        Parse a ``go`` command and start the search thread.

        Supported sub-commands (others silently ignored):
            movetime <ms>
            nodes <N>
            wtime <ms> btime <ms> [winc <ms>] [binc <ms>]
            infinite
        """
        # Parse integer-valued sub-commands into a dict
        int_keys = {"movetime", "nodes", "wtime", "btime", "winc", "binc",
                    "movestogo", "depth"}
        args: dict[str, int] = {}
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok in int_keys and i + 1 < len(tokens):
                try:
                    args[tok] = int(tokens[i + 1])
                except ValueError:
                    pass
                i += 2
            else:
                i += 1

        # Refuse to search a finished game
        if self._board.is_game_over():
            _send("bestmove (none)")
            return

        # Determine simulation count
        if "movetime" in args:
            num_sims: Optional[int] = self._sims_from_ms(args["movetime"])

        elif "nodes" in args:
            num_sims = max(1, args["nodes"])

        elif "wtime" in args or "btime" in args:
            movetime = self._movetime_from_clock(
                args.get("wtime"),
                args.get("btime"),
                args.get("winc", 0),
                args.get("binc", 0),
            )
            num_sims = self._sims_from_ms(movetime)

        elif "infinite" in tokens:
            num_sims = None  # infinite mode — loop until stop

        else:
            # No recognised sub-command: fall back to a quick fixed search
            num_sims = 400

        log.info(
            "go: sims=%s  (NPS=%d)  turn=%s",
            "∞" if num_sims is None else num_sims,
            self._opt_nps,
            "white" if self._board.turn == chess.WHITE else "black",
        )
        self._start_search(num_sims)

    def _handle_stop(self) -> None:
        """Signal the search thread to stop and wait for bestmove to be emitted."""
        self._stop_event.set()
        if self._search_thread is not None and self._search_thread.is_alive():
            # Wait long enough for the in-flight MCTS batch to finish.
            # The batch size is bounded by _INFINITE_CHUNK, so this is at most
            # a few seconds even on CPU.
            self._search_thread.join(timeout=30.0)

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        """
        Read UCI commands from stdin and dispatch them.
        Exits on ``quit`` or EOF.
        """
        log.info("%s engine started.", ENGINE_NAME)

        for raw in sys.stdin:
            line   = raw.rstrip("\r\n")
            tokens = line.split()
            if not tokens:
                continue

            cmd, rest = tokens[0], tokens[1:]
            log.debug("← %s", line)

            if cmd == "uci":
                self._handle_uci()

            elif cmd == "debug":
                level = logging.DEBUG if rest and rest[0] == "on" else logging.INFO
                logging.getLogger().setLevel(level)

            elif cmd == "isready":
                self._ensure_model()
                _send("readyok")

            elif cmd == "setoption":
                self._handle_setoption(rest)

            elif cmd == "ucinewgame":
                # Reset to starting position; does not reload the model
                self._board = chess.Board()

            elif cmd == "position":
                self._apply_position(rest)

            elif cmd == "go":
                self._handle_go(rest)

            elif cmd == "stop":
                self._handle_stop()

            elif cmd == "quit":
                self._handle_stop()
                log.info("Quit received — exiting.")
                sys.exit(0)

            else:
                log.debug("Ignoring unknown command: '%s'", cmd)


# ── Entry point ───────────────────────────────────────────────────────────────

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="AlphaZero-style chess engine with UCI interface"
    )
    p.add_argument(
        "--checkpoint_dir", default="checkpoints",
        help="Directory that contains step_*.pt checkpoints (default: checkpoints/)",
    )
    p.add_argument(
        "--num_blocks", type=int, default=20,
        help="Residual blocks in the network tower — must match checkpoint (default: 20)",
    )
    p.add_argument(
        "--num_filters", type=int, default=256,
        help="Conv filter width — must match checkpoint (default: 256)",
    )
    return p


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    UCIEngine(
        checkpoint_dir=args.checkpoint_dir,
        num_blocks=args.num_blocks,
        num_filters=args.num_filters,
    ).run()
