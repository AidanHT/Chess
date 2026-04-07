#!/usr/bin/env python3
"""
smoke_test.py
-------------
End-to-end integration test suite for the AlphaZero-style chess engine.

Three test suites prove the pipeline is structurally sound before committing
to a multi-GPU training run:

  Test1SingleBatchOverfit  -- Loss function + optimiser can actually learn.
  Test2MCTSDryRun          -- MCTS produces legal moves without memory leaks.
  Test3UCISimulation       -- UCI protocol handler speaks correct UCI.

Run:
    python smoke_test.py -v
    python -m pytest smoke_test.py -v
"""

from __future__ import annotations

import io
import re
import sys
import tempfile
import time
import unittest
from typing import Optional

import chess
import numpy as np
import torch
from torch.optim import AdamW

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from chess_engine import (
    ACTION_SPACE_SIZE,
    BOARD_SIZE,
    TOTAL_PLANES,
    ChessResNet,
    MCTS,
    UCIEngine,
    decode_move_perspective,
    encode_board,
    encode_move_perspective,
)
from chess_engine.training.train import compute_loss


# --- Shared constants ---------------------------------------------------------

# Scaled-down model: fast enough for smoke testing, still exercises all code paths.
_NUM_BLOCKS  = 2
_NUM_FILTERS = 64

# Overfit test hyper-parameters
_OVERFIT_LR      = 0.01   # lr=0.05 causes dying ReLU in the value head; 0.01 is stable
_OVERFIT_EPOCHS  = 100
_LOSS_THRESHOLD  = 0.05   # Both policy_loss AND value_loss must fall below this

# MCTS dry-run parameters
_MCTS_SIMS    = 50
_VRAM_LEAK_MB = 10.0   # Max acceptable VRAM growth (proves no graph leaks)


# --- Terminal formatting helpers ----------------------------------------------

def _banner(msg: str) -> None:
    sep = "=" * 68
    print("\n" + sep)
    print("  " + msg)
    print(sep)


def _info(msg: str) -> None:
    print("  [INFO] " + msg)


def _ok(msg: str) -> None:
    print("  [PASS] " + msg)


# =============================================================================
# TEST 1 -- Single-Batch Overfit
# =============================================================================

class Test1SingleBatchOverfit(unittest.TestCase):
    """
    Trains a tiny ChessResNet on 4 fixed positions for 100 epochs using AdamW
    and asserts that both policy_loss and value_loss drop below 0.05.

    Why this is the definitive smoke test
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    A neural network with ~800 K parameters memorising 4 examples is trivial.
    If it fails, the gradient graph is broken.  The two most likely culprits:

      1. Policy head: a .detach() call or a non-differentiable op inside the
         forward pass severs the gradient path from the cross-entropy loss back
         to the Conv/BN weights.

      2. Value head: extreme tanh saturation (targets +/-1 push pre-activations
         toward +/-inf, making tanh'(x) -> 0) or a detached path through the
         FC layers.

    In both cases the RuntimeError message names the suspected root cause.
    """

    # --- Batch construction ---------------------------------------------------

    @staticmethod
    def _make_boards_and_moves() -> tuple:
        """
        Return (boards, target_moves) with 4 truly distinct positions and
        4 moves whose perspective-encoded action indices are all different.

        IMPORTANT: Using the same action index for multiple positions creates
        a degenerate policy task -- the backbone need not differentiate between
        those positions, so the value head cannot fit their different targets.

        Verified action indices (perspective-encoded):
          e2e4  (white, pos 0): from_sq=12, N dir, dist 2  -> action 877
          c7c5  (black, pos 1): mirrors to c2c4 in white coords -> action 731
          g1f3  (white, pos 2): knight (2,-1)               -> action 495
          e7e6  (black, pos 3): mirrors to e2e3 in white coords -> action 876
        All four are distinct: {877, 731, 495, 876}.
        """
        # Position 0: starting board, white to move
        b0 = chess.Board()
        m0 = chess.Move.from_uci("e2e4")    # e4 pawn push

        # Position 1: after 1.e4, black to move (Sicilian response)
        b1 = chess.Board()
        b1.push_uci("e2e4")
        m1 = chess.Move.from_uci("c7c5")    # c5 (black perspective mirrors to c2c4)

        # Position 2: after 1.e4 c5, white to move
        b2 = chess.Board()
        b2.push_uci("e2e4")
        b2.push_uci("c7c5")
        m2 = chess.Move.from_uci("g1f3")    # Nf3 knight development

        # Position 3: after 1.d4, black to move (French preamble)
        b3 = chess.Board()
        b3.push_uci("d2d4")
        m3 = chess.Move.from_uci("e7e6")    # e6 (black perspective mirrors to e2e3)

        boards = [b0, b1, b2, b3]
        moves  = [m0, m1, m2, m3]

        # Verify all moves are legal and all action indices are distinct.
        action_indices = []
        for i, (b, m) in enumerate(zip(boards, moves)):
            assert m in b.legal_moves, \
                "Test setup error: %s is not legal in position %d" % (m.uci(), i)
            action_indices.append(encode_move_perspective(m, b.turn))
        assert len(set(action_indices)) == len(action_indices), \
            "Test setup error: duplicate action indices %s" % action_indices

        return boards, moves

    @staticmethod
    def _encode_boards(boards: list) -> torch.Tensor:
        """Return (B, 119, 8, 8) float32 tensor on CPU."""
        B   = len(boards)
        buf = np.empty((B, TOTAL_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        for i, b in enumerate(boards):
            # encode_board returns (8, 8, 119); transpose to channels-first layout.
            buf[i] = encode_board(b).transpose(2, 0, 1)
        return torch.from_numpy(buf)

    @staticmethod
    def _make_policy_targets(boards: list, target_moves: list) -> torch.Tensor:
        """One-hot over the specified target move for each position. Shape (B, 4672)."""
        B  = len(boards)
        pt = np.zeros((B, ACTION_SPACE_SIZE), dtype=np.float32)
        for i, (b, move) in enumerate(zip(boards, target_moves)):
            action_idx = encode_move_perspective(move, b.turn)
            pt[i, action_idx] = 1.0
            _info(
                "  Position %d (%s to move): target = %s  (action index %d)"
                % (i, "white" if b.turn == chess.WHITE else "black",
                   move.uci(), action_idx)
            )
        return torch.from_numpy(pt)

    # --- Test body ------------------------------------------------------------

    def test_policy_and_value_converge(self) -> None:
        _banner(
            "TEST 1 -- Single-Batch Overfit  "
            "(blocks=%d, filters=%d, epochs=%d, lr=%.3f)"
            % (_NUM_BLOCKS, _NUM_FILTERS, _OVERFIT_EPOCHS, _OVERFIT_LR)
        )
        _info(
            "Goal: memorise 4 positions -- both losses must fall below %.2f."
            % _LOSS_THRESHOLD
        )

        device = torch.device("cpu")
        _info("Device: %s" % device)

        # --- Build batch ------------------------------------------------------
        boards, target_moves = self._make_boards_and_moves()
        board_t  = self._encode_boards(boards).to(device)
        policy_t = self._make_policy_targets(boards, target_moves).to(device)
        # Value targets span positive, negative, and zero.  Avoid exactly +-1
        # since tanh(x) = +-1 only at x -> +-inf, causing vanishing gradients.
        value_t  = torch.tensor([1.0, -1.0, 0.0, 1.0], dtype=torch.float32,
                                 device=device)

        _info("Board tensor  : %s" % str(tuple(board_t.shape)))
        _info("Policy target : %s  (one-hot, 1 hot per row)" % str(tuple(policy_t.shape)))
        _info("Value targets : %s" % str(value_t.tolist()))

        # --- Instantiate model ------------------------------------------------
        model = ChessResNet(num_blocks=_NUM_BLOCKS, num_filters=_NUM_FILTERS).to(device)
        model.train()
        n_params = sum(p.numel() for p in model.parameters())
        _info("Model parameter count: %d" % n_params)

        optimizer = AdamW(model.parameters(), lr=_OVERFIT_LR, weight_decay=1e-4)

        # --- Training loop ----------------------------------------------------
        print()
        final_p_loss: float = float("inf")
        final_v_loss: float = float("inf")

        for epoch in range(_OVERFIT_EPOCHS):
            optimizer.zero_grad(set_to_none=True)
            policy_logits, value_pred = model(board_t)
            total_loss, p_loss, v_loss = compute_loss(
                policy_logits, value_pred, policy_t, value_t
            )
            total_loss.backward()
            optimizer.step()

            final_p_loss = p_loss.item()
            final_v_loss = v_loss.item()

            # Print progress every 10 epochs (and the very first step so the
            # user can verify that training started with a reasonable initial loss).
            if epoch == 0 or (epoch + 1) % 10 == 0:
                _info(
                    "  Epoch %3d/%d | policy_loss = %.6f  value_loss = %.6f"
                    % (epoch + 1, _OVERFIT_EPOCHS, final_p_loss, final_v_loss)
                )

        print()
        _info("Final policy_loss : %.6f  (threshold < %.2f)"
              % (final_p_loss, _LOSS_THRESHOLD))
        _info("Final value_loss  : %.6f  (threshold < %.2f)"
              % (final_v_loss, _LOSS_THRESHOLD))

        # --- Assertions -------------------------------------------------------
        if final_p_loss >= _LOSS_THRESHOLD:
            raise RuntimeError(
                "OVERFIT TEST FAILED -- policy_loss = %.6f did not converge "
                "below %.2f in %d epochs.\n\n"
                "Likely causes:\n"
                "  1. Gradients are detached somewhere in the policy head "
                "     (e.g. a .detach() call, or np.argmax used instead of "
                "     torch.argmax inside the forward pass).\n"
                "  2. F.cross_entropy receives already-softmaxed logits "
                "     (double-softmax produces near-uniform probabilities that "
                "     can never collapse to one-hot, keeping loss near "
                "     log(4672) ~= 8.45).\n"
                "  3. AdamW learning rate is too small for the number of epochs."
                % (final_p_loss, _LOSS_THRESHOLD, _OVERFIT_EPOCHS)
            )

        if final_v_loss >= _LOSS_THRESHOLD:
            raise RuntimeError(
                "OVERFIT TEST FAILED -- value_loss = %.6f did not converge "
                "below %.2f in %d epochs.\n\n"
                "Likely causes:\n"
                "  1. Gradients detached in the value head (check value_fc1 / "
                "     value_fc2 layers -- they must stay in the computation graph).\n"
                "  2. Tanh saturation: if the value head pre-activations grow "
                "     very large, tanh'(x) -> 0 and gradient ceases to flow. "
                "     This typically indicates incorrect weight initialisation "
                "     or an extremely large learning rate that overshoots the minimum."
                % (final_v_loss, _LOSS_THRESHOLD, _OVERFIT_EPOCHS)
            )

        _ok(
            "Both losses converged: policy = %.6f,  value = %.6f  (both < %.2f)."
            % (final_p_loss, final_v_loss, _LOSS_THRESHOLD)
        )


# =============================================================================
# TEST 2 -- MCTS Dry Run
# =============================================================================

class Test2MCTSDryRun(unittest.TestCase):
    """
    Passes a randomly-initialised small model into MCTS and runs 50 simulations
    from the starting chess position.

    Three assertions are checked:
      1. Legal move   -- the argmax of the policy is a legal move, proving that
                         encode_move_perspective() / decode_move_perspective()
                         round-trip correctly.
      2. Tree branching -- more than one action is assigned non-zero probability,
                         proving PUCT selection is not collapsing into an infinite
                         loop on a single path.
      3. VRAM stability -- allocated GPU memory does not grow materially during
                         the search.  A large leak would indicate that
                         torch.Tensor objects (with live grad_fn chains) are
                         stored inside MCTSNode attributes, defeating
                         @torch.no_grad().
    """

    def test_legal_move_and_stable_vram(self) -> None:
        _banner(
            "TEST 2 -- MCTS Dry Run  "
            "(%d simulations from starting position, blocks=%d, filters=%d)"
            % (_MCTS_SIMS, _NUM_BLOCKS, _NUM_FILTERS)
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _info("Device: %s" % device)
        if torch.cuda.is_available():
            _info("GPU: %s" % torch.cuda.get_device_name(0))
        else:
            _info("CUDA not available -- VRAM stability check will be skipped.")

        # --- Model + MCTS setup -----------------------------------------------
        model = ChessResNet(num_blocks=_NUM_BLOCKS, num_filters=_NUM_FILTERS).to(device)
        model.eval()
        mcts  = MCTS(model, device, batch_size=8)
        board = chess.Board()

        legal_count = sum(1 for _ in board.legal_moves)
        _info("Starting position -- %d legal moves available." % legal_count)
        _info(
            "MCTS batch_size = 8  ->  %d simulations in "
            "%d forward-pass batches."
            % (_MCTS_SIMS, (_MCTS_SIMS + 7) // 8)
        )

        # --- Snapshot VRAM before search --------------------------------------
        vram_before_mb: Optional[float] = None
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            vram_before_mb = torch.cuda.memory_allocated() / (1024 ** 2)
            _info("VRAM allocated before search: %.3f MB" % vram_before_mb)

        # --- Run MCTS ---------------------------------------------------------
        _info("Running %d simulations ..." % _MCTS_SIMS)
        t0 = time.perf_counter()
        policy_vector: np.ndarray = mcts.run(
            board, num_simulations=_MCTS_SIMS, temperature=1.0
        )
        elapsed = time.perf_counter() - t0
        _info(
            "Search completed in %.3f s  (%.1f sims/s)"
            % (elapsed, _MCTS_SIMS / elapsed)
        )

        # --- VRAM stability check ---------------------------------------------
        if torch.cuda.is_available() and vram_before_mb is not None:
            torch.cuda.synchronize()
            vram_after_mb = torch.cuda.memory_allocated() / (1024 ** 2)
            vram_delta    = vram_after_mb - vram_before_mb
            _info(
                "VRAM allocated after search:  %.3f MB  (delta = %+.3f MB)"
                % (vram_after_mb, vram_delta)
            )
            self.assertLess(
                vram_delta,
                _VRAM_LEAK_MB,
                "VRAM grew by %.3f MB during MCTS (cap = %.1f MB). "
                "Computational graphs are accumulating inside the search tree.  "
                "Verify that @torch.no_grad() decorates MCTS._infer() so that "
                "policy_logits_t and values_t are plain tensors with no grad_fn."
                % (vram_delta, _VRAM_LEAK_MB),
            )
            _ok(
                "VRAM delta (%+.3f MB) is within the %.1f MB cap -- "
                "no graph leaks detected." % (vram_delta, _VRAM_LEAK_MB)
            )

        # --- Policy vector sanity --------------------------------------------
        _info("Policy vector shape : %s" % str(policy_vector.shape))

        policy_sum = float(policy_vector.sum())
        _info("Policy vector sum   : %.6f  (expected ~= 1.0)" % policy_sum)
        self.assertAlmostEqual(
            policy_sum,
            1.0,
            places=3,
            msg=(
                "Policy vector sums to %.6f instead of 1.0.  "
                "_policy_vector() has a normalisation error: either the "
                "visit-count exponentiation produces NaN (temperature too large) "
                "or the final sum used for division is zero (no simulations "
                "completed)." % policy_sum
            ),
        )

        # --- Tree branching check --------------------------------------------
        # With 50 simulations in a 20-move starting position, PUCT should
        # explore many branches.  Fewer than 2 unique actions indicates that
        # selection is stuck on the same child every time.
        num_explored = int((policy_vector > 0.0).sum())
        _info("Non-zero policy entries (distinct explored actions): %d" % num_explored)
        self.assertGreater(
            num_explored,
            1,
            "Only %d action(s) received non-zero probability after %d simulations.  "
            "PUCT selection may be broken: confirm that virtual-loss is applied "
            "correctly so sibling simulations explore different branches."
            % (num_explored, _MCTS_SIMS),
        )

        # --- Strict legality check -------------------------------------------
        best_action = int(np.argmax(policy_vector))
        best_move   = decode_move_perspective(best_action, board.turn)
        _info(
            "Argmax action index : %d  ->  decoded move : %s"
            % (best_action, best_move.uci())
        )

        self.assertIn(
            best_move,
            board.legal_moves,
            "MCTS recommended move '%s' which is NOT legal in the starting "
            "position.  This indicates a bug in encode_move_perspective() or "
            "decode_move_perspective(): the perspective-flip for Black's moves "
            "(chess.square_mirror) may be applied when it should not be, or the "
            "action-space look-up table was built with wrong direction vectors."
            % best_move.uci(),
        )

        _ok("MCTS returned strictly legal move: %s." % best_move.uci())
        _ok(
            "Tree expanded to %d distinct actions in %d simulations "
            "-- no infinite loops detected." % (num_explored, _MCTS_SIMS)
        )


# =============================================================================
# TEST 3 -- UCI I/O Simulation
# =============================================================================

class Test3UCISimulation(unittest.TestCase):
    """
    Injects UCI commands directly into UCIEngine.run() via io.StringIO so
    there are no subprocess, pipe-buffering, or platform-specific gotchas.

    Protocol sequence tested
    ~~~~~~~~~~~~~~~~~~~~~~~~
    ENGINE <- uci
    ENGINE -> id name ..., id author ..., option ..., uciok
    ENGINE <- setoption name NumBlocks value 2
    ENGINE <- setoption name NumFilters value 64
    ENGINE <- isready
    ENGINE -> readyok
    ENGINE <- position startpos
    ENGINE <- go nodes 10
    ENGINE -> bestmove <uci_move>
    ENGINE <- quit   (join search thread first, then sys.exit)

    How the in-process redirect works
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    sys.stdin  is replaced with io.StringIO(commands) before engine.run().
    sys.stdout is replaced with io.StringIO() to capture responses.
    engine.run() calls _send() which always writes to the current sys.stdout,
    so all UCI responses land in our capture buffer.

    The "quit" command triggers UCIEngine._handle_stop(), which *joins* the
    search thread before calling sys.exit(0).  This guarantees that
    "bestmove ..." is written to the capture buffer before SystemExit is
    raised, making the timing deterministic with no races.
    """

    def _build_commands(self) -> str:
        """Return the newline-terminated command string sent to the engine."""
        lines = [
            "uci",
            # Override to the tiny model so isready is fast on CPU.
            "setoption name NumBlocks value %d" % _NUM_BLOCKS,
            "setoption name NumFilters value %d" % _NUM_FILTERS,
            "isready",
            "position startpos",
            "go nodes 10",
            # "quit" triggers _handle_stop() which joins the search thread,
            # ensuring bestmove is written before sys.exit(0) is raised.
            "quit",
        ]
        return "\n".join(lines) + "\n"

    def test_readyok_and_bestmove(self) -> None:
        _banner("TEST 3 -- UCI I/O Simulation (in-process io.StringIO)")

        commands = self._build_commands()

        _info("Commands injected into engine stdin:")
        for line in commands.strip().splitlines():
            print("    -> " + line)
        print()

        # tempfile.TemporaryDirectory is used as the checkpoint directory so
        # Path.glob("step_*.pt") always returns an empty list -> random weights.
        with tempfile.TemporaryDirectory() as tmp_ckpt_dir:
            engine = UCIEngine(
                checkpoint_dir=tmp_ckpt_dir,
                num_blocks=_NUM_BLOCKS,
                num_filters=_NUM_FILTERS,
            )

            fake_in  = io.StringIO(commands)
            fake_out = io.StringIO()

            old_stdin  = sys.stdin
            old_stdout = sys.stdout
            sys.stdin  = fake_in
            sys.stdout = fake_out

            try:
                engine.run()
            except SystemExit:
                # "quit" causes sys.exit(0) -- that is the expected success path.
                # By this point _handle_stop() has already joined the search
                # thread, so "bestmove ..." is already written to fake_out.
                pass
            except Exception as exc:
                # Restore I/O before re-raising so subsequent tests are not broken.
                sys.stdin  = old_stdin
                sys.stdout = old_stdout
                self.fail(
                    "engine.run() raised an unexpected exception: "
                    "%s: %s\n"
                    "Check stderr above for the full traceback."
                    % (type(exc).__name__, exc)
                )
            finally:
                sys.stdin  = old_stdin
                sys.stdout = old_stdout

        output = fake_out.getvalue()

        _info("Raw engine output captured from stdout:")
        for line in output.strip().splitlines():
            print("    <- " + line)
        print()

        # --- Assertion 1: readyok --------------------------------------------
        self.assertIn(
            "readyok",
            output,
            "Engine did not emit 'readyok' after 'isready'.\n"
            "Possible causes:\n"
            "  * _ensure_model() raised an exception (model init failed).\n"
            "  * The isready command was not recognised by the command dispatcher.\n"
            "Check stderr for Python tracebacks.",
        )
        _ok("Received 'readyok'  -- model loaded successfully.")

        # --- Assertion 2: bestmove present -----------------------------------
        bestmove_match = re.search(
            r"bestmove\s+([a-h][1-8][a-h][1-8][qrbn]?)", output
        )
        self.assertIsNotNone(
            bestmove_match,
            "Engine did not emit 'bestmove <move>' after 'go nodes 10'.\n"
            "Possible causes:\n"
            "  * The search thread crashed before calling _send('bestmove ...').\n"
            "  * _handle_stop() did not join the search thread before sys.exit.\n"
            "  * MCTS.run() raised an exception inside _worker_bounded().\n"
            "Check stderr for Python tracebacks from the daemon thread.",
        )

        move_str = bestmove_match.group(1)  # type: ignore[union-attr]
        _info("Extracted bestmove: %s" % move_str)

        # --- Assertion 3: bestmove is strictly legal -------------------------
        starting_board = chess.Board()
        try:
            candidate = chess.Move.from_uci(move_str)
        except ValueError:
            self.fail(
                "Engine emitted malformed move string: '%s'.  "
                "UCIEngine._best_move_from_policy() returned a move that "
                "chess.Move.from_uci() cannot parse." % move_str
            )

        self.assertIn(
            candidate,
            starting_board.legal_moves,
            "Engine returned '%s' which is NOT a legal move from the starting "
            "position.\n"
            "Possible causes:\n"
            "  * decode_move_perspective() decoded to the wrong square (off-by-one "
            "    in the direction / distance tables).\n"
            "  * _best_move_from_policy() returned None and fell through to the "
            "    chess.Move.null() fallback." % move_str,
        )

        _ok("bestmove %s is strictly legal in the starting position." % move_str)
        _ok("Full UCI handshake (uci -> isready -> position -> go -> bestmove) passed.")


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    print("AlphaZero Chess Engine -- Smoke Test Suite")
    print("-" * 45)
    print("Tests run in order:  overfit  ->  MCTS  ->  UCI")
    print("Run with:  python smoke_test.py -v")
    print("           python -m pytest smoke_test.py -v")
    print()
    unittest.main(verbosity=2)
