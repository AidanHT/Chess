"""
Batched Monte Carlo Tree Search for the AlphaZero-style chess engine.

Key design decisions
────────────────────
Virtual loss
    Applied to every node in the traversal path during selection (N += 1,
    W -= 1).  Backpropagation corrects W afterwards.  Because N is incremented
    as part of the virtual-loss step, no separate N update is needed in
    backprop — only W (and hence Q) requires correction.  Net effect per
    simulation: N += 1, W += value (sign-adjusted for perspective).

Batched GPU inference
    Within each batch, unique unvisited leaves are stacked into a single
    (B, 119, 8, 8) tensor and forwarded through ChessResNet in one call.
    Policy logits and values are then routed back to their respective paths.
    Duplicate leaves in the same batch are backpropagated with the single
    evaluated value; the double virtual-loss increment on duplicate leaves
    self-corrects through two matching backpropagation passes.

Perspective convention for Q
    Q(s, a) is stored from the perspective of the player who CHOSE to take
    action a — i.e., the player to move at the *parent* of the resulting node.
    This means all players maximise Q during PUCT selection (no per-node sign
    flip needed in the formula).  During backpropagation the scalar is negated
    at each ply because adjacent nodes belong to opposite colours.

Legal-move masking
    _expand() collects raw logits only at legal action indices, shifts them by
    −max for numerical stability, then applies softmax over that subset.
    Illegal actions receive prior probability 0 and are never given children.
"""

from __future__ import annotations

import math
from typing import Optional

import chess
import numpy as np
import torch

from ..core import ACTION_SPACE_SIZE, BOARD_SIZE, TOTAL_PLANES, decode_move_perspective, encode_board, encode_move_perspective
from ..models import ChessResNet

# ── Module constants ──────────────────────────────────────────────────────────

_C_PUCT: float = 1.25
_DIR_ALPHA: float = 0.3
_DIR_EPS: float = 0.25
_BATCH_SIZE: int = 8

# Virtual-loss magnitude applied to W during selection.
# Using 1.0 means each in-flight simulation temporarily counts as a full loss,
# strongly discouraging sibling simulations from repeating the same path.
_VIRTUAL_LOSS: float = 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# MCTSNode
# ═══════════════════════════════════════════════════════════════════════════════

class MCTSNode:
    """
    A single node in the MCTS search tree.

    Attributes
    ----------
    N : int
        Visit count.  Incremented once per simulation during selection (via
        virtual loss); the resulting count is the permanent visit count because
        no separate backprop increment is performed.
    W : float
        Accumulated action value from the perspective of the player who CHOSE to
        enter this node (i.e., the player to move at the parent).  Temporarily
        depressed by virtual loss and corrected during backpropagation.
    Q : float
        Mean action value  W / N.  Kept in sync after every change to W or N.
    P : float
        Prior probability assigned by the parent's masked, re-normalised policy
        output.
    children : dict[int, MCTSNode]
        Maps action index in [0, ACTION_SPACE_SIZE) to child node.
        Empty until the node is expanded by the search.
    """

    __slots__ = ("N", "W", "Q", "P", "children")

    def __init__(self, prior: float) -> None:
        self.N: int = 0
        self.W: float = 0.0
        self.Q: float = 0.0
        self.P: float = prior
        self.children: dict[int, MCTSNode] = {}

    def is_expanded(self) -> bool:
        """True once ``_expand`` has created children for this node."""
        return bool(self.children)

    def select_child(self, c_puct: float) -> tuple[int, "MCTSNode"]:
        """
        Return ``(action, child)`` maximising the PUCT score.

        .. math::

            \\mathrm{PUCT}(s, a) =
            Q(s, a) + C \\cdot P(s, a) \\cdot
            \\frac{\\sqrt{N(s)}}{1 + N(s, a)}

        Parameters
        ----------
        c_puct : float
            Exploration constant *C* in the PUCT formula.
        """
        sqrt_n = math.sqrt(self.N)
        best_score = -math.inf
        best_action = -1
        best_child: Optional[MCTSNode] = None

        for action, child in self.children.items():
            score = child.Q + c_puct * child.P * sqrt_n / (1.0 + child.N)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        assert best_child is not None, "select_child called on unexpanded node"
        return best_action, best_child


# ═══════════════════════════════════════════════════════════════════════════════
# MCTS
# ═══════════════════════════════════════════════════════════════════════════════

class MCTS:
    """
    Batched Monte Carlo Tree Search backed by :class:`~model.ChessResNet`.

    Parameters
    ----------
    model : ChessResNet
        Shared policy-value network (placed in ``eval`` mode during search).
    device : torch.device
        Device on which to run GPU inference.
    c_puct : float
        Exploration constant in the PUCT formula (AlphaZero default ≈ 1.25).
    dirichlet_alpha : float
        Dirichlet noise concentration α applied to root priors.
    dirichlet_eps : float
        Noise mixing weight ε:
        ``P′(a) = (1 − ε) P(a) + ε η(a)``,  ``η ~ Dir(α)``.
    batch_size : int
        Maximum number of leaf positions packed into one forward pass.
    """

    def __init__(
        self,
        model: ChessResNet,
        device: torch.device,
        *,
        c_puct: float = _C_PUCT,
        dirichlet_alpha: float = _DIR_ALPHA,
        dirichlet_eps: float = _DIR_EPS,
        batch_size: int = _BATCH_SIZE,
    ) -> None:
        self.model = model
        self.device = device
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps
        self.batch_size = batch_size

    # ──────────────────────────────────────────────────────────────────────────
    # Public interface
    # ──────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def run(
        self,
        board: chess.Board,
        num_simulations: int,
        temperature: float = 1.0,
    ) -> np.ndarray:
        # Set eval mode once per search, not on every _infer() call.
        self.model.eval()
        """
        Run batched MCTS from *board* and return a visit-count policy vector.

        Algorithm outline
        ~~~~~~~~~~~~~~~~~
        1. Expand root with one network call; apply Dirichlet noise to priors.
        2. In batches of ``batch_size``:
             a. **Selection** — run ``batch_size`` tree traversals in sequence,
                applying virtual loss at every node so sibling simulations
                diverge naturally.
             b. **Inference** — collect unique unvisited leaves, encode them
                into a single tensor, and run one forward pass.
             c. **Expansion** — create children for each evaluated leaf using
                the masked, re-normalised policy logits.
             d. **Backpropagation** — propagate the scalar value from each leaf
                to the root, alternating sign at every ply.
        3. Convert root's child visit counts to a probability distribution.

        Parameters
        ----------
        board : chess.Board
            Root position.  Must not already be in a terminal state.
        num_simulations : int
            Total number of leaf evaluations to perform.
        temperature : float
            Policy temperature τ: ``π(a) ∝ N(root, a)^(1/τ)``.
            Pass ``0.0`` for deterministic (greedy) play.

        Returns
        -------
        numpy.ndarray
            Shape ``(4672,)`` float32.  Probability distribution over actions
            proportional to visit counts raised to power ``1/τ``.
        """
        # ── Initialise and expand root ────────────────────────────────────────
        root = MCTSNode(prior=0.0)
        root_logits, _ = self._infer([board])
        self._expand(root, board, root_logits[0])
        self._apply_dirichlet_noise(root)

        # ── Simulation loop (batched) ─────────────────────────────────────────
        done = 0
        while done < num_simulations:
            batch = min(self.batch_size, num_simulations - done)

            # ── Phase 1: Selection ─────────────────────────────────────────────
            # Each call to _select applies virtual loss along the path, so
            # subsequent simulations in the same batch explore different branches.
            trajectories: list[tuple[list[MCTSNode], chess.Board, MCTSNode]] = []
            for _ in range(batch):
                path, leaf_board, leaf = self._select(root, board.copy())
                trajectories.append((path, leaf_board, leaf))

            # ── Phase 2: Batch inference on unique unvisited leaves ────────────
            # Deduplicate leaves; terminals are resolved without the network.
            seen_ids: set[int] = set()
            infer_items: list[tuple[chess.Board, MCTSNode]] = []
            node_value: dict[int, float] = {}

            for _, lb, leaf in trajectories:
                nid = id(leaf)
                if nid in seen_ids:
                    continue
                seen_ids.add(nid)
                if lb.is_game_over():
                    node_value[nid] = _terminal_value(lb.result(), lb.turn)
                else:
                    infer_items.append((lb, leaf))
                    # Placeholder; overwritten below after the forward pass.
                    node_value[nid] = 0.0

            if infer_items:
                boards_batch = [lb for lb, _ in infer_items]
                logits_list, values_np = self._infer(boards_batch)

                for (lb, leaf), logits, val in zip(
                    infer_items, logits_list, values_np
                ):
                    node_value[id(leaf)] = float(val)
                    # Expansion: create children with masked policy priors.
                    # Guard against the rare case of a duplicate leaf that was
                    # already expanded by an earlier simulation in this batch.
                    if not leaf.is_expanded():
                        self._expand(leaf, lb, logits)

            # ── Phase 3: Backpropagation ───────────────────────────────────────
            for path, _, leaf in trajectories:
                self._backpropagate(path, node_value[id(leaf)])

            done += batch

        return self._policy_vector(root, temperature)

    # ──────────────────────────────────────────────────────────────────────────
    # Selection
    # ──────────────────────────────────────────────────────────────────────────

    def _select(
        self, root: MCTSNode, board: chess.Board
    ) -> tuple[list[MCTSNode], chess.Board, MCTSNode]:
        """
        Traverse the tree from *root* using PUCT until an unexpanded or terminal
        node is reached, applying virtual loss at every node entered.

        Virtual-loss bookkeeping
        ~~~~~~~~~~~~~~~~~~~~~~~~
        At each node: ``N += 1`` (counted immediately as the real visit) and
        ``W -= _VIRTUAL_LOSS`` (temporary penalty, undone in _backpropagate).
        This makes the node look like a losing state, steering sibling
        simulations in the same batch towards less-visited branches.

        Parameters
        ----------
        board : chess.Board
            A *copy* of the root board that is advanced in-place as moves are
            pushed during traversal.

        Returns
        -------
        path : list[MCTSNode]
            All nodes visited from root to leaf (both inclusive).
        board : chess.Board
            Board state at the selected leaf.
        leaf : MCTSNode
            The leaf node (same object as ``path[-1]``).
        """
        node = root
        path: list[MCTSNode] = []

        while True:
            # Apply virtual loss: real N increment + temporary W penalty.
            node.N += 1
            node.W -= _VIRTUAL_LOSS
            if node.N > 0:
                node.Q = node.W / node.N
            path.append(node)

            # Stop at unexpanded nodes (leaves) or terminal positions.
            if not node.is_expanded() or board.is_game_over():
                break

            action, child = node.select_child(self.c_puct)
            board.push(decode_move_perspective(action, board.turn))
            node = child

        return path, board, node

    # ──────────────────────────────────────────────────────────────────────────
    # Backpropagation
    # ──────────────────────────────────────────────────────────────────────────

    def _backpropagate(self, path: list[MCTSNode], value: float) -> None:
        """
        Propagate *value* from leaf to root and correct virtual-loss artifacts.

        *value* is the network's scalar output for the player to move at the
        leaf.  It is negated at each level because parent and child alternate
        between opposing colours.

        Because ``N`` was already incremented by the virtual-loss step in
        ``_select``, only ``W`` (and consequently ``Q``) needs adjustment here:

        .. code-block:: text

            W_final = W_after_vl + _VIRTUAL_LOSS + value_at_this_ply
                    = (W_orig − _VIRTUAL_LOSS) + _VIRTUAL_LOSS + value
                    = W_orig + value                              ✓

        Parameters
        ----------
        path : list[MCTSNode]
            Nodes in traversal order (root first, leaf last), as returned by
            ``_select``.
        value : float
            Network value at the leaf, in (−1, 1), from the perspective of the
            player to move at the leaf.
        """
        for node in reversed(path):
            value = -value                      # flip perspective at each ply
            node.W += _VIRTUAL_LOSS + value     # undo VL penalty, add real value
            if node.N > 0:
                node.Q = node.W / node.N

    # ──────────────────────────────────────────────────────────────────────────
    # Expansion
    # ──────────────────────────────────────────────────────────────────────────

    def _expand(
        self,
        node: MCTSNode,
        board: chess.Board,
        policy_logits: np.ndarray,
    ) -> None:
        """
        Create child nodes for every legal move with masked, re-normalised priors.

        Masking strategy
        ~~~~~~~~~~~~~~~~
        Only the logits at legal action indices are retained; the softmax is
        applied over *that subset only*.  This is algebraically equivalent to
        setting all illegal logits to ``−∞`` before a full softmax, but avoids
        touching the full 4672-dimensional vector.

        Parameters
        ----------
        policy_logits : numpy.ndarray
            Raw policy-head output, shape ``(4672,)`` float32.
        """
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return  # terminal node — no children to create

        # Convert each legal move to its perspective-correct action index.
        action_indices: list[int] = [
            encode_move_perspective(m, board.turn) for m in legal_moves
        ]

        # Extract logits for legal moves via numpy fancy indexing (one vectorised
        # C call) rather than a Python loop over individual elements.
        legal_logits = policy_logits[np.array(action_indices, dtype=np.int64)]
        legal_logits = legal_logits - legal_logits.max()  # new array; stable shift
        exp_l = np.exp(legal_logits)
        probs: np.ndarray = exp_l / exp_l.sum()

        for action_idx, prob in zip(action_indices, probs):
            node.children[action_idx] = MCTSNode(prior=float(prob))

    # ──────────────────────────────────────────────────────────────────────────
    # Dirichlet noise
    # ──────────────────────────────────────────────────────────────────────────

    def _apply_dirichlet_noise(self, root: MCTSNode) -> None:
        """
        Mix Dirichlet noise into root prior probabilities to promote exploration.

        Applied once after the root is expanded, before any simulations begin:

        .. math::

            P'(a) = (1 - \\varepsilon)\\, P(a) + \\varepsilon\\, \\eta(a),
            \\qquad \\eta \\sim \\operatorname{Dir}(\\alpha)
        """
        if not root.children:
            return
        actions = list(root.children.keys())
        noise: np.ndarray = np.random.dirichlet(
            np.full(len(actions), self.dirichlet_alpha)
        ).astype(np.float32)
        eps = self.dirichlet_eps
        for action, eta in zip(actions, noise):
            child = root.children[action]
            child.P = (1.0 - eps) * child.P + eps * float(eta)

    # ──────────────────────────────────────────────────────────────────────────
    # Neural-network inference (batched)
    # ──────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _infer(
        self, boards: list[chess.Board]
    ) -> tuple[list[np.ndarray], np.ndarray]:
        """
        Encode *boards*, batch them into a single tensor, and run one forward
        pass through the network.

        This is the sole point of contact with the GPU.  All leaf positions
        accumulated during a selection phase are forwarded together, amortising
        the overhead of data transfer and kernel launches over the full batch.

        Parameters
        ----------
        boards : list[chess.Board]
            Positions to evaluate.  Any batch size ≥ 1 is valid.

        Returns
        -------
        policy_logits : list[numpy.ndarray]
            One raw-logit array per board, each shape ``(4672,)`` float32.
            Masking and softmax are deferred to :meth:`_expand`.
        values : numpy.ndarray
            Shape ``(len(boards),)`` float32; each scalar in ``(−1, 1)``.
        """
        B = len(boards)
        # Pre-allocate one contiguous buffer and fill in-place — avoids
        # creating B intermediate tensors and an extra torch.stack() copy.
        batch_np = np.empty((B, TOTAL_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        for i, b in enumerate(boards):
            # encode_board returns (8, 8, 119); np.copyto handles the
            # non-contiguous transpose view without an intermediate allocation.
            np.copyto(batch_np[i], encode_board(b).transpose(2, 0, 1))

        batch_t = torch.from_numpy(batch_np).to(self.device)  # (B, 119, 8, 8)

        policy_logits_t, values_t = self.model(batch_t)
        # policy_logits_t : (B, 4672),  values_t : (B,)

        logits_list = [policy_logits_t[i].cpu().numpy() for i in range(B)]
        values_np: np.ndarray = values_t.cpu().float().numpy()
        return logits_list, values_np

    # ──────────────────────────────────────────────────────────────────────────
    # Policy extraction
    # ──────────────────────────────────────────────────────────────────────────

    def _policy_vector(self, root: MCTSNode, temperature: float) -> np.ndarray:
        """
        Convert root's child visit counts into a full policy probability vector.

        .. math::

            \\pi(a) \\propto N(s_{\\text{root}}, a)^{1/\\tau}

        With ``τ = 0``, the distribution collapses to a one-hot on the action
        with the highest visit count (deterministic play).

        Returns
        -------
        numpy.ndarray
            Shape ``(4672,)`` float32.
        """
        policy = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
        if not root.children:
            return policy

        actions = list(root.children.keys())
        visits = np.array([root.children[a].N for a in actions], dtype=np.float64)

        if temperature < 1e-8:
            policy[actions[int(np.argmax(visits))]] = 1.0
        else:
            powered = visits ** (1.0 / temperature)
            total = powered.sum()
            if total > 0.0:
                policy[np.array(actions, dtype=np.int64)] = (
                    powered / total
                ).astype(np.float32)

        return policy


# ── Terminal-value helper ─────────────────────────────────────────────────────

def _terminal_value(result: str, turn: chess.Color) -> float:
    """
    Map a finished-game result string to a scalar in ``{−1, 0, +1}``.

    In a checkmate position ``board.turn`` is the *losing* side (the one that
    was mated), so the returned value is almost always ``−1`` in decisive games.
    The general formula is provided for robustness (e.g., adjudicated results).

    Parameters
    ----------
    result : str
        python-chess result: ``'1-0'``, ``'0-1'``, or ``'1/2-1/2'``.
    turn : chess.Color
        The side to move at the terminal position.

    Returns
    -------
    float
        Value in ``{−1, 0, +1}`` from ``turn``'s perspective.
    """
    if result == "1/2-1/2":
        return 0.0
    white_won = result == "1-0"
    current_is_white = turn == chess.WHITE
    # winner == current player → +1 (unusual but handles adjudicated positions)
    # winner != current player → −1 (normal checkmate case)
    if white_won == current_is_white:
        return 1.0
    return -1.0
