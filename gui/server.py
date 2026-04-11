"""
FastAPI server for the chess GUI.

Serves the web interface and handles game logic + engine integration via WebSocket.

Usage:
    python gui/server.py                        # default settings
    python gui/server.py --port 3000            # custom port
    python gui/server.py --no-engine            # UI dev mode (no model)
    python gui/server.py --checkpoint path.pt   # custom checkpoint
    python gui/server.py --num-sims 1600        # stronger engine
"""

from __future__ import annotations

import argparse
import asyncio
import random
import sys
import time
from pathlib import Path
from typing import Optional

import chess
import chess.svg
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

# ── Path setup ───────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from chess_engine.core import (  # noqa: E402
    ACTION_SPACE_SIZE,
    decode_move_perspective,
    encode_move_perspective,
)
from chess_engine.mcts import MCTS  # noqa: E402
from chess_engine.models import ChessResNet  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════════════
# Game Engine — wraps model + MCTS for play
# ═══════════════════════════════════════════════════════════════════════════════


class GameEngine:
    """Neural network + MCTS wrapper for interactive play."""

    def __init__(self, checkpoint_path: Path, num_sims: int = 800) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(checkpoint_path)
        # Large batch size for GPU inference — fewer Python round-trips
        self.mcts = MCTS(self.model, self.device, batch_size=64)
        self.num_sims = num_sims
        self.thinking = False
        self._search_start: float = 0.0

    def _load_model(self, path: Path) -> ChessResNet:
        model = ChessResNet(num_blocks=20, num_filters=256)
        if path.exists():
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
            state = ckpt["model"]
            # Strip `_orig_mod.` prefix added by torch.compile()
            state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
            model.load_state_dict(state)
            print(f"  Loaded: {path.name} (step {ckpt.get('global_step', '?')})")
        else:
            print(f"  WARNING: {path} not found — using random weights")
        model.to(self.device)
        model.eval()
        # Half-precision for faster GPU inference
        if self.device.type == "cuda":
            model.half()
        return model

    # ── Search ────────────────────────────────────────────────────────────────

    def search(self, board: chess.Board) -> dict:
        """Run MCTS and return best move + analysis.  Called from a worker thread."""
        self.thinking = True
        self._search_start = time.time()

        policy = self.mcts.run(board.copy(stack=True), self.num_sims, temperature=0.0)

        best_move = self._pick_move(policy, board)
        analysis = self._build_analysis(board)
        self.thinking = False
        return {"move": best_move, "analysis": analysis}

    def _pick_move(self, policy: np.ndarray, board: chess.Board) -> chess.Move:
        best_action = int(np.argmax(policy))
        candidate = decode_move_perspective(best_action, board.turn)
        if candidate in board.legal_moves:
            return candidate
        # Fallback: scan legal moves for highest policy probability
        best, best_p = None, -1.0
        for m in board.legal_moves:
            p = float(policy[encode_move_perspective(m, board.turn)])
            if p > best_p:
                best, best_p = m, p
        return best or next(iter(board.legal_moves))

    def _build_analysis(self, board: chess.Board) -> dict:
        root = self.mcts._root
        if not root or not root.children:
            return {"top_moves": [], "eval": 0.0, "sims": 0, "elapsed": 0.0}

        top = []
        for action, child in sorted(root.children.items(), key=lambda x: -x[1].N):
            move = decode_move_perspective(action, board.turn)
            try:
                san = board.san(move) if move in board.legal_moves else move.uci()
            except Exception:
                san = move.uci()
            top.append(
                {
                    "san": san,
                    "uci": move.uci(),
                    "visits": child.N,
                    "q": round(child.Q, 4),
                    "winrate": round((child.Q + 1) / 2 * 100, 1),
                }
            )
            if len(top) >= 5:
                break

        # Best child Q from White's perspective (for the eval bar)
        best_q = top[0]["q"] if top else 0.0
        eval_white = best_q if board.turn == chess.WHITE else -best_q

        return {
            "top_moves": top,
            "eval": round(eval_white, 4),
            "sims": self.num_sims,
            "elapsed": round(time.time() - self._search_start, 2),
        }

    # ── Progress polling (called from async loop while search runs) ───────────

    def get_progress(self) -> Optional[dict]:
        if not self.thinking:
            return None
        return {
            "sims": self.mcts._sims_done,
            "total": self.num_sims,
            "elapsed": round(time.time() - self._search_start, 2),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Game Session — server-authoritative game state
# ═══════════════════════════════════════════════════════════════════════════════


class GameSession:

    def __init__(self) -> None:
        self.board = chess.Board()
        self.player_color: chess.Color = chess.WHITE
        self.last_move: Optional[chess.Move] = None

    def reset(self, player_color: chess.Color = chess.WHITE) -> None:
        self.board = chess.Board()
        self.player_color = player_color
        self.last_move = None

    def try_move(self, uci: str) -> Optional[chess.Move]:
        """Validate and push a UCI move.  Returns the Move or None."""
        try:
            move = self.board.parse_uci(uci)
            if move in self.board.legal_moves:
                self.board.push(move)
                self.last_move = move
                return move
        except (ValueError, AssertionError):
            pass
        return None

    def push_engine_move(self, move: chess.Move) -> str:
        """Push a pre-validated engine move.  Returns SAN."""
        san = self.board.san(move)
        self.board.push(move)
        self.last_move = move
        return san

    def undo_pair(self) -> bool:
        """Undo last two half-moves (engine + player)."""
        if len(self.board.move_stack) >= 2:
            self.board.pop()
            self.board.pop()
            self.last_move = self.board.peek() if self.board.move_stack else None
            return True
        return False

    # ── Serialise full state for the client ───────────────────────────────────

    def get_state(self) -> dict:
        legal = [m.uci() for m in self.board.legal_moves]

        check_sq = None
        if self.board.is_check():
            k = self.board.king(self.board.turn)
            if k is not None:
                check_sq = chess.square_name(k)

        game_over = None
        if self.board.is_game_over():
            result = self.board.result()
            reason = (
                "Checkmate"
                if self.board.is_checkmate()
                else "Stalemate"
                if self.board.is_stalemate()
                else "Insufficient material"
                if self.board.is_insufficient_material()
                else "Fifty-move rule"
                if self.board.is_fifty_moves()
                else "Threefold repetition"
                if self.board.is_repetition()
                else "Game over"
            )
            game_over = {"result": result, "reason": reason}

        return {
            "fen": self.board.fen(),
            "turn": "white" if self.board.turn == chess.WHITE else "black",
            "legal_moves": legal,
            "last_move": self.last_move.uci() if self.last_move else None,
            "check": check_sq,
            "game_over": game_over,
            "move_history": self._build_history(),
            "captured": self._captured(),
            "player_color": "white" if self.player_color == chess.WHITE else "black",
            "is_player_turn": self.board.turn == self.player_color,
        }

    def _build_history(self) -> list:
        history: list[dict] = []
        temp = chess.Board()
        for i, move in enumerate(self.board.move_stack):
            san = temp.san(move)
            temp.push(move)
            if i % 2 == 0:
                history.append({"num": i // 2 + 1, "white": san, "black": None})
            else:
                if history:
                    history[-1]["black"] = san
        return history

    def _captured(self) -> dict:
        start = chess.Board()
        white_taken: list[str] = []  # pieces captured BY White (Black pieces gone)
        black_taken: list[str] = []  # pieces captured BY Black (White pieces gone)

        for pt in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN]:
            sym = chess.piece_symbol(pt)
            sw = bin(start.pieces_mask(pt, chess.WHITE)).count("1")
            sb = bin(start.pieces_mask(pt, chess.BLACK)).count("1")
            cw = bin(self.board.pieces_mask(pt, chess.WHITE)).count("1")
            cb = bin(self.board.pieces_mask(pt, chess.BLACK)).count("1")

            for _ in range(sw - cw):
                black_taken.append(sym.upper())
            for _ in range(sb - cb):
                white_taken.append(sym.lower())

        return {"white": white_taken, "black": black_taken}


# ═══════════════════════════════════════════════════════════════════════════════
# FastAPI Application
# ═══════════════════════════════════════════════════════════════════════════════

GUI_DIR = Path(__file__).resolve().parent

app = FastAPI(title="Chess Engine")
app.mount("/static", StaticFiles(directory=GUI_DIR / "static"), name="static")

engine: Optional[GameEngine] = None
game = GameSession()

# ── Piece SVG endpoint (cached) ──────────────────────────────────────────────

_svg_cache: dict[str, str] = {}

_PIECE_MAP = {
    "p": chess.PAWN,
    "n": chess.KNIGHT,
    "b": chess.BISHOP,
    "r": chess.ROOK,
    "q": chess.QUEEN,
    "k": chess.KING,
}


@app.get("/")
async def index():
    return HTMLResponse((GUI_DIR / "templates" / "index.html").read_text(encoding="utf-8"))


@app.get("/pieces/{color}/{piece}")
async def piece_svg(color: str, piece: str):
    key = f"{color}{piece}"
    if key not in _svg_cache:
        try:
            c = chess.WHITE if color == "w" else chess.BLACK
            _svg_cache[key] = chess.svg.piece(chess.Piece(_PIECE_MAP[piece], c), size=80)
        except Exception:
            return Response(status_code=404)
    return Response(
        content=_svg_cache[key],
        media_type="image/svg+xml",
        headers={"Cache-Control": "public, max-age=86400"},
    )


# ── WebSocket game endpoint ──────────────────────────────────────────────────


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    await ws.send_json({"type": "state", **game.get_state()})

    try:
        while True:
            data = await ws.receive_json()
            t = data.get("type")

            if t == "new_game":
                color_str = data.get("color", "white")
                strength = data.get("strength", 800)

                if color_str == "random":
                    color = random.choice([chess.WHITE, chess.BLACK])
                else:
                    color = chess.WHITE if color_str == "white" else chess.BLACK

                game.reset(player_color=color)
                if engine:
                    engine.num_sims = strength

                await ws.send_json({"type": "state", **game.get_state()})

                # Engine plays first if player is Black
                if game.board.turn != game.player_color and engine:
                    await _run_engine(ws)

            elif t == "move":
                uci = data.get("uci", "")
                move = game.try_move(uci)
                if not move:
                    await ws.send_json({"type": "error", "message": f"Illegal: {uci}"})
                    continue

                await ws.send_json({"type": "state", **game.get_state()})

                # Engine responds if game isn't over
                if not game.board.is_game_over() and engine:
                    await _run_engine(ws)

            elif t == "resign":
                r = "0-1" if game.player_color == chess.WHITE else "1-0"
                await ws.send_json(
                    {"type": "game_over", "result": r, "reason": "Resignation"}
                )

            elif t == "undo":
                if game.undo_pair():
                    await ws.send_json({"type": "state", **game.get_state()})

            elif t == "update_settings":
                if engine and "strength" in data:
                    engine.num_sims = data["strength"]

    except WebSocketDisconnect:
        pass


async def _run_engine(ws: WebSocket):
    """Run MCTS search in a thread with periodic progress updates via WS."""
    if not engine:
        return

    await ws.send_json({"type": "thinking_start"})

    loop = asyncio.get_event_loop()
    future = loop.run_in_executor(None, engine.search, game.board)

    # Send progress updates while the search thread runs
    while not future.done():
        prog = engine.get_progress()
        if prog:
            try:
                await ws.send_json({"type": "thinking", **prog})
            except Exception:
                break
        await asyncio.sleep(0.3)

    try:
        result = future.result()
    except Exception as e:
        await ws.send_json({"type": "error", "message": str(e)})
        return

    move = result["move"]
    san = game.push_engine_move(move)

    await ws.send_json(
        {
            "type": "engine_move",
            "uci": move.uci(),
            "san": san,
            "analysis": result["analysis"],
        }
    )
    await ws.send_json({"type": "state", **game.get_state()})


# ═══════════════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    global engine

    parser = argparse.ArgumentParser(description="Chess GUI — play against your engine")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--checkpoint",
        default=str(PROJECT_ROOT / "checkpoints" / "final_model.pt"),
    )
    parser.add_argument("--num-sims", type=int, default=800)
    parser.add_argument(
        "--no-engine", action="store_true", help="Run without engine (UI development)"
    )
    args = parser.parse_args()

    print("=== Chess Engine GUI ===\n")

    if not args.no_engine:
        print("Loading model...")
        engine = GameEngine(Path(args.checkpoint), num_sims=args.num_sims)
        print(f"  Device: {engine.device}")
        print(f"  Simulations: {args.num_sims}")
    else:
        print("Running without engine (--no-engine)")

    url = f"http://{args.host}:{args.port}"
    print(f"\n  Open {url} in your browser to play\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
