#!/usr/bin/env bash
# ec2_setup.sh — Run ONCE after SSH-ing into a fresh EC2 instance.
# Downloads training data from S3, installs missing packages, and sets up the project.
#
# Usage (on EC2):
#   bash aws/ec2_setup.sh
#   -- OR if you haven't uploaded the script to S3 --
#   bash -c "$(aws s3 cp s3://chess-az-training/code/ec2_setup.sh -)"

set -euo pipefail

# ── Load .env ─────────────────────────────────────────────────────────────────
ENV_FILE="${BASH_SOURCE[0]%/*}/.env"
if [ ! -f "$ENV_FILE" ]; then
    echo "ERROR: aws/.env not found at $ENV_FILE"
    exit 1
fi
# shellcheck disable=SC1090
source "$ENV_FILE"

BUCKET="${CHESS_BUCKET:-chess-az-training}"
DATA_DIR="$HOME/data/pgn"
CKPT_DIR="$HOME/checkpoints"
PROJECT_DIR="$HOME/chess-engine"

echo "=== Chess Engine: EC2 Instance Setup ==="
echo ""

# ── Verify GPU ────────────────────────────────────────────────────────────────
echo "[1/5] GPU check:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# ── Activate DLAMI PyTorch environment ───────────────────────────────────────
echo "[2/5] Activating PyTorch environment ..."

# DLAMI (PyTorch 2.7+) uses a virtualenv at /opt/pytorch/
# shellcheck disable=SC1091
source /opt/pytorch/bin/activate
echo "  Python: $(python3 --version)"
echo "  PyTorch: $(python3 -c 'import torch; print(torch.__version__)')"
echo "  CUDA available: $(python3 -c 'import torch; print(torch.cuda.is_available())')"

# ── Install missing packages ──────────────────────────────────────────────────
echo ""
echo "[3/5] Installing missing packages (wandb, python-chess) ..."
pip install --quiet wandb python-chess

echo "  wandb: $(python -c 'import wandb; print(wandb.__version__)')"
echo "  python-chess: $(python -c 'import chess; print(chess.__version__)')"

# ── Download data ─────────────────────────────────────────────────────────────
echo ""
echo "[4/5] Downloading training data ..."

mkdir -p "$DATA_DIR" "$HOME/data/puzzles" "$CKPT_DIR"

# Small PGNs from S3 (already uploaded, fast intra-region)
echo "  [S3] Syncing small PGN files ..."
aws s3 sync "s3://$BUCKET/data/pgn/" "$DATA_DIR/" --no-progress

# Large PGN direct from Lichess (data center speeds, ~2-5 min for 7.6 GB)
LARGE_PGN="$DATA_DIR/lichess_db_standard_rated_2026-03.pgn"
if [ ! -f "$LARGE_PGN" ]; then
    echo "  [Lichess] Downloading lichess_db_standard_rated_2026-03.pgn (7.6 GB) ..."
    pip install --quiet zstandard
    wget -q --show-progress \
        -O "$DATA_DIR/lichess_db_standard_rated_2026-03.pgn.zst" \
        "https://database.lichess.org/standard/lichess_db_standard_rated_2026-03.pgn.zst"
    echo "  Decompressing ..."
    python3 -c "
import zstandard, pathlib
src = pathlib.Path('$DATA_DIR/lichess_db_standard_rated_2026-03.pgn.zst')
dst = src.with_suffix('')
zstandard.ZstdDecompressor().copy_stream(open(src,'rb'), open(dst,'wb'))
src.unlink()
print('  Done:', dst)
"
else
    echo "  Large PGN already present — skipping."
fi

# Puzzle database direct from Lichess
PUZZLE_CSV="$HOME/data/puzzles/lichess_db_puzzle.csv"
if [ ! -f "$PUZZLE_CSV" ]; then
    echo "  [Lichess] Downloading puzzle database (~500 MB compressed) ..."
    wget -q --show-progress \
        -O "${PUZZLE_CSV}.zst" \
        "https://database.lichess.org/lichess_db_puzzle.csv.zst"
    echo "  Decompressing ..."
    python3 -c "
import zstandard, pathlib
src = pathlib.Path('${PUZZLE_CSV}.zst')
dst = src.with_suffix('')
zstandard.ZstdDecompressor().copy_stream(open(src,'rb'), open(dst,'wb'))
src.unlink()
print('  Done:', dst)
"
else
    echo "  Puzzle CSV already present — skipping."
fi

echo ""
echo "  PGN files:"
ls -lh "$DATA_DIR"/*.pgn 2>/dev/null || echo "  WARNING: No .pgn files found!"

echo ""
echo "  Downloading checkpoints from S3 ..."
aws s3 sync "s3://$BUCKET/checkpoints/" "$CKPT_DIR/" --no-progress
ls -lh "$CKPT_DIR"/*.pt 2>/dev/null || echo "  (none — starting from scratch)"

# ── Set up project source code ────────────────────────────────────────────────
echo ""
echo "[5/5] Setting up project source code ..."

if [ -d "$PROJECT_DIR" ]; then
    echo "  $PROJECT_DIR already exists — skipping clone."
else
    # Option A: Clone from GitHub (uses GITHUB_REPO_URL from .env)
    if [ -n "${GITHUB_REPO_URL:-}" ]; then
        git clone "$GITHUB_REPO_URL" "$PROJECT_DIR"
        echo "  Cloned from $GITHUB_REPO_URL"
    # Option B: Download packed tarball from S3
    elif aws s3 ls "s3://$BUCKET/code/chess-engine.tar.gz" &>/dev/null; then
        echo "  Downloading source tarball from S3 ..."
        aws s3 cp "s3://$BUCKET/code/chess-engine.tar.gz" /tmp/chess-engine.tar.gz
        mkdir -p "$PROJECT_DIR"
        tar -xzf /tmp/chess-engine.tar.gz -C "$PROJECT_DIR" --strip-components=1
        rm /tmp/chess-engine.tar.gz
        echo "  Source extracted to $PROJECT_DIR"
    else
        echo ""
        echo "  ACTION REQUIRED: Set GITHUB_REPO_URL in aws/.env, or upload a source tarball:"
        echo "    tar -czf chess-engine.tar.gz --exclude=data --exclude=checkpoints --exclude='*.zst' ."
        echo "    aws s3 cp chess-engine.tar.gz s3://$BUCKET/code/"
        echo "  Then re-run this script."
        exit 1
    fi
fi

# ── W&B login (auto if key set in .env, interactive otherwise) ────────────────
echo ""
if [ -n "${WANDB_API_KEY:-}" ]; then
    wandb login "$WANDB_API_KEY" --relogin
    echo "  Logged in to W&B using key from aws/.env"
else
    echo "  No WANDB_API_KEY in aws/.env — run 'wandb login' manually if you want W&B."
fi

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Start training:"
echo "  tmux new -s train"
echo "  bash aws/train_spot.sh"
echo "  Ctrl+B D  (detach)"
