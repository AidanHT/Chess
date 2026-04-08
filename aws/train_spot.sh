#!/usr/bin/env bash
# train_spot.sh — Two-phase training launcher for EC2 spot instances.
# Run inside a tmux session to persist across SSH disconnections.
#
# Phase 1 — Broad supervised training on all PGN data (standard + 2000+ games).
#            High learning rate, 10 epochs. Builds positional understanding.
#
# Phase 2 — Fine-tune on elite (2500+) data + Lichess puzzles.
#            Low learning rate, 5 epochs, fresh optimizer.
#            Sharpens the policy head on high-quality moves.
#
# Spot interruption safety:
#   - SIGTERM trap → final S3 sync
#   - Background sync every 5 minutes
#   - Metadata poller detects 2-minute warning and syncs immediately
#
# Usage (on EC2, inside tmux):
#   bash aws/train_spot.sh

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
CKPT_DIR="$HOME/checkpoints"
DATA_DIR="$HOME/data/pgn"          # Phase 1: all standard PGNs
ELITE_DIR="$HOME/data/elite"       # Phase 2: 2500+ filtered games
PUZZLE_CSV="$HOME/data/puzzles/lichess_db_puzzle.csv"
PROJECT_DIR="$HOME/chess-engine"
LOG_FILE="$HOME/training.log"

# Phase configuration
P1_EPOCHS=10
P1_LR=2e-3
P1_STEPS=50000

P2_EPOCHS=5
P2_LR=5e-4
P2_STEPS=20000
P2_PUZZLE_RATIO=0.2

echo "=== Chess Engine: Two-Phase Training ==="
echo "  Phase 1 : All PGN data  (${P1_EPOCHS} epochs, lr=${P1_LR})"
echo "  Phase 2 : Elite + Puzzles (${P2_EPOCHS} epochs, lr=${P2_LR}, fine-tune)"
echo ""

# ── Activate DLAMI PyTorch virtualenv ────────────────────────────────────────
# shellcheck disable=SC1091
source /opt/pytorch/bin/activate

# ── Export W&B key if set ─────────────────────────────────────────────────────
[ -n "${WANDB_API_KEY:-}" ] && export WANDB_API_KEY

# ── Spot interruption handler (SIGTERM) ───────────────────────────────────────
cleanup() {
    echo ""
    echo "[SPOT/SIGTERM] Syncing checkpoints to S3 ..."
    aws s3 sync "$CKPT_DIR" "s3://$BUCKET/checkpoints/" --quiet || true
    kill "$SYNC_PID" "$WATCH_PID" 2>/dev/null || true
    exit 0
}
trap cleanup SIGTERM SIGINT

# ── Background checkpoint sync every 5 minutes ────────────────────────────────
(
    while true; do
        sleep 300
        aws s3 sync "$CKPT_DIR" "s3://$BUCKET/checkpoints/" --quiet 2>/dev/null || true
    done
) &
SYNC_PID=$!

# ── Spot metadata watcher (2-minute warning) ───────────────────────────────────
(
    while true; do
        HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
            --connect-timeout 1 --max-time 2 \
            "http://169.254.169.254/latest/meta-data/spot/interruption-action" \
            2>/dev/null || echo "000")
        if [ "$HTTP_CODE" = "200" ]; then
            echo "[SPOT] Interruption notice — syncing immediately ..."
            aws s3 sync "$CKPT_DIR" "s3://$BUCKET/checkpoints/" --quiet || true
            break
        fi
        sleep 5
    done
) &
WATCH_PID=$!

cd "$PROJECT_DIR"

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — All data, broad training
# ═══════════════════════════════════════════════════════════════════════════════

# Find latest checkpoint for auto-resume (handles mid-phase restarts)
LATEST=$(find "$CKPT_DIR" -name "step_*.pt" 2>/dev/null | sort | tail -1)
P1_RESUME=""
[ -n "$LATEST" ] && P1_RESUME="--resume $LATEST"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  PHASE 1 — Broad training on all PGN data"
echo "  Epochs: ${P1_EPOCHS}  |  LR: ${P1_LR}  |  Batch: 1024"
[ -n "$P1_RESUME" ] && echo "  Resuming from: $LATEST"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

python train_engine.py \
    --data_dir "$DATA_DIR" \
    --batch_size 1024 \
    --num_workers 8 \
    --prefetch_factor 8 \
    --lr "$P1_LR" \
    --epochs "$P1_EPOCHS" \
    --steps_per_epoch "$P1_STEPS" \
    --compile \
    --wandb_project "chess-alphazero" \
    --wandb_run_name "aws-phase1-$(date +%Y%m%d)" \
    $P1_RESUME \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "Phase 1 complete at $(date)."
aws s3 sync "$CKPT_DIR" "s3://$BUCKET/checkpoints/" --no-progress

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Elite + puzzle fine-tuning
# ═══════════════════════════════════════════════════════════════════════════════

P1_BEST=$(find "$CKPT_DIR" -name "step_*.pt" 2>/dev/null | sort | tail -1)

if [ -z "$P1_BEST" ]; then
    echo "ERROR: No Phase 1 checkpoint found — skipping Phase 2."
    exit 1
fi

echo ""

# Decide data dir for Phase 2: prefer elite (2500+), fall back to 2000+ data
if [ -d "$ELITE_DIR" ] && compgen -G "${ELITE_DIR}/*.pgn" > /dev/null 2>&1; then
    P2_DATA_DIR="$ELITE_DIR"
    echo "  Phase 2 data: Elite games (${ELITE_DIR})"
else
    # Fall back to the 2000+ filtered files already in the main pgn dir
    # by creating a symlink directory pointing only to *2000plus* files
    P2_DATA_DIR="$HOME/data/pgn_2000plus"
    mkdir -p "$P2_DATA_DIR"
    find "$DATA_DIR" -name "*2000plus*.pgn" -exec ln -sf {} "$P2_DATA_DIR/" \; 2>/dev/null || true
    LINKED=$(ls "$P2_DATA_DIR"/*.pgn 2>/dev/null | wc -l)
    if [ "$LINKED" -eq 0 ]; then
        echo "  WARNING: No 2000+ or elite PGN files found. Using all PGN data for Phase 2."
        P2_DATA_DIR="$DATA_DIR"
    else
        echo "  Phase 2 data: 2000+ filtered games (${LINKED} files)"
    fi
fi

# Decide whether to include puzzles
P2_PUZZLE_ARGS=""
if [ -f "$PUZZLE_CSV" ]; then
    P2_PUZZLE_ARGS="--puzzle_dir $PUZZLE_CSV --puzzle_ratio $P2_PUZZLE_RATIO"
    echo "  Puzzle CSV : $PUZZLE_CSV (ratio ${P2_PUZZLE_RATIO})"
else
    echo "  Puzzle CSV : not found — run scripts/download_datasets.sh --puzzles first"
    echo "               Continuing Phase 2 without puzzles."
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  PHASE 2 — Fine-tune: elite games + puzzles"
echo "  Epochs: ${P2_EPOCHS}  |  LR: ${P2_LR}  |  Batch: 1024"
echo "  Resuming weights from: $P1_BEST"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

P2_LOG="$HOME/training_phase2.log"

# shellcheck disable=SC2086
python train_engine.py \
    --data_dir "$P2_DATA_DIR" \
    $P2_PUZZLE_ARGS \
    --batch_size 1024 \
    --num_workers 8 \
    --prefetch_factor 8 \
    --lr "$P2_LR" \
    --epochs "$P2_EPOCHS" \
    --steps_per_epoch "$P2_STEPS" \
    --compile \
    --fine_tune \
    --resume "$P1_BEST" \
    --wandb_project "chess-alphazero" \
    --wandb_run_name "aws-phase2-$(date +%Y%m%d)" \
    2>&1 | tee "$P2_LOG"

# ── Final sync ────────────────────────────────────────────────────────────────
kill "$SYNC_PID" "$WATCH_PID" 2>/dev/null || true

echo ""
echo "Both phases complete at $(date). Final sync ..."
aws s3 sync "$CKPT_DIR" "s3://$BUCKET/checkpoints/" --no-progress

STAMP=$(date +%Y%m%d-%H%M%S)
aws s3 cp "$LOG_FILE"  "s3://$BUCKET/logs/phase1-${STAMP}.log"
aws s3 cp "$P2_LOG"    "s3://$BUCKET/logs/phase2-${STAMP}.log"

echo ""
echo "=== Training Complete ==="
echo "  Checkpoints : s3://$BUCKET/checkpoints/"
echo "  Phase 1 log : s3://$BUCKET/logs/phase1-${STAMP}.log"
echo "  Phase 2 log : s3://$BUCKET/logs/phase2-${STAMP}.log"
echo ""
echo "From your local machine:"
echo "  bash aws/download_results.sh"
