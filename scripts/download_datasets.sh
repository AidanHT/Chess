#!/usr/bin/env bash
# download_datasets.sh — Download additional Lichess datasets and upload to S3.
#
# Downloads:
#   1. Lichess puzzle database (lichess_db_puzzle.csv) — ~500 MB compressed
#   2. Additional monthly standard-rated PGN dumps for specified months
#
# Run on EC2 (or locally) after aws/ec2_setup.sh has been run.
# Requires: zstd, wget, aws CLI, python3
#
# Usage:
#   bash scripts/download_datasets.sh [options]
#
# Options:
#   --puzzles           Download the puzzle database (default: on)
#   --no-puzzles        Skip puzzle download
#   --months YYYY-MM    Download standard PGN for this month (repeat for multiple)
#   --filter-elo N      After download, filter PGNs to N+ ELO (creates elite/ subdir)
#   --upload            Upload downloaded data to S3 after decompression
#   --data-dir DIR      Local directory to save files (default: ~/data)
#   --bucket NAME       S3 bucket name (default: chess-az-training)
#
# Examples:
#   # Download puzzles + 6 months of games, filter to 2500+ elite, upload
#   bash scripts/download_datasets.sh \
#       --months 2025-07 --months 2025-08 --months 2025-09 \
#       --months 2025-10 --months 2025-11 --months 2025-12 \
#       --filter-elo 2500 --upload
#
#   # Just download puzzles
#   bash scripts/download_datasets.sh --puzzles --upload

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
DO_PUZZLES=true
MONTHS=()
FILTER_ELO=0       # 0 = no filtering
DO_UPLOAD=false
DATA_DIR="${HOME}/data"
BUCKET="chess-az-training"

# ── Parse arguments ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --puzzles)       DO_PUZZLES=true;   shift ;;
        --no-puzzles)    DO_PUZZLES=false;  shift ;;
        --months)        MONTHS+=("$2");    shift 2 ;;
        --filter-elo)    FILTER_ELO="$2";   shift 2 ;;
        --upload)        DO_UPLOAD=true;    shift ;;
        --data-dir)      DATA_DIR="$2";     shift 2 ;;
        --bucket)        BUCKET="$2";       shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

PGN_DIR="${DATA_DIR}/pgn"
PUZZLE_DIR="${DATA_DIR}/puzzles"
ELITE_DIR="${DATA_DIR}/elite"
LICHESS_BASE="https://database.lichess.org"

mkdir -p "$PGN_DIR" "$PUZZLE_DIR"

echo "=== Lichess Dataset Downloader ==="
echo "  Data dir : $DATA_DIR"
echo "  Upload   : $DO_UPLOAD"
echo ""

# ── Helper: decompress .zst ───────────────────────────────────────────────────
decompress_zst() {
    local src="$1"
    local dst="$2"
    echo "  Decompressing $(basename "$src") ..."
    if command -v zstd &>/dev/null; then
        zstd -d --rm -o "$dst" "$src"
    else
        # Fallback: Python zstandard (pip install zstandard)
        python3 -c "
import zstandard, sys, pathlib
src, dst = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
ctx = zstandard.ZstdDecompressor()
with open(src, 'rb') as fin, open(dst, 'wb') as fout:
    ctx.copy_stream(fin, fout)
src.unlink()
print(f'  Decompressed to {dst}')
" "$src" "$dst"
    fi
}

# ── 1. Puzzle database ────────────────────────────────────────────────────────
if $DO_PUZZLES; then
    echo "[PUZZLES] Downloading Lichess puzzle database ..."
    PUZZLE_CSV="${PUZZLE_DIR}/lichess_db_puzzle.csv"
    PUZZLE_ZST="${PUZZLE_DIR}/lichess_db_puzzle.csv.zst"

    if [ -f "$PUZZLE_CSV" ]; then
        echo "  Already exists — skipping download."
    else
        wget -q --show-progress -O "$PUZZLE_ZST" \
            "${LICHESS_BASE}/lichess_db_puzzle.csv.zst"
        decompress_zst "$PUZZLE_ZST" "$PUZZLE_CSV"
        echo "  Puzzle CSV: $(du -h "$PUZZLE_CSV" | cut -f1)"
    fi

    PUZZLE_COUNT=$(tail -n +2 "$PUZZLE_CSV" | wc -l)
    echo "  Puzzles: $(printf "%'.f" $PUZZLE_COUNT)"

    if $DO_UPLOAD; then
        echo "  Uploading to s3://${BUCKET}/data/puzzles/ ..."
        aws s3 cp "$PUZZLE_CSV" "s3://${BUCKET}/data/puzzles/lichess_db_puzzle.csv" \
            --no-progress
        echo "  Upload complete."
    fi
    echo ""
fi

# ── 2. Monthly standard PGN dumps ─────────────────────────────────────────────
for MONTH in "${MONTHS[@]}"; do
    echo "[PGN] Downloading standard rated games for ${MONTH} ..."
    FNAME="lichess_db_standard_rated_${MONTH}.pgn"
    PGN_PATH="${PGN_DIR}/${FNAME}"
    ZST_PATH="${PGN_DIR}/${FNAME}.zst"

    if [ -f "$PGN_PATH" ]; then
        echo "  Already exists — skipping."
        continue
    fi

    wget -q --show-progress -O "$ZST_PATH" \
        "${LICHESS_BASE}/standard/${FNAME}.zst"
    decompress_zst "$ZST_PATH" "$PGN_PATH"
    echo "  PGN size: $(du -h "$PGN_PATH" | cut -f1)"
    echo ""
done

# ── 3. Optional ELO filtering → elite/ subdir ─────────────────────────────────
if [ "$FILTER_ELO" -gt 0 ]; then
    echo "[FILTER] Filtering downloaded PGNs to ${FILTER_ELO}+ ELO ..."
    mkdir -p "$ELITE_DIR"

    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    FILTER_SCRIPT="${SCRIPT_DIR}/filter_pgn_by_elo.py"

    for PGN in "$PGN_DIR"/*.pgn; do
        BASENAME=$(basename "$PGN" .pgn)
        OUT="${ELITE_DIR}/${BASENAME}_${FILTER_ELO}plus.pgn"
        if [ -f "$OUT" ]; then
            echo "  $OUT already exists — skipping."
            continue
        fi
        echo "  Filtering $(basename "$PGN") ..."
        python3 "$FILTER_SCRIPT" "$PGN" -o "$OUT" -e "$FILTER_ELO" --quiet
        echo "  Elite output: $(du -h "$OUT" | cut -f1)"
    done

    if $DO_UPLOAD; then
        echo "  Uploading elite PGNs to s3://${BUCKET}/data/elite/ ..."
        aws s3 sync "$ELITE_DIR/" "s3://${BUCKET}/data/elite/" --no-progress
        echo "  Upload complete."
    fi
    echo ""
fi

# ── 4. Upload standard PGNs if requested ─────────────────────────────────────
if $DO_UPLOAD && [ "${#MONTHS[@]}" -gt 0 ]; then
    echo "[UPLOAD] Syncing new PGNs to s3://${BUCKET}/data/pgn/ ..."
    aws s3 sync "$PGN_DIR/" "s3://${BUCKET}/data/pgn/" \
        --exclude "*.zst" --no-progress
    echo "  Done."
    echo ""
fi

echo "=== All done! ==="
echo ""
echo "Summary:"
[ -f "${PUZZLE_DIR}/lichess_db_puzzle.csv" ] && \
    echo "  Puzzles : ${PUZZLE_DIR}/lichess_db_puzzle.csv"
[ -d "$ELITE_DIR" ] && ls "$ELITE_DIR"/*.pgn 2>/dev/null | \
    while read -r f; do echo "  Elite   : $f ($(du -h "$f" | cut -f1))"; done
echo ""
echo "To start Phase 2 fine-tuning with puzzles, pass these flags to train_engine.py:"
echo "  --puzzle_dir ${PUZZLE_DIR}/lichess_db_puzzle.csv"
echo "  --puzzle_ratio 0.2"
echo "  --fine_tune"
echo "  --resume <phase1_checkpoint.pt>"
