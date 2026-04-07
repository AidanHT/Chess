#!/usr/bin/env bash
# upload_data.sh — Run ONCE from your local machine (project root).
# Creates the S3 bucket and uploads all PGN files + the existing checkpoint.
# Upload time: ~60-90 min on a 50 Mbps uplink for the full 7.9 GB of .pgn files.
#
# Usage:
#   bash aws/upload_data.sh

set -euo pipefail

# ── Load .env ─────────────────────────────────────────────────────────────────
ENV_FILE="${BASH_SOURCE[0]%/*}/.env"
if [ ! -f "$ENV_FILE" ]; then
    echo "ERROR: aws/.env not found. Fill in aws/.env and try again."
    exit 1
fi
# shellcheck disable=SC1090
source "$ENV_FILE"

BUCKET="${CHESS_BUCKET:-chess-az-training}"
REGION="${AWS_REGION:-us-east-1}"

echo "=== Chess Engine: Upload Training Data to S3 ==="
echo ""

# ── Sanity checks ─────────────────────────────────────────────────────────────
if ! command -v aws &>/dev/null; then
    echo "ERROR: AWS CLI not found. Install from https://aws.amazon.com/cli/"
    exit 1
fi

if [ ! -d "data/pgn" ]; then
    echo "ERROR: data/pgn directory not found. Run from the project root."
    exit 1
fi

PGN_COUNT=$(find data/pgn -name "*.pgn" | wc -l)
if [ "$PGN_COUNT" -eq 0 ]; then
    echo "ERROR: No .pgn files found in data/pgn/. Nothing to upload."
    exit 1
fi
echo "Found $PGN_COUNT .pgn files to upload."

# ── Create S3 bucket ──────────────────────────────────────────────────────────
echo ""
echo "[1/4] Creating S3 bucket: s3://$BUCKET ..."

# us-east-1 does not accept --create-bucket-configuration (it's the default)
if aws s3api head-bucket --bucket "$BUCKET" 2>/dev/null; then
    echo "  Bucket already exists — skipping creation."
else
    aws s3api create-bucket \
        --bucket "$BUCKET" \
        --region "$REGION"
    echo "  Bucket created."
fi

# Block all public access (safety)
aws s3api put-public-access-block \
    --bucket "$BUCKET" \
    --public-access-block-configuration \
    "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true"

# ── Upload PGN files ──────────────────────────────────────────────────────────
echo ""
echo "[2/4] Uploading PGN files to s3://$BUCKET/data/pgn/ ..."
echo "  Excluding .zst files (find_pgn_files() only globs *.pgn)"
echo "  This may take 60-90 minutes on a typical home connection..."
echo ""

aws s3 sync data/pgn/ "s3://$BUCKET/data/pgn/" \
    --exclude "*.zst" \
    --exclude "*.gz" \
    --no-progress \
    --storage-class STANDARD

echo "  PGN upload complete."

# ── Upload existing checkpoint ────────────────────────────────────────────────
echo ""
echo "[3/4] Uploading existing checkpoint(s) to s3://$BUCKET/checkpoints/ ..."

CKPT_COUNT=$(find checkpoints -name "step_*.pt" 2>/dev/null | wc -l)
if [ "$CKPT_COUNT" -gt 0 ]; then
    aws s3 sync checkpoints/ "s3://$BUCKET/checkpoints/" --no-progress
    echo "  Uploaded $CKPT_COUNT checkpoint(s)."
else
    echo "  No checkpoints found — skipping."
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "[4/4] Bucket summary:"
aws s3 ls --summarize --human-readable --recursive "s3://$BUCKET/" 2>/dev/null | tail -3

echo ""
echo "=== Upload complete! ==="
echo ""
echo "Next step: launch your EC2 spot instance:"
echo "  bash aws/launch_spot.sh"
