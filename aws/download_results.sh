#!/usr/bin/env bash
# download_results.sh — Run from your local machine to pull trained checkpoints.
#
# Usage:
#   cd /path/to/chess-project
#   bash aws/download_results.sh

set -euo pipefail

ENV_FILE="${BASH_SOURCE[0]%/*}/.env"
if [ ! -f "$ENV_FILE" ]; then
    echo "ERROR: aws/.env not found."
    exit 1
fi
# shellcheck disable=SC1090
source "$ENV_FILE"

BUCKET="${CHESS_BUCKET:-chess-az-training}"

echo "=== Downloading checkpoints from S3 ==="
echo ""

aws s3 sync "s3://$BUCKET/checkpoints/" checkpoints/ --no-progress

echo ""
echo "Checkpoints:"
ls -lh checkpoints/step_*.pt 2>/dev/null || echo "  (none found)"

echo ""
echo "Latest:"
find checkpoints -name "step_*.pt" 2>/dev/null | sort | tail -1
