#!/usr/bin/env bash
# launch_spot.sh — Launch a g5.2xlarge spot instance from your local machine.
#
# Usage:
#   bash aws/launch_spot.sh

set -euo pipefail

# ── Load .env ─────────────────────────────────────────────────────────────────
ENV_FILE="${BASH_SOURCE[0]%/*}/.env"
if [ ! -f "$ENV_FILE" ]; then
    echo "ERROR: aws/.env not found. Fill in aws/.env and try again."
    exit 1
fi
# shellcheck disable=SC1090
source "$ENV_FILE"

REGION="${AWS_REGION:-us-east-1}"
INSTANCE_TYPE="g5.2xlarge"
VOLUME_SIZE=150   # GB — holds 7.9 GB data + OS + env + checkpoints
IAM_PROFILE="chess-training-role"

echo "=== Chess Engine: Launch EC2 Spot Instance ==="
echo ""

# ── Validate required env vars ────────────────────────────────────────────────
if [ -z "${CHESS_KEY_NAME:-}" ]; then
    echo "ERROR: CHESS_KEY_NAME is not set in aws/.env"
    exit 1
fi

if [ -z "${CHESS_SG_ID:-}" ]; then
    echo "ERROR: CHESS_SG_ID is not set in aws/.env"
    exit 1
fi

KEY_PATH="${CHESS_KEY_PATH:-~/.ssh/${CHESS_KEY_NAME}.pem}"

# ── Find latest Deep Learning AMI ─────────────────────────────────────────────
echo "[1/3] Finding latest AWS Deep Learning AMI (PyTorch 2.4, Ubuntu 22.04) ..."

AMI_ID=$(aws ec2 describe-images \
    --region "$REGION" \
    --owners amazon \
    --filters \
        "Name=name,Values=Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.4 (Ubuntu 22.04)*" \
        "Name=state,Values=available" \
    --query "sort_by(Images, &CreationDate)[-1].ImageId" \
    --output text)

if [ -z "$AMI_ID" ] || [ "$AMI_ID" = "None" ]; then
    echo "ERROR: Could not find DLAMI. Check your AWS region and credentials."
    exit 1
fi

echo "  AMI: $AMI_ID"

# ── Launch spot instance ───────────────────────────────────────────────────────
echo ""
echo "[2/3] Launching $INSTANCE_TYPE spot instance ..."

INSTANCE_ID=$(aws ec2 run-instances \
    --region "$REGION" \
    --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    --key-name "$CHESS_KEY_NAME" \
    --security-group-ids "$CHESS_SG_ID" \
    --iam-instance-profile "Name=$IAM_PROFILE" \
    --instance-market-options '{
        "MarketType": "spot",
        "SpotOptions": {
            "SpotInstanceType": "one-time",
            "InstanceInterruptionBehavior": "terminate"
        }
    }' \
    --block-device-mappings "[{
        \"DeviceName\": \"/dev/sda1\",
        \"Ebs\": {
            \"VolumeSize\": $VOLUME_SIZE,
            \"VolumeType\": \"gp3\",
            \"DeleteOnTermination\": true
        }
    }]" \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=chess-training}]' \
    --query "Instances[0].InstanceId" \
    --output text)

echo "  Instance ID: $INSTANCE_ID"

# ── Wait for instance to be running ───────────────────────────────────────────
echo ""
echo "[3/3] Waiting for instance to start ..."

aws ec2 wait instance-running --region "$REGION" --instance-ids "$INSTANCE_ID"

PUBLIC_IP=$(aws ec2 describe-instances \
    --region "$REGION" \
    --instance-ids "$INSTANCE_ID" \
    --query "Reservations[0].Instances[0].PublicIpAddress" \
    --output text)

echo ""
echo "=== Instance ready! ==="
echo ""
echo "  Instance ID : $INSTANCE_ID"
echo "  Public IP   : $PUBLIC_IP"
echo "  Cost        : ~\$0.36/hr spot (g5.2xlarge)"
echo ""
echo "SSH command:"
echo "  ssh -i ${KEY_PATH} ubuntu@${PUBLIC_IP}"
echo ""
echo "IMPORTANT: Terminate when done to stop billing:"
echo "  aws ec2 terminate-instances --instance-ids $INSTANCE_ID"
echo ""
echo "Next steps on the instance:"
echo "  1. bash aws/ec2_setup.sh    # download data (~5 min)"
echo "  2. tmux new -s train"
echo "  3. bash aws/train_spot.sh"
echo "  4. Ctrl+B D to detach"

# Save instance ID so terminate reminder is easy
echo "$INSTANCE_ID" > /tmp/chess_training_instance_id.txt
echo "(Instance ID saved to /tmp/chess_training_instance_id.txt)"
