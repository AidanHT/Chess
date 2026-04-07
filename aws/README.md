# AWS Training Setup

Train the AlphaZero chess engine on a `g5.2xlarge` spot instance (NVIDIA A10G, 24 GB VRAM) for ~$0.36/hr.

## Instance specs

| | Local | AWS |
|---|---|---|
| GPU | ~6 GB VRAM | A10G 24 GB VRAM |
| Batch size | 256–512 | **1024** |
| Expected throughput | ~500 pos/s | ~4,000–6,000 pos/s |
| Full 10-epoch run | days | ~20–30 hours (~$10) |

---

## One-Time Prerequisites

### 1. AWS CLI

Install AWS CLI v2 and configure it:

```bash
aws configure
# AWS Access Key ID: ...
# AWS Secret Access Key: ...
# Default region: us-east-1
# Default output format: json
```

### 2. IAM Role for EC2

In the AWS Console: **IAM → Roles → Create Role**
- Trusted entity: **EC2**
- Permissions: paste the contents of `aws/iam_policy.json` as an inline policy
- Role name: `chess-training-role`

### 3. EC2 Key Pair

```bash
aws ec2 create-key-pair \
    --key-name chess-training \
    --query 'KeyMaterial' \
    --output text > ~/.ssh/chess-training.pem

chmod 400 ~/.ssh/chess-training.pem
```

### 4. Security Group (SSH-only)

```bash
# Create the group
SG_ID=$(aws ec2 create-security-group \
    --group-name chess-training-sg \
    --description "Chess training SSH access" \
    --query 'GroupId' --output text)

# Allow SSH from your IP only
MY_IP=$(curl -s https://checkip.amazonaws.com)
aws ec2 authorize-security-group-ingress \
    --group-id "$SG_ID" \
    --protocol tcp --port 22 \
    --cidr "${MY_IP}/32"

echo "Security Group ID: $SG_ID"
```

### 5. Set Environment Variables

Add these to your shell profile (`~/.bashrc` or `~/.zshrc`):

```bash
export CHESS_KEY_NAME="chess-training"
export CHESS_SG_ID="sg-xxxxxxxxxxxxxxxxx"   # from step 4
```

### 6. Upload Training Data (one-time, ~60–90 min)

From the project root:

```bash
bash aws/upload_data.sh
```

This uploads:
- All `.pgn` files from `data/pgn/` (~7.9 GB)
- Any existing checkpoints from `checkpoints/`

---

## Per-Training-Session Workflow

### Step 1: Launch the spot instance (~30 seconds)

```bash
bash aws/launch_spot.sh
```

Prints the instance ID, public IP, and SSH command.

### Step 2: Upload source code (if not on GitHub)

If your repo is private or not on GitHub, pack and upload the source:

```bash
# From project root (local machine):
tar -czf chess-engine.tar.gz \
    --exclude='data' \
    --exclude='checkpoints' \
    --exclude='*.zst' \
    --exclude='__pycache__' \
    --exclude='.git' \
    .

aws s3 cp chess-engine.tar.gz s3://chess-az-training/code/
```

### Step 3: SSH in and run setup (~5–10 min)

```bash
ssh -i ~/.ssh/chess-training.pem ubuntu@<PUBLIC_IP>

# On the instance:
# Copy the setup script from S3 (or paste it manually)
aws s3 cp s3://chess-az-training/code/ec2_setup.sh ~/
bash ec2_setup.sh
```

If your repo is on GitHub, edit `ec2_setup.sh` to uncomment the `git clone` line.

### Step 4: Log in to Weights & Biases

```bash
wandb login
# Paste your API key from wandb.ai/settings
```

Or set it as an env variable instead (no interactive prompt):

```bash
export WANDB_API_KEY="your-api-key"
```

### Step 5: Start training in tmux

```bash
tmux new -s train
bash aws/train_spot.sh
```

Detach (keep running after SSH disconnect):

```
Ctrl+B, then D
```

Re-attach later:

```bash
ssh -i ~/.ssh/chess-training.pem ubuntu@<PUBLIC_IP>
tmux attach -t train
```

### Step 6: Monitor

- **Weights & Biases**: your dashboard at wandb.ai shows live loss curves
- **SSH tail**: `ssh ubuntu@<IP> -t "tail -f ~/training.log"`

### Step 7: Download checkpoints

From your local machine (checkpoints are auto-synced to S3):

```bash
bash aws/download_results.sh
```

### Step 8: TERMINATE THE INSTANCE (stops billing)

```bash
aws ec2 terminate-instances --instance-ids i-xxxxxxxxxxxxxxxxx
```

The instance ID was printed by `launch_spot.sh` and saved to `/tmp/chess_training_instance_id.txt`.

---

## Spot Interruption Handling

`train_spot.sh` has three layers of protection:

1. **Background sync every 5 min** — pushes new checkpoints to S3 independently of training
2. **Metadata watcher** — polls the EC2 interruption endpoint every 5 seconds; syncs immediately when a 2-minute warning is issued
3. **SIGTERM trap** — fires when EC2 sends the termination signal; runs a final `aws s3 sync` before the instance is reclaimed

On the next session, `train_spot.sh` auto-detects the latest checkpoint in `~/checkpoints/` and passes `--resume` automatically.

---

## Cost Reference

| Item | Cost |
|---|---|
| g5.2xlarge spot (us-east-1) | ~$0.36/hr |
| Full 10-epoch run (~25 hrs) | ~$9 |
| EBS 150 GB gp3 (per day) | ~$0.40 |
| S3 storage 30 GB (per month) | ~$0.70 |
| S3 ↔ EC2 transfer (same region) | **Free** |

---

## Troubleshooting

**`g5.2xlarge` capacity not available in us-east-1:**
Try `us-east-2` or `us-west-2`. Change `REGION` in `launch_spot.sh`.

**OOM on batch_size=1024:**
Reduce to 512 in `train_spot.sh` and halve the learning rate to `1e-3`.

**`find_pgn_files` returns 0 files:**
Check that S3 upload excluded `.zst` files and that `~/data/pgn/*.pgn` exists on the instance.

**`wandb` not logging:**
Set `--wandb_disabled` in `train_spot.sh` to skip W&B entirely and rely on the tqdm output + `training.log`.
