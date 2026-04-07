"""
Dual-headed ResNet for the AlphaZero-style chess engine.

Input:  (B, 119, 8, 8) float32 tensor  — channels-first layout expected by Conv2d.
        Produced by transposing the (8, 8, 119) output of state_encoding.encode_board().

Output: (policy_logits, value)
    policy_logits : (B, 4672)  raw logits; pair with F.cross_entropy during training.
    value         : (B,)       scalar in (-1, 1) via tanh.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..core import ACTION_SPACE_SIZE, BOARD_SIZE, TOTAL_PLANES


# ─── Residual Block ───────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """
    Standard pre-activation residual block used in AlphaZero:
      Conv → BN → ReLU → Conv → BN → (+ skip) → ReLU
    """

    def __init__(self, num_filters: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(num_filters)
        self.relu  = nn.ReLU(inplace=True)

        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.relu(x + residual)
        return x


# ─── Main Network ─────────────────────────────────────────────────────────────

class ChessResNet(nn.Module):
    """
    AlphaZero-style dual-headed ResNet for chess.

    Parameters
    ----------
    num_blocks : int
        Number of residual blocks in the tower (AlphaZero uses 20 or 40).
    num_filters : int
        Channel width throughout the residual tower (AlphaZero uses 256).
    """

    def __init__(self, num_blocks: int = 20, num_filters: int = 256) -> None:
        super().__init__()

        # ── Input block ───────────────────────────────────────────────────────
        self.input_block = nn.Sequential(
            nn.Conv2d(TOTAL_PLANES, num_filters, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True),
        )
        nn.init.kaiming_normal_(
            self.input_block[0].weight, nonlinearity="relu"
        )

        # ── Residual tower ────────────────────────────────────────────────────
        self.tower = nn.Sequential(
            *[ResidualBlock(num_filters) for _ in range(num_blocks)]
        )

        # ── Policy head ───────────────────────────────────────────────────────
        # 2 channels → flatten → 4672 logits (no softmax; use CrossEntropyLoss)
        policy_conv_channels = 2
        self.policy_conv = nn.Conv2d(num_filters, policy_conv_channels, kernel_size=1, bias=False)
        self.policy_bn   = nn.BatchNorm2d(policy_conv_channels)
        self.policy_relu = nn.ReLU(inplace=True)
        self.policy_fc   = nn.Linear(
            policy_conv_channels * BOARD_SIZE * BOARD_SIZE, ACTION_SPACE_SIZE
        )
        nn.init.kaiming_normal_(self.policy_conv.weight, nonlinearity="relu")

        # ── Value head ────────────────────────────────────────────────────────
        # 1 channel → flatten → 256 → 1 → tanh
        value_conv_channels = 1
        self.value_conv  = nn.Conv2d(num_filters, value_conv_channels, kernel_size=1, bias=False)
        self.value_bn    = nn.BatchNorm2d(value_conv_channels)
        self.value_relu1 = nn.ReLU(inplace=True)
        self.value_fc1   = nn.Linear(value_conv_channels * BOARD_SIZE * BOARD_SIZE, 256)
        self.value_relu2 = nn.ReLU(inplace=True)
        self.value_fc2   = nn.Linear(256, 1)
        nn.init.kaiming_normal_(self.value_conv.weight, nonlinearity="relu")

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (B, 119, 8, 8), dtype float32.

        Returns
        -------
        policy_logits : torch.Tensor  shape (B, 4672)
        value         : torch.Tensor  shape (B,)
        """
        x = self.input_block(x)
        x = self.tower(x)

        # Policy head
        p = self.policy_relu(self.policy_bn(self.policy_conv(x)))
        p = p.flatten(start_dim=1)
        policy_logits = self.policy_fc(p)

        # Value head
        v = self.value_relu1(self.value_bn(self.value_conv(x)))
        v = v.flatten(start_dim=1)
        v = self.value_relu2(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v)).squeeze(-1)  # (B, 1) → (B,)

        return policy_logits, value
