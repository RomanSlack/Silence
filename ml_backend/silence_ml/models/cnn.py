from __future__ import annotations

import torch
from torch import nn


class EMGClassifier(nn.Module):
    """1D CNN over multichannel EMG windows.

    Input:  (B, C, T) where C = n_channels, T = window length in samples.
    Output: (B, n_classes) logits.
    """

    def __init__(self, n_channels: int = 8, n_classes: int = 20, hidden: int = 64):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv1d(n_channels, hidden, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Conv1d(hidden, hidden * 2, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(hidden * 2),
            nn.GELU(),
            nn.Conv1d(hidden * 2, hidden * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(hidden * 4),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(hidden * 4, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x).squeeze(-1)
        return self.head(h)
