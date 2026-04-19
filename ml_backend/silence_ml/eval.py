from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from silence_ml.data.base import EMGDataset
from silence_ml.models import EMGClassifier


@torch.no_grad()
def evaluate(model: EMGClassifier, dataset: EMGDataset, device: str = "cpu",
             batch_size: int = 128) -> dict:
    model.eval().to(device)
    dl = DataLoader(dataset, batch_size=batch_size)
    correct = total = 0
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(-1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return {"accuracy": correct / max(1, total), "n": total}
