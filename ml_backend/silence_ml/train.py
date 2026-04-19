from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from silence_ml.data.base import EMGDataset
from silence_ml.models import EMGClassifier


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 1e-4
    val_split: float = 0.15
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_dir: Path = Path("checkpoints")


def train(dataset: EMGDataset, n_channels: int, n_classes: int,
          cfg: TrainConfig = TrainConfig()) -> dict:
    cfg.ckpt_dir.mkdir(parents=True, exist_ok=True)
    n_val = max(1, int(len(dataset) * cfg.val_split))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0)
    )

    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    model = EMGClassifier(n_channels=n_channels, n_classes=n_classes).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(cfg.epochs):
        model.train()
        train_loss = 0.0
        for x, y in train_dl:
            x, y = x.to(cfg.device), y.to(cfg.device)
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_ds)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(cfg.device), y.to(cfg.device)
                pred = model(x).argmax(-1)
                correct += (pred == y).sum().item()
                total += y.numel()
        val_acc = correct / max(1, total)

        print(f"[epoch {epoch:02d}] train_loss={train_loss:.4f}  val_acc={val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), cfg.ckpt_dir / "best.pt")

    return {"best_val_acc": best_acc}
