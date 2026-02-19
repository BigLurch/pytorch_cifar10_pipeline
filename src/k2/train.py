import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass
class TrainConfig:
    epochs: int = 5
    lr: float = 1e-3
    weight_decay: float = 0.0
    device: str = "auto"
    seed: int = 42
    save_dir: Path = Path("outputs")
    run_name: str = "exp1"


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> Dict[str, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)

    return {
        "loss": total_loss / max(total, 1),
        "acc": correct / max(total, 1),
    }


def train(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    cfg: TrainConfig,
) -> Dict[str, float]:
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    model = model.to(device)

    cfg.save_dir.mkdir(parents=True, exist_ok=True)
    run_dir = cfg.save_dir / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    best_acc = -1.0
    best_path = run_dir / "best.pt"

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs}", leave=False)

        running_loss = 0.0
        seen = 0

        for x, y in pbar:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            seen += x.size(0)
            pbar.set_postfix(train_loss=running_loss / max(seen, 1))

        metrics = evaluate(model, test_loader, device)
        print(
            f"Epoch {epoch}: test_loss={metrics['loss']:.4f} test_acc={metrics['acc']:.4f}"
        )

        if metrics["acc"] > best_acc:
            best_acc = metrics["acc"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "cfg": cfg.__dict__,
                    "best_acc": best_acc,
                },
                best_path,
            )

    import json

    # Save metrics for DVC tracking
    metrics_dir = Path("metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / f"{cfg.run_name}.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_acc": best_acc,
                "epochs": cfg.epochs,
                "lr": cfg.lr,
                "weight_decay": cfg.weight_decay,
                "run_name": cfg.run_name,
            },
            f,
            indent=2,
        )

    print(f"Saved metrics to {metrics_path}")

    return {"best_acc": best_acc, "checkpoint": str(best_path)}
