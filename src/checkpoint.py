from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: optim.Optimizer | None = None,
    step: int | None = None,
    loss: float | None = None,
    config: dict[str, Any] | None = None,
) -> None:
    """
    Save model checkpoint.

    Contains:
        - model_state_dict
        - optimizer_state_dict, optional
        - step, optional
        - loss, optional
        - config, optional
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "step": step,
        "loss": loss,
        "config": config,
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(checkpoint, path)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: optim.Optimizer | None = None,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """
    Load model checkpoint.

    Returns checkpoint metadata so caller can inspect:
        - step
        - loss
        - config
    """
    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint