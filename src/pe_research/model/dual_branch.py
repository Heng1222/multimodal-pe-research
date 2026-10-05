"""Dual-branch sequence embedding autoencoder and masked multitask losses."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as functional


def mlp(dims: list[int], *, final_activation: bool = False) -> nn.Sequential:
    layers: list[nn.Module] = []
    for index, (left, right) in enumerate(zip(dims[:-1], dims[1:], strict=True)):
        layers.append(nn.Linear(left, right))
        if index < len(dims) - 2 or final_activation:
            layers.append(nn.GELU())
    return nn.Sequential(*layers)


class FactorClassifier(nn.Module):
    """Inference-only path; classification necessarily passes through c."""

    def __init__(self, dimension: int, families: int) -> None:
        super().__init__()
        self.encoder = mlp([dimension, 256, 128], final_activation=True)
        self.factor = mlp([128, 64, 32])
        self.binary = nn.Linear(32, 1)
        self.family = nn.Linear(32, families)

    def forward(self, x: Any) -> dict[str, Any]:
        c = torch.sigmoid(self.factor(self.encoder(x)))
        return {"c": c, "binary": self.binary(c).squeeze(-1), "family": self.family(c)}


class DualBranchAE(FactorClassifier):
    """Whitepaper v1.2 architecture, parameterized at its input/output boundary."""

    def __init__(self, dimension: int, families: int) -> None:
        super().__init__(dimension, families)
        self.residual = mlp([128, 64, 32])
        self.joint_decoder = mlp([64, 128, dimension])
        self.factor_decoder = mlp([32, 64, dimension])

    def forward(self, x: Any) -> dict[str, Any]:
        h = self.encoder(x)
        c = torch.sigmoid(self.factor(h))
        residual = self.residual(h)
        return {
            "c": c,
            "z_r": residual,
            "binary": self.binary(c).squeeze(-1),
            "family": self.family(c),
            "joint": self.joint_decoder(torch.cat([c, residual], dim=1)),
            "bottleneck": self.factor_decoder(c),
        }


class MLPClassifier(nn.Module):
    """Independent direct-embedding classifier without a factor bottleneck or decoder."""

    def __init__(self, dimension: int, classes: int) -> None:
        super().__init__()
        self.encoder = mlp([dimension, 256, 128], final_activation=True)
        self.head = nn.Linear(128, classes)
        self.binary_task = classes == 1

    def forward(self, x: Any) -> Any:
        logits = self.head(self.encoder(x))
        return logits.squeeze(-1) if self.binary_task else logits


def losses(
    outputs: dict[str, Any],
    x: Any,
    y: Any,
    target: Any,
    known: Any,
    variance: float,
) -> dict[str, Any]:
    """Average family CE over valid families only; never broadcast binary targets."""
    if not math.isfinite(variance) or not variance > 1e-12:
        raise ValueError("degenerate training variance")
    if x.ndim != 2 or outputs["binary"].shape != y.shape or y.shape != (len(x),):
        raise ValueError("invalid input/binary shapes")
    if target.shape != y.shape or known.shape != y.shape:
        raise ValueError("invalid family mask shapes")
    if target.dtype != torch.int64 or known.dtype != torch.bool:
        raise ValueError("invalid family target/mask dtype")
    if x.dtype != torch.float32 or y.dtype != torch.float32:
        raise ValueError("inputs and binary labels must be float32")
    if (
        outputs["family"].ndim != 2
        or outputs["family"].shape[0] != len(x)
        or outputs["joint"].shape != x.shape
        or outputs["bottleneck"].shape != x.shape
    ):
        raise ValueError("invalid family or reconstruction output shapes")
    if not all(torch.isfinite(value).all() for value in [x, y, *outputs.values()]):
        raise ValueError("non-finite model inputs or outputs")
    mask = (y == 1) & known & (target >= 0) & (target < outputs["family"].shape[1])
    family = (
        functional.cross_entropy(outputs["family"][mask], target[mask])
        if mask.any()
        else outputs["family"].sum() * 0
    )
    return {
        "binary": functional.binary_cross_entropy_with_logits(outputs["binary"], y),
        "family": family,
        "joint": functional.mse_loss(outputs["joint"], x) / variance,
        "bottleneck": functional.mse_loss(outputs["bottleneck"], x) / variance,
    }
