"""Validated, version-controlled experiment configuration (JSON subset of YAML)."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class PilotConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str = "dual_branch_api_pilot/v1"
    data_config: str = "configs/data/api_traces_pilot_v1.yaml"
    model_config_path: str = "configs/model/dual_branch_ae_v1.yaml"
    seed: int = 42
    split_seed: int = 20261002
    device: str = "cpu"
    batch_size: int = Field(default=32, ge=2)
    warmup_epochs: int = Field(default=2, ge=1, le=3)
    epochs: int = Field(default=50, ge=1)
    patience: int = Field(default=8, ge=1)
    learning_rate: float = Field(default=1e-3, gt=0)
    weight_decay: float = Field(default=1e-4, ge=0)
    gradient_clip: float = Field(default=1.0, gt=0)
    gradient_interval: int = Field(default=10, ge=1)
    target_fpr: float = Field(default=0.01, ge=0, lt=1)
    binary_weight: float = Field(default=1.0, gt=0)
    family_weight: float = Field(default=0.5, gt=0)
    joint_weight: float = Field(default=0.5, gt=0)
    bottleneck_weight: float = Field(default=0.125, gt=0)
    threads: int = Field(default=2, ge=1)


def load_pilot_config(path: Path) -> PilotConfig:
    return PilotConfig.model_validate_json(path.read_text(encoding="utf-8-sig"))
