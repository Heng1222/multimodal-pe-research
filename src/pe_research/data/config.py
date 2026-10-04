"""Configuration loading for the AU-PEMal pilot pipeline.

The checked-in ``.yaml`` file deliberately uses JSON syntax. JSON is a YAML 1.2
subset, which keeps this first pipeline free of a runtime YAML dependency.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast


@dataclass(frozen=True)
class SourceConfig:
    name: str
    commit: str
    url: str
    filename: str
    expected_sha256: str
    required_columns: tuple[str, ...]


@dataclass(frozen=True)
class PilotConfig:
    seed: int
    target_benign: int
    target_malicious_per_category: int
    max_candidates: int
    minimum_per_class: int
    minimum_events_per_trace: int


@dataclass(frozen=True)
class VirusTotalConfig:
    base_url: str
    api_key_env: str
    minimum_interval_seconds: float
    request_budget: int
    behaviours_limit: int
    max_retries: int


@dataclass(frozen=True)
class EmbeddingConfig:
    model_name: str
    model_revision: str
    dimension: int
    normalize: bool
    batch_size: int


@dataclass(frozen=True)
class PipelineConfig:
    schema_version: str
    workspace: Path
    source: SourceConfig
    pilot: PilotConfig
    virustotal: VirusTotalConfig
    embedding: EmbeddingConfig


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return cast(dict[str, Any], value)


def load_config(path: Path) -> PipelineConfig:
    """Load and validate the versioned pipeline configuration."""
    raw = _mapping(json.loads(path.read_text(encoding="utf-8")), "config")
    source = _mapping(raw.get("source"), "source")
    pilot = _mapping(raw.get("pilot"), "pilot")
    vt = _mapping(raw.get("virustotal"), "virustotal")
    embedding = _mapping(raw.get("embedding"), "embedding")

    config = PipelineConfig(
        schema_version=str(raw["schema_version"]),
        workspace=Path(str(raw["workspace"])),
        source=SourceConfig(
            name=str(source["name"]),
            commit=str(source["commit"]),
            url=str(source["url"]),
            filename=str(source["filename"]),
            expected_sha256=str(source.get("expected_sha256", "")),
            required_columns=tuple(str(item) for item in source["required_columns"]),
        ),
        pilot=PilotConfig(
            seed=int(pilot["seed"]),
            target_benign=int(pilot["target_benign"]),
            target_malicious_per_category=int(pilot["target_malicious_per_category"]),
            max_candidates=int(pilot["max_candidates"]),
            minimum_per_class=int(pilot["minimum_per_class"]),
            minimum_events_per_trace=int(pilot["minimum_events_per_trace"]),
        ),
        virustotal=VirusTotalConfig(
            base_url=str(vt["base_url"]).rstrip("/"),
            api_key_env=str(vt["api_key_env"]),
            minimum_interval_seconds=float(vt["minimum_interval_seconds"]),
            request_budget=int(vt["request_budget"]),
            behaviours_limit=int(vt["behaviours_limit"]),
            max_retries=int(vt["max_retries"]),
        ),
        embedding=EmbeddingConfig(
            model_name=str(embedding["model_name"]),
            model_revision=str(embedding["model_revision"]),
            dimension=int(embedding["dimension"]),
            normalize=bool(embedding["normalize"]),
            batch_size=int(embedding["batch_size"]),
        ),
    )
    if config.pilot.max_candidates < (
        config.pilot.target_benign + config.pilot.target_malicious_per_category
    ):
        raise ValueError("max_candidates is too small for the requested pilot")
    if config.virustotal.minimum_interval_seconds < 15.0:
        raise ValueError("VirusTotal free-tier interval must be at least 15 seconds")
    return config


def resolve_workspace(config: PipelineConfig, repository_root: Path) -> Path:
    """Resolve a configured workspace without allowing it outside the repository."""
    root = repository_root.resolve()
    candidate = (root / config.workspace).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"workspace escapes repository: {candidate}")
    return candidate
