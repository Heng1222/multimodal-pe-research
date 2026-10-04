"""Versioned configuration for the Zenodo API-trace pilot."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast


@dataclass(frozen=True)
class ApiTraceSourceConfig:
    record_id: str
    version: str
    metadata_url: str
    metadata_filename: str
    metadata_md5: str
    archive_url: str
    archive_filename: str
    archive_md5: str
    minimum_free_gib: float


@dataclass(frozen=True)
class ApiTracePilotConfig:
    seed: int
    target_benign: int
    family_count: int
    target_per_family: int
    expected_families: tuple[str, ...]
    candidate_multiplier: int
    minimum_events: int
    maximum_events: int
    maximum_trace_bytes: int
    maximum_parse_error_fraction: float


@dataclass(frozen=True)
class SampleEmbeddingConfig:
    model_name: str
    model_revision: str
    dimension: int
    maximum_tokens: int
    batch_size: int
    normalize: bool
    pooling: str
    input_format: str
    device: str


@dataclass(frozen=True)
class ApiTraceConfig:
    schema_version: str
    workspace: Path
    source: ApiTraceSourceConfig
    pilot: ApiTracePilotConfig
    sample_embedding: SampleEmbeddingConfig


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return cast(dict[str, Any], value)


def load_api_trace_config(path: Path) -> ApiTraceConfig:
    """Load the JSON-compatible YAML config and enforce pilot invariants."""
    raw = _mapping(json.loads(path.read_text(encoding="utf-8")), "config")
    source = _mapping(raw.get("source"), "source")
    pilot = _mapping(raw.get("pilot"), "pilot")
    sample_embedding = _mapping(raw.get("sample_embedding"), "sample_embedding")
    config = ApiTraceConfig(
        schema_version=str(raw["schema_version"]),
        workspace=Path(str(raw["workspace"])),
        source=ApiTraceSourceConfig(
            record_id=str(source["record_id"]),
            version=str(source["version"]),
            metadata_url=str(source["metadata_url"]),
            metadata_filename=str(source["metadata_filename"]),
            metadata_md5=str(source["metadata_md5"]).lower(),
            archive_url=str(source["archive_url"]),
            archive_filename=str(source["archive_filename"]),
            archive_md5=str(source["archive_md5"]).lower(),
            minimum_free_gib=float(source["minimum_free_gib"]),
        ),
        pilot=ApiTracePilotConfig(
            seed=int(pilot["seed"]),
            target_benign=int(pilot["target_benign"]),
            family_count=int(pilot["family_count"]),
            target_per_family=int(pilot["target_per_family"]),
            expected_families=tuple(str(value) for value in pilot["expected_families"]),
            candidate_multiplier=int(pilot["candidate_multiplier"]),
            minimum_events=int(pilot["minimum_events"]),
            maximum_events=int(pilot["maximum_events"]),
            maximum_trace_bytes=int(pilot["maximum_trace_bytes"]),
            maximum_parse_error_fraction=float(pilot["maximum_parse_error_fraction"]),
        ),
        sample_embedding=SampleEmbeddingConfig(
            model_name=str(sample_embedding["model_name"]),
            model_revision=str(sample_embedding["model_revision"]),
            dimension=int(sample_embedding["dimension"]),
            maximum_tokens=int(sample_embedding["maximum_tokens"]),
            batch_size=int(sample_embedding["batch_size"]),
            normalize=bool(sample_embedding["normalize"]),
            pooling=str(sample_embedding["pooling"]),
            input_format=str(sample_embedding["input_format"]),
            device=str(sample_embedding["device"]),
        ),
    )
    if config.pilot.target_benign != (
        config.pilot.family_count * config.pilot.target_per_family
    ):
        raise ValueError("pilot must balance benign and total malicious samples")
    if len(config.pilot.expected_families) != config.pilot.family_count:
        raise ValueError("expected_families must contain exactly family_count values")
    if config.pilot.candidate_multiplier < 1:
        raise ValueError("candidate_multiplier must be positive")
    if config.pilot.minimum_events > config.pilot.maximum_events:
        raise ValueError("minimum_events cannot exceed maximum_events")
    if config.sample_embedding.dimension <= 0:
        raise ValueError("sample embedding dimension must be positive")
    if config.sample_embedding.maximum_tokens <= 0:
        raise ValueError("sample embedding maximum_tokens must be positive")
    if config.sample_embedding.batch_size <= 0:
        raise ValueError("sample embedding batch_size must be positive")
    if config.sample_embedding.pooling != "last_hidden_state_cls":
        raise ValueError("sample embedding pooling must be last_hidden_state_cls")
    if config.sample_embedding.input_format != "api_name":
        raise ValueError("sample embedding input_format must be api_name")
    return config


def resolve_api_workspace(config: ApiTraceConfig, repository_root: Path) -> Path:
    root = repository_root.resolve()
    workspace = (root / config.workspace).resolve()
    if workspace != root and root not in workspace.parents:
        raise ValueError(f"workspace escapes repository: {workspace}")
    return workspace
