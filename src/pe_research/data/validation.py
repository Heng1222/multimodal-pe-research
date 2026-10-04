"""Artifact validation, quality reporting, and immutable snapshot generation."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from pe_research.data.config import PipelineConfig
from pe_research.data.io import (
    atomic_json,
    load_json,
    package_version,
    read_csv,
    read_csv_header,
    sha256_file,
)
from pe_research.data.normalization import NORMALIZATION_VERSION

FORBIDDEN_EMBEDDING_COLUMNS = {
    "y",
    "class",
    "category",
    "family",
    "sha1",
    "sha256",
    "source_dataset",
    "sandbox_name",
}


def _keys(rows: list[dict[str, str]]) -> set[tuple[str, str]]:
    return {(row["trace_id"], row["event_index"]) for row in rows}


def _ledger_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def validate_artifacts(
    config: PipelineConfig,
    workspace: Path,
    *,
    require_minimum: bool = False,
) -> tuple[list[str], dict[str, Any]]:
    artifacts = workspace / "artifacts"
    required = {
        name: artifacts / name
        for name in (
            "samples.csv",
            "source_features.csv",
            "traces.csv",
            "events.csv",
            "sample_labels.csv",
            "concept_labels.csv",
            "event_embeddings_minilm384.csv",
        )
    }
    errors = [f"missing artifact: {path}" for path in required.values() if not path.exists()]
    if errors:
        return errors, {"valid": False, "errors": errors}

    samples = read_csv(required["samples.csv"])
    labels = read_csv(required["sample_labels.csv"])
    traces = read_csv(required["traces.csv"])
    events = read_csv(required["events.csv"])
    embeddings = read_csv(required["event_embeddings_minilm384.csv"])
    source_features = read_csv(required["source_features.csv"])

    sample_keys = {row["sample_key"] for row in samples}
    label_keys = {row["sample_key"] for row in labels}
    if sample_keys != label_keys:
        errors.append("samples.csv and sample_labels.csv keys differ")
    if source_features:
        feature_columns = {column.casefold() for column in source_features[0]}
        leaked = feature_columns & {"sha1", "sha256", "class", "category", "family", "y"}
        if leaked:
            errors.append(f"source feature CSV contains labels or raw hashes: {sorted(leaked)}")

    sample_splits = {row["sample_key"]: row["split"] for row in samples}
    seen_trace_splits: dict[str, set[str]] = defaultdict(set)
    for trace in traces:
        seen_trace_splits[trace["sample_key"]].add(trace["split"])
        expected = sample_splits.get(trace["sample_key"])
        if expected is None or expected != trace["split"]:
            errors.append(f"trace split mismatch for {trace['sample_key']}")
    if any(len(splits) != 1 for splits in seen_trace_splits.values()):
        errors.append("a sample appears in multiple trace splits")

    if _keys(events) != _keys(embeddings):
        errors.append("events and embeddings do not have a one-to-one key match")
    embedding_columns = list(embeddings[0]) if embeddings else read_csv_header(
        required["event_embeddings_minilm384.csv"]
    )
    lowered = {column.casefold() for column in embedding_columns}
    forbidden = lowered & FORBIDDEN_EMBEDDING_COLUMNS
    if forbidden:
        errors.append(f"embedding CSV contains forbidden columns: {sorted(forbidden)}")
    dimensions = [column for column in embedding_columns if column.startswith("embedding_")]
    if len(dimensions) != config.embedding.dimension:
        errors.append(
            f"embedding dimension mismatch: expected {config.embedding.dimension}, "
            f"got {len(dimensions)}"
        )
    for row in embeddings:
        try:
            if any(not math.isfinite(float(row[column])) for column in dimensions):
                errors.append("embedding CSV contains NaN or infinite values")
                break
        except ValueError:
            errors.append("embedding CSV contains a non-numeric value")
            break

    successful = {row["sample_key"] for row in samples if row["lookup_status"] == "success"}
    label_by_key = {row["sample_key"]: row for row in labels}
    class_success = Counter(
        "benign" if label_by_key[key]["y"] == "0" else "malicious" for key in successful
    )
    canonical_valid = [
        row
        for row in traces
        if row["is_canonical"] == "1" and row["trace_quality"] == "valid"
    ]
    below_minimum = [
        row["trace_id"]
        for row in canonical_valid
        if int(row["event_count"]) < config.pilot.minimum_events_per_trace
    ]
    if below_minimum:
        errors.append("a canonical valid trace is below the event minimum")
    if require_minimum:
        for class_name in ("benign", "malicious"):
            if class_success[class_name] < config.pilot.minimum_per_class:
                errors.append(
                    f"minimum dynamic coverage not met for {class_name}: "
                    f"{class_success[class_name]}/{config.pilot.minimum_per_class}"
                )

    status_counts = Counter(row["lookup_status"] for row in samples)
    enrichment_state_path = artifacts / "enrichment_state.json"
    source_validation_path = artifacts / "source_validation.json"
    enrichment_state = (
        load_json(enrichment_state_path) if enrichment_state_path.exists() else {}
    )
    source_validation = (
        load_json(source_validation_path) if source_validation_path.exists() else {}
    )
    sample_count = len(samples)
    quality: dict[str, Any] = {
        "valid": not errors,
        "errors": errors,
        "counts": {
            "samples": len(samples),
            "traces": len(traces),
            "canonical_valid_traces": len(canonical_valid),
            "events": len(events),
            "embeddings": len(embeddings),
        },
        "successful_dynamic_samples": dict(sorted(class_success.items())),
        "lookup_status": dict(sorted(status_counts.items())),
        "failure_reasons": {
            "pending_not_enriched": status_counts["pending"],
            "hash_miss": status_counts["not_found"],
            "no_behavior": status_counts["no_behaviour"],
            "trace_quality_insufficient": status_counts["insufficient_events"],
            "permission_or_authentication": int(
                enrichment_state.get("reason") == "authentication_or_permission"
            ),
            "quota_exhausted": int(enrichment_state.get("reason") == "quota"),
        },
        "missingness": {
            "dynamic_report_fraction": (
                1.0 - (len(successful) / sample_count) if sample_count else 1.0
            ),
            "ordering_unknown_fraction_among_canonical": (
                1.0
                - (sum(row["ordering_known"] == "1" for row in canonical_valid)
                   / len(canonical_valid))
                if canonical_valid
                else 1.0
            ),
        },
        "ordering": {
            "known_canonical_traces": sum(
                row["ordering_known"] == "1" for row in canonical_valid
            ),
            "sequence_eligible_traces": sum(
                row["sequence_eligible"] == "1" for row in canonical_valid
            ),
        },
        "virustotal_requests_recorded": _ledger_count(workspace / "logs" / "vt_requests.jsonl"),
        "enrichment_state": enrichment_state,
        "source_validation": source_validation,
    }
    return errors, quality


def write_snapshot(config: PipelineConfig, workspace: Path, quality: dict[str, Any]) -> None:
    artifacts = workspace / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    files = sorted(
        path
        for path in artifacts.iterdir()
        if path.is_file() and path.name not in {"snapshot.json", "quality_report.json"}
    )
    source_path = workspace / "raw" / config.source.filename
    source_checksum = sha256_file(source_path) if source_path.exists() else ""
    payload = {
        "schema_version": config.schema_version,
        "source": {
            "name": config.source.name,
            "commit": config.source.commit,
            "url": config.source.url,
            "sha256": source_checksum,
        },
        "embedding": {
            "model_name": config.embedding.model_name,
            "model_revision": config.embedding.model_revision,
            "dimension": config.embedding.dimension,
            "normalized": config.embedding.normalize,
            "normalization_version": NORMALIZATION_VERSION,
        },
        "dependencies": {
            "python_package": package_version("multimodal-pe-research"),
            "sentence_transformers": package_version("sentence-transformers"),
            "torch": package_version("torch"),
        },
        "artifacts": {
            path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in files
        },
        "quality": quality,
    }
    atomic_json(artifacts / "snapshot.json", payload)
    atomic_json(artifacts / "quality_report.json", quality)
