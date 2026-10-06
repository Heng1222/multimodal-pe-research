"""Materialize cached source and VT data into immutable, joinable CSV tables."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from pe_research.data.au_pemal_2025.config import PipelineConfig
from pe_research.data.au_pemal_2025.normalization import behavior_coverage, events_from_behavior
from pe_research.data.au_pemal_2025.records import EventRecord
from pe_research.data.au_pemal_2025.virustotal import read_candidates, read_lookup_results
from pe_research.data.io import load_json, write_csv


def _trace_id(sample_key: str, behavior_id: str) -> str:
    return hashlib.sha256(f"{sample_key}:{behavior_id}".encode()).hexdigest()


def _reports(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = load_json(path).get("data", [])
    if not isinstance(data, list):
        return []
    return [cast(dict[str, Any], item) for item in data if isinstance(item, dict)]


def _attributes(report: Mapping[str, Any]) -> dict[str, Any]:
    value = report.get("attributes", {})
    return cast(dict[str, Any], value) if isinstance(value, dict) else {}


def _report_rank(report: Mapping[str, Any]) -> tuple[int, int, int, str]:
    attributes = _attributes(report)
    trace_id = "rank-only"
    event_count = len(events_from_behavior(trace_id, attributes))
    try:
        analysis_date = int(attributes.get("analysis_date", 0))
    except (TypeError, ValueError):
        analysis_date = 0
    sandbox = str(attributes.get("sandbox_name", ""))
    return (-behavior_coverage(attributes), -event_count, -analysis_date, sandbox)


def _concepts(attributes: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    value = attributes.get("mitre_attack_techniques", [])
    if not isinstance(value, list):
        return {}
    result: dict[str, dict[str, str]] = {}
    for item in value:
        if isinstance(item, dict) and item.get("id"):
            technique_id = str(item["id"]).strip()
            tactic_value = item.get("tactics", item.get("tactic", ""))
            if isinstance(tactic_value, list):
                tactic = "|".join(sorted(str(part) for part in tactic_value if part))
            else:
                tactic = str(tactic_value or "")
            result[technique_id] = {
                "technique_name": str(
                    item.get("name", item.get("signature_description", ""))
                ),
                "tactic": tactic,
            }
    return result


def materialize(config: PipelineConfig, workspace: Path) -> dict[str, int]:
    artifacts = workspace / "artifacts"
    source = workspace / "raw" / config.source.filename
    candidates_path = artifacts / "candidates.csv"
    lookup_path = artifacts / "lookup_results.csv"
    candidates = read_candidates(candidates_path)
    lookup = {item.sample_key: item for item in read_lookup_results(lookup_path)}

    samples: list[dict[str, object]] = []
    sample_labels: list[dict[str, object]] = []
    traces: list[dict[str, object]] = []
    events: list[EventRecord] = []
    canonical_concepts: dict[str, set[str]] = {}
    concept_registry: dict[str, dict[str, str]] = {}

    for candidate in candidates:
        result = lookup.get(candidate.sample_key)
        samples.append(
            {
                "sample_key": candidate.sample_key,
                "source_sha1": candidate.source_sha1,
                "sha256": result.canonical_sha256 if result else "",
                "split": candidate.split,
                "lookup_status": result.lookup_status if result else "pending",
                "source_dataset": config.source.name,
                "schema_version": config.schema_version,
            }
        )
        is_benign = candidate.class_name.casefold() == "benign"
        sample_labels.append(
            {
                "sample_key": candidate.sample_key,
                "y": 0 if is_benign else 1,
                "y_known": 1,
                "category": candidate.category,
                "family": candidate.family,
                "label_source": config.source.name,
                "label_confidence": "published_dataset",
            }
        )
        if result is None or not result.canonical_sha256:
            continue
        behavior_path = workspace / "raw" / "vt" / "behaviours" / (
            f"{result.canonical_sha256}.json"
        )
        reports = _reports(behavior_path)
        if not reports:
            continue
        canonical = sorted(reports, key=_report_rank)[0]
        canonical_id = str(canonical.get("id", ""))
        for report in reports:
            behavior_id = str(report.get("id", ""))
            attributes = _attributes(report)
            report_trace_id = _trace_id(candidate.sample_key, behavior_id)
            report_events = events_from_behavior(report_trace_id, attributes)
            ordering_known = bool(report_events) and bool(report_events[0].ordering_known)
            is_canonical = behavior_id == canonical_id
            valid = len(report_events) >= config.pilot.minimum_events_per_trace
            traces.append(
                {
                    "trace_id": report_trace_id,
                    "sample_key": candidate.sample_key,
                    "split": candidate.split,
                    "vt_behavior_id": behavior_id,
                    "sandbox_name": str(attributes.get("sandbox_name", "")),
                    "analysis_date": str(attributes.get("analysis_date", "")),
                    "event_count": len(report_events),
                    "field_coverage": behavior_coverage(attributes),
                    "is_canonical": int(is_canonical),
                    "trace_quality": "valid" if valid else "insufficient_events",
                    "ordering_known": int(ordering_known),
                    "sequence_eligible": int(is_canonical and valid and ordering_known),
                    "raw_report_path": behavior_path.relative_to(workspace).as_posix(),
                    "network_policy": "unknown",
                }
            )
            if is_canonical and valid:
                events.extend(report_events)
                concepts = _concepts(attributes)
                canonical_concepts[report_trace_id] = set(concepts)
                concept_registry.update(concepts)

    concept_labels: list[dict[str, object]] = []
    for trace_id, positives in sorted(canonical_concepts.items()):
        for concept_id in sorted(concept_registry):
            positive = concept_id in positives
            metadata = concept_registry[concept_id]
            concept_labels.append(
                {
                    "trace_id": trace_id,
                    "concept_id": concept_id,
                    "concept_type": "attack_technique",
                    "tactic": metadata["tactic"] if positive else "",
                    "tactic_known": int(positive and bool(metadata["tactic"])),
                    "technique_id": concept_id,
                    "technique_name": metadata["technique_name"] if positive else "",
                    "value": 1 if positive else "",
                    "known_mask": 1 if positive else 0,
                    "evidence_source": "virustotal_behavior" if positive else "",
                    "evidence_scope": "canonical_trace" if positive else "",
                }
            )

    counts = {
        "samples": write_csv(
            artifacts / "samples.csv",
            [
                "sample_key",
                "source_sha1",
                "sha256",
                "split",
                "lookup_status",
                "source_dataset",
                "schema_version",
            ],
            samples,
        ),
        "sample_labels": write_csv(
            artifacts / "sample_labels.csv",
            [
                "sample_key",
                "y",
                "y_known",
                "category",
                "family",
                "label_source",
                "label_confidence",
            ],
            sample_labels,
        ),
        "traces": write_csv(
            artifacts / "traces.csv",
            [
                "trace_id",
                "sample_key",
                "split",
                "vt_behavior_id",
                "sandbox_name",
                "analysis_date",
                "event_count",
                "field_coverage",
                "is_canonical",
                "trace_quality",
                "ordering_known",
                "sequence_eligible",
                "raw_report_path",
                "network_policy",
            ],
            traces,
        ),
        "events": write_csv(
            artifacts / "events.csv",
            list(EventRecord.__dataclass_fields__),
            (event.as_row() for event in events),
        ),
        "concept_labels": write_csv(
            artifacts / "concept_labels.csv",
            [
                "trace_id",
                "concept_id",
                "concept_type",
                "tactic",
                "tactic_known",
                "technique_id",
                "technique_name",
                "value",
                "known_mask",
                "evidence_source",
                "evidence_scope",
            ],
            concept_labels,
        ),
    }
    if not source.exists():
        raise FileNotFoundError(source)
    return counts
