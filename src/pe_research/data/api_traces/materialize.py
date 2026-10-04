"""Stream the large archive and materialize only a clean ordered pilot."""

from __future__ import annotations

import json
import os
import re
import shutil
import tarfile
from collections import Counter, defaultdict
from collections.abc import Iterator
from pathlib import Path, PurePosixPath
from typing import Any, cast

from pe_research.data.api_traces.config import ApiTraceConfig
from pe_research.data.api_traces.normalization import (
    normalize_api_event,
    parse_trace_payload,
    timestamp_sort_value,
)
from pe_research.data.api_traces.source import deterministic_split, update_candidate_statuses
from pe_research.data.io import atomic_json, read_csv, write_csv

_SHA_IN_NAME = re.compile(r"(?i)(?<![0-9a-f])([0-9a-f]{64})(?![0-9a-f])")


def _safe_member_sha(name: str) -> str | None:
    normalized = name.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts:
        return None
    match = _SHA_IN_NAME.search(path.name)
    return match.group(1).lower() if match else None


def _analyze_trace(
    payload: bytes,
    config: ApiTraceConfig,
) -> dict[str, Any]:
    records, parse_errors = parse_trace_payload(payload)
    total_items = len(records) + parse_errors
    error_fraction = parse_errors / total_items if total_items else 1.0
    process_tokens: dict[str, str] = {}
    thread_tokens: dict[str, str] = {}
    valid: list[dict[str, object]] = []
    missing_function = 0
    for record in records:
        event = normalize_api_event(record, process_tokens, thread_tokens)
        if event is None:
            missing_function += 1
            continue
        valid.append(event)

    reason = ""
    if not records:
        reason = "empty_or_unparseable_trace"
    elif error_fraction > config.pilot.maximum_parse_error_fraction:
        reason = "parse_error_fraction_exceeded"
    elif len(valid) < config.pilot.minimum_events:
        reason = "insufficient_valid_events"
    kept = valid[: config.pilot.maximum_events] if not reason else []
    timestamps = [
        timestamp_sort_value(str(event["vmi_ts"] or event["collector_ts"]))
        for event in kept
    ]
    known_timestamps = [value for value in timestamps if value is not None]
    monotonic_violations = sum(
        current < previous
        for previous, current in zip(known_timestamps, known_timestamps[1:], strict=False)
    )
    return {
        "status": "valid" if not reason else "rejected",
        "failure_reason": reason,
        "raw_record_count": len(records),
        "parse_error_count": parse_errors,
        "missing_function_count": missing_function,
        "valid_event_count": len(valid),
        "kept_event_count": len(kept),
        "truncated": int(len(valid) > config.pilot.maximum_events),
        "timestamp_coverage": (
            sum(value is not None for value in timestamps) / len(kept) if kept else 0.0
        ),
        "timestamp_monotonic_violations": monotonic_violations,
        "events": kept,
    }


def _cache_trace(
    sha256: str,
    payload: bytes,
    analysis: dict[str, Any],
    cache: Path,
) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    raw_part = cache / f"{sha256}.json.part"
    raw_path = cache / f"{sha256}.json"
    raw_part.write_bytes(payload)
    os.replace(raw_part, raw_path)
    atomic_json(cache / f"{sha256}.analysis.json", analysis)


def _load_cached_analysis(sha256: str, cache: Path) -> dict[str, Any] | None:
    path = cache / f"{sha256}.analysis.json"
    raw = cache / f"{sha256}.json"
    if not path.exists() or not raw.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    return cast(dict[str, Any], value) if isinstance(value, dict) else None


def _scan_candidates(
    archive: Path,
    candidates: list[dict[str, str]],
    config: ApiTraceConfig,
    cache: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, tuple[str, str]], dict[str, int]]:
    wanted = {row["source_sha256"] for row in candidates}
    analyses: dict[str, dict[str, Any]] = {}
    statuses: dict[str, tuple[str, str]] = {}
    counters: Counter[str] = Counter()

    for sha in sorted(wanted):
        cached = _load_cached_analysis(sha, cache)
        if cached is not None:
            analyses[sha] = cached
            statuses[sha] = (
                "eligible" if cached.get("status") == "valid" else "rejected",
                str(cached.get("failure_reason", "")),
            )
            counters["cache_hits"] += 1

    seen: set[str] = set()
    with tarfile.open(archive, mode="r|xz") as stream:
        for member in stream:
            if not member.isfile():
                continue
            member_sha = _safe_member_sha(member.name)
            if member_sha is None:
                counters["unrecognized_or_unsafe_members"] += 1
                continue
            sha = member_sha
            if sha not in wanted:
                continue
            if sha in seen:
                counters["duplicate_candidate_members"] += 1
                continue
            seen.add(sha)
            if sha in analyses:
                continue
            if member.size > config.pilot.maximum_trace_bytes:
                statuses[sha] = ("rejected", "trace_too_large")
                counters["trace_too_large"] += 1
                continue
            extracted = stream.extractfile(member)
            if extracted is None:
                statuses[sha] = ("rejected", "archive_member_unreadable")
                continue
            payload = extracted.read(config.pilot.maximum_trace_bytes + 1)
            if len(payload) > config.pilot.maximum_trace_bytes:
                statuses[sha] = ("rejected", "trace_too_large")
                continue
            analysis = _analyze_trace(payload, config)
            analyses[sha] = analysis
            _cache_trace(sha, payload, analysis, cache)
            status = "eligible" if analysis["status"] == "valid" else "rejected"
            statuses[sha] = (status, str(analysis["failure_reason"]))
            counters[status] += 1
    for sha in wanted - set(statuses):
        statuses[sha] = ("missing", "archive_member_missing")
    return analyses, statuses, dict(counters)


def _select_final(
    candidates: list[dict[str, str]],
    analyses: dict[str, dict[str, Any]],
    config: ApiTraceConfig,
) -> list[dict[str, str]]:
    by_family: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in candidates:
        analysis = analyses.get(row["source_sha256"])
        if analysis is not None and analysis.get("status") == "valid":
            by_family[row["family"].casefold()].append(row)

    family_order = ["benign", *(name.casefold() for name in config.pilot.expected_families)]
    selected: list[dict[str, str]] = []
    shortages: dict[str, dict[str, int]] = {}
    for family_index, family in enumerate(family_order):
        target = (
            config.pilot.target_benign
            if family == "benign"
            else config.pilot.target_per_family
        )
        available = sorted(by_family[family], key=lambda row: int(row["candidate_rank"]))
        if len(available) < target:
            shortages[family] = {"required": target, "eligible": len(available)}
            continue
        chosen = available[:target]
        train_count = max(1, int(target * 0.70))
        remaining = target - train_count
        if family == "benign" or (family_index - 1) % 2:
            validation_count = remaining // 2
        else:
            validation_count = (remaining + 1) // 2
        test_count = remaining - validation_count
        split_by_sha = deterministic_split(
            [row["source_sha256"] for row in chosen],
            seed=config.pilot.seed,
            stratum=family,
            validation_count=validation_count,
            test_count=test_count,
        )
        for row in chosen:
            selected.append({**row, "split": split_by_sha[row["source_sha256"]]})
    if shortages:
        raise ValueError(f"eligible candidate quota not met: {shortages}")
    return selected


def _event_rows(
    selected: list[dict[str, str]],
    analyses: dict[str, dict[str, Any]],
) -> Iterator[dict[str, object]]:
    for item in selected:
        sample_id = item["sample_id"]
        trace_id = f"trace_{sample_id}"
        events = cast(list[dict[str, object]], analyses[item["source_sha256"]]["events"])
        for index, event in enumerate(events):
            yield {
                "sample_id": sample_id,
                "trace_id": trace_id,
                "event_index": index,
                **event,
                "ordering_known": 1,
            }


def materialize_api_traces(config: ApiTraceConfig, workspace: Path) -> dict[str, int]:
    """Scan, validate, select, and write the final label-separated dataset."""
    artifacts = workspace / "artifacts"
    candidates_path = artifacts / "candidate_manifest.csv"
    archive = workspace / "raw" / config.source.archive_filename
    if not candidates_path.exists():
        raise FileNotFoundError("candidate_manifest.csv is missing; run select first")
    if not archive.exists():
        raise FileNotFoundError("source archive is missing; run fetch first")
    candidates = read_csv(candidates_path)
    cache = workspace / "cache" / "candidate_traces"
    analyses, statuses, scan_counts = _scan_candidates(
        archive, candidates, config, cache
    )
    update_candidate_statuses(candidates_path, statuses)
    try:
        selected = _select_final(candidates, analyses, config)
    except ValueError as error:
        atomic_json(
            artifacts / "materialization_state.json",
            {"status": "failed", "error": str(error), "scan_counts": scan_counts},
        )
        raise

    family_order = {"benign": 0}
    family_order.update(
        {
            family.casefold(): index + 1
            for index, family in enumerate(config.pilot.expected_families)
        }
    )
    selected.sort(
        key=lambda row: (family_order[row["family"].casefold()], int(row["candidate_rank"]))
    )
    for index, row in enumerate(selected, start=1):
        row["sample_id"] = f"api_trace_{index:06d}"
        statuses[row["source_sha256"]] = ("selected", "")
    update_candidate_statuses(candidates_path, statuses)

    samples = [
        {
            "sample_id": row["sample_id"],
            "source_sha256": row["source_sha256"],
            "split": row["split"],
            "collection_status": "success",
            "candidate_rank": row["candidate_rank"],
        }
        for row in selected
    ]
    labels = [
        {
            "sample_id": row["sample_id"],
            "y": 0 if row["class_name"] == "benign" else 1,
            "y_known": 1,
            "family": "" if row["class_name"] == "benign" else row["family"],
            "family_known": 0 if row["class_name"] == "benign" else 1,
            "label_source": config.source.metadata_filename,
            "confidence": "source_provided_single_label",
        }
        for row in selected
    ]
    traces: list[dict[str, object]] = []
    selected_raw = workspace / "raw" / "selected_traces"
    selected_raw.mkdir(parents=True, exist_ok=True)
    for row in selected:
        analysis = analyses[row["source_sha256"]]
        traces.append(
            {
                "trace_id": f"trace_{row['sample_id']}",
                "sample_id": row["sample_id"],
                "split": row["split"],
                "raw_record_count": analysis["raw_record_count"],
                "parse_error_count": analysis["parse_error_count"],
                "missing_function_count": analysis["missing_function_count"],
                "valid_event_count": analysis["valid_event_count"],
                "event_count": analysis["kept_event_count"],
                "timestamp_coverage": format(float(analysis["timestamp_coverage"]), ".9g"),
                "timestamp_monotonic_violations": analysis["timestamp_monotonic_violations"],
                "ordering_source": "source_json_line_order",
                "ordering_known": 1,
                "truncated": analysis["truncated"],
                "sequence_eligible": 1,
                "trace_quality": "valid",
            }
        )
        shutil.copyfile(
            cache / f"{row['source_sha256']}.json",
            selected_raw / f"{row['source_sha256']}.json",
        )

    write_csv(
        artifacts / "samples.csv",
        ["sample_id", "source_sha256", "split", "collection_status", "candidate_rank"],
        samples,
    )
    write_csv(
        artifacts / "sample_labels.csv",
        [
            "sample_id",
            "y",
            "y_known",
            "family",
            "family_known",
            "label_source",
            "confidence",
        ],
        labels,
    )
    write_csv(
        artifacts / "traces.csv",
        [
            "trace_id",
            "sample_id",
            "split",
            "raw_record_count",
            "parse_error_count",
            "missing_function_count",
            "valid_event_count",
            "event_count",
            "timestamp_coverage",
            "timestamp_monotonic_violations",
            "ordering_source",
            "ordering_known",
            "truncated",
            "sequence_eligible",
            "trace_quality",
        ],
        traces,
    )
    event_count = write_csv(
        artifacts / "events.csv",
        [
            "sample_id",
            "trace_id",
            "event_index",
            "api_name",
            "module_name",
            "process_token",
            "thread_token",
            "parameters_normalized",
            "canonical_text",
            "collector_ts",
            "vmi_ts",
            "function_known",
            "module_known",
            "parameters_known",
            "timestamp_known",
            "ordering_known",
        ],
        _event_rows(selected, analyses),
    )
    counts = {
        "samples": len(samples),
        "traces": len(traces),
        "events": event_count,
        "benign": sum(row["class_name"] == "benign" for row in selected),
        "malicious": sum(row["class_name"] == "malicious" for row in selected),
    }
    atomic_json(
        artifacts / "materialization_state.json",
        {
            "status": "complete",
            "counts": counts,
            "scan_counts": scan_counts,
            "failure_reasons": dict(
                Counter(reason for _status, reason in statuses.values() if reason)
            ),
        },
    )
    return counts
