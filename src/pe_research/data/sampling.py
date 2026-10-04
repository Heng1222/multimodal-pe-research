"""Deterministic, class-balanced candidate selection."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict, deque
from collections.abc import Iterable
from pathlib import Path

from pe_research.data.io import atomic_json, read_csv, write_csv
from pe_research.data.records import CandidateRecord

SHA1_PATTERN = re.compile(r"^[0-9a-f]{40}$")


def stable_split(sha1: str) -> str:
    bucket = int(sha1[:8], 16) % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "validation"
    return "test"


def _rank(seed: int, sha1: str) -> str:
    return hashlib.sha256(f"{seed}:{sha1}".encode()).hexdigest()


def _round_robin(groups: dict[str, list[dict[str, str]]]) -> list[dict[str, str]]:
    queues = {
        name: deque(rows)
        for name, rows in sorted(groups.items())
        if rows
    }
    result: list[dict[str, str]] = []
    while queues:
        for name in list(queues):
            queue = queues[name]
            result.append(queue.popleft())
            if not queue:
                del queues[name]
    return result


def _malicious_order(rows: Iterable[dict[str, str]], seed: int) -> list[dict[str, str]]:
    by_category_family: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        category = row["Category"].strip() or "Unknown"
        family = row["Family"].strip() or "Unknown"
        by_category_family[category][family].append(row)
    per_category: dict[str, list[dict[str, str]]] = {}
    for category, families in by_category_family.items():
        for family_rows in families.values():
            family_rows.sort(key=lambda item: _rank(seed, item["sha1"]))
        per_category[category] = _round_robin(dict(families))
    return _round_robin(per_category)


def build_candidates(
    source_path: Path,
    output_path: Path,
    *,
    seed: int,
    max_candidates: int,
    required_columns: tuple[str, ...],
    exclude_conflicts: bool = False,
    validation_path: Path | None = None,
) -> list[CandidateRecord]:
    rows = read_csv(source_path)
    if not rows:
        raise ValueError("source dataset is empty")
    missing = set(required_columns) - set(rows[0])
    if missing:
        raise ValueError(f"source dataset is missing columns: {sorted(missing)}")

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        sha1 = row.get("sha1", "").strip().lower()
        if not SHA1_PATTERN.fullmatch(sha1):
            raise ValueError(f"invalid SHA-1 in source dataset: {sha1!r}")
        row["sha1"] = sha1
        grouped[sha1].append(row)

    labels = ("Class", "Category", "Family")
    conflicts = {
        sha1: sorted({tuple(row[key].strip() for key in labels) for row in hash_rows})
        for sha1, hash_rows in grouped.items()
        if len({tuple(row[key].strip() for key in labels) for row in hash_rows}) > 1
    }
    validation = {
        "source_rows": len(rows),
        "unique_sha1": len(grouped),
        "duplicate_sha1": sum(len(hash_rows) > 1 for hash_rows in grouped.values()),
        "conflicting_sha1": len(conflicts),
        "conflicts_excluded": bool(exclude_conflicts),
        "conflicts": [
            {
                "sha1": sha1,
                "labels": [dict(zip(labels, values, strict=True)) for values in values_list],
            }
            for sha1, values_list in sorted(conflicts.items())
        ],
    }
    if validation_path is not None:
        atomic_json(validation_path, validation)
    if conflicts and not exclude_conflicts:
        first = min(conflicts)
        raise ValueError(
            f"source has {len(conflicts)} conflicting duplicate SHA-1 labels; "
            f"first conflict: {first}"
        )
    unique = {
        sha1: hash_rows[0]
        for sha1, hash_rows in grouped.items()
        if sha1 not in conflicts
    }

    benign = [
        row for row in unique.values() if row["Class"].strip().casefold() == "benign"
    ]
    malicious = [
        row for row in unique.values() if row["Class"].strip().casefold() != "benign"
    ]
    benign.sort(key=lambda item: _rank(seed, item["sha1"]))
    malicious = _malicious_order(malicious, seed)

    ordered: list[dict[str, str]] = []
    benign_queue = deque(benign)
    malicious_queue = deque(malicious)
    while len(ordered) < max_candidates and (benign_queue or malicious_queue):
        if benign_queue:
            ordered.append(benign_queue.popleft())
        if len(ordered) < max_candidates and malicious_queue:
            ordered.append(malicious_queue.popleft())

    candidates = [
        CandidateRecord(
            queue_index=index,
            sample_key=f"sha1:{row['sha1']}",
            source_sha1=row["sha1"],
            class_name=row["Class"].strip(),
            category=row["Category"].strip(),
            family=row["Family"].strip(),
            split=stable_split(row["sha1"]),
        )
        for index, row in enumerate(ordered)
    ]
    write_csv(
        output_path,
        list(CandidateRecord.__dataclass_fields__),
        (candidate.as_row() for candidate in candidates),
    )
    return candidates


def write_source_features(
    source_path: Path,
    candidates: Iterable[CandidateRecord],
    output_path: Path,
) -> int:
    source = {row["sha1"].strip().lower(): row for row in read_csv(source_path)}
    candidate_list = list(candidates)
    if not candidate_list:
        raise ValueError("candidate list is empty")
    excluded = {"sha1", "Class", "Category", "Family"}
    source_fields = [
        field for field in next(iter(source.values())) if field not in excluded
    ]
    fields = ["sample_key", *source_fields]
    rows = (
        {
            "sample_key": candidate.sample_key,
            **{
                field: source[candidate.source_sha1][field]
                for field in source_fields
            },
        }
        for candidate in candidate_list
    )
    return write_csv(output_path, fields, rows)
