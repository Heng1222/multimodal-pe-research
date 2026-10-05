"""Validated, label-free embedding input and deterministic connected-group splitting."""

from __future__ import annotations

import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from pe_research.data.io import read_csv, read_csv_header

SPLITS = ("train", "validation", "test")
REVISION = "e7f32e3c00f91d699e8c43b53106206bcc72bb22"


def indexed(path: Path) -> dict[str, dict[str, str]]:
    rows = read_csv(path)
    result = {row["sample_id"]: row for row in rows}
    if not rows or len(result) != len(rows) or "" in result:
        raise ValueError(f"empty or duplicate sample IDs: {path}")
    return result


def load_embeddings(path: Path, dimension: int = 768) -> tuple[list[str], Any, dict[str, Any]]:
    columns = [f"embedding_{i:03}" for i in range(dimension)]
    if set(read_csv_header(path)) != {"sample_id", *columns}:
        raise ValueError("expected sample_id and continuous embedding columns; check LFS objects")
    rows = indexed(path)
    ids = sorted(rows)
    x = np.asarray([[float(rows[i][key]) for key in columns] for i in ids], dtype=np.float32)
    norms = np.linalg.norm(x.astype(np.float64), axis=1)
    if not np.isfinite(x).all() or np.any(norms == 0) or np.any(abs(norms - 1) > 1e-3):
        raise ValueError("non-finite, zero, or non-normalized embedding beyond tolerance 1e-3")
    raw = hashlib.sha256(x.tobytes()).hexdigest()
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return (
        ids,
        x,
        {
            "version": "float32_l2/v1",
            "norm_min": float(norms.min()),
            "norm_max": float(norms.max()),
            "renormalized_count": len(x),
            "outside_original_tolerance": int((abs(norms - 1) > 1e-4).sum()),
            "raw_matrix_sha256": raw,
            "normalized_matrix_sha256": hashlib.sha256(x.tobytes()).hexdigest(),
        },
    )


def sequence_fingerprints(path: Path, ids: set[str]) -> dict[str, str]:
    hashes: dict[str, Any] = {}
    counts: Counter[str] = Counter()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            sid = row["sample_id"]
            if sid not in ids or int(row["event_index"]) != counts[sid]:
                raise ValueError("unknown sample or unordered/non-contiguous API events")
            if not row["api_name"].strip():
                raise ValueError("empty API name")
            hashes.setdefault(sid, hashlib.sha256()).update(
                row["api_name"].strip().encode() + b"\x00"
            )
            counts[sid] += 1
    if set(hashes) != ids:
        raise ValueError("missing API sequences")
    return {sid: digest.hexdigest() for sid, digest in hashes.items()}


def group_split(
    samples: dict[str, dict[str, str]],
    labels: dict[str, dict[str, str]],
    fingerprints: dict[str, str],
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    parent = {sid: sid for sid in samples}

    def root(sid: str) -> str:
        while parent[sid] != sid:
            parent[sid] = parent[parent[sid]]
            sid = parent[sid]
        return sid

    seen: dict[tuple[str, str], str] = {}
    for sid in sorted(samples):
        for key in [("sha", samples[sid]["source_sha256"]), ("sequence", fingerprints[sid])]:
            if key in seen:
                a, b = sorted([root(sid), root(seen[key])])
                parent[b] = a
            else:
                seen[key] = sid
    groups: dict[str, list[str]] = defaultdict(list)
    for sid in sorted(samples):
        groups[root(sid)].append(sid)
    strata = {
        sid: "benign"
        if labels[sid]["y"] == "0"
        else labels[sid]["family"]
        if labels[sid]["family_known"] == "1"
        else "malicious_unlabelled"
        for sid in samples
    }
    totals = Counter(strata.values())
    targets = {
        split: {key: count * ratio for key, count in totals.items()}
        for split, ratio in zip(SPLITS, [0.7, 0.15, 0.15], strict=True)
    }
    counts: dict[str, Counter[str]] = {split: Counter() for split in SPLITS}
    assignments: dict[str, str] = {}
    ordered = sorted(
        groups.items(),
        key=lambda item: (-len(item[1]), hashlib.sha256(f"{seed}:{item[0]}".encode()).hexdigest()),
    )
    for gid, members in ordered:
        additions = Counter(strata[sid] for sid in members)

        def cost(
            candidate: str, additions: Counter[str] = additions, size: int = len(members)
        ) -> float:
            # Increment in normalized squared deviation, including overall split sizes.
            score = 0.0
            for key, amount in additions.items():
                current = counts[candidate][key] - targets[candidate][key]
                score += ((current + amount) ** 2 - current**2) / totals[key]
            total_target = sum(targets[candidate].values())
            current_total = sum(counts[candidate].values()) - total_target
            return score + ((current_total + size) ** 2 - current_total**2) / len(samples)

        split = min(SPLITS, key=cost)
        counts[split].update(additions)
        assignments[gid] = split
    manifest = []
    for gid, members in sorted(groups.items()):
        for sid in members:
            label = labels[sid]
            manifest.append(
                {
                    "sample_id": sid,
                    "group_id": gid,
                    "original_split": samples[sid]["split"],
                    "split": assignments[gid],
                    "sequence_fingerprint": fingerprints[sid],
                    "binary_label": int(label["y"]),
                    "family_label": label["family"],
                    "family_known": int(label["family_known"] == "1" and label["y"] == "1"),
                }
            )
    for split in SPLITS:
        subset = [row for row in manifest if row["split"] == split]
        if {row["binary_label"] for row in subset} != {0, 1}:
            raise ValueError(f"split lacks both binary classes: {split}")
        for family in set(strata.values()) - {"benign", "malicious_unlabelled"}:
            if not any(row["family_label"] == family and row["family_known"] for row in subset):
                raise ValueError(f"split lacks family support: {split}/{family}")
    duplicates = [members for members in groups.values() if len(members) > 1]
    return manifest, {
        "split_stratum_counts": {key: dict(value) for key, value in counts.items()},
        "group_count": len(groups),
        "duplicate_groups": len(duplicates),
        "label_conflicts": [
            members for members in duplicates if len({strata[s] for s in members}) > 1
        ],
        "split_seed": seed,
        "algorithm": "connected_groups_normalized_squared_deviation/v1",
        "source": "Zenodo 11079764 / SmartVMI",
        "near_duplicate_detection": "exact API sequences only",
    }
