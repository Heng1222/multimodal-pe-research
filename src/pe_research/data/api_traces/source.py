"""Source download, label cleanup, and deterministic pilot candidates."""

from __future__ import annotations

import csv
import hashlib
import http.client
import json
import os
import re
import shutil
import ssl
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, cast

from pe_research.data.api_traces.config import ApiTraceConfig
from pe_research.data.io import atomic_json, write_csv

HEX_SHA256_LENGTH = 64
_CONTENT_RANGE = re.compile(r"bytes\s+(\d+)-(\d+)/(\d+|\*)", re.IGNORECASE)


class DownloadInterruptedError(RuntimeError):
    """A resumable download exhausted automatic reconnect attempts."""


def _source_digests(path: Path) -> tuple[str, str]:
    md5 = hashlib.md5()  # noqa: S324 - source integrity, not cryptographic security
    sha256 = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            md5.update(chunk)
            sha256.update(chunk)
    return md5.hexdigest(), sha256.hexdigest()


def download_resumable(
    url: str,
    destination: Path,
    expected_md5: str,
    *,
    minimum_free_gib: float = 0,
    read_timeout_seconds: float = 300,
    max_reconnects: int = 100,
    retry_backoff_seconds: float = 5,
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Download with persistent Range resumption and automatic reconnects."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        actual_md5, actual_sha256 = _source_digests(destination)
        if actual_md5 != expected_md5.lower():
            raise ValueError(f"existing source MD5 mismatch: {destination}")
        return {
            "path": str(destination),
            "bytes": destination.stat().st_size,
            "md5": actual_md5,
            "sha256": actual_sha256,
            "resumed": False,
        }

    free = shutil.disk_usage(destination.parent).free
    required = int(minimum_free_gib * 1024**3)
    if required and free < required:
        raise OSError(
            f"insufficient free space for source download: {free / 1024**3:.2f} GiB "
            f"available, {minimum_free_gib:.2f} GiB required"
        )

    temporary = destination.with_suffix(destination.suffix + ".part")
    initial_offset = temporary.stat().st_size if temporary.exists() else 0
    reconnects = 0
    expected_total: int | None = None
    last_reported = initial_offset
    while True:
        offset = temporary.stat().st_size if temporary.exists() else 0
        headers = {"User-Agent": "pe-research/0.1"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(url, headers=headers)
        if progress is not None:
            total_text = f"/{expected_total}" if expected_total is not None else ""
            progress(f"connecting at byte {offset}{total_text}: {destination.name}")
        try:
            with urllib.request.urlopen(request, timeout=read_timeout_seconds) as response:
                status = int(getattr(response, "status", response.getcode()))
                if offset and status != 206:
                    raise DownloadInterruptedError(
                        "server ignored the Range request; partial file was preserved"
                    )
                content_range = response.headers.get("Content-Range", "")
                range_match = _CONTENT_RANGE.fullmatch(content_range.strip())
                if offset and range_match is not None and int(range_match.group(1)) != offset:
                    raise DownloadInterruptedError(
                        f"unexpected Content-Range start: {content_range}"
                    )
                if range_match is not None and range_match.group(3) != "*":
                    expected_total = int(range_match.group(3))
                elif not offset:
                    content_length = response.headers.get("Content-Length", "")
                    if content_length.isdigit():
                        expected_total = int(content_length)
                mode = "ab" if offset else "wb"
                with temporary.open(mode) as output:
                    while chunk := response.read(4 * 1024 * 1024):
                        output.write(chunk)
                        current = output.tell()
                        if progress is not None and current - last_reported >= 256 * 1024 * 1024:
                            total_text = (
                                f"/{expected_total} bytes"
                                if expected_total is not None
                                else " bytes"
                            )
                            progress(f"downloaded {current}{total_text}: {destination.name}")
                            last_reported = current
            current_size = temporary.stat().st_size
            if expected_total is not None and current_size < expected_total:
                raise http.client.IncompleteRead(b"", expected_total - current_size)
            if expected_total is not None and current_size > expected_total:
                raise DownloadInterruptedError(
                    f"partial file exceeds server size: {current_size}/{expected_total}"
                )
            break
        except urllib.error.HTTPError as error:
            if error.code == 416 and temporary.exists():
                actual_md5, actual_sha256 = _source_digests(temporary)
                if actual_md5 != expected_md5.lower():
                    raise DownloadInterruptedError(
                        "server rejected Range and partial file is incomplete"
                    ) from error
                os.replace(temporary, destination)
                return {
                    "path": str(destination),
                    "bytes": destination.stat().st_size,
                    "md5": actual_md5,
                    "sha256": actual_sha256,
                    "resumed": True,
                }
            if error.code not in {408, 429, 500, 502, 503, 504}:
                raise
            transient_error: BaseException = error
        except (
            TimeoutError,
            urllib.error.URLError,
            http.client.IncompleteRead,
            ssl.SSLError,
        ) as error:
            transient_error = error

        reconnects += 1
        if reconnects > max_reconnects:
            current_size = temporary.stat().st_size if temporary.exists() else 0
            raise DownloadInterruptedError(
                f"download stopped after {max_reconnects} reconnects at byte {current_size}; "
                "rerun the same fetch command to resume"
            ) from transient_error
        delay = retry_backoff_seconds
        if isinstance(transient_error, urllib.error.HTTPError):
            retry_after = transient_error.headers.get("Retry-After", "")
            if retry_after.isdigit():
                delay = max(delay, float(retry_after))
        current_size = temporary.stat().st_size if temporary.exists() else 0
        if progress is not None:
            progress(
                f"transient download error ({type(transient_error).__name__}); "
                f"saved byte {current_size}, reconnect {reconnects}/{max_reconnects} "
                f"in {delay:g}s"
            )
        time.sleep(delay)

    actual_md5, actual_sha256 = _source_digests(temporary)
    if actual_md5 != expected_md5.lower():
        raise ValueError(
            f"download MD5 mismatch for {destination.name}: expected {expected_md5}, "
            f"got {actual_md5}; partial file retained"
        )
    os.replace(temporary, destination)
    return {
        "path": str(destination),
        "bytes": destination.stat().st_size,
        "md5": actual_md5,
        "sha256": actual_sha256,
        "resumed": bool(initial_offset or reconnects),
    }


def fetch_sources(
    config: ApiTraceConfig,
    workspace: Path,
    *,
    metadata_only: bool = False,
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    raw = workspace / "raw"
    metadata = download_resumable(
        config.source.metadata_url,
        raw / config.source.metadata_filename,
        config.source.metadata_md5,
        progress=progress,
    )
    files: dict[str, object] = {"metadata": metadata}
    if not metadata_only:
        files["archive"] = download_resumable(
            config.source.archive_url,
            raw / config.source.archive_filename,
            config.source.archive_md5,
            minimum_free_gib=config.source.minimum_free_gib,
            progress=progress,
        )
    provenance: dict[str, object] = {
        "record_id": config.source.record_id,
        "version": config.source.version,
        "files": files,
    }
    atomic_json(raw / "source_provenance.json", provenance)
    return provenance


def _normalize_sha(value: object) -> str | None:
    text = str(value).strip().lower()
    if len(text) != HEX_SHA256_LENGTH:
        return None
    try:
        int(text, 16)
    except ValueError:
        return None
    return text


def _iter_assignments(payload: object) -> Iterable[tuple[str, object]]:
    if isinstance(payload, dict):
        for family, values in payload.items():
            if not isinstance(values, list):
                continue
            for sha in values:
                yield str(family).strip(), sha
        return
    if isinstance(payload, list):
        for item in payload:
            if not isinstance(item, dict):
                continue
            family = item.get("family") or item.get("label")
            sha = item.get("sha256") or item.get("sha")
            if family is not None and sha is not None:
                yield str(family).strip(), sha


def load_single_label_metadata(
    path: Path,
) -> tuple[dict[str, str], dict[str, object]]:
    """Return only unique SHA-to-one-label assignments plus audit statistics."""
    payload = cast(Any, json.loads(path.read_text(encoding="utf-8")))
    labels_by_sha: dict[str, set[str]] = defaultdict(set)
    total_assignments = 0
    invalid_sha_assignments = 0
    empty_family_assignments = 0
    for family, raw_sha in _iter_assignments(payload):
        total_assignments += 1
        sha = _normalize_sha(raw_sha)
        if sha is None:
            invalid_sha_assignments += 1
            continue
        if not family:
            empty_family_assignments += 1
            continue
        labels_by_sha[sha].add(family)

    single = {
        sha: next(iter(labels))
        for sha, labels in labels_by_sha.items()
        if len(labels) == 1
    }
    family_counts = Counter(single.values())
    report: dict[str, object] = {
        "total_assignments": total_assignments,
        "unique_sha256": len(labels_by_sha),
        "single_label_sha256": len(single),
        "multi_label_sha256_excluded": sum(
            len(labels) > 1 for labels in labels_by_sha.values()
        ),
        "invalid_sha_assignments": invalid_sha_assignments,
        "empty_family_assignments": empty_family_assignments,
        "family_count": len(family_counts),
        "single_label_counts_by_family": dict(sorted(family_counts.items())),
    }
    return single, report


def _rank(seed: int, stratum: str, sha256: str, purpose: str = "candidate") -> str:
    return hashlib.sha256(
        f"{seed}|{purpose}|{stratum.casefold()}|{sha256}".encode()
    ).hexdigest()


def build_candidate_manifest(
    config: ApiTraceConfig,
    workspace: Path,
) -> list[dict[str, object]]:
    """Select a deterministic four-times reserve without assigning final sample IDs."""
    metadata_path = workspace / "raw" / config.source.metadata_filename
    labels_by_sha, report = load_single_label_metadata(metadata_path)
    by_family: dict[str, list[str]] = defaultdict(list)
    canonical_family: dict[str, str] = {}
    for sha, family in labels_by_sha.items():
        key = family.casefold()
        by_family[key].append(sha)
        canonical_family.setdefault(key, family)

    benign_key = "benign"
    if benign_key not in by_family:
        raise ValueError("metadata has no single-label benign stratum")
    malicious = sorted(
        (
            (key, len(shas))
            for key, shas in by_family.items()
            if key != benign_key
        ),
        key=lambda item: (-item[1], item[0]),
    )
    selected_family_keys = [key for key, _count in malicious[: config.pilot.family_count]]
    expected = [name.casefold() for name in config.pilot.expected_families]
    if selected_family_keys != expected:
        raise ValueError(
            "top families differ from the pinned pilot expectation: "
            f"expected {expected}, got {selected_family_keys}"
        )

    strata = [benign_key, *selected_family_keys]
    rows: list[dict[str, object]] = []
    for stratum in strata:
        target = (
            config.pilot.target_benign
            if stratum == benign_key
            else config.pilot.target_per_family
        )
        reserve = target * config.pilot.candidate_multiplier
        ranked = sorted(by_family[stratum], key=lambda sha: _rank(config.pilot.seed, stratum, sha))
        if len(ranked) < reserve:
            raise ValueError(
                f"stratum {stratum} has {len(ranked)} single-label samples; {reserve} required"
            )
        for index, sha in enumerate(ranked[:reserve]):
            rows.append(
                {
                    "candidate_rank": index,
                    "source_sha256": sha,
                    "class_name": "benign" if stratum == benign_key else "malicious",
                    "family": canonical_family[stratum],
                    "status": "pending",
                    "failure_reason": "",
                }
            )

    report.update(
        {
            "selected_families": [canonical_family[key] for key in selected_family_keys],
            "candidate_multiplier": config.pilot.candidate_multiplier,
            "candidate_count": len(rows),
        }
    )
    artifacts = workspace / "artifacts"
    write_csv(
        artifacts / "candidate_manifest.csv",
        [
            "candidate_rank",
            "source_sha256",
            "class_name",
            "family",
            "status",
            "failure_reason",
        ],
        rows,
    )
    atomic_json(artifacts / "source_validation.json", report)
    return rows


def deterministic_split(
    shas: list[str],
    *,
    seed: int,
    stratum: str,
    validation_count: int,
    test_count: int,
) -> dict[str, str]:
    ranked = sorted(shas, key=lambda sha: _rank(seed, stratum, sha, "split"))
    train_end = len(ranked) - validation_count - test_count
    result = {sha: "train" for sha in ranked[:train_end]}
    result.update({sha: "validation" for sha in ranked[train_end : train_end + validation_count]})
    result.update({sha: "test" for sha in ranked[train_end + validation_count :]})
    return result


def update_candidate_statuses(
    path: Path,
    statuses: dict[str, tuple[str, str]],
) -> None:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        status, reason = statuses.get(row["source_sha256"], ("missing", "archive_member_missing"))
        row["status"] = status
        row["failure_reason"] = reason
    write_csv(path, list(rows[0]) if rows else [], rows)
