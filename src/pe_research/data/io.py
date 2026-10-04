"""Filesystem, CSV, hashing, and source-download helpers."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import urllib.request
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, cast


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return cast(dict[str, Any], value)


def write_csv(
    path: Path,
    fieldnames: list[str],
    rows: Iterable[Mapping[str, object]],
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    count = 0
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    os.replace(temporary, path)
    return count


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def read_csv_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream).fieldnames or [])


def fetch_file(url: str, destination: Path, expected_sha256: str = "") -> str:
    """Download a public data file atomically and verify its optional checksum."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        actual = sha256_file(destination)
        if not expected_sha256 or actual == expected_sha256:
            return actual
        raise ValueError(f"existing source checksum mismatch: {destination}")

    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "pe-research/0.1"})
    digest = hashlib.sha256()
    try:
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as out:
            while chunk := response.read(1024 * 1024):
                digest.update(chunk)
                out.write(chunk)
        actual = digest.hexdigest()
        if expected_sha256 and actual != expected_sha256:
            raise ValueError(
                f"download checksum mismatch: expected {expected_sha256}, got {actual}"
            )
        os.replace(temporary, destination)
        return actual
    finally:
        if temporary.exists():
            temporary.unlink()


def package_version(distribution: str) -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(distribution)
    except PackageNotFoundError:
        return "not-installed"
