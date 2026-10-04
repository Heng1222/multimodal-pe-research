from __future__ import annotations

import csv
from pathlib import Path

import pytest

from pe_research.data.config import load_config
from pe_research.data.sampling import build_candidates, stable_split


def _write_source(path: Path) -> None:
    fields = ["sha1", "feature", "Class", "Category", "Family"]
    rows: list[dict[str, object]] = []
    for index in range(20):
        rows.append(
            {
                "sha1": f"{index:040x}",
                "feature": index,
                "Class": "Benign",
                "Category": "Benign",
                "Family": "Benign",
            }
        )
    categories = ["RAT", "Ransomware", "InfoStealer", "Trojan"]
    for index in range(40):
        rows.append(
            {
                "sha1": f"{index + 100:040x}",
                "feature": index,
                "Class": "Malware",
                "Category": categories[index % len(categories)],
                "Family": f"family-{index % 8}",
            }
        )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_checked_in_config_is_pinned_and_free_tier_safe() -> None:
    config = load_config(Path("configs/data/au_pemal_pilot_v1.yaml"))

    assert len(config.source.commit) == 40
    assert config.source.commit in config.source.url
    assert len(config.source.expected_sha256) == 64
    assert config.virustotal.minimum_interval_seconds >= 16
    assert config.virustotal.request_budget == 200
    assert config.embedding.dimension == 384
    assert len(config.embedding.model_revision) == 40


def test_candidate_selection_is_deterministic_and_balanced(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    _write_source(source)
    first = build_candidates(
        source,
        tmp_path / "first.csv",
        seed=20261002,
        max_candidates=40,
        required_columns=("sha1", "Class", "Category", "Family"),
    )
    second = build_candidates(
        source,
        tmp_path / "second.csv",
        seed=20261002,
        max_candidates=40,
        required_columns=("sha1", "Class", "Category", "Family"),
    )

    assert first == second
    assert sum(row.class_name == "Benign" for row in first) == 20
    assert sum(row.class_name == "Malware" for row in first) == 20
    assert {row.category for row in first if row.class_name == "Malware"} == {
        "RAT",
        "Ransomware",
        "InfoStealer",
        "Trojan",
    }
    assert all(row.split == stable_split(row.source_sha1) for row in first)


def test_conflicting_duplicate_hash_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    with source.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["sha1", "Class", "Category", "Family"]
        )
        writer.writeheader()
        writer.writerow(
            {"sha1": "a" * 40, "Class": "Benign", "Category": "Benign", "Family": "Benign"}
        )
        writer.writerow(
            {"sha1": "a" * 40, "Class": "Malware", "Category": "RAT", "Family": "x"}
        )

    with pytest.raises(ValueError, match="conflicting duplicate"):
        build_candidates(
            source,
            tmp_path / "out.csv",
            seed=1,
            max_candidates=2,
            required_columns=("sha1", "Class", "Category", "Family"),
        )


def test_conflicting_duplicate_can_be_excluded_and_reported(tmp_path: Path) -> None:
    source = tmp_path / "source.csv"
    with source.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["sha1", "Class", "Category", "Family"]
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "sha1": "a" * 40,
                    "Class": "Benign",
                    "Category": "Benign",
                    "Family": "Benign",
                },
                {
                    "sha1": "a" * 40,
                    "Class": "Malware",
                    "Category": "RAT",
                    "Family": "x",
                },
                {
                    "sha1": "b" * 40,
                    "Class": "Benign",
                    "Category": "Benign",
                    "Family": "Benign",
                },
            ]
        )

    report = tmp_path / "source_validation.json"
    candidates = build_candidates(
        source,
        tmp_path / "out.csv",
        seed=1,
        max_candidates=2,
        required_columns=("sha1", "Class", "Category", "Family"),
        exclude_conflicts=True,
        validation_path=report,
    )

    assert [item.source_sha1 for item in candidates] == ["b" * 40]
    assert '"conflicting_sha1": 1' in report.read_text(encoding="utf-8")
