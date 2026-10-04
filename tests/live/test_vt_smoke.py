"""Explicitly opt-in VirusTotal smoke test (four requests maximum)."""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import pytest

from pe_research.data.config import load_config
from pe_research.data.virustotal import VirusTotalClient, read_candidates

pytestmark = pytest.mark.live


def test_one_benign_and_one_malicious_hash_lookup(tmp_path: Path) -> None:
    if os.getenv("RUN_VT_LIVE") != "1":
        pytest.skip("set RUN_VT_LIVE=1 to consume up to four VT requests")
    api_key = os.getenv("VT_API_KEY", "")
    if not api_key:
        pytest.skip("VT_API_KEY is not set")

    config = load_config(Path("configs/data/au_pemal_pilot_v1.yaml"))
    candidates = read_candidates(
        Path("data/au_pemal_2025/pilot_v1/artifacts/candidates.csv")
    )
    selected = [
        next(item for item in candidates if item.class_name.casefold() == "benign"),
        next(item for item in candidates if item.class_name.casefold() != "benign"),
    ]
    client = VirusTotalClient(
        replace(config.virustotal, request_budget=4, max_retries=0),
        api_key,
        tmp_path / "vt_smoke_ledger.jsonl",
    )

    for candidate in selected:
        file_payload = client.get_json(f"/files/{candidate.source_sha1}")
        if file_payload is None:
            continue
        data = file_payload.get("data", {})
        attributes = data.get("attributes", {}) if isinstance(data, dict) else {}
        sha256 = attributes.get("sha256", "") if isinstance(attributes, dict) else ""
        assert len(str(sha256)) == 64
        behaviors = client.get_json(f"/files/{sha256}/behaviours?limit=40")
        assert behaviors is None or isinstance(behaviors.get("data", []), list)
