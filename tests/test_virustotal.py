from __future__ import annotations

import io
import json
import urllib.error
from dataclasses import replace
from pathlib import Path

import pytest

from pe_research.data.config import PilotConfig, VirusTotalConfig
from pe_research.data.io import write_csv
from pe_research.data.records import CandidateRecord
from pe_research.data.virustotal import (
    AuthenticationError,
    QuotaError,
    RequestBudgetExhausted,
    VirusTotalClient,
    enrich_candidates,
)


class FakeResponse:
    def __init__(self, payload: object, status: int = 200) -> None:
        self.payload = json.dumps(payload).encode()
        self.status = status

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.payload


def _config(*, budget: int = 3) -> VirusTotalConfig:
    return VirusTotalConfig(
        base_url="https://example.invalid/api/v3",
        api_key_env="VT_API_KEY",
        minimum_interval_seconds=16,
        request_budget=budget,
        behaviours_limit=40,
        max_retries=0,
    )


def _pilot() -> PilotConfig:
    return PilotConfig(
        seed=20261002,
        target_benign=1,
        target_malicious_per_category=1,
        max_candidates=1,
        minimum_per_class=1,
        minimum_events_per_trace=5,
    )


def _candidate_csv(path: Path) -> None:
    candidate = CandidateRecord(
        queue_index=0,
        sample_key=f"sha1:{'a' * 40}",
        source_sha1="a" * 40,
        class_name="Benign",
        category="Benign",
        family="Benign",
        split="train",
    )
    write_csv(
        path,
        list(CandidateRecord.__dataclass_fields__),
        [candidate.as_row()],
    )


def test_client_rate_limits_without_logging_key(tmp_path: Path) -> None:
    now = [100.0]
    sleeps: list[float] = []

    def clock() -> float:
        return now[0]

    def sleeper(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    client = VirusTotalClient(
        _config(),
        "top-secret-key",
        tmp_path / "ledger.jsonl",
        opener=lambda *_args, **_kwargs: FakeResponse({"data": {}}),
        clock=clock,
        sleeper=sleeper,
    )
    client.get_json("/files/a")
    client.get_json("/files/b")

    assert sleeps == [16]
    ledger = (tmp_path / "ledger.jsonl").read_text(encoding="utf-8")
    assert "top-secret-key" not in ledger
    assert "/files/a" in ledger


def test_client_stops_on_429(tmp_path: Path) -> None:
    def opener(*_args: object, **_kwargs: object) -> object:
        raise urllib.error.HTTPError(
            "https://example.invalid",
            429,
            "limited",
            {"Retry-After": "60"},
            io.BytesIO(b"{}"),
        )

    client = VirusTotalClient(
        _config(),
        "secret",
        tmp_path / "ledger.jsonl",
        opener=opener,
        sleeper=lambda _seconds: None,
    )
    with pytest.raises(QuotaError) as error:
        client.get_json("/files/a")
    assert error.value.retry_after == "60"


@pytest.mark.parametrize("status", [401, 403])
def test_client_stops_on_auth_failure_without_leaking_key(
    tmp_path: Path, status: int
) -> None:
    def opener(*_args: object, **_kwargs: object) -> object:
        raise urllib.error.HTTPError(
            "https://example.invalid",
            status,
            "denied",
            {},
            io.BytesIO(b"{}"),
        )

    secret = "must-not-appear"
    client = VirusTotalClient(
        _config(),
        secret,
        tmp_path / "ledger.jsonl",
        opener=opener,
        sleeper=lambda _seconds: None,
    )
    with pytest.raises(AuthenticationError) as error:
        client.get_json("/files/a")
    assert secret not in str(error.value)
    assert secret not in (tmp_path / "ledger.jsonl").read_text(encoding="utf-8")


def test_client_returns_none_for_404(tmp_path: Path) -> None:
    def opener(*_args: object, **_kwargs: object) -> object:
        raise urllib.error.HTTPError(
            "https://example.invalid",
            404,
            "missing",
            {},
            io.BytesIO(b"{}"),
        )

    client = VirusTotalClient(
        _config(),
        "secret",
        tmp_path / "ledger.jsonl",
        opener=opener,
        sleeper=lambda _seconds: None,
    )
    assert client.get_json("/files/a") is None


def test_client_retries_5xx(tmp_path: Path) -> None:
    calls = [0]

    def opener(*_args: object, **_kwargs: object) -> object:
        calls[0] += 1
        if calls[0] == 1:
            raise urllib.error.HTTPError(
                "https://example.invalid",
                503,
                "temporary",
                {},
                io.BytesIO(b"{}"),
            )
        return FakeResponse({"data": {}})

    client = VirusTotalClient(
        replace(_config(), max_retries=1),
        "secret",
        tmp_path / "ledger.jsonl",
        opener=opener,
        sleeper=lambda _seconds: None,
    )
    assert client.get_json("/files/a") == {"data": {}}
    assert calls == [2]


def test_client_enforces_local_budget(tmp_path: Path) -> None:
    client = VirusTotalClient(
        _config(budget=1),
        "secret",
        tmp_path / "ledger.jsonl",
        opener=lambda *_args, **_kwargs: FakeResponse({"data": {}}),
        sleeper=lambda _seconds: None,
    )
    client.get_json("/files/a")
    with pytest.raises(RequestBudgetExhausted):
        client.get_json("/files/b")


def test_request_budget_is_cumulative_across_resumed_clients(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    first = VirusTotalClient(
        _config(budget=1),
        "secret",
        ledger,
        opener=lambda *_args, **_kwargs: FakeResponse({"data": {}}),
        sleeper=lambda _seconds: None,
    )
    first.get_json("/files/a")
    resumed = VirusTotalClient(
        _config(budget=1),
        "secret",
        ledger,
        opener=lambda *_args, **_kwargs: FakeResponse({"data": {}}),
        sleeper=lambda _seconds: None,
    )
    with pytest.raises(RequestBudgetExhausted):
        resumed.get_json("/files/b")
    assert resumed.requests_made == 0


def test_enrichment_caches_success_and_resumes_without_requests(tmp_path: Path) -> None:
    candidates = tmp_path / "candidates.csv"
    output = tmp_path / "lookup_results.csv"
    cache = tmp_path / "cache"
    _candidate_csv(candidates)
    calls = [0]
    sha256 = "b" * 64

    def opener(*_args: object, **_kwargs: object) -> object:
        calls[0] += 1
        if calls[0] == 1:
            return FakeResponse({"data": {"id": sha256, "attributes": {"sha256": sha256}}})
        return FakeResponse(
            {
                "data": [
                    {
                        "id": "report-1",
                        "attributes": {"files_opened": [f"C:/temp/{index}" for index in range(5)]},
                    }
                ]
            }
        )

    first_client = VirusTotalClient(
        _config(),
        "secret",
        tmp_path / "ledger.jsonl",
        opener=opener,
        sleeper=lambda _seconds: None,
    )
    first = enrich_candidates(
        candidates,
        output,
        cache,
        tmp_path / "ledger.jsonl",
        _pilot(),
        _config(),
        client=first_client,
    )
    assert first[0].lookup_status == "success"
    assert calls == [2]

    output.unlink()

    def forbidden_opener(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("a cached result must not trigger a request")

    second_client = VirusTotalClient(
        _config(),
        "secret",
        tmp_path / "ledger.jsonl",
        opener=forbidden_opener,
        sleeper=lambda _seconds: None,
    )
    second = enrich_candidates(
        candidates,
        output,
        cache,
        tmp_path / "ledger.jsonl",
        _pilot(),
        _config(),
        client=second_client,
    )
    assert second[0].lookup_status == "success"
    assert second_client.requests_made == 0
