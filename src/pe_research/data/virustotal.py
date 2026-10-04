"""Read-only VirusTotal enrichment with conservative free-tier quota handling."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from pe_research.data.config import PilotConfig, VirusTotalConfig
from pe_research.data.io import atomic_json, load_json, read_csv, write_csv
from pe_research.data.records import CandidateRecord


class VirusTotalError(RuntimeError):
    """Base error for safe, read-only VT enrichment."""


class AuthenticationError(VirusTotalError):
    """The configured key cannot access the requested public endpoint."""


class QuotaError(VirusTotalError):
    def __init__(self, message: str, retry_after: str = "") -> None:
        super().__init__(message)
        self.retry_after = retry_after


class RequestBudgetExhausted(VirusTotalError):
    """The local, per-run request budget was reached."""


@dataclass(frozen=True)
class LookupResult:
    queue_index: int
    sample_key: str
    source_sha1: str
    canonical_sha256: str
    class_name: str
    category: str
    family: str
    split: str
    lookup_status: str
    behaviour_report_count: int
    valid_report_count: int
    error_code: str

    def as_row(self) -> dict[str, object]:
        return asdict(self)


def read_candidates(path: Path) -> list[CandidateRecord]:
    return [
        CandidateRecord(
            queue_index=int(row["queue_index"]),
            sample_key=row["sample_key"],
            source_sha1=row["source_sha1"],
            class_name=row["class_name"],
            category=row["category"],
            family=row["family"],
            split=row["split"],
        )
        for row in read_csv(path)
    ]


def read_lookup_results(path: Path) -> list[LookupResult]:
    if not path.exists():
        return []
    return [
        LookupResult(
            queue_index=int(row["queue_index"]),
            sample_key=row["sample_key"],
            source_sha1=row["source_sha1"],
            canonical_sha256=row["canonical_sha256"],
            class_name=row["class_name"],
            category=row["category"],
            family=row["family"],
            split=row["split"],
            lookup_status=row["lookup_status"],
            behaviour_report_count=int(row["behaviour_report_count"]),
            valid_report_count=int(row["valid_report_count"]),
            error_code=row["error_code"],
        )
        for row in read_csv(path)
    ]


def _write_results(path: Path, results: dict[str, LookupResult]) -> None:
    ordered = sorted(results.values(), key=lambda item: item.queue_index)
    write_csv(
        path,
        list(LookupResult.__dataclass_fields__),
        (item.as_row() for item in ordered),
    )


class RateLimiter:
    """A no-burst limiter that also respects the most recent persisted request."""

    def __init__(
        self,
        interval_seconds: float,
        ledger_path: Path,
        *,
        clock: Callable[[], float] = time.time,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.interval_seconds = interval_seconds
        self.ledger_path = ledger_path
        self.clock = clock
        self.sleeper = sleeper
        self.last_request_epoch, self.persisted_request_count = self._read_history()

    def _read_history(self) -> tuple[float | None, int]:
        if not self.ledger_path.exists():
            return None, 0
        lines = self.ledger_path.read_text(encoding="utf-8").splitlines()
        epochs: list[float] = []
        for line in lines:
            try:
                value = json.loads(line).get("epoch")
                epochs.append(float(value))
            except (AttributeError, TypeError, ValueError, json.JSONDecodeError):
                continue
        return (epochs[-1] if epochs else None), len(epochs)

    def wait(self) -> None:
        if self.last_request_epoch is not None:
            delay = self.interval_seconds - (self.clock() - self.last_request_epoch)
            if delay > 0:
                self.sleeper(delay)

    def record(self, *, path: str, status: int) -> None:
        epoch = self.clock()
        self.last_request_epoch = epoch
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "epoch": epoch,
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(epoch)),
            "path": path,
            "status": status,
        }
        with self.ledger_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")


class VirusTotalClient:
    """Small VT v3 client intentionally exposing GET only."""

    def __init__(
        self,
        config: VirusTotalConfig,
        api_key: str,
        ledger_path: Path,
        *,
        opener: Callable[..., Any] = urllib.request.urlopen,
        clock: Callable[[], float] = time.time,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if not api_key.strip():
            raise AuthenticationError(f"set {config.api_key_env} before VT enrichment")
        self.config = config
        self.api_key = api_key.strip()
        self.opener = opener
        self.sleeper = sleeper
        self.rate_limiter = RateLimiter(
            config.minimum_interval_seconds,
            ledger_path,
            clock=clock,
            sleeper=sleeper,
        )
        self.requests_made = 0
        self.budget_used = self.rate_limiter.persisted_request_count

    def _request_once(self, path: str) -> dict[str, Any] | None:
        if self.budget_used >= self.config.request_budget:
            raise RequestBudgetExhausted(
                f"local VirusTotal request budget reached ({self.config.request_budget})"
            )
        self.rate_limiter.wait()
        request = urllib.request.Request(
            f"{self.config.base_url}{path}",
            headers={"x-apikey": self.api_key, "User-Agent": "pe-research/0.1"},
            method="GET",
        )
        self.requests_made += 1
        self.budget_used += 1
        try:
            with self.opener(request, timeout=60) as response:
                status = int(getattr(response, "status", 200))
                payload = response.read()
            self.rate_limiter.record(path=path, status=status)
            parsed = json.loads(payload.decode("utf-8"))
            if not isinstance(parsed, dict):
                raise VirusTotalError("VirusTotal returned a non-object JSON response")
            return cast(dict[str, Any], parsed)
        except urllib.error.HTTPError as error:
            self.rate_limiter.record(path=path, status=error.code)
            if error.code == 404:
                return None
            if error.code in {401, 403}:
                raise AuthenticationError(
                    f"VirusTotal rejected the configured key with HTTP {error.code}"
                ) from error
            if error.code == 429:
                retry_after = error.headers.get("Retry-After", "")
                raise QuotaError("VirusTotal quota or rate limit reached", retry_after) from error
            raise
        except urllib.error.URLError:
            self.rate_limiter.record(path=path, status=0)
            raise

    def get_json(self, path: str) -> dict[str, Any] | None:
        for attempt in range(self.config.max_retries + 1):
            try:
                return self._request_once(path)
            except urllib.error.HTTPError as error:
                if error.code < 500 or attempt >= self.config.max_retries:
                    raise VirusTotalError(f"VirusTotal HTTP error {error.code}") from error
                self.sleeper(float(2**attempt))
            except urllib.error.URLError as error:
                if attempt >= self.config.max_retries:
                    raise VirusTotalError("VirusTotal network request failed") from error
                self.sleeper(float(2**attempt))
        raise AssertionError("unreachable")


EVENT_LIST_FIELDS = (
    "calls_highlighted",
    "command_executions",
    "files_opened",
    "files_written",
    "files_deleted",
    "files_attribute_changed",
    "processes_created",
    "processes_terminated",
    "processes_killed",
    "processes_injected",
    "registry_keys_opened",
    "registry_keys_set",
    "registry_keys_deleted",
    "services_opened",
    "services_created",
    "services_started",
    "services_stopped",
    "services_deleted",
    "modules_loaded",
    "dns_lookups",
    "http_conversations",
    "ip_traffic",
)


def rough_event_count(attributes: dict[str, Any]) -> int:
    total = 0
    for field in EVENT_LIST_FIELDS:
        value = attributes.get(field)
        if isinstance(value, list):
            total += len(value)
    return total


def _behaviour_data(payload: dict[str, Any]) -> list[dict[str, Any]]:
    value = payload.get("data", [])
    if not isinstance(value, list):
        return []
    return [cast(dict[str, Any], item) for item in value if isinstance(item, dict)]


def _cache_get(client: VirusTotalClient, path: str, cache_path: Path) -> dict[str, Any] | None:
    status_path = cache_path.with_suffix(cache_path.suffix + ".status.json")
    if cache_path.exists():
        return load_json(cache_path)
    if status_path.exists() and load_json(status_path).get("status") == "not_found":
        return None
    payload = client.get_json(path)
    if payload is None:
        atomic_json(status_path, {"status": "not_found"})
        return None
    atomic_json(cache_path, payload)
    return payload


def _success_counts(results: dict[str, LookupResult]) -> tuple[int, Counter[str]]:
    benign = 0
    malicious: Counter[str] = Counter()
    for result in results.values():
        if result.lookup_status != "success":
            continue
        if result.class_name.casefold() == "benign":
            benign += 1
        else:
            malicious[result.category or "Unknown"] += 1
    return benign, malicious


def _result(
    candidate: CandidateRecord,
    *,
    canonical_sha256: str,
    lookup_status: str,
    behaviour_report_count: int,
    valid_report_count: int,
    error_code: str,
) -> LookupResult:
    return LookupResult(
        queue_index=candidate.queue_index,
        sample_key=candidate.sample_key,
        source_sha1=candidate.source_sha1,
        canonical_sha256=canonical_sha256,
        class_name=candidate.class_name,
        category=candidate.category,
        family=candidate.family,
        split=candidate.split,
        lookup_status=lookup_status,
        behaviour_report_count=behaviour_report_count,
        valid_report_count=valid_report_count,
        error_code=error_code,
    )


def enrich_candidates(
    candidates_path: Path,
    output_path: Path,
    cache_root: Path,
    ledger_path: Path,
    pilot: PilotConfig,
    vt_config: VirusTotalConfig,
    *,
    api_key: str | None = None,
    client: VirusTotalClient | None = None,
) -> list[LookupResult]:
    candidates = read_candidates(candidates_path)
    existing = {item.sample_key: item for item in read_lookup_results(output_path)}
    actual_key = api_key if api_key is not None else os.getenv(vt_config.api_key_env, "")
    vt = client or VirusTotalClient(vt_config, actual_key, ledger_path)

    for candidate in candidates:
        benign_count, malicious_counts = _success_counts(existing)
        goals_met = benign_count >= pilot.target_benign and all(
            count >= pilot.target_malicious_per_category
            for count in malicious_counts.values()
        ) and len(malicious_counts) >= 4
        if goals_met:
            break
        if candidate.sample_key in existing:
            continue
        is_benign = candidate.class_name.casefold() == "benign"
        if is_benign and benign_count >= pilot.target_benign:
            continue
        if (
            not is_benign
            and malicious_counts[candidate.category or "Unknown"]
            >= pilot.target_malicious_per_category
        ):
            continue

        file_cache = cache_root / "files" / f"{candidate.source_sha1}.json"
        file_payload = _cache_get(
            vt,
            f"/files/{urllib.parse.quote(candidate.source_sha1)}",
            file_cache,
        )
        if file_payload is None:
            result = _result(
                candidate,
                canonical_sha256="",
                lookup_status="not_found",
                behaviour_report_count=0,
                valid_report_count=0,
                error_code="vt_file_404",
            )
            existing[candidate.sample_key] = result
            _write_results(output_path, existing)
            continue

        file_data = file_payload.get("data", {})
        if not isinstance(file_data, dict):
            raise VirusTotalError("VirusTotal file response has no data object")
        attributes = file_data.get("attributes", {})
        sha256 = ""
        if isinstance(attributes, dict):
            sha256 = str(attributes.get("sha256", ""))
        if not sha256:
            sha256 = str(file_data.get("id", ""))
        if len(sha256) != 64:
            raise VirusTotalError("VirusTotal file response has no canonical SHA-256")

        behavior_cache = cache_root / "behaviours" / f"{sha256}.json"
        behaviour_payload = _cache_get(
            vt,
            f"/files/{sha256}/behaviours?limit={vt_config.behaviours_limit}",
            behavior_cache,
        )
        reports = _behaviour_data(behaviour_payload or {})
        valid = 0
        for report in reports:
            report_attributes = report.get("attributes", {})
            if isinstance(report_attributes, dict) and rough_event_count(
                cast(dict[str, Any], report_attributes)
            ) >= pilot.minimum_events_per_trace:
                valid += 1
        if not reports:
            status, error_code = "no_behaviour", "vt_no_behaviour"
        elif not valid:
            status, error_code = "insufficient_events", "trace_below_event_minimum"
        else:
            status, error_code = "success", ""
        result = _result(
            candidate,
            canonical_sha256=sha256,
            lookup_status=status,
            behaviour_report_count=len(reports),
            valid_report_count=valid,
            error_code=error_code,
        )
        existing[candidate.sample_key] = result
        _write_results(output_path, existing)

    return sorted(existing.values(), key=lambda item: item.queue_index)
