"""Stable record schemas shared by the materialization stages."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class CandidateRecord:
    queue_index: int
    sample_key: str
    source_sha1: str
    class_name: str
    category: str
    family: str
    split: str

    def as_row(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class EventRecord:
    trace_id: str
    event_index: int
    event_type: str
    operation: str
    object_role: str
    result: str
    canonical_text: str
    observed_at: str
    ordering_known: int
    object_known: int
    result_known: int

    def as_row(self) -> dict[str, object]:
        return asdict(self)
