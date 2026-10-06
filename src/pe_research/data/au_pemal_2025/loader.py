"""Training-side readers that enforce trace eligibility contracts."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from pe_research.data.io import read_csv


def load_sequence_events(
    events_path: Path,
    traces_path: Path,
    *,
    trace_ids: Iterable[str] | None = None,
    allow_ineligible: bool = False,
) -> dict[str, list[dict[str, str]]]:
    """Load canonical sequences, rejecting unordered traces unless explicitly allowed."""
    traces = {row["trace_id"]: row for row in read_csv(traces_path)}
    if trace_ids is None:
        selected = {
            trace_id
            for trace_id, trace in traces.items()
            if trace["is_canonical"] == "1"
            and trace["trace_quality"] == "valid"
            and (allow_ineligible or trace["sequence_eligible"] == "1")
        }
    else:
        selected = set(trace_ids)
        missing = selected - set(traces)
        if missing:
            raise KeyError(f"unknown trace IDs: {sorted(missing)}")
        ineligible = {
            trace_id
            for trace_id in selected
            if traces[trace_id]["sequence_eligible"] != "1"
        }
        if ineligible and not allow_ineligible:
            raise ValueError(
                "sequence-ineligible traces requested; ordering is not trustworthy: "
                f"{sorted(ineligible)}"
            )

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for event in read_csv(events_path):
        if event["trace_id"] in selected:
            grouped[event["trace_id"]].append(event)
    for sequence in grouped.values():
        sequence.sort(key=lambda row: int(row["event_index"]))
    return dict(grouped)
