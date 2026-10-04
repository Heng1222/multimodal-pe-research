from __future__ import annotations

from pathlib import Path

import pytest

from pe_research.data.io import write_csv
from pe_research.data.loader import load_sequence_events


def test_sequence_loader_rejects_unknown_order_by_default(tmp_path: Path) -> None:
    traces = tmp_path / "traces.csv"
    events = tmp_path / "events.csv"
    write_csv(
        traces,
        ["trace_id", "is_canonical", "trace_quality", "sequence_eligible"],
        [
            {
                "trace_id": "trace-1",
                "is_canonical": "1",
                "trace_quality": "valid",
                "sequence_eligible": "0",
            }
        ],
    )
    write_csv(
        events,
        ["trace_id", "event_index", "canonical_text"],
        [{"trace_id": "trace-1", "event_index": 0, "canonical_text": "event"}],
    )

    assert load_sequence_events(events, traces) == {}
    with pytest.raises(ValueError, match="ordering is not trustworthy"):
        load_sequence_events(events, traces, trace_ids=["trace-1"])
    loaded = load_sequence_events(
        events,
        traces,
        trace_ids=["trace-1"],
        allow_ineligible=True,
    )
    assert loaded["trace-1"][0]["canonical_text"] == "event"
