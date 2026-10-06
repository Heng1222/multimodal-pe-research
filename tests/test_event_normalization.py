from pe_research.data.au_pemal_2025.normalization import events_from_behavior, normalize_text


def test_normalization_masks_identifiers_and_secrets() -> None:
    text = normalize_text(
        "C:\\Users\\alice\\tool.exe pid=123 token=hunter2 "
        "8.8.8.8 192.168.1.2 aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    )

    assert "alice" not in text
    assert "hunter2" not in text
    assert "123" not in text
    assert "8.8.8.8" not in text
    assert "192.168.1.2" not in text
    assert "a" * 40 not in text
    assert "<USER>" in text
    assert "<PUBLIC_IP>" in text
    assert "<PRIVATE_IP>" in text


def test_unordered_summary_is_not_sequence_eligible_input() -> None:
    events = events_from_behavior(
        "trace",
        {
            "calls_highlighted": ["CreateFileW", "RegSetValueExW"],
            "files_written": [r"C:\Users\bob\out.bin"],
        },
    )

    assert len(events) == 3
    assert all(event.ordering_known == 0 for event in events)
    assert all("bob" not in event.canonical_text for event in events)
    assert all("sha1" not in event.canonical_text for event in events)


def test_timestamped_events_are_sorted() -> None:
    events = events_from_behavior(
        "trace",
        {
            "ip_traffic": [
                {"timestamp": "2026-01-01T00:00:02Z", "destination_ip": "8.8.8.8"},
                {"timestamp": "2026-01-01T00:00:01Z", "destination_ip": "1.1.1.1"},
            ]
        },
    )

    assert [event.observed_at for event in events] == [
        "2026-01-01T00:00:01Z",
        "2026-01-01T00:00:02Z",
    ]
    assert all(event.ordering_known == 1 for event in events)
