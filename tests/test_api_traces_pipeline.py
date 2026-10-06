from __future__ import annotations

import hashlib
import io
import json
import math
import tarfile
from pathlib import Path
from typing import Any

from pe_research.data.api_traces_malware_detection.config import (
    ApiTraceConfig,
    ApiTracePilotConfig,
    ApiTraceSourceConfig,
    SampleEmbeddingConfig,
    load_api_trace_config,
)
from pe_research.data.api_traces_malware_detection.eda import run_api_trace_eda
from pe_research.data.api_traces_malware_detection.materialize import (
    _safe_member_sha,
    materialize_api_traces,
)
from pe_research.data.api_traces_malware_detection.normalization import normalize_api_event
from pe_research.data.api_traces_malware_detection.sample_embedding import (
    EncodedBatch,
    TokenCoverage,
    write_frozen_sample_embeddings,
)
from pe_research.data.api_traces_malware_detection.source import (
    build_candidate_manifest,
    download_resumable,
    load_single_label_metadata,
)
from pe_research.data.api_traces_malware_detection.validation import validate_api_trace_artifacts
from pe_research.data.io import read_csv


class TinySampleEncoder:
    def __init__(self) -> None:
        self.sequences: list[list[str]] = []

    @property
    def runtime_metadata(self) -> dict[str, object]:
        return {"device": "test", "separator_token": "[SEP]"}

    def encode(self, ordered_events: list[list[str]]) -> EncodedBatch:
        vectors: list[list[float]] = []
        coverage: list[TokenCoverage] = []
        for events in ordered_events:
            self.sequences.append(list(events))
            raw = [float(len(events) + index + 1) for index in range(8)]
            norm = math.sqrt(sum(value * value for value in raw))
            vectors.append([value / norm for value in raw])
            token_count = len(events) * 5 + 2
            coverage.append(
                TokenCoverage(
                    event_count=len(events),
                    original_token_count=token_count,
                    retained_token_count=token_count,
                    truncated=False,
                )
            )
        return EncodedBatch(vectors=vectors, coverage=coverage)


def _config(tmp_path: Path) -> ApiTraceConfig:
    return ApiTraceConfig(
        schema_version="test/api-traces/v1",
        workspace=tmp_path / "workspace",
        source=ApiTraceSourceConfig(
            record_id="test",
            version="1",
            metadata_url="https://example.invalid/metadata",
            metadata_filename="shas_by_families.json",
            metadata_md5="0" * 32,
            archive_url="https://example.invalid/archive",
            archive_filename="traces.tar.xz",
            archive_md5="0" * 32,
            minimum_free_gib=0,
        ),
        pilot=ApiTracePilotConfig(
            seed=20261002,
            target_benign=3,
            family_count=1,
            target_per_family=3,
            expected_families=("fam",),
            candidate_multiplier=1,
            minimum_events=3,
            maximum_events=4,
            maximum_trace_bytes=1024 * 1024,
            maximum_parse_error_fraction=0.05,
        ),
        sample_embedding=SampleEmbeddingConfig(
            model_name="fake",
            model_revision="1" * 40,
            dimension=8,
            maximum_tokens=64,
            batch_size=2,
            normalize=True,
            pooling="last_hidden_state_cls",
            input_format="api_name",
            device="cpu",
        ),
    )


def _trace_payload(sample_number: int) -> bytes:
    rows = []
    functions = ["NtOpenFile", "NtOpenFile", "NtWriteFile", "NtClose", "NtCreateSection"]
    for index, function in enumerate(functions):
        rows.append(
            {
                "ts": str(index),
                "vmi_ts": str(index),
                "vmi_FunctionName": function,
                "vmi_ModuleName": "ntdll.dll",
                "vmi_ProcessDtb": f"0x{sample_number:08x}",
                "vmi_ProcessTeb": f"0x{sample_number + 100:08x}",
                "vmi_Parameterlist": {
                    "FileName": f"C:\\Users\\alice\\tmp\\random-{sample_number}.exe",
                    "Handle": f"0x{sample_number + index:08x}",
                },
            }
        )
    return ("\n".join(json.dumps(row) for row in rows) + "\n").encode()


def _write_fixture(workspace: Path) -> list[str]:
    raw = workspace / "raw"
    raw.mkdir(parents=True)
    benign = [f"{index:064x}" for index in range(1, 4)]
    malicious = [f"{index:064x}" for index in range(101, 104)]
    multi = f"{999:064x}"
    metadata = {"benign": benign, "fam": [*malicious, multi], "other": [multi]}
    (raw / "shas_by_families.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    with tarfile.open(raw / "traces.tar.xz", "w:xz") as archive:
        for sample_number, sha in enumerate([*benign, *malicious], start=1):
            payload = _trace_payload(sample_number)
            info = tarfile.TarInfo(f"traces/{sha}.json")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return [*benign, *malicious]


def test_checked_in_api_trace_config_is_pinned() -> None:
    config = load_api_trace_config(Path("configs/data/api_traces_pilot_v1.yaml"))
    assert config.pilot.target_benign == 500
    assert config.pilot.target_per_family == 50
    assert config.pilot.maximum_events == 512
    assert config.sample_embedding.model_name == "Alibaba-NLP/gte-modernbert-base"
    assert config.sample_embedding.dimension == 768
    assert config.sample_embedding.maximum_tokens == 8192
    assert config.sample_embedding.pooling == "last_hidden_state_cls"
    assert config.sample_embedding.input_format == "api_name"
    assert len(config.pilot.expected_families) == 10
    assert config.source.archive_md5 == "03d9610cadb30ee5d6cad50251f111bf"


def test_metadata_excludes_multi_label_and_reports_it(tmp_path: Path) -> None:
    path = tmp_path / "labels.json"
    shared = "a" * 64
    path.write_text(
        json.dumps({"benign": ["b" * 64], "one": [shared], "two": [shared]}),
        encoding="utf-8",
    )
    labels, report = load_single_label_metadata(path)
    assert labels == {"b" * 64: "benign"}
    assert report["multi_label_sha256_excluded"] == 1


def test_normalization_masks_identifiers_and_keeps_trace_local_context() -> None:
    event = normalize_api_event(
        {
            "vmi_FunctionName": "NtCreateFile",
            "vmi_ModuleName": "NTDLL.DLL",
            "vmi_ProcessDtb": "0x12345678",
            "vmi_ProcessTeb": "0x87654321",
            "vmi_Parameterlist": {
                "Path": "C:\\Users\\secret-user\\dropper.exe",
                "Handle": "0xabcdef12",
                "Token": "do-not-keep",
            },
        },
        {},
        {},
    )
    assert event is not None
    text = str(event["canonical_text"])
    assert "secret-user" not in text
    assert "do-not-keep" not in text
    assert "abcdef12" not in text
    assert "<PROCESS_0>" in text
    assert "<THREAD_0>" in text


def test_archive_member_path_safety() -> None:
    sha = "a" * 64
    assert _safe_member_sha(f"traces/{sha}.json") == sha
    assert _safe_member_sha(f"../{sha}.json") is None
    assert _safe_member_sha(f"/absolute/{sha}.json") is None


def test_download_resumes_from_part_file(tmp_path: Path, monkeypatch: Any) -> None:
    destination = tmp_path / "source.bin"
    partial = destination.with_suffix(".bin.part")
    partial.write_bytes(b"abc")
    requests = []

    class Response(io.BytesIO):
        status = 206
        headers: dict[str, str] = {}

        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_args: object) -> None:
            self.close()

        def getcode(self) -> int:
            return self.status

    def fake_urlopen(request: Any, timeout: int) -> Response:
        requests.append((request, timeout))
        return Response(b"def")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    result = download_resumable(
        "https://example.invalid/source.bin",
        destination,
        hashlib.md5(b"abcdef").hexdigest(),
    )
    assert destination.read_bytes() == b"abcdef"
    assert result["resumed"] is True
    assert requests[0][0].get_header("Range") == "bytes=3-"


def test_download_reconnects_after_read_timeout(tmp_path: Path, monkeypatch: Any) -> None:
    destination = tmp_path / "source.bin"
    requests = []

    class FlakyResponse:
        status = 200
        headers = {"Content-Length": "6"}

        def __init__(self) -> None:
            self.read_count = 0

        def __enter__(self) -> FlakyResponse:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def getcode(self) -> int:
            return self.status

        def read(self, _size: int) -> bytes:
            self.read_count += 1
            if self.read_count == 1:
                return b"abc"
            raise TimeoutError("simulated timeout")

    class ResumedResponse(io.BytesIO):
        status = 206
        headers = {"Content-Range": "bytes 3-5/6"}

        def __enter__(self) -> ResumedResponse:
            return self

        def __exit__(self, *_args: object) -> None:
            self.close()

        def getcode(self) -> int:
            return self.status

    responses = [FlakyResponse(), ResumedResponse(b"def")]

    def fake_urlopen(request: Any, timeout: float) -> Any:
        requests.append((request, timeout))
        return responses.pop(0)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("time.sleep", lambda _seconds: None)
    messages: list[str] = []
    result = download_resumable(
        "https://example.invalid/source.bin",
        destination,
        hashlib.md5(b"abcdef").hexdigest(),
        retry_backoff_seconds=0,
        progress=messages.append,
    )
    assert destination.read_bytes() == b"abcdef"
    assert result["resumed"] is True
    assert requests[1][0].get_header("Range") == "bytes=3-"
    assert any("transient download error" in message for message in messages)


def test_small_archive_end_to_end(tmp_path: Path) -> None:
    config = _config(tmp_path)
    workspace = config.workspace
    source_shas = _write_fixture(workspace)
    candidates = build_candidate_manifest(config, workspace)
    assert len(candidates) == 6

    counts = materialize_api_traces(config, workspace)
    assert counts == {
        "samples": 6,
        "traces": 6,
        "events": 24,
        "benign": 3,
        "malicious": 3,
    }
    events = read_csv(workspace / "artifacts" / "events.csv")
    first_functions = [row["api_name"] for row in events[:4]]
    assert first_functions == ["NtOpenFile", "NtOpenFile", "NtWriteFile", "NtClose"]
    assert all(row["ordering_known"] == "1" for row in events)
    assert not any(sha in row["canonical_text"] for sha in source_shas for row in events)

    encoder = TinySampleEncoder()
    embedded = write_frozen_sample_embeddings(
        workspace / "artifacts" / "events.csv",
        workspace / "artifacts" / "samples.csv",
        workspace / "artifacts" / "sample_embeddings_gte_modernbert768.csv",
        workspace / "cache" / "sample_embeddings_gte_modernbert768.npy",
        workspace / "artifacts" / "sample_embedding_coverage.csv",
        workspace / "artifacts" / "sample_embedding_metadata.json",
        config.sample_embedding,
        workspace / "cache" / "huggingface",
        encoder=encoder,
    )
    assert embedded == 6
    assert len(encoder.sequences) == 6
    assert [
        text.split(" ", maxsplit=1)[0].removeprefix("api=")
        for text in encoder.sequences[0]
    ] == first_functions
    errors, quality = validate_api_trace_artifacts(
        config, workspace, require_complete=False
    )
    assert errors == []
    assert quality["counts"]["sample_embeddings"] == 6
    metadata = json.loads(
        (workspace / "artifacts" / "sample_embedding_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    assert metadata["frozen"] is True
    assert metadata["training_performed"] is False
    assert metadata["pooling"] == "last_hidden_state_cls"
    assert len(metadata["ordered_api_text_digest"]) == hashlib.sha256().digest_size * 2
    embedding_rows = read_csv(
        workspace / "artifacts" / "sample_embeddings_gte_modernbert768.csv"
    )
    assert set(embedding_rows[0]) == {
        "sample_id",
        *(f"embedding_{index:03d}" for index in range(8)),
    }
    eda_outputs = run_api_trace_eda(config, workspace)
    assert Path(eda_outputs["sample_binary"]).exists()
    assert Path(eda_outputs["sample_family"]).exists()
