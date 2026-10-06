from __future__ import annotations

import csv
from pathlib import Path

from pe_research.data.au_pemal_2025.config import (
    EmbeddingConfig,
    PilotConfig,
    PipelineConfig,
    SourceConfig,
    VirusTotalConfig,
)
from pe_research.data.au_pemal_2025.embedding import write_event_embeddings
from pe_research.data.au_pemal_2025.materialize import materialize
from pe_research.data.au_pemal_2025.records import CandidateRecord
from pe_research.data.au_pemal_2025.validation import validate_artifacts
from pe_research.data.au_pemal_2025.virustotal import LookupResult
from pe_research.data.io import atomic_json, read_csv, write_csv


class FakeEncoder:
    def encode(self, texts: list[str]) -> list[list[float]]:
        return [[float(index % 7) / 7 for index in range(384)] for _text in texts]


def _config() -> PipelineConfig:
    return PipelineConfig(
        schema_version="test/v1",
        workspace=Path("unused"),
        source=SourceConfig(
            name="test",
            commit="a" * 40,
            url="https://example.invalid/source.csv",
            filename="source.csv",
            expected_sha256="",
            required_columns=("sha1", "Class", "Category", "Family"),
        ),
        pilot=PilotConfig(
            seed=1,
            target_benign=1,
            target_malicious_per_category=1,
            max_candidates=2,
            minimum_per_class=1,
            minimum_events_per_trace=5,
        ),
        virustotal=VirusTotalConfig(
            base_url="https://example.invalid",
            api_key_env="VT_API_KEY",
            minimum_interval_seconds=16,
            request_budget=2,
            behaviours_limit=40,
            max_retries=0,
        ),
        embedding=EmbeddingConfig(
            model_name="fake",
            model_revision="b" * 40,
            dimension=384,
            normalize=True,
            batch_size=2,
        ),
    )


def test_materialize_embed_and_validate_round_trip(tmp_path: Path) -> None:
    config = _config()
    workspace = tmp_path / "workspace"
    raw = workspace / "raw"
    artifacts = workspace / "artifacts"
    raw.mkdir(parents=True)
    artifacts.mkdir(parents=True)
    (raw / "source.csv").write_text("sha1,Class,Category,Family\n", encoding="utf-8")

    candidate = CandidateRecord(0, f"sha1:{'1' * 40}", "1" * 40, "Malware", "RAT", "x", "train")
    write_csv(
        artifacts / "candidates.csv",
        list(CandidateRecord.__dataclass_fields__),
        [candidate.as_row()],
    )
    write_csv(
        artifacts / "source_features.csv",
        ["sample_key", "feature"],
        [
            {
                "sample_key": candidate.sample_key,
                "feature": "1",
            }
        ],
    )
    result = LookupResult(
        0,
        candidate.sample_key,
        candidate.source_sha1,
        "2" * 64,
        "Malware",
        "RAT",
        "x",
        "train",
        "success",
        1,
        1,
        "",
    )
    write_csv(
        artifacts / "lookup_results.csv",
        list(LookupResult.__dataclass_fields__),
        [result.as_row()],
    )
    behavior_path = raw / "vt" / "behaviours" / f"{'2' * 64}.json"
    atomic_json(
        behavior_path,
        {
            "data": [
                {
                    "id": f"{'2' * 64}_sandbox",
                    "attributes": {
                        "sandbox_name": "sandbox",
                        "analysis_date": 1,
                        "calls_highlighted": ["a", "b", "c"],
                        "files_written": ["d", "e", "f"],
                        "mitre_attack_techniques": [{"id": "T1059"}],
                    },
                }
            ]
        },
    )

    counts = materialize(config, workspace)
    assert counts["events"] == 6
    events = read_csv(artifacts / "events.csv")
    assert all(row["ordering_known"] == "0" for row in events)
    assert all("Malware" not in row["canonical_text"] for row in events)

    embedded = write_event_embeddings(
        artifacts / "events.csv",
        artifacts / "event_embeddings_minilm384.csv",
        artifacts / "embedding_metadata.json",
        config.embedding,
        workspace / "cache",
        encoder=FakeEncoder(),
    )
    assert embedded == 6
    errors, quality = validate_artifacts(config, workspace)
    assert errors == []
    assert quality["counts"]["events"] == 6

    with (artifacts / "event_embeddings_minilm384.csv").open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        header = next(csv.reader(stream))
    assert len([column for column in header if column.startswith("embedding_")]) == 384
    assert not {"y", "category", "family", "sha1", "sha256"} & set(header)
