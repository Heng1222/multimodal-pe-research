"""Precompute version-pinned MiniLM event embeddings as a standalone CSV."""

from __future__ import annotations

import hashlib
import importlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol, cast

from pe_research.data.config import EmbeddingConfig
from pe_research.data.io import atomic_json, package_version, read_csv, write_csv
from pe_research.data.normalization import NORMALIZATION_VERSION


class TextEncoder(Protocol):
    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...


class SentenceTransformerEncoder:
    def __init__(self, config: EmbeddingConfig, cache_folder: Path) -> None:
        try:
            module = importlib.import_module("sentence_transformers")
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "embedding dependencies are missing; run `uv sync --extra embedding`"
            ) from error
        model_class = cast(Any, module.SentenceTransformer)
        self.model = model_class(
            config.model_name,
            revision=config.model_revision,
            device="cpu",
            cache_folder=str(cache_folder),
        )
        self.config = config

    def encode(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        result = self.model.encode(
            list(texts),
            batch_size=self.config.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalize,
        )
        return cast(list[list[float]], result.tolist())


def write_event_embeddings(
    events_path: Path,
    output_path: Path,
    metadata_path: Path,
    config: EmbeddingConfig,
    cache_folder: Path,
    *,
    encoder: TextEncoder | None = None,
) -> int:
    events = read_csv(events_path)
    texts = [row["canonical_text"] for row in events]
    if texts:
        actual_encoder = encoder or SentenceTransformerEncoder(config, cache_folder)
        vectors = list(actual_encoder.encode(texts))
    else:
        vectors = []
    if len(vectors) != len(events):
        raise ValueError("embedding row count does not match event row count")

    dimensions = [f"embedding_{index:03d}" for index in range(config.dimension)]
    rows: list[dict[str, object]] = []
    content_digest = hashlib.sha256()
    for event, vector in zip(events, vectors, strict=True):
        if len(vector) != config.dimension:
            raise ValueError(
                f"expected {config.dimension} embedding dimensions, got {len(vector)}"
            )
        content_digest.update(event["trace_id"].encode())
        content_digest.update(event["event_index"].encode())
        content_digest.update(event["canonical_text"].encode())
        row: dict[str, object] = {
            "trace_id": event["trace_id"],
            "event_index": event["event_index"],
        }
        row.update(
            {
                name: format(float(value), ".9g")
                for name, value in zip(dimensions, vector, strict=True)
            }
        )
        rows.append(row)

    count = write_csv(output_path, ["trace_id", "event_index", *dimensions], rows)
    atomic_json(
        metadata_path,
        {
            "model_name": config.model_name,
            "model_revision": config.model_revision,
            "dimension": config.dimension,
            "normalize_embeddings": config.normalize,
            "normalization_version": NORMALIZATION_VERSION,
            "canonical_text_digest": content_digest.hexdigest(),
            "sentence_transformers_version": package_version("sentence-transformers"),
            "torch_version": package_version("torch"),
            "row_count": count,
        },
    )
    return count
