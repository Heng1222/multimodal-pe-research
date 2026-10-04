# AU-PEMal-2025 dynamic-behavior pilot

This pipeline selects a reproducible AU-PEMal-2025 V2 pilot, performs read-only
VirusTotal hash lookups, normalizes cached behavior reports, and precomputes
label-free MiniLM embeddings. It never uploads, downloads, rescans, or executes
PE files.

## Security prerequisites

The VirusTotal key previously pasted into a conversation must be revoked. Put a
new key only in the current process environment as `VT_API_KEY`; never add it to
the configuration, command line, logs, or Git. The client only implements HTTP
GET against `/files/{hash}` and `/files/{sha256}/behaviours`.

The checked-in configuration enforces a minimum 16-second interval, a persisted
200-request ceiling for this pilot, cache-based resume, and immediate stops for
authentication and quota errors. A response cache and request ledger are written
under the ignored data workspace.

## Reproducible stages

From the repository root:

```powershell
uv sync --dev --extra embedding --extra eda
uv run pe-research data fetch
uv run pe-research data select-pilot

# Set the rotated key in this PowerShell session without checking it into a file.
$env:VT_API_KEY = "<rotated-key>"
uv run pe-research data enrich-vt
Remove-Item Env:VT_API_KEY

uv run pe-research data materialize
uv run pe-research data embed
uv run pe-research data validate --require-minimum
uv run pe-research data eda
```

Each command is independently resumable. Do not delete `raw/vt`,
`lookup_results.csv`, or `logs/vt_requests.jsonl` between enrichment runs.
VirusTotal 404 responses are cached, so they are not requested again.

## Output contract

Generated data lives under `data/au_pemal_2025/pilot_v1/`, which is ignored by
Git. `raw/` holds the immutable source and complete VT JSON responses. The
`artifacts/` directory holds `samples.csv`, `source_features.csv`, `traces.csv`,
`events.csv`, `sample_labels.csv`, `concept_labels.csv`, and
`event_embeddings_minilm384.csv`, plus model metadata, a quality report, and the
snapshot manifest.

`event_embeddings_minilm384.csv` contains only `trace_id`, `event_index`, and
384 floating-point embedding columns. Labels stay in separate files. A trace
without trustworthy timestamps has `ordering_known=0` and
`sequence_eligible=0`; its generated row order must not be interpreted as time.

The non-live test suite is safe to run without a key:

```powershell
uv run pytest
uv run ruff check .
uv run mypy
```

After enrichment has at least one candidate from each class, an opt-in live smoke
test consumes at most four requests:

```powershell
$env:RUN_VT_LIVE = "1"
uv run pytest -m live tests/live/test_vt_smoke.py
Remove-Item Env:RUN_VT_LIVE
```
