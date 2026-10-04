"""Command-line entry point for reproducible PE research data stages."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from pe_research.data.api_traces.config import (
    ApiTraceConfig,
    load_api_trace_config,
    resolve_api_workspace,
)
from pe_research.data.api_traces.eda import run_api_trace_eda
from pe_research.data.api_traces.materialize import materialize_api_traces
from pe_research.data.api_traces.sample_embedding import write_frozen_sample_embeddings
from pe_research.data.api_traces.source import (
    DownloadInterruptedError,
    build_candidate_manifest,
    fetch_sources,
)
from pe_research.data.api_traces.validation import (
    validate_api_trace_artifacts,
    write_api_snapshot,
)
from pe_research.data.config import PipelineConfig, load_config, resolve_workspace
from pe_research.data.eda import run_eda
from pe_research.data.embedding import write_event_embeddings
from pe_research.data.io import atomic_json, fetch_file
from pe_research.data.materialize import materialize
from pe_research.data.sampling import build_candidates, write_source_features
from pe_research.data.validation import validate_artifacts, write_snapshot
from pe_research.data.virustotal import (
    AuthenticationError,
    QuotaError,
    RequestBudgetExhausted,
    VirusTotalError,
    enrich_candidates,
)

app = typer.Typer(help="Multimodal PE research utilities.")
data_app = typer.Typer(help="AU-PEMal pilot data pipeline.")
api_traces_app = typer.Typer(help="Ordered Zenodo API-trace pilot pipeline.")
app.add_typer(data_app, name="data")
app.add_typer(api_traces_app, name="api-traces")

DEFAULT_CONFIG = Path("configs/data/au_pemal_pilot_v1.yaml")
DEFAULT_API_TRACE_CONFIG = Path("configs/data/api_traces_pilot_v1.yaml")
ConfigOption = Annotated[Path, typer.Option("--config")]
MinimumOption = Annotated[bool, typer.Option("--require-minimum")]


def _context(config_path: Path) -> tuple[PipelineConfig, Path]:
    config = load_config(config_path)
    workspace = resolve_workspace(config, Path.cwd())
    return config, workspace


def _api_context(config_path: Path) -> tuple[ApiTraceConfig, Path]:
    config = load_api_trace_config(config_path)
    workspace = resolve_api_workspace(config, Path.cwd())
    return config, workspace


def _require_api_artifact(path: Path, instruction: str) -> None:
    if path.exists():
        return
    partial = path.with_suffix(path.suffix + ".part")
    partial_note = ""
    if partial.exists():
        partial_note = f" ({partial.stat().st_size} bytes are safely saved in {partial.name})"
    typer.echo(f"missing prerequisite: {path}{partial_note}; {instruction}", err=True)
    raise typer.Exit(2)


def _fetch_api_sources(
    config: ApiTraceConfig,
    workspace: Path,
    *,
    metadata_only: bool,
) -> dict[str, object]:
    try:
        return fetch_sources(
            config,
            workspace,
            metadata_only=metadata_only,
            progress=typer.echo,
        )
    except (DownloadInterruptedError, OSError, ValueError) as error:
        typer.echo(f"fetch stopped: {error}", err=True)
        typer.echo("The .part file was preserved; rerun the same fetch command.", err=True)
        raise typer.Exit(2) from error


@data_app.command("fetch")
def fetch(config_path: ConfigOption = DEFAULT_CONFIG) -> None:
    """Download and checksum the pinned public AU-PEMal CSV."""
    config, workspace = _context(config_path)
    destination = workspace / "raw" / config.source.filename
    checksum = fetch_file(config.source.url, destination, config.source.expected_sha256)
    atomic_json(
        workspace / "raw" / "source_provenance.json",
        {
            "name": config.source.name,
            "commit": config.source.commit,
            "url": config.source.url,
            "sha256": checksum,
        },
    )
    typer.echo(f"source ready: {destination} ({checksum})")


@data_app.command("select-pilot")
def select_pilot(config_path: ConfigOption = DEFAULT_CONFIG) -> None:
    """Build a deterministic candidate queue and source feature table."""
    config, workspace = _context(config_path)
    source = workspace / "raw" / config.source.filename
    candidates = build_candidates(
        source,
        workspace / "artifacts" / "candidates.csv",
        seed=config.pilot.seed,
        max_candidates=config.pilot.max_candidates,
        required_columns=config.source.required_columns,
        exclude_conflicts=True,
        validation_path=workspace / "artifacts" / "source_validation.json",
    )
    feature_count = write_source_features(
        source,
        candidates,
        workspace / "artifacts" / "source_features.csv",
    )
    typer.echo(f"selected {len(candidates)} candidates and {feature_count} feature rows")


@data_app.command("enrich-vt")
def enrich_vt(config_path: ConfigOption = DEFAULT_CONFIG) -> None:
    """Query existing VT reports with no upload or rescan capability."""
    config, workspace = _context(config_path)
    state_path = workspace / "artifacts" / "enrichment_state.json"
    try:
        results = enrich_candidates(
            workspace / "artifacts" / "candidates.csv",
            workspace / "artifacts" / "lookup_results.csv",
            workspace / "raw" / "vt",
            workspace / "logs" / "vt_requests.jsonl",
            config.pilot,
            config.virustotal,
        )
    except QuotaError as error:
        atomic_json(
            state_path,
            {"status": "stopped", "reason": "quota", "retry_after": error.retry_after},
        )
        suffix = f"; Retry-After={error.retry_after}" if error.retry_after else ""
        typer.echo(f"VT quota stop; progress saved{suffix}", err=True)
        raise typer.Exit(3) from error
    except (AuthenticationError, RequestBudgetExhausted, VirusTotalError) as error:
        if isinstance(error, AuthenticationError):
            reason = "authentication_or_permission"
        elif isinstance(error, RequestBudgetExhausted):
            reason = "local_request_budget"
        else:
            reason = "virustotal_error"
        atomic_json(state_path, {"status": "stopped", "reason": reason})
        typer.echo(str(error), err=True)
        raise typer.Exit(2) from error
    success = sum(item.lookup_status == "success" for item in results)
    atomic_json(
        state_path,
        {
            "status": "complete",
            "reason": "targets_met_or_candidates_exhausted",
            "successful_candidates": success,
            "processed_candidates": len(results),
        },
    )
    typer.echo(f"VT enrichment complete: {success}/{len(results)} successful candidates")


@data_app.command("materialize")
def materialize_command(
    config_path: ConfigOption = DEFAULT_CONFIG,
) -> None:
    """Create normalized, label-separated CSV artifacts."""
    config, workspace = _context(config_path)
    counts = materialize(config, workspace)
    typer.echo(f"materialized: {counts}")


@data_app.command("embed")
def embed(config_path: ConfigOption = DEFAULT_CONFIG) -> None:
    """Precompute pinned MiniLM embeddings for canonical events."""
    config, workspace = _context(config_path)
    artifacts = workspace / "artifacts"
    try:
        count = write_event_embeddings(
            artifacts / "events.csv",
            artifacts / "event_embeddings_minilm384.csv",
            artifacts / "embedding_metadata.json",
            config.embedding,
            workspace / "cache" / "huggingface",
        )
    except RuntimeError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(2) from error
    typer.echo(f"embedded {count} events")


@data_app.command("validate")
def validate(
    config_path: ConfigOption = DEFAULT_CONFIG,
    require_minimum: MinimumOption = False,
) -> None:
    """Validate joins, leakage barriers, dimensions, and pilot coverage."""
    config, workspace = _context(config_path)
    errors, quality = validate_artifacts(
        config,
        workspace,
        require_minimum=require_minimum,
    )
    write_snapshot(config, workspace, quality)
    if errors:
        for error in errors:
            typer.echo(f"ERROR: {error}", err=True)
        raise typer.Exit(1)
    typer.echo("dataset artifacts validated")


@data_app.command("eda")
def eda(config_path: ConfigOption = DEFAULT_CONFIG) -> None:
    """Generate summary tables and interactive 3D UMAP visualizations."""
    config, workspace = _context(config_path)
    try:
        outputs = run_eda(config, workspace)
    except RuntimeError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(2) from error
    typer.echo(f"EDA complete: {outputs['report']}")


@api_traces_app.command("fetch")
def api_fetch(
    config_path: ConfigOption = DEFAULT_API_TRACE_CONFIG,
    metadata_only: Annotated[bool, typer.Option("--metadata-only")] = False,
) -> None:
    """Resume and verify the Zenodo metadata and trace archive downloads."""
    config, workspace = _api_context(config_path)
    result = _fetch_api_sources(config, workspace, metadata_only=metadata_only)
    typer.echo(f"API-trace sources ready: {result['files']}")


@api_traces_app.command("select")
def api_select(config_path: ConfigOption = DEFAULT_API_TRACE_CONFIG) -> None:
    """Exclude multi-label SHA values and create deterministic reserve candidates."""
    config, workspace = _api_context(config_path)
    _require_api_artifact(
        workspace / "raw" / config.source.metadata_filename,
        "run `pe-research api-traces fetch --metadata-only` first",
    )
    rows = build_candidate_manifest(config, workspace)
    typer.echo(f"selected {len(rows)} reserve candidates")


@api_traces_app.command("materialize")
def api_materialize(config_path: ConfigOption = DEFAULT_API_TRACE_CONFIG) -> None:
    """Stream the archive and materialize 1,000 clean ordered traces."""
    config, workspace = _api_context(config_path)
    _require_api_artifact(
        workspace / "artifacts" / "candidate_manifest.csv",
        "run `pe-research api-traces select` first",
    )
    _require_api_artifact(
        workspace / "raw" / config.source.archive_filename,
        "rerun `pe-research api-traces fetch` until checksum verification completes",
    )
    try:
        counts = materialize_api_traces(config, workspace)
    except ValueError as error:
        typer.echo(f"materialization stopped: {error}", err=True)
        raise typer.Exit(2) from error
    typer.echo(f"materialized API traces: {counts}")


@api_traces_app.command("embed")
def api_embed(config_path: ConfigOption = DEFAULT_API_TRACE_CONFIG) -> None:
    """Create one frozen GTE 768-D CLS embedding per ordered API trace."""
    config, workspace = _api_context(config_path)
    artifacts = workspace / "artifacts"
    cache = workspace / "cache"
    _require_api_artifact(
        artifacts / "events.csv",
        "run `pe-research api-traces materialize` first",
    )
    _require_api_artifact(
        artifacts / "samples.csv",
        "run `pe-research api-traces materialize` first",
    )
    try:
        count = write_frozen_sample_embeddings(
            artifacts / "events.csv",
            artifacts / "samples.csv",
            artifacts / "sample_embeddings_gte_modernbert768.csv",
            cache / "sample_embeddings_gte_modernbert768.npy",
            artifacts / "sample_embedding_coverage.csv",
            artifacts / "sample_embedding_metadata.json",
            config.sample_embedding,
            cache / "huggingface",
            progress=typer.echo,
        )
    except (RuntimeError, ValueError) as error:
        typer.echo(f"embedding stopped: {error}", err=True)
        raise typer.Exit(2) from error
    typer.echo(f"embedded {count} ordered API traces with frozen GTE CLS")


@api_traces_app.command("validate")
def api_validate(
    config_path: ConfigOption = DEFAULT_API_TRACE_CONFIG,
    require_complete: Annotated[bool, typer.Option("--require-complete/--allow-partial")] = True,
) -> None:
    """Validate balance, joins, ordering, leakage barriers, and dimensions."""
    config, workspace = _api_context(config_path)
    errors, quality = validate_api_trace_artifacts(
        config, workspace, require_complete=require_complete
    )
    write_api_snapshot(config, workspace, quality)
    if errors:
        for error in errors:
            typer.echo(f"ERROR: {error}", err=True)
        raise typer.Exit(1)
    typer.echo("API-trace dataset artifacts validated")


@api_traces_app.command("eda")
def api_eda(config_path: ConfigOption = DEFAULT_API_TRACE_CONFIG) -> None:
    """Produce EDA plus interactive sample-level GTE CLS 3D UMAP plots."""
    config, workspace = _api_context(config_path)
    _require_api_artifact(
        workspace / "artifacts" / "sample_embeddings_gte_modernbert768.csv",
        "run the fetch, materialize, and embed stages first",
    )
    try:
        outputs = run_api_trace_eda(config, workspace)
    except (RuntimeError, ValueError) as error:
        typer.echo(f"EDA stopped: {error}", err=True)
        raise typer.Exit(2) from error
    typer.echo(f"API-trace EDA complete: {outputs['report']}")


@api_traces_app.command("all")
def api_all(
    config_path: ConfigOption = DEFAULT_API_TRACE_CONFIG,
    skip_eda: Annotated[bool, typer.Option("--skip-eda")] = False,
) -> None:
    """Run the end-to-end API-trace pilot pipeline."""
    config, workspace = _api_context(config_path)
    _fetch_api_sources(config, workspace, metadata_only=False)
    build_candidate_manifest(config, workspace)
    materialize_api_traces(config, workspace)
    artifacts = workspace / "artifacts"
    cache = workspace / "cache"
    write_frozen_sample_embeddings(
        artifacts / "events.csv",
        artifacts / "samples.csv",
        artifacts / "sample_embeddings_gte_modernbert768.csv",
        cache / "sample_embeddings_gte_modernbert768.npy",
        artifacts / "sample_embedding_coverage.csv",
        artifacts / "sample_embedding_metadata.json",
        config.sample_embedding,
        cache / "huggingface",
        progress=typer.echo,
    )
    errors, quality = validate_api_trace_artifacts(config, workspace)
    write_api_snapshot(config, workspace, quality)
    if errors:
        for error in errors:
            typer.echo(f"ERROR: {error}", err=True)
        raise typer.Exit(1)
    if not skip_eda:
        run_api_trace_eda(config, workspace)
    typer.echo("end-to-end API-trace pilot complete")
