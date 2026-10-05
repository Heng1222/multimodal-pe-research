"""Lazy experiment commands with automatic run-directory selection."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(help="API embedding dual-branch AE Pilot.")
DEFAULT_CONFIG = Path("configs/experiment/api_dual_branch_pilot_v1.yaml")
RunOption = Annotated[Path | None, typer.Option("--run", exists=True, file_okay=False)]


def latest_run(marker: str) -> Path:
    from pe_research.data.io import load_json

    candidates = sorted(Path("experiment").glob("api_ae_*"), reverse=True)
    for candidate in candidates:
        policy = candidate / "optimization_candidate.json"
        if policy.exists() and load_json(policy)["test_locked"]:
            continue
        if candidate.is_dir() and (candidate / marker).exists():
            return candidate
    raise typer.BadParameter("沒有可用實驗；執行 pe-research experiment run 自動建立完整實驗。")


@app.command(name="run")
def run_all(config: Annotated[Path, typer.Option(exists=True)] = DEFAULT_CONFIG) -> None:
    """Automatically prepare, train, evaluate, and visualize a new experiment."""
    from pe_research.experiment.pilot import evaluate, prepare, train
    from pe_research.experiment.visualization import visualize

    run = prepare(config)
    typer.echo(f"實驗資料夾：{run}")
    train(run)
    evaluate(run)
    visualize(run)
    typer.echo(f"報告：{run / 'PILOT_REPORT.md'}")
    typer.echo(f"UMAP：{run / 'visualization/latent_3d.html'}")


@app.command()
def optimize(run: RunOption = None) -> None:
    """Calibrate four configurations using validation, then evaluate the winner."""
    from pe_research.experiment.optimization import optimize as execute

    selected = execute(run or latest_run("metrics.json"))
    typer.echo(selected / "PILOT_REPORT.md")
    typer.echo(selected / "visualization/latent_3d.html")


@app.command()
def prepare(config: Annotated[Path, typer.Option(exists=True)] = DEFAULT_CONFIG) -> None:
    from pe_research.experiment.pilot import prepare as execute

    typer.echo(execute(config))


@app.command()
def train(run: RunOption = None) -> None:
    from pe_research.experiment.pilot import prepare
    from pe_research.experiment.pilot import train as execute

    if run is None:
        candidates = sorted(Path("experiment").glob("api_ae_*"), reverse=True)
        newest = candidates[0] if candidates else None
        run = (
            newest
            if (
                newest is not None
                and (newest / "prepared.json").exists()
                and not (Path("model") / newest.name).exists()
            )
            else None
        )
        if run is None:
            run = prepare()
    typer.echo(f"實驗資料夾：{run}")
    typer.echo(execute(run))


@app.command()
def evaluate(run: RunOption = None) -> None:
    from pe_research.experiment.pilot import evaluate as execute
    from pe_research.experiment.report import write_report

    selected = run or latest_run("trained.json")
    if run is None and (selected / "metrics.json").exists():
        write_report(selected)
    else:
        execute(selected)
    typer.echo(selected / "PILOT_REPORT.md")


@app.command()
def visualize(run: RunOption = None) -> None:
    from pe_research.experiment.visualization import visualize as execute

    selected = run or latest_run("metrics.json")
    existing = selected / "visualization/latent_3d.html"
    if run is None and existing.exists():
        typer.echo(existing)
    else:
        typer.echo(execute(selected))


@app.command()
def baseline(run: RunOption = None) -> None:
    """Add independent binary/family MLP controls to an existing experiment."""
    from pe_research.experiment.baseline import evaluate_baselines, train_baselines
    from pe_research.experiment.report import write_report

    selected = run or latest_run("trained.json")
    if not (selected / "mlp_baseline/trained.json").exists():
        train_baselines(selected)
    evaluate_baselines(selected)
    if (selected / "metrics.json").exists():
        write_report(selected)
    typer.echo(selected / "PILOT_REPORT.md")


@app.command()
def report(run: RunOption = None) -> None:
    """Refresh the report from saved results; no training or test reevaluation."""
    from pe_research.experiment.report import write_report

    typer.echo(write_report(run or latest_run("metrics.json")))


@app.command()
def predict(
    model_dir: Annotated[Path | None, typer.Option(exists=True, file_okay=False)] = None,
    embeddings: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    metadata: Annotated[Path | None, typer.Option(exists=True, dir_okay=False)] = None,
    output: Annotated[Path | None, typer.Option()] = None,
) -> None:
    from pe_research.data.io import load_json
    from pe_research.experiment.pilot import predict as execute

    selected = None
    if model_dir is None:
        selected = latest_run("trained.json")
        model_dir = Path("model") / selected.name
    if embeddings is None or metadata is None:
        if selected is None:
            selected = Path("experiment") / model_dir.name
        source = Path(load_json(selected / "prepared.json")["source"])
        embeddings = embeddings or source / "sample_embeddings_gte_modernbert768.csv"
        metadata = metadata or source / "sample_embedding_metadata.json"
    if output is None:
        from datetime import datetime, timedelta, timezone

        directory = selected or Path("experiment") / model_dir.name
        stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%dT%H%M%S%f")
        output = directory / f"inference_predictions_{stamp}.csv"
    execute(model_dir, embeddings, metadata, output)
    typer.echo(output)
