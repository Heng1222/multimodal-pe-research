"""Reproducible EDA and interactive UMAP visualizations for the pilot dataset."""

from __future__ import annotations

import csv
import importlib
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from pe_research.data.au_pemal_2025.config import PipelineConfig
from pe_research.data.io import atomic_json, read_csv, write_csv

BENIGN_COLOR = "#9E9E9E"
MALICIOUS_COLOR = "#D62728"


def _modules() -> tuple[Any, Any, Any]:
    try:
        numpy = importlib.import_module("numpy")
        plotly_graph_objects = importlib.import_module("plotly.graph_objects")
        plotly_colors = importlib.import_module("plotly.colors")
        umap = importlib.import_module("umap")
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "EDA dependencies are missing; run `uv sync --extra embedding --extra eda`"
        ) from error
    return numpy, (plotly_graph_objects, plotly_colors), umap


def _load_embeddings(
    path: Path,
    expected_rows: int,
    dimension: int,
    numpy: Any,
) -> tuple[Any, list[tuple[str, str]]]:
    matrix = numpy.empty((expected_rows, dimension), dtype=numpy.float32)
    keys: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        embedding_start = header.index("embedding_000")
        if len(header) - embedding_start != dimension:
            raise ValueError("embedding CSV dimension does not match the configured dimension")
        for row_index, row in enumerate(reader):
            if row_index >= expected_rows:
                raise ValueError("embedding CSV contains more rows than events.csv")
            keys.append((row[0], row[1]))
            matrix[row_index] = numpy.asarray(row[embedding_start:], dtype=numpy.float32)
    if len(keys) != expected_rows:
        raise ValueError("embedding CSV and events.csv row counts differ")
    return matrix, keys


def _umap_3d(matrix: Any, seed: int, umap: Any) -> Any:
    if len(matrix) < 4:
        raise ValueError("at least four rows are required for a 3D UMAP")
    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=min(15, len(matrix) - 1),
        min_dist=0.1,
        metric="cosine",
        random_state=seed,
        transform_seed=seed,
        n_jobs=1,
    )
    return reducer.fit_transform(matrix)


def _family_colors(families: list[str], plotly_colors: Any) -> dict[str, str]:
    palette = list(plotly_colors.qualitative.Alphabet) + list(
        plotly_colors.qualitative.Dark24
    )
    malicious = sorted({family for family in families if family != "Benign"})
    result = {"Benign": BENIGN_COLOR}
    result.update(
        {family: palette[index % len(palette)] for index, family in enumerate(malicious)}
    )
    return result


def _write_event_plot(
    coords: Any,
    metadata: list[dict[str, str]],
    output: Path,
    *,
    color_by: str,
    graph_objects: Any,
    plotly_colors: Any,
) -> None:
    if color_by == "class":
        groups = ["Benign", "Malicious"]
        colors = {"Benign": BENIGN_COLOR, "Malicious": MALICIOUS_COLOR}
        title = "Event embeddings — Benign vs Malicious"
    else:
        malicious_families = {
            row["family"] for row in metadata if row["family"] != "Benign"
        }
        groups = ["Benign", *sorted(malicious_families)]
        colors = _family_colors([row["family"] for row in metadata], plotly_colors)
        title = "Event embeddings — Malware family (Benign in gray)"

    figure = graph_objects.Figure()
    for group in groups:
        indices = [
            index
            for index, row in enumerate(metadata)
            if row[color_by] == group
        ]
        if not indices:
            continue
        customdata = [
            [
                metadata[index]["sample_key"],
                metadata[index]["class"],
                metadata[index]["category"],
                metadata[index]["family"],
                metadata[index]["split"],
                metadata[index]["event_type"],
                metadata[index]["operation"],
                metadata[index]["canonical_text"][:240],
            ]
            for index in indices
        ]
        figure.add_trace(
            graph_objects.Scatter3d(
                x=coords[indices, 0],
                y=coords[indices, 1],
                z=coords[indices, 2],
                mode="markers",
                name=group,
                marker={"size": 2.3, "opacity": 0.55, "color": colors[group]},
                customdata=customdata,
                hovertemplate=(
                    "sample=%{customdata[0]}<br>class=%{customdata[1]}"
                    "<br>category=%{customdata[2]}<br>family=%{customdata[3]}"
                    "<br>split=%{customdata[4]}<br>event=%{customdata[5]}"
                    " / %{customdata[6]}<br>%{customdata[7]}<extra></extra>"
                ),
            )
        )
    _finish_figure(figure, title, output)


def _write_sample_plot(
    coords: Any,
    metadata: list[dict[str, str]],
    output: Path,
    *,
    color_by: str,
    graph_objects: Any,
    plotly_colors: Any,
) -> None:
    if color_by == "class":
        groups = ["Benign", "Malicious"]
        colors = {"Benign": BENIGN_COLOR, "Malicious": MALICIOUS_COLOR}
        title = "Mean-pooled sample embeddings — Benign vs Malicious"
    else:
        malicious_families = {
            row["family"] for row in metadata if row["family"] != "Benign"
        }
        groups = ["Benign", *sorted(malicious_families)]
        colors = _family_colors([row["family"] for row in metadata], plotly_colors)
        title = "Mean-pooled samples — Malware family (Benign in gray)"

    figure = graph_objects.Figure()
    for group in groups:
        indices = [
            index
            for index, row in enumerate(metadata)
            if row[color_by] == group
        ]
        if not indices:
            continue
        customdata = [
            [
                metadata[index]["sample_key"],
                metadata[index]["class"],
                metadata[index]["category"],
                metadata[index]["family"],
                metadata[index]["split"],
                metadata[index]["event_count"],
            ]
            for index in indices
        ]
        figure.add_trace(
            graph_objects.Scatter3d(
                x=coords[indices, 0],
                y=coords[indices, 1],
                z=coords[indices, 2],
                mode="markers",
                name=group,
                marker={
                    "size": 7,
                    "opacity": 0.85,
                    "color": colors[group],
                    "line": {"width": 0.5, "color": "#333333"},
                },
                customdata=customdata,
                hovertemplate=(
                    "sample=%{customdata[0]}<br>class=%{customdata[1]}"
                    "<br>category=%{customdata[2]}<br>family=%{customdata[3]}"
                    "<br>split=%{customdata[4]}<br>events=%{customdata[5]}"
                    "<extra></extra>"
                ),
            )
        )
    _finish_figure(figure, title, output)


def _finish_figure(figure: Any, title: str, output: Path) -> None:
    figure.update_layout(
        title=(
            f"{title}<br><sup>UMAP coordinates are visualization-only "
            "and have no temporal meaning.</sup>"
        ),
        template="plotly_white",
        scene={
            "xaxis_title": "UMAP-1",
            "yaxis_title": "UMAP-2",
            "zaxis_title": "UMAP-3",
        },
        legend={"itemsizing": "constant"},
        margin={"l": 0, "r": 0, "b": 0, "t": 70},
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(str(output), include_plotlyjs="directory", full_html=True)


def _numeric_feature_summary(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    if not rows:
        return []
    result: list[dict[str, object]] = []
    for field in rows[0]:
        if field == "sample_key":
            continue
        values: list[float] = []
        missing = 0
        for row in rows:
            raw = row[field].strip()
            if not raw:
                missing += 1
                continue
            try:
                value = float(raw)
            except ValueError:
                missing += 1
                continue
            if math.isfinite(value):
                values.append(value)
            else:
                missing += 1
        result.append(
            {
                "feature": field,
                "count": len(values),
                "missing": missing,
                "missing_fraction": format(missing / len(rows), ".6g"),
                "min": format(min(values), ".9g") if values else "",
                "mean": format(statistics.fmean(values), ".9g") if values else "",
                "median": format(statistics.median(values), ".9g") if values else "",
                "max": format(max(values), ".9g") if values else "",
            }
        )
    return result


def _table(counter: Counter[str]) -> str:
    lines = ["| Value | Count |", "|---|---:|"]
    lines.extend(f"| {name} | {count} |" for name, count in counter.most_common())
    return "\n".join(lines)


def run_eda(config: PipelineConfig, workspace: Path) -> dict[str, str]:
    numpy, plotly_modules, umap = _modules()
    graph_objects, plotly_colors = plotly_modules
    artifacts = workspace / "artifacts"
    output = artifacts / "eda"
    output.mkdir(parents=True, exist_ok=True)

    samples = read_csv(artifacts / "samples.csv")
    labels = read_csv(artifacts / "sample_labels.csv")
    traces = read_csv(artifacts / "traces.csv")
    events = read_csv(artifacts / "events.csv")
    source_features = read_csv(artifacts / "source_features.csv")
    label_by_sample = {row["sample_key"]: row for row in labels}
    trace_by_id = {row["trace_id"]: row for row in traces}
    event_by_key = {(row["trace_id"], row["event_index"]): row for row in events}

    matrix, embedding_keys = _load_embeddings(
        artifacts / "event_embeddings_minilm384.csv",
        len(events),
        config.embedding.dimension,
        numpy,
    )
    if not numpy.isfinite(matrix).all():
        raise ValueError("embedding matrix contains NaN or infinite values")

    event_metadata: list[dict[str, str]] = []
    indices_by_sample: dict[str, list[int]] = defaultdict(list)
    for index, key in enumerate(embedding_keys):
        event = event_by_key[key]
        trace = trace_by_id[event["trace_id"]]
        sample_key = trace["sample_key"]
        label = label_by_sample[sample_key]
        class_name = "Benign" if label["y"] == "0" else "Malicious"
        family = "Benign" if class_name == "Benign" else label["family"]
        event_metadata.append(
            {
                "trace_id": event["trace_id"],
                "event_index": event["event_index"],
                "sample_key": sample_key,
                "class": class_name,
                "category": label["category"],
                "family": family,
                "split": trace["split"],
                "event_type": event["event_type"],
                "operation": event["operation"],
                "canonical_text": event["canonical_text"],
            }
        )
        indices_by_sample[sample_key].append(index)

    sample_keys = sorted(indices_by_sample)
    sample_matrix = numpy.vstack(
        [matrix[indices_by_sample[sample_key]].mean(axis=0) for sample_key in sample_keys]
    )
    sample_metadata: list[dict[str, str]] = []
    canonical_by_sample = {
        row["sample_key"]: row
        for row in traces
        if row["is_canonical"] == "1" and row["trace_quality"] == "valid"
    }
    for sample_key in sample_keys:
        label = label_by_sample[sample_key]
        trace = canonical_by_sample[sample_key]
        class_name = "Benign" if label["y"] == "0" else "Malicious"
        sample_metadata.append(
            {
                "sample_key": sample_key,
                "class": class_name,
                "category": label["category"],
                "family": "Benign" if class_name == "Benign" else label["family"],
                "split": trace["split"],
                "event_count": str(len(indices_by_sample[sample_key])),
            }
        )

    event_coords = _umap_3d(matrix, config.pilot.seed, umap)
    sample_coords = _umap_3d(sample_matrix, config.pilot.seed, umap)

    event_coordinate_rows = [
        {
            **metadata,
            "umap_1": format(float(event_coords[index, 0]), ".9g"),
            "umap_2": format(float(event_coords[index, 1]), ".9g"),
            "umap_3": format(float(event_coords[index, 2]), ".9g"),
        }
        for index, metadata in enumerate(event_metadata)
    ]
    sample_coordinate_rows = [
        {
            **metadata,
            "umap_1": format(float(sample_coords[index, 0]), ".9g"),
            "umap_2": format(float(sample_coords[index, 1]), ".9g"),
            "umap_3": format(float(sample_coords[index, 2]), ".9g"),
        }
        for index, metadata in enumerate(sample_metadata)
    ]
    write_csv(
        output / "umap_event_3d.csv",
        list(event_coordinate_rows[0]),
        event_coordinate_rows,
    )
    write_csv(
        output / "umap_sample_mean_3d.csv",
        list(sample_coordinate_rows[0]),
        sample_coordinate_rows,
    )

    _write_event_plot(
        event_coords,
        event_metadata,
        output / "umap_event_binary_3d.html",
        color_by="class",
        graph_objects=graph_objects,
        plotly_colors=plotly_colors,
    )
    _write_event_plot(
        event_coords,
        event_metadata,
        output / "umap_event_family_3d.html",
        color_by="family",
        graph_objects=graph_objects,
        plotly_colors=plotly_colors,
    )
    _write_sample_plot(
        sample_coords,
        sample_metadata,
        output / "umap_sample_binary_3d.html",
        color_by="class",
        graph_objects=graph_objects,
        plotly_colors=plotly_colors,
    )
    _write_sample_plot(
        sample_coords,
        sample_metadata,
        output / "umap_sample_family_3d.html",
        color_by="family",
        graph_objects=graph_objects,
        plotly_colors=plotly_colors,
    )

    event_type_counts = Counter(row["event_type"] for row in events)
    operation_counts = Counter(row["operation"] for row in events)
    class_counts = Counter(row["class"] for row in sample_metadata)
    family_counts = Counter(row["family"] for row in sample_metadata)
    category_counts = Counter(row["category"] for row in sample_metadata)
    split_counts = Counter(row["split"] for row in sample_metadata)
    sample_event_counts = [int(row["event_count"]) for row in sample_metadata]
    norms = numpy.linalg.norm(matrix, axis=1)
    canonical_text_counts = Counter(row["canonical_text"] for row in events)
    summary = {
        "seed": config.pilot.seed,
        "umap": {
            "n_components": 3,
            "n_neighbors": 15,
            "min_dist": 0.1,
            "metric": "cosine",
        },
        "counts": {
            "candidate_samples": len(samples),
            "dynamic_samples": len(sample_metadata),
            "traces": len(traces),
            "events": len(events),
            "embeddings": len(embedding_keys),
            "unique_canonical_texts": len(canonical_text_counts),
            "duplicate_event_text_rows": sum(
                count - 1 for count in canonical_text_counts.values() if count > 1
            ),
        },
        "sample_event_count": {
            "min": min(sample_event_counts),
            "mean": statistics.fmean(sample_event_counts),
            "median": statistics.median(sample_event_counts),
            "max": max(sample_event_counts),
        },
        "embedding_norm": {
            "min": float(norms.min()),
            "mean": float(norms.mean()),
            "max": float(norms.max()),
        },
        "class_counts": dict(class_counts),
        "category_counts": dict(category_counts),
        "family_counts": dict(family_counts),
        "split_counts": dict(split_counts),
        "event_type_counts": dict(event_type_counts),
        "operation_counts": dict(operation_counts),
        "ordering_known_traces": sum(
            row["ordering_known"] == "1" for row in canonical_by_sample.values()
        ),
    }
    atomic_json(output / "eda_summary.json", summary)

    write_csv(
        output / "sample_summary.csv",
        list(sample_metadata[0]),
        sample_metadata,
    )
    write_csv(
        output / "event_type_counts.csv",
        ["event_type", "count"],
        ({"event_type": name, "count": count} for name, count in event_type_counts.most_common()),
    )
    feature_summary = _numeric_feature_summary(source_features)
    write_csv(
        output / "source_feature_summary.csv",
        ["feature", "count", "missing", "missing_fraction", "min", "mean", "median", "max"],
        feature_summary,
    )

    report = "\n".join(
        [
            "# AU-PEMal pilot EDA",
            "",
            "## Dataset overview",
            "",
            f"- Candidate samples: {len(samples)}",
            f"- Dynamic samples: {len(sample_metadata)}",
            f"- Sandbox reports: {len(traces)}",
            f"- Canonical events and embeddings: {len(events)}",
            f"- Unique canonical event texts: {len(canonical_text_counts)}",
            f"- Canonical traces with known ordering: {summary['ordering_known_traces']}",
            "",
            "## Dynamic sample class distribution",
            "",
            _table(class_counts),
            "",
            "## Malware category distribution",
            "",
            _table(
                Counter(
                    {
                        key: value
                        for key, value in category_counts.items()
                        if key != "Benign"
                    }
                )
            ),
            "",
            "## Dynamic sample split",
            "",
            _table(split_counts),
            "",
            "## Event types",
            "",
            _table(event_type_counts),
            "",
            "## Event-count distribution per sample",
            "",
            f"Min {min(sample_event_counts)}, mean {statistics.fmean(sample_event_counts):.2f}, "
            f"median {statistics.median(sample_event_counts):.1f}, max {max(sample_event_counts)}.",
            "",
            "## Embedding checks",
            "",
            f"All {len(embedding_keys)} vectors are finite. L2 norm: "
            f"min {float(norms.min()):.6f}, mean {float(norms.mean()):.6f}, "
            f"max {float(norms.max()):.6f}.",
            "",
            "## Interpretation warning",
            "",
            "All canonical traces have `ordering_known=0`. UMAP visualizes semantic event "
            "similarity only; neither event indices nor UMAP axes represent execution time. "
            "Sample-level plots use mean pooling over each sample's event embeddings.",
        ]
    )
    report_path = output / "EDA_REPORT.md"
    report_path.write_text(report + "\n", encoding="utf-8")
    return {
        "report": str(report_path),
        "event_binary": str(output / "umap_event_binary_3d.html"),
        "event_family": str(output / "umap_event_family_3d.html"),
        "sample_binary": str(output / "umap_sample_binary_3d.html"),
        "sample_family": str(output / "umap_sample_family_3d.html"),
    }
