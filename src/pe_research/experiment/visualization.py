"""Train-only projections and self-contained offline Plotly controls."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from pe_research.data.io import atomic_json, package_version, read_csv, write_csv
from pe_research.experiment.pilot import load_model, prepared_data

UMAP_PARAMETERS = {
    "n_components": 3,
    "min_dist": 0.1,
    "metric": "euclidean",
    "init": "random",
    "random_state": 42,
    "transform_seed": 42,
    "n_jobs": 1,
}


def project(matrix: Any, train_mask: Any, output: Path) -> tuple[Any, dict[str, Any]]:
    import umap

    mean = matrix[train_mask].mean(0)
    std = matrix[train_mask].std(0)
    keep = std > 1e-6
    info: dict[str, Any] = {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "retained_dimensions": np.flatnonzero(keep).tolist(),
        "excluded_dimensions": np.flatnonzero(~keep).tolist(),
        "raw_variance": matrix[train_mask].var(0).tolist(),
        "parameters": {**UMAP_PARAMETERS, "n_neighbors": min(15, int(train_mask.sum()) - 1)},
        "train_fit_count": int(train_mask.sum()),
        "status": "ok",
    }
    if train_mask.sum() < 5 or not keep.any():
        info["status"] = (
            "insufficient train samples" if train_mask.sum() < 5 else "collapsed subspace"
        )
        return np.zeros((len(matrix), 3)), info
    scaled = (matrix[:, keep] - mean[keep]) / std[keep]
    reducer = umap.UMAP(**info["parameters"])
    coordinates = np.zeros((len(matrix), 3))
    coordinates[train_mask] = reducer.fit_transform(scaled[train_mask])
    if (~train_mask).any():
        coordinates[~train_mask] = reducer.transform(scaled[~train_mask])
    with output.open("wb") as stream:
        pickle.dump(reducer, stream)
    return coordinates, info


def visualize(run: Path) -> Path:
    import plotly.colors
    import plotly.graph_objects as go
    import plotly.io as pio

    if not (run / "evaluation_frozen.json").exists() or not (run / "metrics.json").exists():
        raise ValueError("evaluate fixed selection before viewing test representations")
    load_model(run)
    _, rows, _, _ = prepared_data(run)
    destination = run / "visualization"
    destination.mkdir(exist_ok=False)
    with np.load(run / "representations.npz", allow_pickle=False) as data:
        if list(data["sample_ids"]) != [row["sample_id"] for row in rows]:
            raise ValueError("representation sample IDs are not aligned")
        c, residual = data["c"], data["z_r"]
    predictions = {row["sample_id"]: row for row in read_csv(run / "predictions.csv")}
    train_mask = np.asarray([row["split"] == "train" for row in rows])
    matrices = {"c": c, "z_r": residual, "combined": np.concatenate([c, residual], axis=1)}
    coordinates, stats = {}, {}
    for key, matrix in matrices.items():
        coordinates[key], stats[key] = project(
            matrix, train_mask, destination / f"{key}_reducer.pkl"
        )
    # Concatenated per-dimension standardization equals concatenated standardized branches.
    atomic_json(
        destination / "projection.json",
        {
            "views": stats,
            "versions": {name: package_version(name) for name in ["numpy", "umap-learn", "plotly"]},
        },
    )
    families = sorted(
        {str(row["family_label"]) for row in rows if row["binary_label"] and row["family_known"]}
    )
    palette = {"benign": "#9E9E9E", "malicious": "#D62728", "malicious/unlabelled": "#202020"}
    palette.update(
        {
            family: plotly.colors.qualitative.Alphabet[i % len(plotly.colors.qualitative.Alphabet)]
            for i, family in enumerate(families)
        }
    )
    atomic_json(destination / "palette.json", palette)
    figure = go.Figure()
    trace_keys = []
    coordinate_rows = []
    for view, xyz in coordinates.items():
        for i, row in enumerate(rows):
            coordinate_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "view": view,
                    "x": xyz[i, 0],
                    "y": xyz[i, 1],
                    "z": xyz[i, 2],
                    "coordinate_type": "train fit" if train_mask[i] else "new sample transform",
                    **row,
                }
            )
        for coloring in ["binary", "family"]:
            labels = [
                "benign"
                if not row["binary_label"]
                else "malicious"
                if coloring == "binary"
                else str(row["family_label"])
                if row["family_known"]
                else "malicious/unlabelled"
                for row in rows
            ]
            for split in ["train", "validation", "test"]:
                for label in sorted(set(labels)):
                    indices = [
                        i
                        for i, row in enumerate(rows)
                        if row["split"] == split and labels[i] == label
                    ]
                    if not indices:
                        continue
                    custom = []
                    for i in indices:
                        row = rows[i]
                        pred = predictions[row["sample_id"]]
                        custom.append(
                            [
                                row["sample_id"],
                                row["group_id"],
                                split,
                                row["binary_label"],
                                row["family_label"] or "N/A",
                                row["family_target"] >= 0,
                                pred["malicious_score"],
                                pred["predicted_family"],
                                pred["binary_correct"],
                                pred["family_correct"],
                                "train fit" if train_mask[i] else "new sample transform",
                                pred["family_head_prediction"],
                            ]
                        )
                    figure.add_trace(
                        go.Scatter3d(
                            x=xyz[indices, 0],
                            y=xyz[indices, 1],
                            z=xyz[indices, 2],
                            mode="markers",
                            name=f"{label} / {split}",
                            legendgroup=label,
                            marker={"size": 4, "color": palette[label], "opacity": 0.75},
                            customdata=custom,
                            visible=view == "c" and coloring == "binary",
                            hovertemplate=(
                                "sample=%{customdata[0]}<br>group=%{customdata[1]}"
                                "<br>split=%{customdata[2]}<br>true binary=%{customdata[3]}"
                                "<br>true family=%{customdata[4]}"
                                "<br>in dictionary=%{customdata[5]}"
                                "<br>malicious score=%{customdata[6]}"
                                "<br>gated family=%{customdata[7]}"
                                "<br>binary correct=%{customdata[8]}"
                                "<br>family head correct=%{customdata[9]}"
                                "<br>coordinates=%{customdata[10]}"
                                "<br>family head=%{customdata[11]}<extra></extra>"
                            ),
                        )
                    )
                    trace_keys.append([view, coloring, split])
    write_csv(destination / "coordinates.csv", list(coordinate_rows[0]), coordinate_rows)
    write_csv(
        destination / "metadata.csv",
        list(next(iter(predictions.values()))),
        list(predictions.values()),
    )
    figure.update_layout(
        title="API AE latent representations", height=820, margin={"l": 0, "r": 0, "b": 0, "t": 50}
    )
    controls = (
        '<label>子空間 <select id="view"><option>c</option><option>z_r</option>'
        "<option>combined</option></select></label> "
        '<label>著色 <select id="coloring"><option value="binary">良／惡</option>'
        '<option value="family">良性＋family</option></select></label> '
        '<label>切分 <select id="split"><option value="all">all</option>'
        "<option>train</option><option>validation</option><option>test</option></select></label>"
    )
    script = """
const keys=KEYS;
const gd=document.getElementById('latent');
function update(){
    const v=document.getElementById('view').value;
    const c=document.getElementById('coloring').value;
    const s=document.getElementById('split').value;
    Plotly.restyle(gd,{visible:keys.map(k=>
        k[0]===v&&k[1]===c&&(s==='all'||k[2]===s))});
}
['view','coloring','split'].forEach(id=>
    document.getElementById(id).addEventListener('change',update));
""".replace("KEYS", json.dumps(trace_keys))
    chart = pio.to_html(
        figure, include_plotlyjs=True, full_html=False, div_id="latent", post_script=script
    )
    warnings = json.dumps(
        {key: value["status"] for key, value in stats.items()}, ensure_ascii=False
    )
    html = (
        '<!doctype html><html lang="zh-Hant"><meta charset="utf-8">'
        "<title>API AE Pilot</title><body><h1>雙分支表示</h1>"
        f"{controls}<p>投影狀態：{warnings}</p>"
        "<p>標籤僅用於著色；train 擬合、validation/test transform。"
        "不同視圖距離不可直接比較；因素未具概念語意。高維診斷見 metrics.json。</p>"
        f"{chart}</body></html>"
    )
    output = destination / "latent_3d.html"
    output.write_text(html, encoding="utf-8")
    from pe_research.experiment.report import write_report

    write_report(run)
    return output
