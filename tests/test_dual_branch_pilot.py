"""Meaningful data, loss, evaluation and end-to-end Pilot contract checks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from typer.testing import CliRunner

from pe_research.cli import app
from pe_research.data.api_traces_malware_detection.training_embeddings import (
    group_split,
    load_embeddings,
)
from pe_research.data.io import write_csv
from pe_research.experiment.metrics import binomial_interval, metrics, threshold_at_fpr
from pe_research.experiment.pilot import infer, interleaved_batches, prediction_rows
from pe_research.experiment.visualization import project
from pe_research.model.dual_branch import DualBranchAE, FactorClassifier, losses


def write_vectors(path: Path, vectors: np.ndarray, ids: list[str]) -> None:
    columns = [f"embedding_{i:03}" for i in range(vectors.shape[1])]
    write_csv(
        path,
        ["sample_id", *columns],
        [
            {"sample_id": sid, **dict(zip(columns, vector, strict=True))}
            for sid, vector in zip(ids, vectors, strict=True)
        ],
    )


def test_embedding_join_and_invalid_values(tmp_path: Path) -> None:
    path = tmp_path / "vectors.csv"
    write_vectors(path, np.array([[1, 0], [0, 1]], dtype=np.float32), ["b", "a"])
    ids, matrix, stats = load_embeddings(path, 2)
    assert ids == ["a", "b"] and matrix[0, 1] == 1
    assert stats["renormalized_count"] == 2
    assert not (tmp_path / "cache.npy").exists()
    for vectors, ids in [
        (np.array([[1, 0], [0, 1]]), ["a", "a"]),
        (np.array([[float("nan"), 0]]), ["a"]),
        (np.array([[0, 0]]), ["a"]),
        (np.array([[2, 0]]), ["a"]),
    ]:
        write_vectors(path, vectors, ids)
        with pytest.raises(ValueError):
            load_embeddings(path, 2)
    path.write_text("version https://git-lfs.github.com/spec/v1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_embeddings(path, 2)


def test_connected_split_is_stable_and_does_not_leak() -> None:
    samples = {}
    labels = {}
    fingerprints = {}
    for i in range(120):
        sid = f"s{i:03}"
        samples[sid] = {"source_sha256": str(i), "split": "train"}
        labels[sid] = {
            "y": str(i % 3 != 0 and 1 or 0),
            "family": "a" if i % 3 == 1 else "b" if i % 3 == 2 else "",
            "family_known": str(int(i % 3 != 0)),
        }
        fingerprints[sid] = str(i)
    # A -> B through sequence; B -> C through SHA, exercising transitive closure.
    fingerprints["s004"] = fingerprints["s001"]
    samples["s007"]["source_sha256"] = samples["s004"]["source_sha256"]
    first, _ = group_split(samples, labels, fingerprints, 20261002)
    second, _ = group_split(dict(reversed(list(samples.items()))), labels, fingerprints, 20261002)
    assert first == second
    lookup = {r["sample_id"]: r for r in first}
    assert len({lookup[s]["split"] for s in ["s001", "s004", "s007"]}) == 1
    assert len({lookup[s]["group_id"] for s in ["s001", "s004", "s007"]}) == 1
    assert len(first) == 120


def test_losses_shapes_masks_gradient_paths_and_scale() -> None:
    torch.manual_seed(42)
    model = DualBranchAE(768, 2)
    x = torch.randn(3, 768)
    y = torch.tensor([0.0, 1.0, 1.0])
    target = torch.tensor([-1, 1, -1])
    known = torch.tensor([False, True, False])
    out = model(x)
    terms = losses(out, x, y, target, known, 2.0)
    assert out["binary"].shape == (3,)
    assert out["family"].shape == (3, 2)
    assert float(terms["family"].detach()) == pytest.approx(
        float(torch.nn.functional.cross_entropy(out["family"][1:2], target[1:2]).detach())
    )
    gradients = torch.autograd.grad(
        terms["binary"], model.residual.parameters(), allow_unused=True, retain_graph=True
    )
    assert all(g is None for g in gradients)
    for key in ["joint", "bottleneck"]:
        expected = torch.mean((out[key] - x) ** 2) / 2
        assert torch.allclose(terms[key], expected)
    joint_grad = torch.autograd.grad(terms["joint"], model.residual[-1].weight, retain_graph=True)[
        0
    ]
    factor_grad = torch.autograd.grad(terms["bottleneck"], model.factor[-1].weight)[0]
    assert joint_grad.norm() > 0 and factor_grad.norm() > 0
    out1 = model(x[:1])
    empty = losses(out1, x[:1], y[:1], target[:1], known[:1], 2.0)
    assert out1["binary"].shape == (1,) and float(empty["family"].detach()) == 0
    with pytest.raises(ValueError):
        losses(out1, x[:1], y[:1, None], target[:1], known[:1], 2.0)
    with pytest.raises(ValueError):
        losses(out1, x[:1], y[:1], target[:1], known[:1], 0.0)


def test_threshold_ties_and_nonzero_upper_bound() -> None:
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.5, 0.5, 0.5, 0.8])
    threshold = threshold_at_fpr(y, scores, 0.01)
    assert threshold == 0.8
    assert (scores[y == 0] >= threshold).sum() == 0
    assert binomial_interval(0, 75)[1] > 0
    tied = np.full(4, 0.5, dtype=np.float32)
    all_negative = threshold_at_fpr(y, tied, 0.01)
    assert not (tied >= all_negative).any()


def test_metrics_exclusions_and_missed_detection_denominator() -> None:
    rows = [
        {"binary_label": 0, "family_target": -1, "family_known": 0, "group_id": "b"},
        {"binary_label": 1, "family_target": 0, "family_known": 1, "group_id": "m"},
        {"binary_label": 1, "family_target": -1, "family_known": 1, "group_id": "u"},
    ]
    x = np.ones((3, 2))
    outputs = {
        "scores": np.array([0.1, 0.2, 0.8]),
        "family_probs": np.array([[1, 0]] * 3),
        "c": np.zeros((3, 32)),
        "z_r": np.zeros((3, 32)),
        "joint": x,
        "bottleneck": x,
    }
    result = metrics(x, outputs, rows, ["a", "b"], x[0], 1.0, 0.5)
    assert result["family_macro_f1"] == 1
    assert result["end_to_end_accuracy"] == 0
    assert result["family_evaluated_count"] == 1 and result["family_excluded_count"] == 1
    assert result["joint_constant_ratio"] is None
    assert result["representation"]["c_collapsed"]


def test_inference_bundle_roundtrip(tmp_path: Path) -> None:
    model = DualBranchAE(768, 2)
    inference = FactorClassifier(768, 2)
    inference.load_state_dict(
        {k: v for k, v in model.state_dict().items() if k in inference.state_dict()}
    )
    torch.save(inference.state_dict(), tmp_path / "inference.pt")
    restored = FactorClassifier(768, 2)
    restored.load_state_dict(torch.load(tmp_path / "inference.pt", weights_only=True))
    x = np.random.default_rng(42).normal(size=(4, 768)).astype(np.float32)
    outputs = infer(model, x, "cpu")
    for other in [inference, restored]:
        np.testing.assert_array_equal(outputs["scores"], infer(other, x, "cpu")["scores"])
    rows = prediction_rows(
        outputs, [{"sample_id": "x"}] * 4, {"threshold": 2.0, "families": ["a", "b"]}
    )
    assert all(r["predicted_family"] == "N/A" for r in rows)
    assert "factor_32" in rows[0]


def test_sampling_no_oversampling() -> None:
    y = np.array([0] * 12 + [1] * 12)
    batches = interleaved_batches(y, 8, np.random.default_rng(42))
    assert sorted(np.concatenate(batches).tolist()) == list(range(24))
    assert all(set(y[b]) == {0, 1} for b in batches)


def test_projection_train_only_and_collapse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import umap

    calls = {}

    class FakeReducer:
        def __init__(self, **kwargs):
            calls["parameters"] = kwargs

        def fit_transform(self, x):
            calls["fit"] = x.copy()
            return np.zeros((len(x), 3))

        def transform(self, x):
            calls["transform"] = x.copy()
            return np.ones((len(x), 3))

    monkeypatch.setattr(umap, "UMAP", FakeReducer)
    monkeypatch.setattr("pe_research.experiment.visualization.pickle.dump", lambda *args: None)
    matrix = np.arange(28, dtype=float).reshape(7, 4)
    mask = np.array([True] * 5 + [False] * 2)
    xyz, info = project(matrix, mask, tmp_path / "reducer.pkl")
    assert len(calls["fit"]) == 5 and len(calls["transform"]) == 2
    np.testing.assert_allclose(info["mean"], matrix[:5].mean(0))
    assert np.all(xyz[~mask] == 1)
    collapsed, info = project(np.ones((7, 4)), mask, tmp_path / "collapsed.pkl")
    assert np.all(collapsed == 0) and info["status"] == "collapsed subspace"


def test_cli_help() -> None:
    result = CliRunner().invoke(app, ["experiment", "--help"])
    assert result.exit_code == 0
    for name in ["prepare", "train", "evaluate", "visualize", "predict"]:
        assert name in result.stdout


def test_synthetic_cli_full_lifecycle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json

    from pe_research.data.io import atomic_json, read_csv, sha256_file
    from pe_research.experiment.config import PilotConfig
    from pe_research.experiment.pilot import DEFAULT_CONFIG

    repository = Path.cwd()
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs/data").mkdir(parents=True)
    (tmp_path / "configs/model").mkdir(parents=True)
    (tmp_path / "configs/experiment").mkdir(parents=True)
    source = tmp_path / "data/synthetic/pilot_v1/artifacts"
    source.mkdir(parents=True)
    data_config = json.loads(
        (repository / "configs/data/api_traces_pilot_v1.yaml").read_text(encoding="utf-8")
    )
    data_config["workspace"] = "data/synthetic/pilot_v1"
    data_config["pilot"].update(
        target_benign=30,
        family_count=2,
        target_per_family=15,
        expected_families=["fam_a", "fam_b"],
        minimum_events=1,
    )
    atomic_json(tmp_path / "configs/data/api_traces_pilot_v1.yaml", data_config)
    (tmp_path / "configs/model/dual_branch_ae_v1.yaml").write_bytes(
        (repository / "configs/model/dual_branch_ae_v1.yaml").read_bytes()
    )
    atomic_json(tmp_path / DEFAULT_CONFIG, PilotConfig(epochs=1, warmup_epochs=1).model_dump())
    (tmp_path / "uv.lock").write_text("synthetic environment", encoding="utf-8")
    samples, labels, traces, events, coverage = [], [], [], [], []
    ids = [f"s{i:03}" for i in range(60)]
    rng = np.random.default_rng(42)
    vectors = rng.normal(size=(60, 768)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    for i, sid in enumerate(ids):
        split = ["train", "validation", "test"][i % 3]
        y = int(i >= 30)
        family = "" if not y else "fam_a" if i < 45 else "fam_b"
        samples.append(dict(sample_id=sid, source_sha256=f"{i:064x}", split=split))
        labels.append(
            dict(sample_id=sid, y=y, y_known=1, family=family, family_known=int(y and i != 59))
        )
        traces.append(
            dict(
                sample_id=sid,
                trace_id=sid,
                split=split,
                event_count=1,
                sequence_eligible=1,
                ordering_known=1,
                truncated=0,
                timestamp_coverage=1,
            )
        )
        events.append(
            dict(sample_id=sid, event_index=0, api_name=f"NtApi{i}", canonical_text=f"api=NtApi{i}")
        )
        coverage.append(
            dict(
                sample_id=sid,
                event_count=1,
                original_token_count=8,
                retained_token_count=8,
                truncated=0,
            )
        )
    for name, table in [
        ("samples.csv", samples),
        ("sample_labels.csv", labels),
        ("traces.csv", traces),
        ("events.csv", events),
        ("sample_embedding_coverage.csv", coverage),
    ]:
        write_csv(source / name, list(table[0]), table)
    vector_path = source / "sample_embeddings_gte_modernbert768.csv"
    write_vectors(vector_path, vectors, ids)
    metadata = {
        **data_config["sample_embedding"],
        "normalize_embeddings": True,
        "frozen": True,
        "training_performed": False,
        "row_count": 60,
        "output_sha256": sha256_file(vector_path),
        "events_sha256": sha256_file(source / "events.csv"),
    }
    atomic_json(source / "sample_embedding_metadata.json", metadata)
    atomic_json(source / "source_validation.json", {})
    atomic_json(source / "materialization_state.json", {})
    atomic_json(source / "quality_report.json", {"valid": True, "errors": []})
    names = [p.name for p in source.iterdir() if p.name != "quality_report.json"]
    atomic_json(
        source / "snapshot.json",
        {"artifacts": {n: {"sha256": sha256_file(source / n)} for n in names}},
    )
    before = {p.name: sha256_file(p) for p in source.iterdir()}
    runner = CliRunner()
    result = runner.invoke(app, ["experiment", "prepare"])
    assert result.exit_code == 0, str(result.exception)
    run = next((tmp_path / "experiment").glob("api_ae_*"))
    for command in ["train", "evaluate", "visualize"]:
        result = runner.invoke(app, ["experiment", command, "--run", str(run)])
        assert result.exit_code == 0, str(result.exception)
    result = runner.invoke(
        app,
        [
            "experiment",
            "predict",
            "--model-dir",
            str(tmp_path / "model" / run.name),
            "--embeddings",
            str(vector_path),
            "--metadata",
            str(source / "sample_embedding_metadata.json"),
            "--output",
            str(run / "inference.csv"),
        ],
    )
    assert result.exit_code == 0, str(result.exception)
    expected = read_csv(run / "predictions.csv")
    actual = read_csv(run / "inference.csv")
    assert [r["malicious_score"] for r in actual] == [r["malicious_score"] for r in expected]
    assert {p.name: sha256_file(p) for p in source.iterdir()} == before
    html = (run / "visualization/latent_3d.html").read_text(encoding="utf-8")
    assert "cdn.plot.ly" not in html.split("<body>")[0]
    for value in [
        'id="view"',
        'id="coloring"',
        'id="split"',
        "Plotly.restyle",
        "family head correct",
    ]:
        assert value in html
    coordinates = read_csv(run / "visualization/coordinates.csv")
    assert len(coordinates) == 180
    assert len({r["sample_id"] for r in coordinates}) == 60
    report_text = (run / "PILOT_REPORT.md").read_text(encoding="utf-8")
    for heading in [
        "資料、品質與切分",
        "Epoch 訓練走勢",
        "各家族分類結果",
        "良惡判定混淆計數",
        "Test 家族混淆矩陣",
        "內容保留與表徵診斷",
        "各加權 loss 的梯度力度",
        "UMAP 使用與投影設定",
    ]:
        assert heading in report_text
    assert "visualization/latent_3d.html" in report_text
    assert "MLP 對照組與 AE 比較" in report_text
    assert "MLP Test 家族混淆矩陣" in report_text
    control = json.loads((run / "mlp_baseline/evaluation_frozen.json").read_text())
    manifest = read_csv(run / "split_manifest.csv")
    assert control["split_manifest_sha256"] == sha256_file(run / "split_manifest.csv")
    assert control["embeddings_sha256"] == sha256_file(run / "embeddings.npy")
    assert (
        control["families"] == json.loads((run / "family_dictionary.json").read_text())["families"]
    )
    assert control["selections"]["binary"]["train_count"] == sum(
        r["split"] == "train" for r in manifest
    )
    assert control["selections"]["family"]["train_count"] == sum(
        r["split"] == "train"
        and r["binary_label"] == "1"
        and r["family_known"] == "1"
        and int(r["family_target"]) >= 0
        for r in manifest
    )
    assert control["test_exposure"] == "both controls selected before test evaluation"
    control_metrics = json.loads((run / "mlp_baseline/metrics.json").read_text())
    assert "joint_mse" not in control_metrics["test"]
    assert "representation" not in control_metrics["test"]
    from pe_research.experiment.baseline import evaluate_baselines

    assert evaluate_baselines(run) == control_metrics
    weights = tmp_path / "model" / run.name / "mlp_baseline/family.pt"
    weights.write_bytes(weights.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="weights changed"):
        evaluate_baselines(run)
    result = runner.invoke(app, ["experiment", "evaluate", "--run", str(run)])
    assert result.exit_code != 0
    # Tampering is rejected instead of silently changing the experiment.
    (run / "split_manifest.csv").write_text("invalid", encoding="utf-8")
    result = runner.invoke(app, ["experiment", "train", "--run", str(run)])
    assert result.exit_code != 0


def test_automatic_run_and_stage_selection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from pe_research.experiment import pilot, report, visualization

    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / pilot.DEFAULT_CONFIG
    config_path.parent.mkdir(parents=True)
    config_path.write_text("{}", encoding="utf-8")
    run = tmp_path / "experiment/api_ae_20261005T120000_seed42"
    run.mkdir(parents=True)
    (run / "prepared.json").write_text("{}")
    calls = []
    monkeypatch.setattr(
        pilot, "prepare", lambda config=pilot.DEFAULT_CONFIG: calls.append("prepare") or run
    )
    monkeypatch.setattr(pilot, "train", lambda selected: calls.append(("train", selected)) or run)
    monkeypatch.setattr(pilot, "evaluate", lambda selected: calls.append(("evaluate", selected)))
    monkeypatch.setattr(
        visualization,
        "visualize",
        lambda selected: (
            calls.append(("visualize", selected)) or run / "visualization/latent_3d.html"
        ),
    )
    monkeypatch.setattr(
        report,
        "write_report",
        lambda selected: calls.append(("report", selected)) or run / "PILOT_REPORT.md",
    )
    runner = CliRunner()
    result = runner.invoke(app, ["experiment", "run"])
    assert result.exit_code == 0, str(result.exception)
    assert calls == ["prepare", ("train", run), ("evaluate", run), ("visualize", run)]
    (run / "trained.json").write_text("{}")
    (run / "metrics.json").write_text("{}")
    calls.clear()
    result = runner.invoke(app, ["experiment", "report"])
    assert result.exit_code == 0, str(result.exception)
    assert calls == [("report", Path("experiment") / run.name)]
    calls.clear()
    result = runner.invoke(app, ["experiment", "evaluate"])
    assert result.exit_code == 0, str(result.exception)
    assert calls == [("report", Path("experiment") / run.name)]
