"""Calibration must preserve data and select without touching losing candidates' test."""

from pathlib import Path

import pytest

from pe_research.data.io import atomic_json, load_json
from pe_research.experiment import optimization
from pe_research.experiment.cli import latest_run
from pe_research.experiment.config import PilotConfig
from pe_research.experiment.optimization_report import optimization_sections


def validation(score: float, ratio: float = 0.5, collapsed: bool = False) -> dict:
    return {
        "selection_score": score,
        "auprc": 0.9,
        "family_macro_f1": 0.3,
        "malicious_recall": 0.4,
        "joint_constant_ratio": ratio,
        "bottleneck_constant_ratio": ratio,
        "representation": {"c_collapsed": collapsed},
    }


def test_predeclared_candidates_preserve_other_parameters() -> None:
    original = PilotConfig()
    variants = optimization.candidates(original)
    assert [name for name, _ in variants] == [
        "reconstruction_half",
        "family_double",
        "balanced",
        "balanced_lower_lr",
    ]
    for _, config in variants:
        for field, value in original.model_dump().items():
            if field not in {"learning_rate", "family_weight", "joint_weight", "bottleneck_weight"}:
                assert config.model_dump()[field] == value
    assert variants[2][1].family_weight == 1
    assert variants[2][1].joint_weight == 0.25
    assert variants[2][1].bottleneck_weight == 0.0625
    assert variants[3][1].learning_rate == 0.0005
    assert original.family_weight == 0.5


@pytest.mark.parametrize(
    "ratio,collapsed,expected",
    [(0.9, False, True), (1.0, False, False), (0.5, True, False), (float("nan"), False, False)],
)
def test_eligibility(ratio: float, collapsed: bool, expected: bool) -> None:
    assert optimization.eligible(validation(0.7, ratio, collapsed)) == expected


@pytest.mark.parametrize("improves", [True, False])
def test_only_validation_winner_can_open_test(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, improves: bool
) -> None:
    from pe_research.experiment import report

    monkeypatch.chdir(tmp_path)
    base = tmp_path / "experiment/api_ae_000_seed42"
    base.mkdir(parents=True)
    (base / "metrics.json").write_text("DO NOT READ TEST")
    (base / "trained.json").write_text("{}")
    original = PilotConfig()
    atomic_json(base / "resolved_config.json", original.model_dump())
    names = ["embeddings.npy", "split_manifest.csv", "family_dictionary.json"]
    for name in names:
        (base / name).write_bytes(b"same-data")
    atomic_json(tmp_path / "model" / base.name / "selection.json", {"validation": validation(0.7)})
    (tmp_path / optimization.DEFAULT_CONFIG).parent.mkdir(parents=True)
    runs = []
    tested = []
    plotted = []
    monkeypatch.setattr(optimization, "prepared_data", lambda path: None)

    def prepare(path: Path) -> Path:
        run = tmp_path / "experiment" / f"api_ae_{len(runs) + 1:03}_seed42"
        run.mkdir()
        for name in names:
            (run / name).write_bytes(b"same-data")
        runs.append(run)
        return run

    def train(run: Path) -> None:
        # Last candidate has the highest score but fails reconstruction eligibility.
        scores = [0.72, 0.73, 0.8, 0.99] if improves else [0.7, 0.69, 0.68, 0.99]
        index = runs.index(run)
        atomic_json(
            tmp_path / "model" / run.name / "selection.json",
            {"validation": validation(scores[index], 1.1 if index == 3 else 0.5)},
        )
        (run / "trained.json").write_text("{}")

    def evaluate(run: Path) -> None:
        assert load_json(run / "optimization.json")["status"] == "selection_frozen"
        assert not load_json(run / "optimization_candidate.json")["test_locked"]
        assert all(
            load_json(r / "optimization_candidate.json")["test_locked"] for r in runs if r != run
        )
        tested.append(run)
        (run / "metrics.json").write_text("{}")

    monkeypatch.setattr(optimization, "prepare", prepare)
    monkeypatch.setattr(optimization, "train", train)
    monkeypatch.setattr(optimization, "evaluate", evaluate)
    monkeypatch.setattr(optimization, "visualize", lambda r: plotted.append(r))
    monkeypatch.setattr(report, "write_report", lambda r: r / "PILOT_REPORT.md")
    selected = optimization.optimize(base.relative_to(tmp_path))
    assert selected == (runs[2] if improves else base)
    assert tested == ([runs[2]] if improves else [])
    assert plotted == tested
    assert latest_run("trained.json") == Path("experiment") / selected.name
    audit = load_json(selected / "optimization.json")
    assert audit["status"] == "complete"
    assert len(audit["results"]) == 5
    assert load_json(tmp_path / optimization.DEFAULT_CONFIG)["family_weight"] == (
        1 if improves else 0.5
    )
    assert (
        load_json(tmp_path / "configs/experiment/api_dual_branch_pilot_v1_original.yaml")
        == original.model_dump()
    )
    text = "\n".join(optimization_sections(selected))
    assert "balanced_lower_lr" in text
    assert "全新盲測" in text
    assert "未開啟其 test" in text


def test_changed_data_stops_before_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    base = tmp_path / "experiment/api_ae_base"
    base.mkdir(parents=True)
    atomic_json(base / "resolved_config.json", PilotConfig().model_dump())
    for name in ["embeddings.npy", "split_manifest.csv", "family_dictionary.json"]:
        (base / name).write_bytes(b"original")
    atomic_json(tmp_path / "model" / base.name / "selection.json", {"validation": validation(0.7)})
    (tmp_path / optimization.DEFAULT_CONFIG).parent.mkdir(parents=True)
    changed = tmp_path / "experiment/api_ae_changed"
    changed.mkdir()
    (changed / "embeddings.npy").write_bytes(b"different")
    monkeypatch.setattr(optimization, "prepared_data", lambda r: None)
    monkeypatch.setattr(optimization, "prepare", lambda p: changed)
    monkeypatch.setattr(optimization, "train", lambda r: pytest.fail("must not train changed data"))
    with pytest.raises(ValueError, match="changed the dataset"):
        optimization.optimize(base)
    assert not (tmp_path / optimization.DEFAULT_CONFIG).exists()


def test_losing_candidate_test_stays_locked(tmp_path: Path) -> None:
    from pe_research.experiment.baseline import evaluate_baselines
    from pe_research.experiment.pilot import evaluate

    atomic_json(tmp_path / "optimization_candidate.json", {"test_locked": True})
    for execute in [evaluate, evaluate_baselines]:
        with pytest.raises(ValueError, match="test remains locked"):
            execute(tmp_path)
