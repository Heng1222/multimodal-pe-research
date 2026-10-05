"""Bounded validation-only loss/LR calibration on an unchanged dataset split."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from pe_research.data.io import atomic_json, load_json, sha256_file
from pe_research.experiment.config import PilotConfig, load_pilot_config
from pe_research.experiment.pilot import DEFAULT_CONFIG, evaluate, prepare, prepared_data, train
from pe_research.experiment.visualization import visualize


def candidates(config: PilotConfig) -> list[tuple[str, PilotConfig]]:
    """Predeclare four small changes rather than exploring an adaptive search grid."""
    lower = {
        "joint_weight": config.joint_weight / 2,
        "bottleneck_weight": config.bottleneck_weight / 2,
    }
    higher = {"family_weight": config.family_weight * 2}
    variants = [
        ("reconstruction_half", lower),
        ("family_double", higher),
        ("balanced", {**lower, **higher}),
        ("balanced_lower_lr", {**lower, **higher, "learning_rate": config.learning_rate / 2}),
    ]
    return [
        (name, PilotConfig.model_validate({**config.model_dump(), **changes}))
        for name, changes in variants
    ]


def eligible(validation: dict[str, Any]) -> bool:
    return (
        all(
            validation.get(f"{key}_constant_ratio") is not None
            and validation[f"{key}_constant_ratio"] < 1
            for key in ["joint", "bottleneck"]
        )
        and not validation["representation"]["c_collapsed"]
    )


def optimize(base: Path) -> Path:
    """Read candidate validation only; freeze the winner before its test evaluation."""
    base = base.resolve()
    prepared_data(base)
    config = load_pilot_config(base / "resolved_config.json")
    root = Path.cwd().resolve()
    model_root = root / "model"
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%dT%H%M%S%f")
    session = root / "experiment" / f"tuning_{stamp}_seed{config.seed}"
    session.mkdir(exist_ok=False)
    expected = {
        name: sha256_file(base / name)
        for name in ["embeddings.npy", "split_manifest.csv", "family_dictionary.json"]
    }
    original_selection = load_json(model_root / base.name / "selection.json")
    original = {
        "name": "original",
        "run": str(base.relative_to(root)),
        "config": config.model_dump(),
        "validation": original_selection["validation"],
        "eligible": eligible(original_selection["validation"]),
    }
    results = [original]
    planned = candidates(config)
    manifest: dict[str, Any] = {
        "base_run": str(base.relative_to(root)),
        "selection_rule": "(AUPRC + 0.5 * family macro-F1) / 1.5",
        "constraints": "both validation reconstructions beat train-mean constant; c not collapsed",
        "candidate_count": len(planned),
        "planned": {name: c.model_dump() for name, c in planned},
        "same_dataset_checksums": expected,
        "results": results,
        "test_policy": (
            "test is not used to choose candidates; only the selected new candidate is evaluated"
        ),
        "prior_test_exposure": (
            "the base test results were already disclosed; reused test is not a fresh blind holdout"
        ),
        "status": "running",
    }
    atomic_json(session / "optimization.json", manifest)
    configs = root / "configs/experiment"
    for name, variant in planned:
        config_path = configs / f"api_dual_branch_tuning_{stamp}_{name}.yaml"
        atomic_json(config_path, variant.model_dump())
        print(f"TUNING start {name}", flush=True)
        run = prepare(config_path)
        atomic_json(run / "optimization_candidate.json", {"test_locked": True, "name": name})
        for filename, digest in expected.items():
            if sha256_file(run / filename) != digest:
                raise ValueError(f"tuning changed the dataset or split: {filename}")
        train(run)  # includes independent MLP controls; neither model reads test here
        selection = load_json(model_root / run.name / "selection.json")
        result = {
            "name": name,
            "run": str(run.relative_to(root)),
            "config": variant.model_dump(),
            "validation": selection["validation"],
            "eligible": eligible(selection["validation"]),
        }
        results.append(result)
        atomic_json(session / "optimization.json", manifest)
        print(
            f"TUNING result {name}: score={result['validation']['selection_score']:.4f}, "
            f"eligible={result['eligible']}",
            flush=True,
        )
    valid = [r for r in results if r["eligible"]]
    if not valid:
        manifest["status"] = "no candidate satisfies reconstruction/noncollapse constraints"
        atomic_json(session / "optimization.json", manifest)
        raise ValueError(manifest["status"])
    # Strict score improvement only; original is first, so ties keep original.
    winner = max(valid, key=lambda r: r["validation"]["selection_score"])
    selected = root / str(winner["run"])
    manifest.update(status="selection_frozen", selected=winner["name"], selected_run=winner["run"])
    atomic_json(session / "optimization.json", manifest)
    atomic_json(selected / "optimization.json", manifest)
    atomic_json(
        selected / "optimization_candidate.json", {"test_locked": False, "name": winner["name"]}
    )
    if winner["name"] != "original":
        evaluate(selected)
        visualize(selected)
    else:
        from pe_research.experiment.report import write_report

        write_report(selected)
    # Publish defaults only after the selected result and visualizations are complete.
    backup = configs / "api_dual_branch_pilot_v1_original.yaml"
    if not backup.exists():
        atomic_json(backup, config.model_dump())
    atomic_json(root / DEFAULT_CONFIG, winner["config"])
    manifest["status"] = "complete"
    manifest["default_config"] = str(DEFAULT_CONFIG)
    atomic_json(session / "optimization.json", manifest)
    atomic_json(selected / "optimization.json", manifest)
    from pe_research.experiment.report import write_report

    write_report(selected)
    print(f"TUNING selected {winner['name']}: {selected}", flush=True)
    return selected
