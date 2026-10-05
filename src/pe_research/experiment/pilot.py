"""Local, reproducible API embedding Pilot orchestration."""

from __future__ import annotations

import hashlib
import os
import random
import shutil
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from pe_research.data.api_traces.config import load_api_trace_config, resolve_api_workspace
from pe_research.data.api_traces.validation import validate_api_trace_artifacts
from pe_research.data.io import (
    atomic_json,
    load_json,
    package_version,
    read_csv,
    sha256_file,
    write_csv,
)
from pe_research.data.training_embeddings import (
    REVISION,
    SPLITS,
    group_split,
    indexed,
    load_embeddings,
    sequence_fingerprints,
)
from pe_research.experiment.config import load_pilot_config
from pe_research.experiment.metrics import metrics, threshold_at_fpr
from pe_research.model.dual_branch import DualBranchAE, FactorClassifier, losses

DEFAULT_CONFIG = Path("configs/experiment/api_dual_branch_pilot_v1.yaml")


def local_path(root: Path, value: str | Path) -> Path:
    path = (root / value).resolve()
    if root.resolve() not in path.parents:
        raise ValueError(f"path outside project: {value}")
    return path


def git_value(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], capture_output=True, text=True, encoding="utf-8", check=False
    )
    return result.stdout.strip()


def verify_prepared(run: Path) -> dict[str, Any]:
    state = load_json(run / "prepared.json")
    for name, expected in state["prepared_checksums"].items():
        if sha256_file(run / name) != expected:
            raise ValueError(f"prepared artifact checksum mismatch: {name}")
    return state


def prepare(config_path: Path = DEFAULT_CONFIG) -> Path:
    root = Path.cwd().resolve()
    config = load_pilot_config(config_path)
    data_config = load_api_trace_config(local_path(root, config.data_config))
    workspace = resolve_api_workspace(data_config, root)
    source = workspace / "artifacts"
    model_config = load_json(local_path(root, config.model_config_path))
    expected_model = {
        "architecture": "dual_branch_ae/v1",
        "input_dimension": 768,
        "encoder": [768, 256, 128],
        "factor": [128, 64, 32],
        "residual": [128, 64, 32],
        "joint_decoder": [64, 128, 768],
        "factor_decoder": [32, 64, 768],
    }
    if model_config != expected_model:
        raise ValueError("unsupported architecture; this Pilot implements whitepaper v1.2 at 768D")
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%dT%H%M%S%f")
    run = root / "experiment" / f"api_ae_{stamp}_seed{config.seed}"
    run.mkdir(parents=True, exist_ok=False)
    atomic_json(run / "resolved_config.json", config.model_dump())
    atomic_json(run / "model_config.json", model_config)
    snapshot = load_json(source / "snapshot.json")
    required_inputs = {
        "samples.csv",
        "sample_labels.csv",
        "traces.csv",
        "events.csv",
        "sample_embeddings_gte_modernbert768.csv",
        "sample_embedding_coverage.csv",
        "sample_embedding_metadata.json",
        "source_validation.json",
        "materialization_state.json",
    }
    if not required_inputs.issubset(snapshot["artifacts"]):
        raise ValueError("dataset snapshot is missing required artifact checksums")
    newline_equivalent = []
    for name, info in snapshot["artifacts"].items():
        if sha256_file(source / name) != info["sha256"]:
            equivalent = (
                name.endswith(".json")
                and hashlib.sha256((source / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                == info["sha256"]
            )
            if not equivalent:
                raise ValueError(f"dataset snapshot checksum mismatch: {name}")
            newline_equivalent.append(name)
    ids, x, normalization = load_embeddings(source / "sample_embeddings_gte_modernbert768.csv")
    samples, labels, traces, coverage = [
        indexed(source / name)
        for name in (
            "samples.csv",
            "sample_labels.csv",
            "traces.csv",
            "sample_embedding_coverage.csv",
        )
    ]
    for table in [samples, labels, traces, coverage]:
        if set(table) != set(ids):
            raise ValueError("incomplete sample join")
    metadata = load_json(source / "sample_embedding_metadata.json")
    if metadata.get("events_sha256") != sha256_file(source / "events.csv"):
        raise ValueError("embedding event checksum mismatch")
    if metadata.get("output_sha256") != sha256_file(
        source / "sample_embeddings_gte_modernbert768.csv"
    ) or metadata.get("row_count") != len(ids):
        raise ValueError("embedding metadata checksum or row count mismatch")
    if metadata["model_revision"] != REVISION or metadata["dimension"] != 768:
        raise ValueError("unexpected embedding revision or dimension")
    # Existing full audit is order-sensitive; audit a run-local view in canonical source order.
    # Original tables and reports remain immutable; training joins remain ID based.
    audit_workspace = run / "audit"
    audit_artifacts = audit_workspace / "artifacts"
    audit_artifacts.mkdir(parents=True)
    canonical_ids = list(samples)
    for name in [
        "samples.csv",
        "sample_labels.csv",
        "traces.csv",
        "sample_embedding_coverage.csv",
        "sample_embeddings_gte_modernbert768.csv",
    ]:
        table = indexed(source / name)
        write_csv(
            audit_artifacts / name,
            list(next(iter(table.values()))),
            [table[s] for s in canonical_ids],
        )
    for name in [
        "events.csv",
        "sample_embedding_metadata.json",
        "source_validation.json",
        "materialization_state.json",
    ]:
        os.link(source / name, audit_artifacts / name)
    # CSV canonicalization changes bytes, so use an audit-only metadata checksum.
    audit_metadata = dict(metadata)
    audit_metadata["output_sha256"] = sha256_file(
        audit_artifacts / "sample_embeddings_gte_modernbert768.csv"
    )
    (audit_artifacts / "sample_embedding_metadata.json").unlink()
    atomic_json(audit_artifacts / "sample_embedding_metadata.json", audit_metadata)
    errors, quality = validate_api_trace_artifacts(
        data_config,
        audit_workspace,
        report_path=run / "source_audit.json",
        require_complete=False,
        training_contract=True,
    )
    blocking = [
        error for error in errors if not error.startswith("sample embedding is not L2 normalized:")
    ]
    if blocking:
        raise ValueError(f"source audit failed: {blocking}")
    for sid in ids:
        if any(labels[sid][key] not in {"0", "1"} for key in ["y_known", "family_known"]):
            raise ValueError("invalid known mask")
        sha = samples[sid]["source_sha256"]
        if len(sha) != 64 or any(char not in "0123456789abcdef" for char in sha):
            raise ValueError("invalid source SHA-256")
        if (
            labels[sid]["y"] == "1"
            and labels[sid]["family_known"] == "1"
            and not labels[sid]["family"].strip()
        ):
            raise ValueError("known family label is empty")
        if labels[sid]["y_known"] == "1" and labels[sid]["y"] not in {"0", "1"}:
            raise ValueError("invalid known binary label")
        if traces[sid]["sequence_eligible"] != "1" or traces[sid]["ordering_known"] != "1":
            raise ValueError("sequence-ineligible trace")
    fingerprints = sequence_fingerprints(source / "events.csv", set(ids))
    excluded_unknown = [sid for sid in ids if labels[sid]["y_known"] != "1"]
    retained = [i for i, sid in enumerate(ids) if sid not in excluded_unknown]
    ids, x = [ids[i] for i in retained], x[retained]
    samples = {sid: samples[sid] for sid in ids}
    manifest, split_report = group_split(samples, labels, fingerprints, config.split_seed)
    by_id = {r["sample_id"]: r for r in manifest}
    rows = [by_id[sid] for sid in ids]
    families = sorted(
        {
            str(row["family_label"])
            for row in rows
            if row["split"] == "train" and row["family_known"] and row["binary_label"] == 1
        }
    )
    if len(families) < 2:
        raise ValueError("at least two training families are required")
    for row in rows:
        row["family_target"] = (
            families.index(row["family_label"])
            if row["family_known"] and row["family_label"] in families
            else -1
        )
    train_x = x[[row["split"] == "train" for row in rows]].astype(np.float64)
    mean = train_x.mean(0)
    variance = float(np.mean((train_x - mean) ** 2))
    if not np.isfinite(variance) or variance <= 1e-12:
        raise ValueError("degenerate training variance")
    np.save(run / "embeddings.npy", x)
    np.save(run / "train_mean.npy", mean)
    write_csv(run / "split_manifest.csv", list(rows[0]), rows)
    atomic_json(run / "family_dictionary.json", {"families": families})
    atomic_json(
        run / "quality_report.json",
        {
            "valid": True,
            "source_audit_valid": quality["valid"],
            "snapshot_json_newline_equivalent": newline_equivalent,
            "resolved_errors": errors,
            "excluded_unknown_binary_ids": excluded_unknown,
            "normalization": normalization,
            **split_report,
            "upstream_overlap": (
                "Frozen pretrained inference; upstream pretraining overlap is unknown"
            ),
            "source_label_counts": {
                "SmartVMI": {
                    "benign": sum(r["binary_label"] == 0 for r in rows),
                    "malicious": sum(r["binary_label"] == 1 for r in rows),
                }
            },
        },
    )
    input_names = [*snapshot["artifacts"], "snapshot.json", "quality_report.json"]
    checksums = {name: sha256_file(source / name) for name in input_names}
    prepared_names = [
        "embeddings.npy",
        "train_mean.npy",
        "split_manifest.csv",
        "family_dictionary.json",
        "resolved_config.json",
        "model_config.json",
        "quality_report.json",
    ]
    atomic_json(
        run / "prepared.json",
        {
            "schema_version": config.schema_version,
            "variance": variance,
            "source": str(source.relative_to(root)),
            "embedding_metadata": metadata,
            "input_checksums": checksums,
            "normalization": normalization,
            "config_sha256": sha256_file(run / "resolved_config.json"),
            "prepared_checksums": {name: sha256_file(run / name) for name in prepared_names},
        },
    )
    atomic_json(
        run / "environment.json",
        {
            "git_commit": git_value("rev-parse", "HEAD"),
            "git_status": git_value("status", "--short"),
            "lock_sha256": sha256_file(root / "uv.lock"),
            "packages": {
                name: package_version(name)
                for name in ["torch", "numpy", "scikit-learn", "scipy", "umap-learn", "plotly"]
            },
            "timezone": "Asia/Taipei",
            "device_requested": config.device,
            "source_checksums": {
                str(path.relative_to(root)): sha256_file(path)
                for path in (root / "src").rglob("*.py")
            },
        },
    )
    (run / "working_tree.patch").write_text(git_value("diff", "HEAD"), encoding="utf-8")
    for path in (root / "src").rglob("*.py"):
        target = run / "code_snapshot" / path.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    shutil.copy2(root / "uv.lock", run / "uv.lock")
    shutil.copy2(local_path(root, config.data_config), run / "data_config.yaml")
    if (root / "pyproject.toml").exists():
        shutil.copy2(root / "pyproject.toml", run / "pyproject.toml")
    return run


def prepared_data(run: Path) -> tuple[Any, list[dict[str, Any]], list[str], dict[str, Any]]:
    state = verify_prepared(run)
    rows: list[dict[str, Any]] = [dict(row) for row in read_csv(run / "split_manifest.csv")]
    for row in rows:
        for key in ["binary_label", "family_known", "family_target"]:
            row[key] = int(row[key])
    return (
        np.load(run / "embeddings.npy"),
        rows,
        load_json(run / "family_dictionary.json")["families"],
        state,
    )


def infer(model: Any, x: Any, device: str, batch_size: int = 128) -> dict[str, Any]:
    model.eval()
    batches: dict[str, list[Any]] = {}
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            output = model(torch.from_numpy(x[start : start + batch_size]).to(device))
            output["scores"] = torch.sigmoid(output["binary"])
            output["family_probs"] = torch.softmax(output["family"], dim=1)
            for key, value in output.items():
                batches.setdefault(key, []).append(value.cpu().numpy())
    return {key: np.concatenate(value) for key, value in batches.items()}


def interleaved_batches(y: Any, batch_size: int, generator: Any) -> list[Any]:
    benign = generator.permutation(np.flatnonzero(y == 0)).tolist()
    malicious = generator.permutation(np.flatnonzero(y == 1)).tolist()
    order = []
    for i in range(max(len(benign), len(malicious))):
        if i < len(benign):
            order.append(benign[i])
        if i < len(malicious):
            order.append(malicious[i])
    return [np.asarray(order[i : i + batch_size]) for i in range(0, len(order), batch_size)]


def train(run: Path) -> Path:
    x, rows, families, state = prepared_data(run)
    config = load_pilot_config(run / "resolved_config.json")
    model_dir = local_path(Path.cwd().resolve(), Path("model") / run.name)
    model_dir.mkdir(parents=True, exist_ok=False)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_num_threads(config.threads)
    torch.use_deterministic_algorithms(True)
    device = (
        "cuda"
        if config.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if config.device == "auto"
        else config.device
    )
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA explicitly requested but unavailable")
    model = DualBranchAE(768, len(families)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    initial_classification = {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
        if k.startswith(("binary.", "family."))
    }
    atomic_json(
        run / "runtime.json",
        {
            "device": device,
            "threads": config.threads,
            "seed": config.seed,
            "deterministic_algorithms": True,
        },
    )
    train_indices = np.asarray([i for i, row in enumerate(rows) if row["split"] == "train"])
    valid_indices = np.asarray([i for i, row in enumerate(rows) if row["split"] == "validation"])
    y = np.asarray([row["binary_label"] for row in rows], dtype=np.float32)
    target = np.asarray([row["family_target"] for row in rows], dtype=np.int64)
    known = np.asarray([bool(row["family_known"]) for row in rows])
    mean = np.load(run / "train_mean.npy")
    weights = {
        key: getattr(config, f"{key}_weight") for key in ["binary", "family", "joint", "bottleneck"]
    }
    rng = np.random.default_rng(config.seed)
    best = -float("inf")
    stale = 0
    history = []
    gradient_history = []
    for epoch in range(config.warmup_epochs + config.epochs):
        warmup = epoch < config.warmup_epochs
        model.train()
        totals = dict.fromkeys(weights, 0.0)
        counts = dict.fromkeys(weights, 0)
        family_count = 0
        for batch_index, local_indices in enumerate(
            interleaved_batches(y[train_indices], config.batch_size, rng)
        ):
            indices = train_indices[local_indices]
            xb = torch.from_numpy(x[indices]).to(device)
            yb = torch.from_numpy(y[indices]).to(device)
            fb = torch.from_numpy(target[indices]).to(device)
            kb = torch.from_numpy(known[indices]).to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(xb)
            terms = losses(output, xb, yb, fb, kb, state["variance"])
            active = ["joint", "bottleneck"] if warmup else list(weights)
            nf = int(((yb == 1) & kb & (fb >= 0)).sum())
            family_count += nf
            if not warmup and batch_index % config.gradient_interval == 0:
                record: dict[str, Any] = {
                    "epoch": epoch + 1,
                    "batch": batch_index,
                    "effective_family_count": nf,
                }
                for key in active:
                    if key == "family" and not nf:
                        continue
                    gradients = torch.autograd.grad(
                        weights[key] * terms[key],
                        [
                            cast(torch.nn.Linear, model.factor[-1]).weight,
                            cast(torch.nn.Linear, model.residual[-1]).weight,
                        ],
                        retain_graph=True,
                        allow_unused=True,
                    )
                    record[key] = {
                        name: float(grad.norm()) if grad is not None else 0.0
                        for name, grad in zip(["factor", "residual"], gradients, strict=True)
                    }
                gradient_history.append(record)
            total = sum(weights[key] * terms[key] for key in active)
            total.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.gradient_clip, error_if_nonfinite=True
            )
            optimizer.step()
            for key in weights:
                count = nf if key == "family" else len(indices)
                totals[key] += float(terms[key].detach()) * count
                counts[key] += count
        outputs = infer(model, x[valid_indices], device)
        threshold = threshold_at_fpr(y[valid_indices], outputs["scores"], config.target_fpr)
        validation = metrics(
            x[valid_indices],
            outputs,
            [rows[i] for i in valid_indices],
            families,
            mean,
            state["variance"],
            threshold,
        )
        raw_losses = {key: totals[key] / counts[key] if counts[key] else None for key in weights}
        record = {
            "epoch": epoch + 1,
            "phase": "warmup" if warmup else "joint",
            "losses": raw_losses,
            "weighted_losses": {
                key: value * weights[key] if value is not None else None
                for key, value in raw_losses.items()
            },
            "effective_family_count": family_count,
            "validation": validation,
        }
        # Keep raw MSE alongside normalized loss values.
        record["raw_joint_mse"] = raw_losses["joint"] * state["variance"]
        record["raw_bottleneck_mse"] = raw_losses["bottleneck"] * state["variance"]
        history.append(record)
        atomic_json(run / "history.json", {"epochs": history})
        atomic_json(run / "gradients.json", {"samples": gradient_history})
        print(
            f"epoch {epoch + 1:02d} {record['phase']} "
            f"score={validation['selection_score']:.4f} "
            f"recall={validation['malicious_recall']:.3f} "
            f"family_F1={validation['family_macro_f1']:.3f} "
            f"reconstruction={validation['joint_constant_ratio']:.3f}/"
            f"{validation['bottleneck_constant_ratio']:.3f}",
            flush=True,
        )
        if warmup:
            continue
        if validation["selection_score"] > best:
            best = validation["selection_score"]
            stale = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch + 1,
                    "seed": config.seed,
                },
                model_dir / "best.pt",
            )
            atomic_json(
                model_dir / "selection.json",
                {"epoch": epoch + 1, "threshold": threshold, "validation": validation},
            )
        else:
            stale += 1
        if stale >= config.patience:
            break
    selected = torch.load(model_dir / "best.pt", map_location="cpu", weights_only=True)
    inference = FactorClassifier(768, len(families))
    inference.load_state_dict(
        {
            key: value
            for key, value in selected["state_dict"].items()
            if key in inference.state_dict()
        }
    )
    torch.save(inference.state_dict(), model_dir / "inference.pt")
    selection = load_json(model_dir / "selection.json")
    updated = {
        prefix: any(
            not torch.equal(initial_classification[key], selected["state_dict"][key])
            for key in initial_classification
            if key.startswith(prefix + ".")
        )
        for prefix in ["binary", "family"]
    }
    atomic_json(
        model_dir / "bundle.json",
        {
            "families": families,
            "dimension": 768,
            "threshold": selection["threshold"],
            "embedding_revision": REVISION,
            "embedding_model": state["embedding_metadata"]["model_name"],
            "normalization": state["normalization"],
            "variance": state["variance"],
            "classification_heads_updated": updated,
            "checkpoint_sha256": sha256_file(model_dir / "best.pt"),
            "inference_sha256": sha256_file(model_dir / "inference.pt"),
            "selection_sha256": sha256_file(model_dir / "selection.json"),
            "prepared_sha256": sha256_file(run / "prepared.json"),
        },
    )
    atomic_json(model_dir / "family_dictionary.json", {"families": families})
    np.save(model_dir / "train_mean.npy", mean)
    atomic_json(
        run / "trained.json",
        {
            "model_directory": str(model_dir.relative_to(Path.cwd())),
            "bundle_sha256": sha256_file(model_dir / "bundle.json"),
        },
    )
    # Test remains unopened until evaluate explicitly freezes this selection.
    from pe_research.experiment.baseline import train_baselines

    train_baselines(run)
    return model_dir


def load_model(run: Path) -> tuple[Any, dict[str, Any], Path]:
    prepared_data(run)
    model_dir = local_path(Path.cwd().resolve(), Path("model") / run.name)
    bundle = load_json(model_dir / "bundle.json")
    if sha256_file(model_dir / "bundle.json") != load_json(run / "trained.json")["bundle_sha256"]:
        raise ValueError("model bundle changed")
    for name, key in [
        ("best.pt", "checkpoint_sha256"),
        ("inference.pt", "inference_sha256"),
        ("selection.json", "selection_sha256"),
    ]:
        if sha256_file(model_dir / name) != bundle[key]:
            raise ValueError(f"model artifact changed: {name}")
    if sha256_file(run / "prepared.json") != bundle["prepared_sha256"]:
        raise ValueError("model and prepared data differ")
    model = DualBranchAE(768, len(bundle["families"]))
    model.load_state_dict(
        torch.load(model_dir / "best.pt", map_location="cpu", weights_only=True)["state_dict"]
    )
    return model, bundle, model_dir


def prediction_rows(
    outputs: dict[str, Any], rows: list[dict[str, Any]], bundle: dict[str, Any]
) -> list[dict[str, Any]]:
    results = []
    for i, row in enumerate(rows):
        malicious = bool(outputs["scores"][i] >= bundle["threshold"])
        fi = int(outputs["family_probs"][i].argmax())
        result = dict(row)
        result.update(
            {
                "malicious_score": float(outputs["scores"][i]),
                "prediction": "malicious" if malicious else "benign",
                "predicted_family": bundle["families"][fi] if malicious else "N/A",
                "family_head_prediction": bundle["families"][fi],
                "family_confidence": float(outputs["family_probs"][i, fi]) if malicious else "N/A",
            }
        )
        if "binary_label" in row:
            result["binary_correct"] = int(malicious == bool(row["binary_label"]))
            result["family_correct"] = (
                int(fi == row["family_target"])
                if row["binary_label"] == 1 and row["family_known"] and row["family_target"] >= 0
                else "N/A"
            )
        result.update(
            {f"factor_{j + 1:02}": float(value) for j, value in enumerate(outputs["c"][i])}
        )
        results.append(result)
    return results


def evaluate(run: Path) -> dict[str, Any]:
    policy = run / "optimization_candidate.json"
    if policy.exists() and load_json(policy)["test_locked"]:
        raise ValueError("unselected optimization candidate: test remains locked")
    model, bundle, _ = load_model(run)
    if (run / "metrics.json").exists():
        raise ValueError("test already evaluated; use a new run for revised settings")
    if (run / "evaluation_frozen.json").exists() and load_json(run / "evaluation_frozen.json")[
        "bundle"
    ] != bundle:
        raise ValueError("cannot retry evaluation with changed model selection")
    x, rows, families, state = prepared_data(run)
    from pe_research.experiment.baseline import evaluate_baselines, train_baselines

    if not (run / "mlp_baseline/trained.json").exists():
        train_baselines(run)
    evaluate_baselines(run)
    atomic_json(
        run / "evaluation_frozen.json",
        {"bundle": bundle, "prepared_sha256": sha256_file(run / "prepared.json")},
    )
    torch.set_num_threads(load_pilot_config(run / "resolved_config.json").threads)
    outputs = infer(model, x, "cpu")
    result = {}
    for split in SPLITS:
        ix = np.asarray([i for i, row in enumerate(rows) if row["split"] == split])
        result[split] = metrics(
            x[ix],
            {key: value[ix] for key, value in outputs.items()},
            [rows[i] for i in ix],
            families,
            np.load(run / "train_mean.npy"),
            state["variance"],
            bundle["threshold"],
        )
    atomic_json(run / "metrics.json", result)
    predictions = prediction_rows(outputs, rows, bundle)
    write_csv(run / "predictions.csv", list(predictions[0]), predictions)
    np.savez(
        run / "representations.npz",
        sample_ids=np.asarray([r["sample_id"] for r in rows]),
        c=outputs["c"],
        z_r=outputs["z_r"],
    )
    validation = result["validation"]
    success = (
        all(bundle["classification_heads_updated"].values())
        and validation["joint_constant_ratio"] is not None
        and validation["bottleneck_constant_ratio"] is not None
        and validation["joint_constant_ratio"] < 1
        and validation["bottleneck_constant_ratio"] < 1
        and not validation["representation"]["c_collapsed"]
    )
    atomic_json(
        run / "pilot_status.json",
        {
            "research_success": success,
            "deployment_acceptance": "not assessed",
            "classification_heads_updated": bundle["classification_heads_updated"],
        },
    )
    from pe_research.experiment.report import write_report

    write_report(run)
    return result


def predict(model_dir: Path, embedding_path: Path, metadata_path: Path, output_path: Path) -> None:
    if output_path.exists():
        raise ValueError("prediction output already exists")
    bundle = load_json(model_dir / "bundle.json")
    metadata = load_json(metadata_path)
    expected = {
        "model_name": bundle["embedding_model"],
        "model_revision": bundle["embedding_revision"],
        "dimension": 768,
        "pooling": "last_hidden_state_cls",
        "input_format": "api_name",
        "frozen": True,
        "training_performed": False,
        "normalize_embeddings": True,
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("incompatible embedding metadata")
    if metadata.get("output_sha256") != sha256_file(embedding_path):
        raise ValueError("prediction input checksum mismatch")
    if sha256_file(model_dir / "inference.pt") != bundle["inference_sha256"]:
        raise ValueError("inference weights changed")
    ids, x, _ = load_embeddings(embedding_path)
    model = FactorClassifier(768, len(bundle["families"]))
    model.load_state_dict(
        torch.load(model_dir / "inference.pt", map_location="cpu", weights_only=True)
    )
    torch.set_num_threads(2)
    predictions = prediction_rows(
        infer(model, x, "cpu"), [{"sample_id": sid} for sid in ids], bundle
    )
    write_csv(output_path, list(predictions[0]), predictions)
