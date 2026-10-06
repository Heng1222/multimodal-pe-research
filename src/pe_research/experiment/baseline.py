"""Independent binary/family MLP controls using the exact AE prepared dataset."""

from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score
from torch.nn import functional as functional

from pe_research.data.api_traces_malware_detection.training_embeddings import SPLITS
from pe_research.data.io import atomic_json, load_json, sha256_file, write_csv
from pe_research.experiment.config import load_pilot_config
from pe_research.experiment.metrics import metrics, threshold_at_fpr
from pe_research.experiment.pilot import interleaved_batches, local_path, prepared_data
from pe_research.model.dual_branch import MLPClassifier


def logits(model: Any, x: Any, device: str) -> Any:
    model.eval()
    results = []
    with torch.inference_mode():
        for start in range(0, len(x), 128):
            output = model(torch.from_numpy(x[start : start + 128]).to(device))
            if not torch.isfinite(output).all():
                raise ValueError("non-finite MLP output")
            results.append(output.cpu().numpy())
    return np.concatenate(results)


def probabilities(raw: Any, *, binary: bool) -> Any:
    tensor = torch.from_numpy(raw)
    return (torch.sigmoid(tensor) if binary else torch.softmax(tensor, dim=1)).numpy()


def train_baselines(run: Path) -> Path:
    x, rows, families, state = prepared_data(run)
    config = load_pilot_config(run / "resolved_config.json")
    directory = local_path(Path.cwd().resolve(), Path("model") / run.name / "mlp_baseline")
    directory.mkdir(parents=True, exist_ok=False)
    output = run / "mlp_baseline"
    output.mkdir(exist_ok=False)
    root = Path.cwd().resolve()
    source_checksums = {}
    for path in (root / "src").rglob("*.py"):
        relative = path.relative_to(root)
        destination = output / "code_snapshot" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        source_checksums[str(relative)] = sha256_file(path)
    device = (
        "cuda"
        if config.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if config.device == "auto"
        else config.device
    )
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA explicitly requested but unavailable")
    torch.set_num_threads(config.threads)
    torch.use_deterministic_algorithms(True)
    y = np.asarray([row["binary_label"] for row in rows], dtype=np.float32)
    target = np.asarray([row["family_target"] for row in rows], dtype=np.int64)
    known = np.asarray([bool(row["family_known"]) for row in rows])
    train_mask = np.asarray([row["split"] == "train" for row in rows])
    valid_mask = np.asarray([row["split"] == "validation" for row in rows])
    family_mask = (y == 1) & known & (target >= 0) & (target < len(families))
    selections = {}
    histories = {}
    for task in ["binary", "family"]:
        binary = task == "binary"
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        rng = np.random.default_rng(config.seed)
        model = MLPClassifier(768, 1 if binary else len(families)).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        ti = np.flatnonzero(train_mask if binary else train_mask & family_mask)
        vi = np.flatnonzero(valid_mask if binary else valid_mask & family_mask)
        if not len(ti) or not len(vi):
            raise ValueError(f"insufficient train/validation support for MLP {task}")
        best = -float("inf")
        stale = 0
        history = []
        for epoch in range(config.epochs):
            model.train()
            if binary:
                batches = interleaved_batches(y[ti], config.batch_size, rng)
            else:
                order = rng.permutation(len(ti))
                batches = [
                    order[i : i + config.batch_size]
                    for i in range(0, len(order), config.batch_size)
                ]
            total = 0.0
            for batch in batches:
                ix = ti[batch]
                optimizer.zero_grad(set_to_none=True)
                raw = model(torch.from_numpy(x[ix]).to(device))
                labels = torch.from_numpy(y[ix] if binary else target[ix]).to(device)
                loss: Any = (
                    functional.binary_cross_entropy_with_logits(raw, labels)
                    if binary
                    else functional.cross_entropy(raw, labels)
                )
                if not torch.isfinite(loss):
                    raise ValueError("non-finite MLP loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.gradient_clip, error_if_nonfinite=True
                )
                optimizer.step()
                total += float(loss.detach()) * len(ix)
            probs = probabilities(logits(model, x[vi], device), binary=binary)
            if binary:
                score = float(average_precision_score(y[vi], probs))
                threshold = threshold_at_fpr(y[vi], probs, config.target_fpr)
                extra = {
                    "threshold": threshold,
                    "recall": float((probs[y[vi] == 1] >= threshold).mean()),
                    "fpr": float((probs[y[vi] == 0] >= threshold).mean()),
                }
            else:
                score = float(
                    f1_score(
                        target[vi],
                        probs.argmax(1),
                        labels=sorted(set(target[vi].tolist())),
                        average="macro",
                        zero_division=0,
                    )
                )
                extra = {}
            history.append(
                {"epoch": epoch + 1, "loss": total / len(ti), "validation_score": score, **extra}
            )
            if score > best:
                best, stale = score, 0
                torch.save(model.state_dict(), directory / f"{task}.pt")
                selections[task] = {
                    "epoch": epoch + 1,
                    "validation_score": score,
                    "train_count": len(ti),
                    "validation_count": len(vi),
                    "parameter_count": sum(p.numel() for p in model.parameters()),
                    **extra,
                }
            else:
                stale += 1
            if stale >= config.patience:
                break
        histories[task] = history
        print(
            f"MLP {task}: selected epoch={selections[task]['epoch']}, "
            f"validation score={best:.4f}, completed epochs={len(history)}",
            flush=True,
        )
    bundle = {
        "architecture": "independent_mlp_768_256_128/v1",
        "families": families,
        "seed": config.seed,
        "device": device,
        "selections": selections,
        "threshold": selections["binary"]["threshold"],
        "prepared_sha256": sha256_file(run / "prepared.json"),
        "split_manifest_sha256": sha256_file(run / "split_manifest.csv"),
        "embeddings_sha256": sha256_file(run / "embeddings.npy"),
        "family_dictionary_sha256": sha256_file(run / "family_dictionary.json"),
        "config_sha256": sha256_file(run / "resolved_config.json"),
        "weights_sha256": {
            task: sha256_file(directory / f"{task}.pt") for task in ["binary", "family"]
        },
        "training_config": config.model_dump(),
        "test_exposure": "retrospective control: AE test results were already available"
        if (run / "metrics.json").exists()
        else "both controls selected before test evaluation",
        "dataset_embedding_revision": state["embedding_metadata"]["model_revision"],
        "source_checksums": source_checksums,
    }
    atomic_json(directory / "bundle.json", bundle)
    atomic_json(output / "history.json", histories)
    atomic_json(output / "trained.json", {"bundle_sha256": sha256_file(directory / "bundle.json")})
    return directory


def evaluate_baselines(run: Path) -> dict[str, Any]:
    policy = run / "optimization_candidate.json"
    if policy.exists() and load_json(policy)["test_locked"]:
        raise ValueError("unselected optimization candidate: test remains locked")
    x, rows, families, state = prepared_data(run)
    output = run / "mlp_baseline"
    directory = local_path(Path.cwd().resolve(), Path("model") / run.name / "mlp_baseline")
    bundle = load_json(directory / "bundle.json")
    if (
        sha256_file(directory / "bundle.json")
        != load_json(output / "trained.json")["bundle_sha256"]
    ):
        raise ValueError("MLP bundle changed")
    for name, key in [
        ("prepared.json", "prepared_sha256"),
        ("split_manifest.csv", "split_manifest_sha256"),
        ("embeddings.npy", "embeddings_sha256"),
        ("family_dictionary.json", "family_dictionary_sha256"),
        ("resolved_config.json", "config_sha256"),
    ]:
        if sha256_file(run / name) != bundle[key]:
            raise ValueError(f"MLP/AE dataset or config mismatch: {name}")
    for task, expected in bundle["weights_sha256"].items():
        if sha256_file(directory / f"{task}.pt") != expected:
            raise ValueError(f"MLP weights changed: {task}")
    if (output / "metrics.json").exists():
        return load_json(output / "metrics.json")
    if (output / "evaluation_frozen.json").exists() and load_json(
        output / "evaluation_frozen.json"
    ) != bundle:
        raise ValueError("MLP evaluation selection changed")
    atomic_json(output / "evaluation_frozen.json", bundle)
    torch.set_num_threads(load_pilot_config(run / "resolved_config.json").threads)
    probabilities_by_task = {}
    for task in ["binary", "family"]:
        model = MLPClassifier(768, 1 if task == "binary" else len(families))
        model.load_state_dict(
            torch.load(directory / f"{task}.pt", map_location="cpu", weights_only=True)
        )
        probabilities_by_task[task] = probabilities(
            logits(model, x, "cpu"), binary=task == "binary"
        )
    outputs = {
        "scores": probabilities_by_task["binary"],
        "family_probs": probabilities_by_task["family"],
    }
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
    predictions = []
    for i, row in enumerate(rows):
        family = families[int(outputs["family_probs"][i].argmax())]
        predictions.append(
            {
                **row,
                "malicious_score": float(outputs["scores"][i]),
                "prediction": "malicious"
                if outputs["scores"][i] >= bundle["threshold"]
                else "benign",
                "family_head_prediction": family,
            }
        )
    write_csv(output / "predictions.csv", list(predictions[0]), predictions)
    atomic_json(output / "metrics.json", result)
    return result
