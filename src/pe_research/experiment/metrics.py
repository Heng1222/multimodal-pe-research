"""Threshold selection and task/content diagnostics with explicit support counts."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import beta
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


def threshold_at_fpr(y: Any, scores: Any, alpha: float) -> float:
    if y.shape != scores.shape or y.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("invalid threshold input shapes or scores")
    if set(np.unique(y)) != {0, 1} or not 0 <= alpha < 1:
        raise ValueError("threshold selection requires both classes and 0 <= FPR < 1")
    # The all-negative threshold must survive float32 inference comparisons.
    upper = float(np.nextafter(np.float32(np.max(scores)), np.float32(np.inf)))
    candidates = np.r_[np.unique(scores), upper]
    feasible = []
    for threshold in candidates:
        predicted = scores >= threshold
        fpr = float(predicted[y == 0].mean())
        recall = float(predicted[y == 1].mean())
        if fpr <= alpha:
            feasible.append((recall, -fpr, float(threshold)))
    return max(feasible)[2]


def binomial_interval(k: int, n: int) -> list[float] | None:
    if not n:
        return None
    return [
        0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1)),
        1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k)),
    ]


def representation_stats(c: Any, residual: Any, y: Any) -> dict[str, Any]:
    return {
        "c_variance": c.var(0).tolist(),
        "z_r_variance": residual.var(0).tolist(),
        "c_within_class_variance": {
            str(label): c[y == label].var(0).tolist() for label in np.unique(y)
        },
        "c_saturation_fraction": float(((c < 0.01) | (c > 0.99)).mean()),
        "c_collapsed": bool(np.all(c.var(0) <= 1e-12)),
    }


def metrics(
    x: Any,
    outputs: dict[str, Any],
    rows: list[dict[str, Any]],
    families: list[str],
    mean: Any,
    variance: float,
    threshold: float,
) -> dict[str, Any]:
    y = np.asarray([row["binary_label"] for row in rows])
    target = np.asarray([row["family_target"] for row in rows])
    mask = (y == 1) & np.asarray([bool(row["family_known"]) for row in rows]) & (target >= 0)
    scores = outputs["scores"]
    predicted = scores >= threshold
    family_predicted = outputs["family_probs"].argmax(1)
    benign = y == 0
    nb = int(benign.sum())
    fp = int(predicted[benign].sum())
    eval_labels = sorted(set(target[mask].tolist()))
    result: dict[str, Any] = {
        "samples": len(rows),
        "groups": len({r["group_id"] for r in rows}),
        "source": "Zenodo 11079764 / SmartVMI",
        "threshold": threshold,
        "auroc": float(roc_auc_score(y, scores)) if len(np.unique(y)) == 2 else None,
        "auprc": float(average_precision_score(y, scores)) if y.any() else None,
        "auprc_no_information": float(y.mean()),
        "benign_count": nb,
        "false_positives": fp,
        "fpr": fp / nb if nb else None,
        "fpr_resolution": 1 / nb if nb else None,
        "fpr_interval": binomial_interval(fp, nb),
        "interval_method": (
            "95% Clopper-Pearson; sample interval assumes independence; groups may correlate"
        ),
        "malicious_count": int(y.sum()),
        "malicious_recall": float(predicted[y == 1].mean()) if y.any() else None,
        "family_evaluated_count": int(mask.sum()),
        "family_excluded_count": int((y == 1).sum() - mask.sum()),
        "family_dictionary_coverage": float(mask.sum() / y.sum()) if y.any() else None,
        "family_evaluation_set": [families[i] for i in eval_labels],
        "family_macro_f1": float(
            f1_score(
                target[mask],
                family_predicted[mask],
                labels=eval_labels,
                average="macro",
                zero_division=0,
            )
        )
        if mask.any()
        else 0.0,
        "family_balanced_accuracy": float(
            balanced_accuracy_score(target[mask], family_predicted[mask])
        )
        if mask.any()
        else None,
        "end_to_end_accuracy": float(
            (predicted[mask] & (family_predicted[mask] == target[mask])).mean()
        )
        if mask.any()
        else None,
        "family_results": {},
    }
    if "c" in outputs and "z_r" in outputs:
        result["representation"] = representation_stats(outputs["c"], outputs["z_r"], y)
    if mask.any():
        precision, recall, f1, support = precision_recall_fscore_support(
            target[mask], family_predicted[mask], labels=eval_labels, zero_division=0
        )
        for j, i in enumerate(eval_labels):
            result["family_results"][families[i]] = {
                "precision": float(precision[j]),
                "recall": float(recall[j]),
                "f1": float(f1[j]),
                "samples": int(support[j]),
                "groups": len({r["group_id"] for r in rows if r["family_target"] == i}),
            }
    benign_groups = {row["group_id"] for row in rows if row["binary_label"] == 0}
    group_fp = {
        row["group_id"]
        for row, flag in zip(rows, predicted, strict=True)
        if row["binary_label"] == 0 and flag
    }
    result["benign_group_count"] = len(benign_groups)
    result["false_positive_groups"] = len(group_fp)
    result["group_fpr"] = len(group_fp) / len(benign_groups) if benign_groups else None
    result["group_fpr_interval"] = binomial_interval(len(group_fp), len(benign_groups))
    baseline = float(np.mean((x.astype(np.float64) - mean) ** 2))
    if "joint" in outputs or "bottleneck" in outputs:
        result["constant_prediction_mse"] = baseline
    for key in ["joint", "bottleneck"]:
        if key not in outputs:
            continue
        mse = float(np.mean((outputs[key].astype(np.float64) - x) ** 2))
        result[f"{key}_mse"] = mse
        result[f"{key}_normalized_mse"] = mse / variance
        result[f"{key}_constant_ratio"] = mse / baseline if baseline > 0 else None
    result["selection_score"] = (result["auprc"] + 0.5 * result["family_macro_f1"]) / 1.5
    result["binary_accuracy"] = float(accuracy_score(y, predicted))
    return result
