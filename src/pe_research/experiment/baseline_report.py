"""Standalone AE versus direct-embedding MLP comparison report sections."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from pe_research.data.io import load_json, read_csv


def comparison_sections(run: Path, ae: dict[str, Any]) -> list[str]:
    source = run / "mlp_baseline"
    if not (source / "metrics.json").exists():
        return [
            "",
            "## MLP 對照組",
            "",
            "此歷史實驗尚無 MLP 對照組；可執行 `pe-research experiment baseline` 自動補跑。",
            "",
        ]
    mlp = load_json(source / "metrics.json")
    bundle = load_json(source / "evaluation_frozen.json")
    history = load_json(source / "history.json")
    config = bundle["training_config"]
    selections = bundle["selections"]
    lines = [
        "",
        "## MLP 對照組與 AE 比較",
        "",
        "兩個獨立單任務 MLP：良惡分類 768→256→128→1，family 分類 768→256→128→K；"
        "hidden GELU，分類 logits 為線性。直接使用相同正規化 embedding，沒有因素瓶頸、"
        "decoder 或重建 loss；兩個 MLP 不共用權重。",
        "使用同一份 prepared embedding、split manifest 與 train family 字典，"
        "所有樣本／群組 ID 均一致，未另行切分。良惡使用全部 train 樣本；"
        "family 僅使用 AE 相同的可信、字典內惡性樣本。",
        f"MLP seed={bundle['seed']}；device={bundle['device']}；batch={config['batch_size']}；"
        f"AdamW lr={config['learning_rate']}、weight decay={config['weight_decay']}；"
        f"global clipping={config['gradient_clip']}；上限 {config['epochs']} epochs、"
        f"patience={config['patience']}。",
        "MLP 無重建暖身；良惡交錯 batch、不過採樣；family 使用隨機打亂、不過採樣。"
        "良惡獨立以 validation AUPRC 選模，family 獨立以 validation macro-F1 選模；"
        "AE 使用兩任務綜合分數。這是單任務分類效用對照，不是只有移除重建項的嚴格消融實驗。",
        f"MLP 良惡選定 epoch {selections['binary']['epoch']}，"
        f"family 選定 epoch {selections['family']['epoch']}；"
        f"實際完成 epochs={len(history['binary'])}/{len(history['family'])}。",
        f"MLP 良惡 train/val={selections['binary']['train_count']}/"
        f"{selections['binary']['validation_count']}；"
        f"family train/val={selections['family']['train_count']}/"
        f"{selections['family']['validation_count']}。"
        f"參數數量 binary/family={selections['binary']['parameter_count']}/"
        f"{selections['family']['parameter_count']}。",
        f"MLP 門檻 **{bundle['threshold']:.10f}**，依 validation 的相同 "
        f"FPR ≤{100 * config['target_fpr']:.2f}% 規則獨立選定，沒有沿用 AE 的數值門檻。",
        "測試揭露狀態："
        + (
            "**歷史實驗補跑**；AE test 結果此前已揭露。MLP 超參數沿用既定設定，"
            "選模仍只看 validation，但不能視為全程盲測。"
            if "retrospective" in bundle["test_exposure"]
            else "兩個 MLP 的 checkpoint 與門檻均在讀取 test 前固定。"
        ),
        "",
        "| 集合 | 模型 | AUROC | AUPRC | Recall | FPR | Family macro-F1 | "
        "Balanced accuracy | 偵測且家族正確 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ["train", "validation", "test"]:
        for model, values in [("AE", ae[split]), ("MLP", mlp[split])]:
            keys = [
                "auroc",
                "auprc",
                "malicious_recall",
                "fpr",
                "family_macro_f1",
                "family_balanced_accuracy",
                "end_to_end_accuracy",
            ]
            cells = ["N/A" if values[key] is None else f"{values[key]:.4f}" for key in keys]
            lines.append(f"| {split} | {model} | {' | '.join(cells)} |")
    test, base = ae["test"], mlp["test"]
    lines.extend(
        [
            "",
            f"Test MLP 相對 AE：AUPRC 差 **{base['auprc'] - test['auprc']:+.4f}**，"
            "recall 差 **"
            f"{100 * (base['malicious_recall'] - test['malicious_recall']):+.2f} 個百分點**，"
            f"family macro-F1 差 **{base['family_macro_f1'] - test['family_macro_f1']:+.4f}**。"
            "差值只描述本次單種子實驗，不代表統計顯著性；門檻效用必須連同誤報與區間判讀。",
            "",
            "### MLP 誤報不確定性與混淆計數",
            "",
            "| 集合 | TP | FN | TN | FP | FPR 95% CI | 群組誤報／良性群組 |",
            "|---|---:|---:|---:|---:|---|---:|",
        ]
    )
    predictions = read_csv(source / "predictions.csv")
    for split in ["train", "validation", "test"]:
        binary_counts = Counter(
            (row["binary_label"], row["prediction"]) for row in predictions if row["split"] == split
        )
        cells = [
            str(binary_counts[pair])
            for pair in [("1", "malicious"), ("1", "benign"), ("0", "benign"), ("0", "malicious")]
        ]
        m = mlp[split]
        ci = "–".join(f"{v:.4f}" for v in m["fpr_interval"]) if m["fpr_interval"] else "N/A"
        lines.append(
            f"| {split} | {' | '.join(cells)} | {ci} | "
            f"{m['false_positive_groups']} / {m['benign_group_count']} |"
        )
    lines.extend(
        [
            "",
            "95% Clopper–Pearson 區間以樣本為單位，群組可能相關；零誤報不證明部署 FPR ≤1%。",
            "",
            "### Test 各家族比較（不先經 detector 篩選）",
            "",
            "| 家族 | 樣本／群組 | AE precision／recall／F1 | MLP precision／recall／F1 |",
            "|---|---:|---|---|",
        ]
    )
    for family in bundle["families"]:
        a, b = test["family_results"].get(family), base["family_results"].get(family)
        if a and b:
            av = " / ".join(f"{a[k]:.4f}" for k in ["precision", "recall", "f1"])
            bv = " / ".join(f"{b[k]:.4f}" for k in ["precision", "recall", "f1"])
            lines.append(f"| {family} | {b['samples']} / {b['groups']} | {av} | {bv} |")
    lines.extend(
        [
            "",
            "### MLP Test 家族混淆矩陣",
            "",
            "| 真實＼預測 | " + " | ".join(bundle["families"]) + " |",
            "|---|" + "---:|" * len(bundle["families"]),
        ]
    )
    for family in bundle["families"]:
        counts = Counter(
            row["family_head_prediction"]
            for row in predictions
            if row["split"] == "test"
            and row["family_label"] == family
            and int(row["family_target"]) >= 0
            and row["family_known"] == "1"
        )
        lines.append(f"| {family} | {' | '.join(str(counts[f]) for f in bundle['families'])} |")
    lines.extend(
        [
            "",
            "### MLP Epoch 走勢",
            "",
            "| 任務 | Epoch | train loss | validation 選模分數 | "
            "validation recall | validation FPR |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for task in ["binary", "family"]:
        for entry in history[task]:
            suffix = " **選定**" if entry["epoch"] == selections[task]["epoch"] else ""
            recall = f"{entry['recall']:.4f}" if "recall" in entry else "N/A"
            fpr = f"{entry['fpr']:.4f}" if "fpr" in entry else "N/A"
            lines.append(
                f"| {task} | {entry['epoch']}{suffix} | {entry['loss']:.4f} | "
                f"{entry['validation_score']:.4f} | {recall} | {fpr} |"
            )
    lines.extend(
        [
            "",
            f"共同 split SHA-256 `{bundle['split_manifest_sha256']}`；"
            f"共同 embedding SHA-256 `{bundle['embeddings_sha256']}`。",
            "MLP 的 checkpoint、門檻、逐 epoch 紀錄與逐樣本結果保存於模型／實驗的 `mlp_baseline/`。"
            "MLP 不具重建或因素表示，這些指標不填入虛構數值；UMAP 仍展示 AE 分支。",
            "",
        ]
    )
    return lines
