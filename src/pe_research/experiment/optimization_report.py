"""Summarize validation-only calibration in the main human-readable report."""

from pathlib import Path
from statistics import mean

from pe_research.data.io import load_json


def optimization_sections(run: Path) -> list[str]:
    path = run / "optimization.json"
    if not path.exists():
        return []
    audit = load_json(path)
    results = audit["results"]
    original = results[0]
    selected = next(r for r in results if r["name"] == audit["selected"])
    gradient_path = run.parent.parent / audit["base_run"] / "gradients.json"
    diagnosis = "依原實驗的分類與重建梯度診斷，"
    if gradient_path.exists():
        records = load_json(gradient_path)["samples"]
        if records:
            diagnosis = (
                "原設定的因素末層平均加權梯度："
                + "、".join(
                    f"{term} {mean(r[term]['factor'] for r in records):.4f}"
                    for term in ["binary", "family", "joint", "bottleneck"]
                )
                + "。依分類與重建的梯度差異，"
            )
    lines = [
        "## 本次參數優化：只依 validation 選擇",
        "",
        diagnosis + "預先固定四組候選，比較重建權重減半、family 加倍，以及 learning rate 減半。",
        "切分、正規化 embedding 與家族字典以 checksum 驗證完全相同；"
        "seed、架構、batch、訓練上限、早停及門檻規則維持相同。每組亦訓練兩個 MLP 對照組。",
        "",
        "| 設定 | LR | family / joint / bottleneck 權重 | Val AUPRC | "
        "Val family F1 | Val recall | 聯合 / 瓶頸常數誤差比 | 選模分數 | 合格 |",
        "|---|---:|---|---:|---:|---:|---|---:|---|",
    ]
    for result in results:
        c, v = result["config"], result["validation"]
        label = result["name"] + (" **選定**" if result["name"] == audit["selected"] else "")
        lines.append(
            f"| {label} | {c['learning_rate']:g} | "
            f"{c['family_weight']:g} / {c['joint_weight']:g} / {c['bottleneck_weight']:g} | "
            f"{v['auprc']:.4f} | {v['family_macro_f1']:.4f} | "
            f"{100 * v['malicious_recall']:.2f}% | "
            f"{v['joint_constant_ratio']:.4f} / {v['bottleneck_constant_ratio']:.4f} | "
            f"{v['selection_score']:.4f} | {'是' if result['eligible'] else '否'} |"
        )
    old, new = original["validation"], selected["validation"]
    lines.extend(
        [
            "",
            f"選定 `{audit['selected']}`；validation 選模分數 "
            f"{old['selection_score']:.4f} → **{new['selection_score']:.4f}**，"
            f"family macro-F1 {old['family_macro_f1']:.4f} → **{new['family_macro_f1']:.4f}**。",
            "以既定 `(AUPRC + 0.5 × family macro-F1) / 1.5` 最大值選擇，"
            "且要求兩項 validation 重建都優於 train 均值常數、因素未全部崩塌；"
            "同分保留原設定。選定設定、checkpoint、threshold 先凍結，才執行 test。",
            "未選定候選僅訓練及保存 validation 結果，未開啟其 test。"
            "四次比較仍可能對小型 validation 過擬合，本輪維持單種子研究範圍。",
            "**歷史 test 已在調參前揭露：本次重用 test 並非全新盲測。**"
            "它僅提供固定設定的描述性結果；獨立泛化驗證需要新 holdout 或新資料。",
            f"優化狀態：`{audit['status']}`。完成後的預設參數已寫入 "
            "`configs/experiment/api_dual_branch_pilot_v1.yaml`；原設定另存 `_original.yaml`。",
            "",
            "| 候選 | 實驗資料夾（保留 checkpoint、validation 與 MLP 訓練紀錄） |",
            "|---|---|",
        ]
    )
    for result in results:
        lines.append(f"| {result['name']} | `{result['run']}` |")
    lines.extend(
        [
            "",
            "### 各候選的 MLP validation 對照",
            "",
            "| 候選 | MLP Val AUPRC | MLP Val recall | MLP Val family F1 | 良惡 / family epoch |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for result in results:
        folder = Path(result["run"]).name
        bundle_path = run.parent.parent / "model" / folder / "mlp_baseline/bundle.json"
        if not bundle_path.exists():
            continue
        selections = load_json(bundle_path)["selections"]
        binary, family = selections["binary"], selections["family"]
        lines.append(
            f"| {result['name']} | {binary['validation_score']:.4f} | "
            f"{100 * binary['recall']:.2f}% | {family['validation_score']:.4f} | "
            f"{binary['epoch']} / {family['epoch']} |"
        )
    lines.extend(
        [
            "",
            "MLP 不使用 AE 的重建／family 加權，因此 LR 相同的候選得到相同對照結果。"
            "較低 LR 的候選遵循原 patience 規則提早停止；未為它額外放寬 epoch／patience。",
        ]
    )
    return [*lines, ""]
