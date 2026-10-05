"""Self-contained human-readable reports generated from frozen experiment results."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from pe_research.data.io import load_json, read_csv


def number(value: Any, digits: int = 4) -> str:
    return "N/A" if value is None else f"{float(value):.{digits}f}"


def write_report(run: Path) -> Path:
    """Regenerate the report without training or reopening test evaluation."""
    metrics = load_json(run / "metrics.json")
    metrics = {split: metrics[split] for split in ["train", "validation", "test"]}
    config = load_json(run / "resolved_config.json")
    prepared = load_json(run / "prepared.json")
    quality = load_json(run / "quality_report.json")
    environment = load_json(run / "environment.json")
    runtime = load_json(run / "runtime.json")
    frozen = load_json(run / "evaluation_frozen.json")["bundle"]
    status = load_json(run / "pilot_status.json")
    history = load_json(run / "history.json")["epochs"]
    gradients = load_json(run / "gradients.json")["samples"]
    rows = read_csv(run / "split_manifest.csv")
    families = frozen["families"]
    selection_epoch = max(
        (e for e in history if e["phase"] == "joint"),
        key=lambda e: e["validation"]["selection_score"],
    )["epoch"]
    val, test = metrics["validation"], metrics["test"]
    lines = [
        "# API 雙分支 AE Pilot 實驗報告",
        "",
        f"實驗：`{run.name}`。本報告與 [互動式 UMAP](visualization/latent_3d.html) "
        "提供完整的實驗判讀；其餘檔案僅供重現、稽核或後續推論。",
        "",
        "## 結果判讀",
        "",
        f"- 研究流程成功條件：**{'達成' if status['research_success'] else '未全部達成'}**。"
        "未訂定產品最低 recall／family F1，不代表部署驗收通過。",
        f"- AE Test AUROC **{number(test['auroc'])}**、AUPRC **{number(test['auprc'])}**；"
        f"固定門檻 recall **{number(100 * test['malicious_recall'], 2)}%**，"
        f"family macro-F1 **{number(test['family_macro_f1'])}**。"
        "排序能力與門檻下的偵測能力必須分開評估。",
        f"- Test {test['benign_count']} 筆良性中 {test['false_positives']} 筆誤報；"
        f"95% FPR 區間為 **{number(100 * test['fpr_interval'][0], 2)}%–"
        f"{number(100 * test['fpr_interval'][1], 2)}%**。零誤報不能證明部署 FPR ≤1%。",
        f"- Validation 聯合／瓶頸重建相對常數預測誤差為 "
        f"**{number(val['joint_constant_ratio'])}／{number(val['bottleneck_constant_ratio'])}**；"
        "小於 1 表示優於 train 均值常數預測。",
        "- 低 FPR recall 與家族分類仍需改善；圖上分群不能取代這些任務指標。",
        "",
        "## 資料、品質與切分",
        "",
        f"來源：`{prepared['source']}`；單一 Zenodo／SmartVMI 來源。"
        "來源與良惡兩類並未完全重合，但尚無跨來源泛化驗證。",
        "",
        f"Embedding：`{prepared['embedding_metadata']['model_name']}`，768 維 CLS、"
        f"frozen inference；revision `{prepared['embedding_metadata']['model_revision']}`。"
        "僅使用有序 API 名稱，不含參數或事件時間特徵；上游預訓練資料重疊未知。",
        "",
        "| 集合 | 樣本 | 群組 | 良性 | 惡性 | 可信字典內家族 | 排除家族樣本 | 字典覆蓋率 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split, m in metrics.items():
        lines.append(
            f"| {split} | {m['samples']} | {m['groups']} | {m['benign_count']} | "
            f"{m['malicious_count']} | {m['family_evaluated_count']} | "
            f"{m['family_excluded_count']} | {number(m['family_dictionary_coverage'])} |"
        )
    groups: dict[str, set[str]] = {}
    for row in rows:
        groups.setdefault(row["group_id"], set()).add(row["split"])
    cross = sum(len(splits) > 1 for splits in groups.values())
    norm = quality["normalization"]
    lines.extend(
        [
            "",
            f"共 {len(rows)} 筆、{quality['group_count']} 個群組；"
            f"{quality['duplicate_groups']} 個同 SHA／同 API 序列群組，跨 split 群組 **{cross}**。",
            f"切分 seed={config['split_seed']}，目標 70／15／15；"
            "群組大小降序、seed hash 打破同分，以整體與 stratum 的平方偏差配置。"
            "保留同 SHA／相同完整 API 名稱序列的連通群組；未全面辨識近重複。",
            f"原始向量 norm 範圍 {norm['norm_min']:.8f}–{norm['norm_max']:.8f}；"
            f"{norm['outside_original_tolerance']} 筆超過 1e-4 偏差。"
            f"{norm['renormalized_count']} 筆做 float32 逐向量 L2 正規化，"
            "僅接受偏差 ≤1e-3，原資料不改寫。",
            f"來源稽核原始錯誤：{'; '.join(quality['resolved_errors']) or '無'}。"
            f"修正後有效：{quality['valid']}；其他品質錯誤會阻擋訓練。",
            "JSON 換行等價驗證："
            + (", ".join(quality.get("snapshot_json_newline_equivalent", [])) or "無")
            + "。"
            "CSV 位元組 checksum 維持嚴格比對。",
            f"未知良惡排除數：{len(quality.get('excluded_unknown_binary_ids', []))}；"
            f"重複序列家族標籤衝突群組：{len(quality['label_conflicts'])}。",
        ]
    )
    for conflict in quality["label_conflicts"]:
        selected = [r for r in rows if r["sample_id"] in conflict]
        labels = ", ".join(sorted({r["family_label"] or "benign" for r in selected}))
        lines.append(f"- 同輸入卻標籤不同：{', '.join(conflict)}；標籤：{labels}。已置於同 split。")
    lines.extend(
        [
            "",
            "### 各家族支持數",
            "",
            "| 家族 | train 樣本／群組 | validation 樣本／群組 | test 樣本／群組 |",
            "|---|---:|---:|---:|",
        ]
    )
    for family in families:
        cells = []
        for split in ["train", "validation", "test"]:
            subset = [r for r in rows if r["split"] == split and r["family_label"] == family]
            cells.append(f"{len(subset)} / {len({r['group_id'] for r in subset})}")
        lines.append(f"| {family} | {' | '.join(cells)} |")
    lines.extend(
        [
            "",
            "## 模型、訓練與選模",
            "",
            "Encoder 768→256→128；因素及重建輔助各 128→64→32；"
            "分類僅讀取 sigmoid 因素 c。聯合 decoder 64→128→768；"
            "瓶頸 decoder 32→64→768；hidden GELU，無 skip connection。",
            "",
            f"Loss = {config['binary_weight']} × BCE + {config['family_weight']} × masked CE + "
            f"{config['joint_weight']} × normalized joint MSE + "
            f"{config['bottleneck_weight']} × normalized bottleneck MSE。",
            f"固定 train 變異量 V={prepared['variance']:.10g}，以 float64 計算；"
            "兩項 MSE 按 batch 與 768 維平均，使用同一輸入與重建目標。",
            f"seed={config['seed']}；裝置={runtime['device']}；threads={runtime['threads']}；"
            f"batch={config['batch_size']}；AdamW lr={config['learning_rate']}、"
            f"weight decay={config['weight_decay']}；global clipping={config['gradient_clip']}。",
            f"暖身 {config['warmup_epochs']} epochs；聯合訓練上限 {config['epochs']}；"
            f"patience={config['patience']}；實際完成 {len(history)} epochs，"
            f"其中聯合訓練 {sum(e['phase'] == 'joint' for e in history)} epochs。"
            f"選定第 **{selection_epoch}** epoch（含暖身）。",
            "train 良惡交錯抽 batch，不反覆過採樣小家族；validation/test 不重採樣。",
            "以 (AUPRC + 0.5 × family macro-F1) / 1.5 選模並早停，同分保留較早 epoch。",
            f"固定門檻 **{frozen['threshold']:.10f}**，score ≥ 門檻判惡性。"
            f"僅在 validation 以 FPR ≤{100 * config['target_fpr']:.2f}% 中最高 recall 選定；"
            "同 recall 選較低 FPR，再選較高門檻；test 未參與選模或調參。",
            "",
            "### Epoch 訓練走勢",
            "",
            "| Epoch | 階段 | train BCE | family CE | 聯合 raw MSE | 瓶頸 raw MSE | "
            "val 選模分數 | val recall | val family F1 | val 聯合／瓶頸常數比 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for e in history:
        m, loss = e["validation"], e["losses"]
        suffix = " **選定**" if e["epoch"] == selection_epoch else ""
        lines.append(
            f"| {e['epoch']}{suffix} | {e['phase']} | {number(loss['binary'])} | "
            f"{number(loss['family'])} | {e['raw_joint_mse']:.6g} | "
            f"{e['raw_bottleneck_mse']:.6g} | {number(m['selection_score'])} | "
            f"{number(m['malicious_recall'])} | {number(m['family_macro_f1'])} | "
            f"{number(m['joint_constant_ratio'])} / {number(m['bottleneck_constant_ratio'])} |"
        )
    lines.extend(
        [
            "",
            "## 固定 checkpoint 的任務表現",
            "",
            "| 集合 | AUROC | AUPRC | AUPRC 無資訊參考 | Recall | FPR | "
            "Family macro-F1 | Family balanced accuracy | 偵測且家族正確 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for split, m in metrics.items():
        keys = [
            "auroc",
            "auprc",
            "auprc_no_information",
            "malicious_recall",
            "fpr",
            "family_macro_f1",
            "family_balanced_accuracy",
            "end_to_end_accuracy",
        ]
        lines.append(f"| {split} | {' | '.join(number(m[k]) for k in keys)} |")
    lines.extend(
        [
            "",
            "Family 指標只對真實惡性、可信且 train 字典內樣本計算，**不先經 detector 篩選**。"
            "評估 family 集合：" + ", ".join(families) + "。"
            "「偵測且家族正確」分母包含漏檢的可信字典內惡性樣本。",
            "",
            "### FPR 支持數與不確定性",
            "",
            "| 集合 | 良性筆數 | 誤報數 | FPR 95% CI | 解析度 | 良性群組 | 誤報群組 | "
            "群組 FPR | 群組 95% CI |",
            "|---|---:|---:|---|---:|---:|---:|---:|---|",
        ]
    )
    for split, m in metrics.items():

        def ci(key: str, data: dict[str, Any] = m) -> str:
            return "–".join(number(v) for v in data[key]) if data[key] else "N/A"

        lines.append(
            f"| {split} | {m['benign_count']} | {m['false_positives']} | "
            f"{ci('fpr_interval')} | {number(m['fpr_resolution'])} | "
            f"{m['benign_group_count']} | "
            f"{m['false_positive_groups']} | {number(m['group_fpr'])} | "
            f"{ci('group_fpr_interval')} |"
        )
    lines.extend(
        [
            "",
            "區間採 95% Clopper–Pearson；樣本區間假設獨立，但同群可能相關。"
            "群組誤報定義為該良性群組任一樣本被判惡性。"
            "少量良性樣本下的 1% 是 validation 經驗約束，不是部署 FPR 保證。",
            "",
            "### 各家族分類結果",
            "",
            "| 家族 | 集合 | 樣本／群組 | Precision | Recall | F1 |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for family in families:
        for split, m in metrics.items():
            f = m["family_results"].get(family)
            if f:
                lines.append(
                    f"| {family} | {split} | {f['samples']} / {f['groups']} | "
                    f"{number(f['precision'])} | {number(f['recall'])} | {number(f['f1'])} |"
                )
    predictions = read_csv(run / "predictions.csv")
    lines.extend(
        [
            "",
            "### 良惡判定混淆計數",
            "",
            "| 集合 | TP（惡性判惡） | FN（惡性漏檢） | TN（良性判良） | FP（良性誤報） |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for split in ["train", "validation", "test"]:
        subset = [r for r in predictions if r["split"] == split]
        binary_counts = Counter((r["binary_label"], r["prediction"]) for r in subset)
        binary_cells = [
            binary_counts[pair]
            for pair in [("1", "malicious"), ("1", "benign"), ("0", "benign"), ("0", "malicious")]
        ]
        lines.append(f"| {split} | {' | '.join(str(value) for value in binary_cells)} |")
    lines.extend(
        [
            "",
            "### Test 家族混淆矩陣（包含 detector 漏檢）",
            "",
            "列為真實家族，欄為 family head 預測；此處不使用 detector gating。",
            "",
            "| 真實＼預測 | " + " | ".join(families) + " |",
            "|---|" + "---:|" * len(families),
        ]
    )
    for family in families:
        counts = Counter(
            r["family_head_prediction"]
            for r in predictions
            if r["split"] == "test" and r["family_label"] == family and r["family_correct"] != "N/A"
        )
        lines.append(f"| {family} | {' | '.join(str(counts[f]) for f in families)} |")
    lines.extend(
        [
            "",
            "## 內容保留與表徵診斷",
            "",
            "| 集合 | 常數 baseline MSE | 聯合 raw MSE | 聯合校正 MSE | 聯合／常數 | "
            "瓶頸 raw MSE | 瓶頸校正 MSE | 瓶頸／常數 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for split, m in metrics.items():
        keys = [
            "constant_prediction_mse",
            "joint_mse",
            "joint_normalized_mse",
            "joint_constant_ratio",
            "bottleneck_mse",
            "bottleneck_normalized_mse",
            "bottleneck_constant_ratio",
        ]
        lines.append(f"| {split} | {' | '.join(number(m[k], 8) for k in keys)} |")
    lines.extend(
        [
            "",
            "校正 MSE 的分母固定為 train 變異量；常數比則以該集合相對 train 平均向量的誤差為分母。"
            "常數比為 N/A 表示分母為零。低 MSE 不證明稀有行為證據已保留。",
            "",
            "| 集合 | c 變異 min／mean／max | z_r 變異 min／mean／max | c 飽和比例 | 全部崩塌 |",
            "|---|---|---|---:|---|",
        ]
    )
    for split, m in metrics.items():
        diag = m["representation"]
        summaries = [
            " / ".join(f"{value:.6g}" for value in [np.min(v), np.mean(v), np.max(v)])
            for v in [diag["c_variance"], diag["z_r_variance"]]
        ]
        lines.append(
            f"| {split} | {summaries[0]} | {summaries[1]} | "
            f"{number(diag['c_saturation_fraction'])} | {diag['c_collapsed']} |"
        )
    lines.extend(
        [
            "",
            "### 各因素高維診斷（固定 checkpoint，validation）",
            "",
            "| 因素 | c 變異 | 良性類內變異 | 惡性類內變異 | z_r 同維變異 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    diag = val["representation"]
    for i in range(32):
        lines.append(
            f"| factor_{i + 1:02} | {diag['c_variance'][i]:.6g} | "
            f"{diag['c_within_class_variance']['0'][i]:.6g} | "
            f"{diag['c_within_class_variance']['1'][i]:.6g} | {diag['z_r_variance'][i]:.6g} |"
        )
    lines.extend(
        [
            "",
            "### 各加權 loss 的梯度力度",
            "",
            f"每 {config['gradient_interval']} batches、global clipping 前，"
            "量測候選因素與重建輔助投影末層權重的 norm；下表跨聯合訓練 batches 彙整。"
            "家族空 batch 不納入。數值相近不代表力度相同，也不要求各項相等。",
            "",
            "| Loss | 有效紀錄 | factor norm mean／median／max | residual norm mean／median／max |",
            "|---|---:|---|---|",
        ]
    )
    for term in ["binary", "family", "joint", "bottleneck"]:
        records = [r[term] for r in gradients if term in r]
        summary = []
        for branch in ["factor", "residual"]:
            values = [r[branch] for r in records]
            summary.append(
                " / ".join(
                    number(value, 6)
                    for value in [np.mean(values), np.median(values), np.max(values)]
                )
                if values
                else "N/A"
            )
        lines.append(f"| {term} | {len(records)} | {summary[0]} | {summary[1]} |")
    lines.extend(
        [
            "",
            f"分類頭更新：binary={status['classification_heads_updated']['binary']}，"
            f"family={status['classification_heads_updated']['family']}。"
            "非零梯度僅表示路徑參與更新，不證明兩分支資訊獨立。",
            "",
            "## UMAP 使用與投影設定",
            "",
            "[開啟離線互動 UMAP](visualization/latent_3d.html)：c、z_r、combined 三視圖；"
            "真實良惡或良性＋family 著色；train／validation／test 篩選。"
            "Hover 包含樣本／群組、真實標籤、分數與預測正確性；切換不改變座標。",
            "",
        ]
    )
    projection_path = run / "visualization/projection.json"
    if projection_path.exists():
        projection = load_json(projection_path)
        lines.extend(["| 視圖 | train 擬合數 | 保留／排除維度 | 狀態 |", "|---|---:|---:|---|"])
        for view, info in projection["views"].items():
            lines.append(
                f"| {view} | {info['train_fit_count']} | "
                f"{len(info['retained_dimensions'])} / {len(info['excluded_dimensions'])} | "
                f"{info['status']} |"
            )
        lines.append("")
    else:
        lines.append("UMAP 尚未產生；完成 visualize 後報告會自動更新。")
    lines.extend(
        [
            "Scaler 僅在 train 擬合，std ≤1e-6 的維度排除；UMAP 只 fit train、其他集合 transform。"
            "無監督 3D、neighbors=min(15,N_train−1)、min_dist=0.1、euclidean、random init、"
            "random_state=42、transform_seed=42、n_jobs=1。",
            "不同視圖軸向與距離不可直接比較；transform 不保證保留新分布群結構。"
            "分群不等於解耦成功；因素不是命名概念、概念機率或因果解釋。",
            "",
            "## 重現、驗證與保存策略",
            "",
            f"Git commit `{environment['git_commit']}`；"
            f"dirty working tree={'有' if environment['git_status'] else '無'}。"
            f"lock SHA-256 `{environment['lock_sha256']}`。",
            "套件版本：" + "；".join(f"{k}={v}" for k, v in environment["packages"].items()) + "。",
            "Embedding CSV SHA-256 `"
            + prepared["input_checksums"]["sample_embeddings_gte_modernbert768.csv"]
            + "`。",
            f"Checkpoint SHA-256 `{frozen['checkpoint_sha256']}`；"
            f"inference SHA-256 `{frozen['inference_sha256']}`。",
        ]
    )
    verification_path = run / "verification.json"
    if verification_path.exists():
        checks = load_json(verification_path)
        lines.append(
            f"驗證：pytest {checks['pytest']['passed']} passed、"
            f"{checks['pytest']['skipped']} skipped（VT live opt-in）；"
            f"ruff={checks['ruff']}；mypy={checks['mypy']}；lock={checks['lock_check']}。"
        )
        lines.append(
            f"AE／推論模型最大分數差={checks['inference_max_abs_score_difference']}；"
            f"跨 split 群組={checks['cross_split_groups']}。"
        )
        if checks.get("browser"):
            lines.append("瀏覽器驗證：離線 WebGL 正常、控制項可切換、篩選後座標不變、無 JS 例外。")
    lines.extend(
        [
            "保留模型權重、字典、門檻、切分、設定、資料／程式指紋、"
            "程式快照、逐 epoch logs、gradients、predictions 與 UMAP reducer／座標，"
            "以支援重現與診斷；這些檔案不必逐一閱讀。"
            "原始資料集不改寫、不刪除；報告與 UMAP 是主要閱讀入口。",
            "",
            "後續改善應先檢查 validation 的家族混淆、低 FPR recall 與梯度取捨；"
            "另建 revision，再凍結設定後測試，不以目前 test 調參。",
            "",
        ]
    )
    output = run / "PILOT_REPORT.md"
    from pe_research.experiment.baseline_report import comparison_sections
    from pe_research.experiment.optimization_report import optimization_sections

    insertion = lines.index("## 資料、品質與切分")
    lines[insertion:insertion] = comparison_sections(run, metrics)
    lines[insertion:insertion] = optimization_sections(run)
    output.write_text("\n".join(lines), encoding="utf-8")
    return output
