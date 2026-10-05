# API embedding 雙分支 AE Pilot

第一版使用 `data/api_traces_malware_detection/pilot_v1/artifacts/` 的正式 768 維 GTE CLS CSV；不讀未完成的 `.npy.part`，不重新執行 Transformer，也不改寫原始資料。

## 操作

在 repository 根目錄執行：

```powershell
uv sync --locked --extra training --extra eda --dev
uv run --locked pe-research experiment run
```

`run` 自動建立臺北時間命名的實驗及模型資料夾，完成 prepare、train、evaluate、visualize，最後印出報告與 UMAP 路徑。全程不需填入 run 路徑。

也可分階段執行，指令自動選取最新適用的實驗；`train` 沒有待訓練的最新實驗時會自行建立新實驗：

```powershell
uv run --locked pe-research experiment prepare
uv run --locked pe-research experiment train
uv run --locked pe-research experiment evaluate
uv run --locked pe-research experiment visualize
uv run --locked pe-research experiment report
```

`report` 直接將已保存結果彙整為完整報告，不重跑訓練或 test。已完成實驗的無參數 `evaluate`／`visualize` 會重用既有結果。仍可用 `--run` 指定歷史實驗。

推論預設自動使用最新已訓練模型及對應的正式 embedding／metadata，輸出到該實驗資料夾中的唯一時間戳檔案：

```powershell
uv run --locked pe-research experiment predict
```

輸出包含良惡分數、門檻判定、惡性樣本的已知 family 信心與 `factor_01`–`factor_32`。良性 family 為 N/A；信心不是校準機率或 unknown-family 拒識證據。

若要推論新資料，再用 `--embeddings`、`--metadata` 覆寫輸入；metadata 必須包含同版本模型資訊及 CSV checksum。

## 設定與可重現性

需要根據最新完成實驗校準參數時，在根目錄執行：

```powershell
uv run --locked pe-research experiment optimize
```

自動使用最新完成的實驗，固定同一 Train／validation／test、seed 和架構，
比較重建權重減半、family 權重加倍、兩者合併、合併且 learning rate 減半四組候選。
每組也訓練相同切分的良惡／family MLP。只依 validation 選擇，
要求兩項重建都優於常數預測且因素未全部崩塌；只評估選定新候選的 test 並產生 UMAP。
完成後自動更新預設 config，原設定另存 `_original.yaml`。
候選比較與選擇理由全部寫入選定實驗的 `PILOT_REPORT.md`，不需要手填資料夾。
已有 test 結果揭露時，新報告會說明重用 test 不是全新盲測。

- 模型設定：`configs/model/dual_branch_ae_v1.yaml`。目前只接受白皮書 v1.2 的 768 維適配架構。
- 實驗設定：`configs/experiment/api_dual_branch_pilot_v1.yaml`。預設 CPU、seed 42、暖身 2 epochs、聯合訓練最多 50 epochs、patience 8、batch 32。
- 2026-10-05 validation 校準後預設：BCE 權重 1、family CE 1、joint MSE 0.25、bottleneck MSE 0.0625，LR 保持 `1e-3`。原設定保留於 `configs/experiment/api_dual_branch_pilot_v1_original.yaml`。
- 相同 SHA 或完整 API-name 序列以連通群組綁定；群組大小降序、seed hash 打破同分，再最小化整體及各 stratum 的 normalized squared deviation。切分目標 70／15／15，群組完整性優先。
- family 字典只由 train 的可信惡性名稱建立。未知良惡排除並記錄；未知／字典外 family 不進 CE，仍參與良惡與重建。
- 逐向量 float32 L2 正規化只接受原 norm 偏差 ≤0.001；不放寬來源 CSV checksum。JSON 的 CRLF→LF 若可重現 snapshot checksum，記錄為換行等價。
- `prepare` 在 run-local audit view 重新執行完整資料品質檢查，不覆寫來源品質報告。只有可修正的 L2 異常可繼續，其他錯誤停止。
- train 統計量以 float64 計算；MSE 按 768 維平均並除以固定 train 變異量。禁止以 validation/test 重算統計量。
- train 不讀 test 表示或指標。`evaluate` 固定模型 bundle 後使用 test；結果完成後禁止重新評估同 run，修訂設定必須另建 run。
- `prepare` 保存 source checksums、resolved config、Git commit／dirty 狀態、lock hash、套件版本與 run-local 正規化資料。跨主機重現需使用同程式版本與 lock。
- run 與 model 目錄禁止覆寫。失敗 run 會保留供診斷；重新 prepare 建立新 run。評估中斷可在未產生 metrics 且 bundle 不變時重試。

## 交付位置

主要閱讀入口只有 **`PILOT_REPORT.md` 與 `visualization/latent_3d.html`**。報告包含資料品質與切分、設定與門檻、逐 epoch 走勢、任務指標與支持數、FPR 信賴區間、各家族結果與混淆矩陣、重建與表徵／梯度診斷、UMAP 規格，以及重現資訊。evaluate 與 visualize 後會自動更新。

底層產物保留供重現、稽核與推論，平常不必逐一查看；原始 dataset 不刪除。

`experiment/<run_id>/` 保存 `split_manifest.csv`、品質／來源稽核、`history.json`、`gradients.json`、`metrics.json`、`predictions.csv`、`representations.npz`、`pilot_status.json` 與 `PILOT_REPORT.md`。

`model/<run_id>/` 保存完整 `best.pt`（含 optimizer）、只有分類路徑的 `inference.pt`、`selection.json`、`bundle.json`、family 字典與 train 平均向量。模型及實驗產物維持 ignored，不進資料集的 Git LFS。

`experiment/<run_id>/visualization/latent_3d.html` 是內嵌 Plotly JavaScript 的單檔離線圖。三個表示視圖可切換真實標籤著色及 split，座標不隨切換變動。train-only scaler／reducer、座標、metadata、色票與版本一併保存。reducer pickle 僅供本機可信產物使用。

## 指標與限制

每次 `experiment run`／`train` 都自動訓練兩個獨立 MLP 對照組：良惡 768→256→128→1、family 768→256→128→K，hidden GELU。直接讀取與 AE 相同的正規化 embedding、Train／validation／test manifest、家族字典和 seed。良惡使用全部 train，family 只用可信字典內惡性 train；不重新切分或過採樣。

MLP 沿用 optimizer、learning rate、weight decay、batch、clipping、epoch 上限與 patience，不做重建暖身。良惡以 validation AUPRC、family 以 validation macro-F1 分別選 checkpoint；偵測門檻依同一 validation FPR 規則獨立選定。AE 與 MLP 比較、各家族結果、混淆矩陣及兩項 MLP 的逐 epoch 紀錄都寫入 `PILOT_REPORT.md`。MLP 不具重建目標，其重建與因素診斷不適用；UMAP 仍展示 AE 表示。

對已有的歷史實驗，執行 `uv run --locked pe-research experiment baseline`，自動在最新已訓練實驗補跑並更新報告，不重訓 AE 或重新切分。若 AE test 結果已揭露，報告會標明歷史補跑；MLP 仍只依 validation 選模。對照產物存於實驗／模型資料夾的 `mlp_baseline/`。

選模使用 `(AUPRC + 0.5 × family macro-F1) / 1.5`。門檻在 validation 以經驗 FPR ≤1% 的最大 recall 選擇，再依較低 FPR／較高門檻打破 ties。family 指標不經 detector 篩選；端到端結果包含漏檢分母。

FPR 回報 benign 數、false positives、解析度與 95% Clopper–Pearson 區間，並提供良性群組任一樣本誤報的群組結果。樣本層級區間有獨立性假設，群組可能相關；零誤報不證明部署 FPR ≤1%。

成功条件僅為分類頭有效更新、validation 兩項重建優於 train 均值常數預測、因素未全部崩塌。家族效果與低 FPR recall 仍需單獨檢視，未設定業務最低門檻，因此不宣稱部署驗收通過。

目前資料是單一來源、抽樣良惡比例 1:1、每家族樣本少。已知同序列跨 split 已消除，但未全面辨識近重複／跨環境差異；上游預訓練重疊未知。因素未具命名概念語意，重建與 UMAP 分群均不證明分支資訊互斥或因果解釋。

## 驗證

```powershell
uv run --locked pytest
uv run --locked ruff check .
uv run --locked mypy
uv lock --check
```

新增測試涵蓋資料契約、連通群組、防洩漏切分、mask／batch size 1、梯度路徑、門檻 ties、family 排除、端到端分母、checkpoint 還原、train-only 投影與 synthetic CLI 全生命週期。live VT 測試仍需明確 opt-in。
