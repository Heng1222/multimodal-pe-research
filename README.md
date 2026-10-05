# Multimodal PE Research

以可重現、安全且可擴充的方式管理多模態 PE 資料管線、模型與實驗。

目前包含 AU-PEMal 與 API Traces 兩套彼此獨立的資料 pipeline；後續模型訓練與實驗沿用相同的 sample identity、schema、設定與產物追蹤方式。

## 快速開始

本專案以 [uv](https://docs.astral.sh/uv/) 管理 Python 3.11、`.venv`、dependencies 與 lockfile。新 Windows 主機只需先用 Git clone repository，然後在專案根目錄執行：

```powershell
powershell -ExecutionPolicy Bypass -File .\bootstrap.ps1
```

[`bootstrap.ps1`](bootstrap.ps1) 可重複安全執行，會依序：

1. 找不到 uv 時，從 Astral 官方 installer 下載並安裝 uv。
2. 由 uv 下載 `.python-version` 指定的 Python 3.11，不依賴系統 Python。
3. 初始化 Git LFS 並執行 `git lfs pull`，取得 dataset CSV、embeddings 與互動 HTML。
4. 使用 committed `uv.lock` 執行 `uv sync --extra embedding --extra eda --dev --locked`。
5. 建立/更新 `.venv`、editable-install 本專案，並用 `pe-research --help` 做 smoke check。

腳本成功後不需要手動 activate，直接使用 `uv run --locked` 即會在專案環境執行：

```powershell
uv run --locked pe-research --help
uv run --locked pytest
uv run --locked ruff check .
uv run --locked mypy
```

若偏好傳統 activated shell，請在 PowerShell 以 dot-source 執行，讓 activation 保留在目前 shell：

```powershell
. .\bootstrap.ps1 -Activate
pe-research --help
pytest
```

Git for Windows 是 clone repository 的前置需求，而且通常已包含 Git LFS。若只想同步程式、不下載 LFS dataset，可使用 `-SkipLfs`；一般研究環境不建議略過。

新增套件時不要直接使用 `pip install`：

```powershell
uv add <package>
uv add --dev <development-package>
uv remove <package>
uv lock --check
```

`uv.lock` 應提交至 Git，讓資料處理、訓練與實驗環境可重現。

## 專案結構

```text
.
├── configs/                  # 可版本控制的資料、模型與實驗設定
│   ├── data/
│   ├── model/
│   └── experiment/
├── data/                     # 版本化的重要 dataset 輸出；raw/cache 仍只留本機
├── model/                    # checkpoint、權重與匯出模型（內容不進 Git）
├── experiment/               # run logs、metrics 與實驗產物（內容不進 Git）
├── docs/                     # 架構、schema、安全與操作文件
├── bootstrap.ps1             # 新 Windows 主機的一鍵環境還原
├── src/pe_research/
│   ├── data/                 # ingest、validation、analysis pipeline
│   ├── model/                # 模型、loss 與訓練元件
│   └── experiment/           # 實驗編排、評估與追蹤
└── tests/                    # 對應上述模組的測試
```

`data/` 會以一般 Git 與 Git LFS 追蹤正式 CSV、labels、embeddings、品質報告和離線互動 HTML；raw archives、真實 PE、VT responses、cache、`.part`、checkpoint 與 secrets 不進 Git。完整清單與跨主機操作方式見 [`data/README.md`](data/README.md)。

## 階段規劃

1. 資料層：實作 canonical manifest、ingest、hash 去重、PE validation、static analysis、report 與 snapshot。
2. Adapter 層：加入 Ghidra、VirusTotal hash lookup 與隔離 sandbox 的安全介面及 mocks。
3. 模型層：定義 feature registry、資料切分、model input contract、訓練與評估。
4. 實驗層：版本化 config、seed、dataset snapshot、commit、metrics 與模型產物。

## 安全原則

- 不在開發主機執行待分析 PE。
- 真實 PE bytes、hash inventory、SQLite manifest、secrets 與完整本機來源路徑不得提交 Git。
- 動態分析只能經明確設定的隔離 sandbox adapter；未設定時安全地標記為 unavailable/skipped。
- VirusTotal 預設只做 hash lookup，未命中時不自動上傳。

## 第一版 API embedding 模型

已提供 768 維雙分支 AE 的資料準備、訓練、評估、推論與離線 3D UMAP，詳見 [操作文件](docs/api_model_pilot.md)。訓練依賴以 `uv sync --locked --extra training --extra eda --dev` 安裝。

直接執行 `uv run --locked pe-research experiment run`，自動建立資料夾並完成整個實驗，不需手動填入 run 路徑。閱讀該資料夾的 `PILOT_REPORT.md` 與 UMAP 即可判讀結果；既有結果可用 `uv run --locked pe-research experiment report` 重新彙整。

每次實驗會自動用相同 embedding、Train／validation／test 切分與家族字典，訓練良惡與 family 的獨立 MLP 對照組；比較結果一併寫入報告。歷史實驗可直接執行 `uv run --locked pe-research experiment baseline` 補跑對照組。
