# Multimodal PE Research

以可重現、安全且可擴充的方式管理多模態 PE 資料管線、模型與實驗。

目前的第一階段是建立「良性優先、未來可直接接入惡意 PE」的資料 pipeline；後續模型訓練與實驗會沿用相同的 sample identity、schema、設定與產物追蹤方式。詳細需求保留在 [`codex_implementation_prompt.md`](codex_implementation_prompt.md)。

## 快速開始

本專案以 [uv](https://docs.astral.sh/uv/) 管理 Python、虛擬環境、dependencies 與 lockfile，目標版本為 Python 3.11+。

```powershell
uv sync --dev
uv run pytest
uv run ruff check .
uv run mypy
```

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
├── data/                     # 本機資料集與 dataset workspace（內容不進 Git）
├── model/                    # checkpoint、權重與匯出模型（內容不進 Git）
├── experiment/               # run logs、metrics 與實驗產物（內容不進 Git）
├── docs/                     # 架構、schema、安全與操作文件
├── src/pe_research/
│   ├── data/                 # ingest、validation、analysis pipeline
│   ├── model/                # 模型、loss 與訓練元件
│   └── experiment/           # 實驗編排、評估與追蹤
└── tests/                    # 對應上述模組的測試
```

根目錄三個產物目錄只追蹤說明文件，不追蹤真實 PE、資料庫、模型權重或實驗輸出。可重現的設定與程式碼放在 `configs/` 和 `src/`。

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
