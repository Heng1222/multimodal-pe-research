# Repository architecture

## 邊界

專案將「可版本控制的研究定義」與「大型、敏感或可再生的產物」分離：

- `src/`、`configs/`、`tests/`、`docs/`：提交 Git。
- `data/`：正式資料產物依 `data/README.md` 使用 Git／Git LFS；raw、cache、`.part` 留在本機。
- `model/`、`experiment/`：checkpoint、metrics、logs 與模型 UMAP 留在本機；程式與設定提交 Git。

## 模組責任

### Data

`src/pe_research/data/` 依 dataset 分成兩個同等層級的 package，名稱與根目錄 `data/` 的分類一致：

- `au_pemal_2025/`：AU-PEMal 設定、採樣、VirusTotal enrichment、event embedding、sequence loader、EDA 與驗證。
- `api_traces_malware_detection/`：API traces 設定、下載、正規化、sample embedding、訓練資料讀取與群組切分、EDA 與驗證。

兩個 dataset 與 experiment 共用的檔案讀寫工具保留在 `data/io.py`。Dataset 設定由各自 package 匯入；CLI 指令與 `configs/data/` 的檔名維持原有介面，產物仍依設定中的 workspace 讀寫。

統一 sample identity、schema、provenance、stage state 與 feature registry。良性與惡意樣本必須經過相同 pipeline 與分析環境，避免來源或工具差異成為模型捷徑。

### Model

只消費由 feature registry 明確標為 `model_eligible` 的特徵。資料來源、簽章、VirusTotal label、collection time 與 pipeline 狀態等欄位保留供稽核，但預設不得進入模型。

### Experiment

每次 run 應記錄 dataset snapshot、config hash、random seed、Git commit、dependency lock、metrics 與產物位置。實驗輸出不直接提交 Git；可重現定義必須提交。

## 建議的後續演進

先完成資料 pipeline 的 schema 與 CLI，再依實際需求加入訓練框架、實驗追蹤工具與外部 artifact storage。這樣可避免在資料契約尚未穩定前過早綁定重量級 ML dependencies。

## 第一版 API 雙分支模型

使用既有 768 維 frozen GTE sample embeddings；data adapter 負責 join、正規化與群組切分，model 負責網路與 loss，experiment 負責 run、選模、評估與圖表。自動建立實驗資料夾；主要閱讀入口為完整 `PILOT_REPORT.md` 與離線 UMAP。操作見 [API Pilot 文件](api_model_pilot.md)。
