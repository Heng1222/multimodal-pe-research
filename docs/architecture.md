# Repository architecture

## 邊界

專案將「可版本控制的研究定義」與「大型、敏感或可再生的產物」分離：

- `src/`、`configs/`、`tests/`、`docs/`：提交 Git。
- `data/`、`model/`、`experiment/`：只保留目錄用途說明，實際產物留在本機或未來的 artifact store。

## 模組責任

### Data

統一 sample identity、schema、provenance、stage state 與 feature registry。良性與惡意樣本必須經過相同 pipeline 與分析環境，避免來源或工具差異成為模型捷徑。

### Model

只消費由 feature registry 明確標為 `model_eligible` 的特徵。資料來源、簽章、VirusTotal label、collection time 與 pipeline 狀態等欄位保留供稽核，但預設不得進入模型。

### Experiment

每次 run 應記錄 dataset snapshot、config hash、random seed、Git commit、dependency lock、metrics 與產物位置。實驗輸出不直接提交 Git；可重現定義必須提交。

## 建議的後續演進

先完成資料 pipeline 的 schema 與 CLI，再依實際需求加入訓練框架、實驗追蹤工具與外部 artifact storage。這樣可避免在資料契約尚未穩定前過早綁定重量級 ML dependencies。
