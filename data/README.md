# Data workspace

此目錄供本機 dataset workspace、raw samples 與衍生資料使用。除本說明與 `.gitignore` 外，內容預設不提交 Git。

第一階段 pipeline 預計建立 `manifest.sqlite`、`raw/`、`artifacts/`、`reports/`、`snapshots/` 與 `logs/`。原始 PE 視為 immutable，衍生產物必須能由樣本、設定與工具版本重建。
